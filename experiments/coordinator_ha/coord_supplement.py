# SPDX-License-Identifier: Apache-2.0
"""Supplementary coordinator fault experiments, closing gaps in the first run.

  sigterm      persistence on, a graceful SIGTERM (what a K8s rollout sends)
               instead of kill -9: does the clean-stop checkpoint keep E?
  long-window  no persistence, kill -9, then look again 120 s after recovery,
               and compare against what the MP servers themselves list in L2
  hang-<N>     SIGSTOP the coordinator for N seconds (port open, no answers),
               measure store latency on the MP servers during the hang, then
               SIGCONT and see what the coordinator makes of it

Run from the repo root with the venv active:
    python <this file> [--only sigterm|long-window|hang-15|hang-40|eviction] [--out f.json]
"""

# Standard
import argparse
import json
import pathlib
import shutil
import signal
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Third Party
import httpx

# First Party
from coord_crash_experiment import (  # noqa: E402
    CHECKPOINT_INTERVAL_S,
    MODEL,
    QUOTAS_GB,
    SEQS,
    Coordinator,
    KVClient,
    ProductionTimingServer,
    read_view,
    seq_settled,
    shared_l2,
    store,
    wait_for,
)

DOWNTIME_S = 8.0


class Fleet:
    """A coordinator, two MP servers sharing an fs L2, and their KV clients."""

    def __init__(self, name: str, persist: bool) -> None:
        self.workdir = pathlib.Path(tempfile.mkdtemp(prefix=f"sup-{name}-"))
        self.coord = Coordinator(self.workdir, persist)
        self.servers: dict = {}
        self.clients: dict = {}

    def __enter__(self) -> "Fleet":
        self.coord.start()
        pool = shared_l2(self.workdir / "shared-l2")
        for sid in ("mp-1", "mp-2"):
            self.servers[sid] = ProductionTimingServer(
                sid, self.workdir, self.coord.url, [pool]
            )
            self.servers[sid].start()
        self.clients = {
            sid: KVClient(s.rpc_url, MODEL) for sid, s in self.servers.items()
        }
        return self

    def __exit__(self, *exc) -> None:
        for c in self.clients.values():
            c.close()
        for s in self.servers.values():
            s.stop()
        if self.coord.proc and self.coord.proc.poll() is None:
            self.coord.proc.send_signal(signal.SIGCONT)
        self.coord.stop()
        shutil.rmtree(self.workdir, ignore_errors=True)

    @property
    def url(self) -> str:
        return self.coord.url

    def setup_ab(self) -> None:
        """Store A and B, set quotas, pin A, wait until the coordinator has them."""
        store(self.clients, "A")
        store(self.clients, "B")
        for salt, gb in QUOTAS_GB.items():
            httpx.put(f"{self.url}/quota/{salt}", json={"limit_gb": gb}).raise_for_status()
        httpx.post(
            f"{self.url}/cache/pins",
            json={"model_name": MODEL, "world_size": 1,
                  "token_ids": SEQS["A"][0], "cache_salt": "alice"},
        ).raise_for_status()
        wait_for(lambda: seq_settled(self.url, "A") and seq_settled(self.url, "B"), 15)

    def restart_and_wait(self) -> dict:
        out = {"restart_to_healthz_s": round(self.coord.start(), 2)}
        t = wait_for(lambda: read_view(self.url).instances == ["mp-1", "mp-2"], 60)
        out["healthz_to_all_reregistered_s"] = None if t is None else round(t, 2)
        return out

    def l2_keys_listed_by_server(self) -> int:
        """Objects the shared L2 actually holds.

        The fs adapter does not implement the MP server's listing API (503),
        so this counts the files in the pool directory instead.
        """
        return sum(1 for p in (self.workdir / "shared-l2").rglob("*") if p.is_file())

    def coordinator_l2_keys(self) -> int:
        page = httpx.get(f"{self.url}/directory/keys", params={"limit": 10000}).json()
        return sum(
            1 for row in page["keys"]
            if any(p["tier"] == "l2" for p in row["placements"])
        )

    def log_count(self, sid: str, needle: str) -> int:
        text = self.servers[sid].log_path.read_text(errors="replace")
        return text.count(needle)


def exp_sigterm() -> dict:
    """Persistence on; SIGTERM instead of kill -9."""
    r: dict = {"experiment": "sigterm"}
    with Fleet("sigterm", persist=True) as f:
        f.setup_ab()
        time.sleep(CHECKPOINT_INTERVAL_S + 0.5)
        store(f.clients, "E")
        wait_for(lambda: seq_settled(f.url, "E"), 15)
        e_seen_at = time.time()
        t0 = time.monotonic()
        f.coord.proc.send_signal(signal.SIGTERM)
        f.coord.proc.wait(30)
        r["sigterm_shutdown_s"] = round(time.monotonic() - t0, 2)
        r["exit_code"] = f.coord.proc.returncode
        r["checkpoint_written_after_E"] = f.coord.checkpoint.stat().st_mtime > e_seen_at
        store(f.clients, "C")
        time.sleep(max(0.0, DOWNTIME_S - (time.monotonic() - t0)))
        r.update(f.restart_and_wait())
        store(f.clients, "D")
        wait_for(lambda: seq_settled(f.url, "D"), 15)
        time.sleep(2)
        r["after_recovery"] = read_view(f.url).__dict__
    return r


def exp_long_window() -> dict:
    """No persistence; kill -9; check again 120 s later; compare to ground truth."""
    r: dict = {"experiment": "long-window"}
    with Fleet("long", persist=False) as f:
        f.setup_ab()
        store(f.clients, "E")
        wait_for(lambda: seq_settled(f.url, "E"), 15)
        f.coord.kill()
        t0 = time.monotonic()
        store(f.clients, "C")
        time.sleep(max(0.0, DOWNTIME_S - (time.monotonic() - t0)))
        r.update(f.restart_and_wait())
        store(f.clients, "D")
        wait_for(lambda: seq_settled(f.url, "D"), 15)
        time.sleep(2)
        r["view_at_+2s"] = read_view(f.url).__dict__
        r["coordinator_l2_keys_at_+2s"] = f.coordinator_l2_keys()
        time.sleep(120)
        r["view_at_+120s"] = read_view(f.url).__dict__
        r["coordinator_l2_keys_at_+120s"] = f.coordinator_l2_keys()
        r["l2_keys_listed_by_mp_server"] = f.l2_keys_listed_by_server()
        r["l2_files_on_disk"] = sum(
            1 for p in (f.workdir / "shared-l2").rglob("*") if p.is_file()
        )
    return r


def timed_stores(f: Fleet, n: int, base: int, every_s: float) -> list[float]:
    """Store n fresh 2-chunk sequences on mp-1, one every ``every_s``; latencies (ms)."""
    lat = []
    for i in range(n):
        t0 = time.monotonic()
        tokens = list(range(base + 10 * i, base + 10 * i + 8))
        f.clients["mp-1"].store(tokens, cache_salt="alice")
        lat.append((time.monotonic() - t0) * 1000)
        time.sleep(max(0.0, every_s - (time.monotonic() - t0)))
    return lat


def summarize(lat: list[float]) -> dict:
    return {
        "n": len(lat),
        "median_ms": round(statistics.median(lat), 1),
        "max_ms": round(max(lat), 1),
    }


def exp_hang(hang_s: int) -> dict:
    """SIGSTOP the coordinator for hang_s seconds; measure the data plane."""
    r: dict = {"experiment": f"hang-{hang_s}", "hang_s": hang_s}
    with Fleet(f"hang{hang_s}", persist=False) as f:
        f.setup_ab()
        r["store_latency_before"] = summarize(timed_stores(f, 10, 10_000, 0.5))
        fails_before = f.log_count("mp-1", "Cache-event publish failed")
        hb_fail_before = f.log_count("mp-1", "registration/heartbeat failed")
        wait_for(lambda: f.coordinator_l2_keys() >= 24, 15)
        keys_before_hang = f.coordinator_l2_keys()

        f.coord.proc.send_signal(signal.SIGSTOP)
        n_during = hang_s  # one store a second for the whole hang
        r["store_latency_during_hang"] = summarize(
            timed_stores(f, n_during, 20_000, 1.0)
        )
        f.coord.proc.send_signal(signal.SIGCONT)
        resumed = time.monotonic()

        r["publish_failures_during_hang"] = (
            f.log_count("mp-1", "Cache-event publish failed") - fails_before
        )
        r["heartbeat_failures_during_hang"] = (
            f.log_count("mp-1", "registration/heartbeat failed") - hb_fail_before
        )
        # What the coordinator does right after it wakes up.
        time.sleep(1)
        r["instances_1s_after_resume"] = read_view(f.url).instances
        t = wait_for(lambda: read_view(f.url).instances == ["mp-1", "mp-2"], 60)
        r["resume_to_all_listed_s"] = (
            None if t is None else round(time.monotonic() - resumed - 0, 2)
        )
        time.sleep(3)
        keys_after = f.coordinator_l2_keys()
        r["coordinator_l2_keys"] = {
            "before_hang": keys_before_hang,
            "stored_during_hang": 2 * n_during,
            "after_resume": keys_after,
            "actually_in_l2": f.l2_keys_listed_by_server(),
        }
        r["store_latency_after"] = summarize(timed_stores(f, 10, 30_000, 0.5))
        r["view_after"] = read_view(f.url).__dict__
    return r


EVICTION_INTERVAL_S = 5.0  # production default


def alice_usage_objects(f: Fleet, chunk_bytes: int) -> float:
    """alice's L2 usage as the coordinator counts it, in objects."""
    st = httpx.get(f"{f.url}/quota/alice").json()
    return round(st["usage_gb"] * 1024**3 / chunk_bytes, 1)


def store_alice(f: Fleet, n: int, base: int) -> None:
    """n fresh 2-chunk alice sequences on mp-1 (2 L2 objects each)."""
    for i in range(n):
        f.clients["mp-1"].store(
            list(range(base + 10 * i, base + 10 * i + 8)), cache_salt="alice"
        )


def exp_eviction() -> dict:
    """Quota enforcement before and after a crash that loses events."""
    r: dict = {"experiment": "eviction", "eviction_interval_s": EVICTION_INTERVAL_S}
    fleet = Fleet("evict", persist=True)
    fleet.coord.eviction_interval = EVICTION_INTERVAL_S
    with fleet as f:
        chunk = f.clients["mp-1"].chunk_bytes
        quota_objects = 8
        httpx.put(
            f"{f.url}/quota/alice",
            json={"limit_gb": quota_objects * chunk / 1024**3},
        ).raise_for_status()
        r["quota_objects"] = quota_objects

        # Control: the coordinator is up the whole time.
        store_alice(f, 10, 50_000)
        time.sleep(4 * EVICTION_INTERVAL_S)
        r["control"] = {
            "stored_objects": 20,
            "actual_l2_objects": f.l2_keys_listed_by_server(),
            "coordinator_usage_objects": alice_usage_objects(f, chunk),
        }

        # Crash: events for the next 20 objects are lost.
        time.sleep(CHECKPOINT_INTERVAL_S + 0.5)
        before_crash = f.l2_keys_listed_by_server()
        f.coord.kill()
        t0 = time.monotonic()
        store_alice(f, 10, 60_000)
        time.sleep(max(0.0, DOWNTIME_S - (time.monotonic() - t0)))
        r.update(f.restart_and_wait())
        r["quota_after_restart"] = httpx.get(f"{f.url}/quota/alice").json()["quota_limit_gb"] * 1024**3 / chunk
        samples = []
        for _ in range(6):  # 6 eviction cycles
            time.sleep(EVICTION_INTERVAL_S)
            samples.append({
                "actual_l2_objects": f.l2_keys_listed_by_server(),
                "coordinator_usage_objects": alice_usage_objects(f, chunk),
            })
        r["crash"] = {
            "l2_objects_before_crash": before_crash,
            "stored_during_downtime": 20,
            "samples_every_5s_after_recovery": samples,
        }
    return r


EXPERIMENTS = {
    "sigterm": exp_sigterm,
    "long-window": exp_long_window,
    "hang-15": lambda: exp_hang(15),
    "hang-40": lambda: exp_hang(40),
    "eviction": exp_eviction,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(EXPERIMENTS), default=None)
    p.add_argument("--out", default="")
    a = p.parse_args()
    names = [a.only] if a.only else list(EXPERIMENTS)
    results = []
    for n in names:
        print(f"=== {n}", flush=True)
        res = EXPERIMENTS[n]()
        print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
        results.append(res)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
