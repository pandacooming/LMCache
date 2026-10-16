# SPDX-License-Identifier: Apache-2.0
"""Experiment: what an MP coordinator crash costs, and what survives a restart.

Two real CPU-only MP servers share one fs-backed L2 pool (standing in for a
shared S3 bucket) and report to a coordinator running as its own process, so
it can be SIGKILLed. Each scenario:

  1. stores A (mp-1, salt alice) and B (mp-2, salt bob); sets quotas; pins A
  2. stores E right before the crash (inside the last checkpoint interval)
  3. SIGKILLs the coordinator
  4. stores C on mp-1 while the coordinator is down
  5. restarts the coordinator on the same port (optionally losing its files,
     as a pod rescheduled to another node without a PVC would)
  6. stores D on mp-2 after recovery
  7. compares the coordinator's view against what was actually stored

Run from the repo root with the venv active:
    python <this file> [--scenario all|no-persist|persist|node-loss]
"""

# Standard
from dataclasses import dataclass, field
import argparse
import json
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path.cwd()))

# Third Party
import httpx

# First Party
from lmcache.v1.distributed.api import Tier
from tests.v1.mp_coordinator.mp_server_harness import (
    CHUNK_SIZE,
    KVClient,
    MPServer,
    directory,
    keys_of,
    shared_l2,
)

MODEL = "exp-model"
HEARTBEAT_S = 5.0  # production default (lmcache.mp heartbeat_interval)
EVENT_FLUSH_S = 1.0  # production default (event_flush_interval)
DOWNTIME_S = 8.0  # longer than one heartbeat, so servers notice the outage
CHECKPOINT_INTERVAL_S = 3.0

SEQS = {
    "A": (list(range(1, 9)), "alice", "mp-1"),  # before crash, pinned
    "B": (list(range(100, 108)), "bob", "mp-2"),  # before crash
    "E": (list(range(200, 208)), "alice", "mp-1"),  # just before crash
    "C": (list(range(300, 308)), "alice", "mp-1"),  # during downtime
    "D": (list(range(400, 408)), "bob", "mp-2"),  # after recovery
}
QUOTAS_GB = {"alice": 1.0, "bob": 2.0}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def answers(url: str) -> bool:
    try:
        return httpx.get(url, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


def wait_for(cond, timeout: float, poll: float = 0.05) -> float | None:
    """Seconds until ``cond()`` held, or None on timeout."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return time.monotonic() - t0
        time.sleep(poll)
    return None


class ProductionTimingServer(MPServer):
    """The harness server, with production heartbeat and flush intervals."""

    def _command(self) -> list[str]:
        cmd = super()._command()
        cmd[cmd.index("--coordinator-heartbeat-interval") + 1] = str(HEARTBEAT_S)
        cmd[cmd.index("--coordinator-event-flush-interval") + 1] = str(EVENT_FLUSH_S)
        return cmd


class Coordinator:
    """``python -m lmcache.v1.mp_coordinator`` as a killable subprocess."""

    def __init__(self, workdir: pathlib.Path, persist: bool) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.workdir = workdir
        self.checkpoint = workdir / "coord.ckpt"
        self.metadata = workdir / "coord-meta.json"
        self.persist = persist
        self.eviction_interval = 0.0  # 0 disables the fleet L2 eviction loop
        self.runs = 0
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> float:
        self.runs += 1
        cmd = [
            sys.executable, "-m", "lmcache.v1.mp_coordinator",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--chunk-size", str(CHUNK_SIZE),
            "--eviction-check-interval", str(self.eviction_interval),
        ]
        if self.persist:
            cmd += [
                "--checkpoint-path", str(self.checkpoint),
                "--checkpoint-interval", str(CHECKPOINT_INTERVAL_S),
                "--metadata-path", str(self.metadata),
            ]
        log = (self.workdir / f"coordinator-{self.runs}.log").open("wb")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        if wait_for(lambda: answers(f"{self.url}/healthz"), 60) is None:
            raise RuntimeError("coordinator did not come up")
        return time.monotonic() - t0

    def kill(self) -> None:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def log(self, run: int) -> str:
        return (self.workdir / f"coordinator-{run}.log").read_text(errors="replace")


@dataclass
class View:
    instances: list[str]
    l1: dict[str, str] = field(default_factory=dict)  # seq -> "k/n"
    l2: dict[str, str] = field(default_factory=dict)
    quotas: dict[str, float] = field(default_factory=dict)
    pinned_keys: int = 0


def read_view(url: str) -> View:
    instances = sorted(
        r["instance_id"] for r in httpx.get(f"{url}/instances").json()["instances"]
    )
    placements = directory(url)
    view = View(instances=instances)
    for name, (tokens, salt, _) in SEQS.items():
        want = set(keys_of(MODEL, tokens, salt))
        for tier, out in ((Tier.L1, view.l1), (Tier.L2, view.l2)):
            have = {p.key for p in placements if p.tier == tier} & want
            out[name] = f"{len(have)}/{len(want)}"
    for row in httpx.get(f"{url}/quota").json()["by_cache_salt"]:
        if row["cache_salt"] in QUOTAS_GB and row["quota_exists"]:
            view.quotas[row["cache_salt"]] = row["quota_limit_gb"]
    view.pinned_keys = httpx.get(f"{url}/cache/pins").json()["total"]
    return view


def store(clients: dict, name: str) -> float:
    tokens, salt, server = SEQS[name]
    t0 = time.monotonic()
    clients[server].store(tokens, cache_salt=salt)
    return time.monotonic() - t0


def seq_settled(url: str, name: str) -> bool:
    tokens, salt, _ = SEQS[name]
    want = set(keys_of(MODEL, tokens, salt))
    have = {p.key for p in directory(url)}
    return want <= have


def run_scenario(name: str, persist: bool, lose_files: bool) -> dict:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix=f"exp-{name}-"))
    coord = Coordinator(workdir, persist)
    result: dict = {"scenario": name}
    servers: dict[str, MPServer] = {}
    clients: dict = {}
    try:
        coord.start()
        pool = shared_l2(workdir / "shared-l2")
        for sid in ("mp-1", "mp-2"):
            servers[sid] = ProductionTimingServer(sid, workdir, coord.url, [pool])
            servers[sid].start()
        clients = {
            sid: KVClient(s.rpc_url, MODEL) for sid, s in servers.items()
        }

        # 1. Normal operation.
        result["store_latency_normal_s"] = round(store(clients, "A"), 3)
        store(clients, "B")
        for salt, gb in QUOTAS_GB.items():
            httpx.put(f"{coord.url}/quota/{salt}", json={"limit_gb": gb}).raise_for_status()
        httpx.post(
            f"{coord.url}/cache/pins",
            json={"model_name": MODEL, "world_size": 1,
                  "token_ids": SEQS["A"][0], "cache_salt": "alice"},
        ).raise_for_status()
        wait_for(lambda: seq_settled(coord.url, "A") and seq_settled(coord.url, "B"), 15)
        if persist:
            time.sleep(CHECKPOINT_INTERVAL_S + 0.5)  # A and B are checkpointed
        result["before_crash"] = read_view(coord.url).__dict__

        # 2. One more store that the coordinator sees, then crash at once.
        store(clients, "E")
        wait_for(lambda: seq_settled(coord.url, "E"), 15)
        e_seen_at = time.time()
        coord.kill()
        t_crash = time.monotonic()
        result["E_seen_to_kill_s"] = round(time.time() - e_seen_at, 3)
        if coord.checkpoint.exists():
            # Only a checkpoint written after the coordinator held E can hold it.
            result["last_checkpoint_after_E_seen"] = (
                coord.checkpoint.stat().st_mtime > e_seen_at
            )

        # 3. Work continues on the servers while the coordinator is down.
        result["store_latency_while_down_s"] = round(store(clients, "C"), 3)
        time.sleep(max(0.0, DOWNTIME_S - (time.monotonic() - t_crash)))

        # 4. Restart, possibly on a fresh node.
        if lose_files:
            for f in (coord.checkpoint, coord.metadata):
                f.unlink(missing_ok=True)
        result["restart_to_healthz_s"] = round(coord.start(), 2)
        t_up = time.monotonic()
        t_reg = wait_for(
            lambda: read_view(coord.url).instances == ["mp-1", "mp-2"], 60
        )
        result["healthz_to_all_reregistered_s"] = None if t_reg is None else round(t_reg, 2)
        result["downtime_total_s"] = round(time.monotonic() - t_crash, 2)

        # 5. Normal operation again.
        store(clients, "D")
        wait_for(lambda: seq_settled(coord.url, "D"), 15)
        time.sleep(2 * EVENT_FLUSH_S)
        result["after_recovery"] = read_view(coord.url).__dict__
        result["objects_actually_in_shared_l2"] = sum(
            1 for f in (workdir / "shared-l2").rglob("*") if f.is_file()
        )

        logs = coord.log(coord.runs)
        result["coordinator_logged_event_gap"] = "Event gap" in logs
        result["servers_logged_dropped_events"] = {
            sid: "dropped and leave a seq gap" in s.log_path.read_text(errors="replace")
            for sid, s in servers.items()
        }
    finally:
        for c in clients.values():
            c.close()
        for s in servers.values():
            s.stop()
        coord.stop()
        shutil.rmtree(workdir, ignore_errors=True)
    return result


SCENARIOS = {
    "no-persist": dict(persist=False, lose_files=False),
    "persist": dict(persist=True, lose_files=False),
    "node-loss": dict(persist=True, lose_files=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    for n in names:
        print(f"=== scenario {n} ===", flush=True)
        r = run_scenario(n, **SCENARIOS[n])
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
