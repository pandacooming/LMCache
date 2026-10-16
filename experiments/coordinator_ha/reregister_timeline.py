# SPDX-License-Identifier: Apache-2.0
"""Timeline of how MP servers re-register after a coordinator restart.

Reuses the crash experiment's pieces, keeps the logs, and lines up three
moments per server against the coordinator becoming healthy again:
  - the heartbeat that got 404 ("Coordinator no longer knows ...")
  - the re-registration ("Registered with coordinator ...")
  - when the coordinator's GET /instances listed it
"""

# Standard
from datetime import datetime
import pathlib
import re
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Third Party
import httpx

# First Party
from coord_crash_experiment import (  # noqa: E402
    Coordinator,
    ProductionTimingServer,
    shared_l2,
    wait_for,
)

TS = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]")


def log_times(path: pathlib.Path, needle: str, after: float) -> list[float]:
    """Wall-clock times of log lines containing ``needle`` after ``after``."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if needle in line and (m := TS.search(line)):
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
            if t >= after:
                out.append(t)
    return out


def listed(url: str) -> list[str]:
    rows = httpx.get(f"{url}/instances", timeout=2).json()["instances"]
    return sorted(r["instance_id"] for r in rows)


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="exp-rereg-"))
    coord = Coordinator(workdir, persist=False)
    coord.start()
    pool = shared_l2(workdir / "shared-l2")
    servers = {
        sid: ProductionTimingServer(sid, workdir, coord.url, [pool])
        for sid in ("mp-1", "mp-2")
    }
    try:
        for s in servers.values():
            s.start()
        time.sleep(3)
        coord.kill()
        killed_at = time.time()
        time.sleep(8)
        coord.start()
        healthy_at = time.time()
        listed_at: dict[str, float] = {}

        def poll() -> bool:
            now = time.time()
            for sid in listed(coord.url):
                listed_at.setdefault(sid, now)
            return len(listed_at) == 2

        wait_for(poll, 60, poll=0.02)
        time.sleep(1)

        print(f"coordinator killed       t = {killed_at - healthy_at:+6.2f} s")
        print(f"coordinator healthy      t = {0:+6.2f} s  (reference)")
        for sid, s in servers.items():
            got_404 = log_times(s.log_path, "no longer knows", healthy_at)
            reg = log_times(s.log_path, "Registered with coordinator", healthy_at)
            print(f"--- {sid}")
            print(f"  heartbeat got 404       t = {got_404[0] - healthy_at:+6.2f} s")
            print(f"  re-registered           t = {reg[0] - healthy_at:+6.2f} s"
                  f"   ({reg[0] - got_404[0]:.2f} s after the 404)")
            print(f"  listed by coordinator   t = {listed_at[sid] - healthy_at:+6.2f} s")
    finally:
        for s in servers.values():
            s.stop()
        coord.stop()


if __name__ == "__main__":
    main()
