# Coordinator fault experiments

Experiments behind the coordinator HA failure-mode report. Not meant for
upstream; kept here so the scripts and raw data survive.

Run from the repository root with the dev venv active (CPU only):

| Script | What it measures |
| --- | --- |
| `coord_crash_experiment.py` | kill -9 / restart in three scenarios: no persistence, persistence, persistence with lost files |
| `coord_supplement.py` | SIGTERM, 120 s observation, coordinator hang (SIGSTOP 15 s / 40 s), quota enforcement after a crash |
| `reregister_timeline.py` | Re-registration timeline from MP server log timestamps |

Scripts import each other, so run them from this directory's path, e.g.
`python experiments/coordinator_ha/coord_supplement.py --only hang-40`.

`results/` holds the raw JSON. `*_v2.json` are re-runs on upstream 1ed386c
(2026-10-16); the rest ran on 7d7ca47 / 1fde7be (2026-10-10/11).
