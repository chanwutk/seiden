# Seiden for the Tetris comparison

Run this command from the Tetris workspace root.

```sh
./seiden/tetris_comparison/run.sh
```

This runs CalDoT1, CalDoT2, Amsterdam, and B3D1–B3D4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`). To run just CalDoT1, pass its dataset name.

```sh
./seiden/tetris_comparison/run.sh caldot1-y05
```

The command syncs this repository to `ace` and runs Seiden's unmodified code in its own container ([Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)), which mounts our data at the paths Seiden hard-codes. Seiden's inputs (the test clips concatenated into `video.mp4`, the reference detector's boxes as `tasti_labels.csv`, and its measured time per frame in `manifest.json`) are built once by `tetris/measurements/r7o3_seiden_prepare.py` in the `polyis` container.

For each dataset, [tetris_comparison/run.py](tetris_comparison/run.py) builds Seiden's index at budgets of 1/4, 1/8, 1/12, ..., 1/32 of the test frames with seeds 0, 1, and 2, and runs the presence and per-frame count queries. Each run's cost charges every labeled frame the reference detector's measured time. `run.py`'s docstring documents this cost model and the one run-time fix to Seiden's bandit. The §7.3 scorer (`tetris/measurements/r7o3_seiden_score.py`) then scores the runs with LEAP's evaluator. Raw runs are under `tetris/measurements-results/r7o3_seiden/<dataset>/`, one JSON per query, budget, and seed, and scores are in `tetris/measurements-results/r7o3_seiden_score/<dataset>.json`.
