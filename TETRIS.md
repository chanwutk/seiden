# Seiden for the Tetris comparison

Run this command from the Tetris workspace root.

```sh
./seiden/tetris_comparison/run.sh
```

This runs CalDoT1, CalDoT2, Amsterdam, and B3D1–B3D4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`). To run just CalDoT1, pass its dataset name.

```sh
./seiden/tetris_comparison/run.sh caldot1-y05
```

The command syncs this repository to `ace` and runs Seiden's unmodified code in its own container ([Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)), which mounts our data at the paths Seiden hard-codes. Seiden's inputs are rebuilt on every run by `tetris/measurements/r7o3_seiden_prepare.py` in the `polyis` container: for each test clip, the clip itself as `video.mp4` and the reference detector's boxes as `tasti_labels.csv`, plus a `manifest.json` with the clip order and the detector time charged per labeled frame (the reference detector's batched per-image time, as Tetris and OTIF batch detection).

Before any measured run, the command runs [tetris_comparison/test_run.py](tetris_comparison/test_run.py), which checks that the driver's timing wrappers and its one fix to Seiden's bandit leave Seiden's answers bit-identical, and that stitching and time accounting are sound. It then runs one container per dataset.

Each test clip is a separate video, as Tetris, OTIF, and LEAP process it, so [tetris_comparison/run.py](tetris_comparison/run.py) builds one Seiden index per clip at budgets of 1/4, 1/8, 1/12, ..., 1/32 of the clip's frames, runs the presence and per-frame count queries, and stitches the clips' answers. Seiden's experiments run each setting once, so each budget runs once (seed 0). Each run charges every labeled frame the detector time above, plus Seiden's own time without setup, video I/O, disk writes, and replay-only ground truth, which are timed separately and recorded. `run.py`'s docstring documents this cost model and the bandit fix. The §7.3 scorer (`tetris/measurements/r7o3_seiden_score.py`) then scores the runs with LEAP's evaluator. Raw runs are under `tetris/measurements-results/r7o3_seiden/<dataset>/`, one JSON per query and budget fraction, and scores are in `tetris/measurements-results/r7o3_seiden_score/<dataset>.json`.
