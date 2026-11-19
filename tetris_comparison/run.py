#!/usr/bin/env python
"""Run Seiden's own index and query code on the Tetris datasets (Tetris paper, R7.O3), without modifying Seiden.

Runs inside this repository's container (Dockerfile, docker-compose.yml), where the repository is mounted at the path
its code hard-codes (/nethome/jbang36/seiden) and the inputs that Tetris's measurements/r7o3_seiden_prepare.py builds
at /srv/data/jbang36/video_data.

Per dataset, query, and index budget, this script only calls Seiden's code:
  EKOPSConfig(dataset, 'car', nb_buckets)     Seiden's index configuration (no mining, training, or embedding DNN).
  EKO_mab(config, images, video.mp4)          Seiden's I-frame + multi-armed-bandit index with its defaults
                                              (anchor_percentage=0.8, c_param=2).
  index.init()                                query-agnostic index construction over I-frame anchors.
  CustomQuery(index).execute()                query-specific bandit anchors (build_additional_anchors) and label
                                              propagation to every frame; returns per-frame y_pred.
Queries: 'presence' uses Seiden's CustomQuery score (1 if any car); 'count' subclasses it with the per-frame object
count, as Seiden's LimitQuery scores. `images` is only used for its length (the target DNN is an identity over the
cached labels), so a placeholder of the right length is passed.

Index budgets (nb_buckets, the frames the detector labels) are 1/4, 1/8, 1/12, ..., 1/32 of the test split's frames,
each with seeds 0, 1, 2.

Cost, as if no frame had been detected beforehand. Seiden replays cached detector outputs, so every index anchor
(index.reps after the query) is charged one detector run, at the time per frame in manifest.json (Tetris's reference
detector on the same split and GPU type), as Seiden's notebooks charge sampled frames x measured per-frame time.
Seiden's own time is the wall clock of index construction and the query. Decoding is not included.

One fix to Seiden's bandit, applied here at run time (Seiden's files are unchanged): EKO_mab.select_rep picks the
segment with the highest UCB value; when that segment has no unlabeled frame left, it sets the segment's reward to 0
and calls itself again, which recurses without end when the segment's exploration bonus alone still ranks it first.
The bandit's segments are length // 100 frames, so a test split whose frame count is not a multiple of 100 (Amsterdam,
54,011 frames) gets a short tail segment that fills early and then always ranks first. The wrapper passes select_rep
only the segments that still have an unlabeled frame. A full segment can never yield a frame, so wherever the original
returns, the wrapper returns the same segment and frame (same order, same random draw); it differs only where the
original recursed without end.

Writes /srv/data/jbang36/tasti_data/results/<dataset>/<query>_b<budget>_s<seed>.json with the run's cost and
Seiden's per-frame y_pred.
"""
import argparse
import importlib
import importlib.abc
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Seiden's code imports itself and its vendored benchmarks from these paths.
SEIDEN = '/nethome/jbang36/seiden'
for p in (SEIDEN, f'{SEIDEN}/benchmarks/stanford/tasti', f'{SEIDEN}/benchmarks/stanford/blazeit',
          f'{SEIDEN}/benchmarks/stanford/swag_python', f'{SEIDEN}/benchmarks/stanford/supg'):
    sys.path.insert(0, p)


class EkoAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve tasti.eko.* to tasti.seiden.*.

    The published repository renamed TASTI's `eko` package to `seiden`, but tasti/examples still imports `tasti.eko`,
    and tasti/__init__.py imports the examples, so `import tasti` fails. This alias resolves the old name to the
    renamed package without editing Seiden's code.
    """

    def find_spec(self, name, path, target=None):
        if name == 'tasti.eko' or name.startswith('tasti.eko.'):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module('tasti.seiden' + spec.name[len('tasti.eko'):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, EkoAlias())

from collections import OrderedDict  # noqa: E402

from src.experiments.query import CustomQuery  # noqa: E402
from src.system_architecture.mab import EKO_mab  # noqa: E402
from src.system_architecture.parameter_search import EKOPSConfig  # noqa: E402

# Seiden's hard-coded data roots, mounted by docker-compose.
VIDEO_DATA = Path('/srv/data/jbang36/video_data')
RESULTS = Path('/srv/data/jbang36/tasti_data/results')
FRACTIONS = [1 / k for k in range(4, 33, 4)]
SEEDS = [0, 1, 2]


class CountQuery(CustomQuery):
    """Seiden's CustomQuery with the per-frame object count as the score (as in Seiden's LimitQuery)."""

    def score(self, target_dnn_output):
        return len(target_dnn_output)


QUERIES = {'presence': CustomQuery, 'count': CountQuery}

# Seiden's select_rep, called only with segments that still have an unlabeled frame (see the module docstring).
_select_rep = EKO_mab.select_rep


def select_open_rep(self, cluster_dict, rep_indices):
    # Segments (start, end) whose frames are not all anchors yet; the dict values are shared, so updates still apply.
    open_clusters = OrderedDict((k, v) for k, v in cluster_dict.items() if len(v['members']) < k[1] - k[0] + 1)
    return _select_rep(self, open_clusters, rep_indices)


EKO_mab.select_rep = select_open_rep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--queries', nargs='+', default=list(QUERIES))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    return parser.parse_args()


def run(dataset: str, query: str, budget: int, seed: int, manifest: dict) -> dict:
    """Build Seiden's index and run one query; returns the run's cost and per-frame answer."""
    # Seed every random source Seiden draws from (I-frame subsets, bandit choices).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = manifest['frames']
    # Only len(images) is used; the target DNN is an identity over the cached labels.
    images = np.zeros((n, 1, 1, 3), dtype=np.uint8)
    config = EKOPSConfig(dataset, category='car', nb_buckets=budget)
    started = time.perf_counter()
    index = EKO_mab(config, images, str(VIDEO_DATA / dataset / 'video.mp4'))
    index.init()
    y_pred, _ = QUERIES[query](index).execute()
    seiden_s = time.perf_counter() - started
    # Every anchor is one detector run.
    oracle_frames = len({int(r) for r in index.reps})
    oracle_s = oracle_frames * manifest['detector_ms_per_frame'] / 1000
    return dict(dataset=dataset, query=query, budget=budget, seed=seed, frames=n,
                iframes=len(index.pyav.get_iframe_indices()), detector_ms_per_frame=manifest['detector_ms_per_frame'],
                oracle_frames=oracle_frames, seiden_s=seiden_s, oracle_s=oracle_s, total_s=seiden_s + oracle_s,
                fps=n / (seiden_s + oracle_s), y_pred=np.asarray(y_pred, dtype=float).tolist())


def main():
    args = parse_args()
    manifest = json.loads((VIDEO_DATA / args.dataset / 'manifest.json').read_text())
    out = RESULTS / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for fraction in FRACTIONS:
        budget = int(round(manifest['frames'] * fraction))
        for seed in args.seeds:
            for query in args.queries:
                result = dict(run(args.dataset, query, budget, seed, manifest), fraction=fraction)
                (out / f'{query}_b{budget}_s{seed}.json').write_text(json.dumps(result) + '\n')
                print(f"{args.dataset} {query} b{budget} ({fraction:g}) s{seed}: {result['oracle_frames']} detector "
                      f"frames, Seiden {result['seiden_s']:.1f}s + detector {result['oracle_s']:.1f}s -> "
                      f"{result['fps']:.1f} FPS", flush=True)


if __name__ == '__main__':
    main()
