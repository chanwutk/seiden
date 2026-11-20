#!/usr/bin/env python
"""Run Seiden's own index and query code on the Tetris datasets (Tetris paper, R7.O3), without modifying Seiden.

Runs inside this repository's container (Dockerfile, docker-compose.yml), where the repository is mounted at the path
its code hard-codes (/nethome/jbang36/seiden) and the inputs that Tetris's measurements/r7o3_seiden_prepare.py builds
at /srv/data/jbang36/video_data/<dataset>/<clip>/{video.mp4,tasti_labels.csv} plus <dataset>/manifest.json.

Each test clip is a separate video, as Tetris, OTIF, and LEAP process it, so Seiden builds one index per clip. Per
dataset, query, budget fraction 1/k, and seed, this script calls only Seiden's code on each clip:
  EKOPSConfig(clip, 'car', nb_buckets)        Seiden's index configuration (no mining, training, or embedding DNN),
                                              with nb_buckets = max(2, round(clip frames / k)).
  EKO_mab(config, images, video.mp4)          Seiden's I-frame + multi-armed-bandit index with its defaults
                                              (anchor_percentage=0.8, c_param=2).
  index.init()                                query-agnostic index construction over I-frame anchors.
  CustomQuery(index).execute()                query-specific bandit anchors (build_additional_anchors) and label
                                              propagation to every frame; returns per-frame y_pred.
Per-clip answers are stitched in manifest order into one answer for the split. Queries: 'presence' uses Seiden's
CustomQuery score (1 if any car); 'count' subclasses it with the per-frame object count, as Seiden's LimitQuery scores.
`images` is only used for its length (the target DNN is an identity over the cached labels), so a placeholder of the
right length is passed. Budgets are 1/4, 1/8, ..., 1/32 of each clip's frames; Seiden's experiments run each setting
once, so one seed (0) is run.

Cost, as if no frame had been detected beforehand. Seiden replays cached detector outputs, so every anchor (index.reps
after the query) is charged one detector run, at the per-frame detection time in manifest.json (the reference
detector's batched per-image time on the same split and GPU, as Tetris and OTIF batch detection). Seiden's own time is
the wall clock of index construction and the query, minus work outside the boundary every system shares (setup,
video decoding and I/O, disk writes) and work that exists only because the run is a replay scored against ground
truth. Each excluded part is timed by wrapping the Seiden function that performs it, and recorded per run:
  label_load         LabelDataset reading tasti_labels.csv (a live run calls the detector instead of reading a file).
  model_load         get_pretrained_embedding_dnn: Seiden loads a ResNet-18 that this configuration never uses.
  iframe_scan        PYAV_wrapper.get_iframe_indices: demuxing the video to find I-frames (video I/O).
  disk_write         numpy.save of Seiden's index cache files.
  ground_truth       building y_true for frames that are not anchors in BaseQuery.propagate, and CustomQuery.execute's
                     conversion of y_true; the answer y_pred reads only anchors' labels.
Work not listed stays charged to Seiden, so an omission over-charges Seiden rather than under-charging it.

One fix to Seiden's bandit, applied here at run time (Seiden's files are unchanged): EKO_mab.select_rep picks the
segment with the highest UCB value; when that segment has no unlabeled frame left, it sets the segment's reward to 0
and calls itself again, which recurses without end when the segment's exploration bonus alone still ranks it first.
The bandit's segments are length // 100 frames, so a clip whose frame count is not a multiple of 100 gets a short tail
segment that can fill early and then always rank first. The wrapper passes select_rep only the segments that still
have an unlabeled frame. A full segment can never yield a frame, so wherever the original returns, the wrapper returns
the same segment and frame (same order, same random draw); it differs only where the original recursed without end.
tetris_comparison/test_run.py checks that neither the timing wrappers nor this fix changes Seiden's answers.

Writes /srv/data/jbang36/tasti_data/results/<dataset>/<query>_f<k>_s<seed>.json with the run's cost, the excluded
time by part, and the stitched per-frame y_pred.
"""
import argparse
import functools
import importlib
import importlib.abc
import importlib.util
import json
import random
import sys
import time
from collections import OrderedDict
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

from src.experiments.query import CustomQuery  # noqa: E402
from src.system_architecture.mab import EKO_mab  # noqa: E402
from src.system_architecture.parameter_search import EKOPSConfig  # noqa: E402

# The modules whose names Seiden's code resolves at call time.
MAB = sys.modules[EKO_mab.__module__]
PROPAGATE_CLASS = next(c for c in CustomQuery.__mro__ if 'propagate' in c.__dict__)
TASTI = sys.modules[PROPAGATE_CLASS.__module__].tasti

# Seiden's hard-coded data roots, mounted by docker-compose.
VIDEO_DATA = Path('/srv/data/jbang36/video_data')
RESULTS = Path('/srv/data/jbang36/tasti_data/results')
# Budgets 1/k of each clip's frames, and the seed of Seiden's single run per setting.
DENOMINATORS = [4, 8, 12, 16, 20, 24, 28, 32]
SEEDS = [0]
# The excluded parts of Seiden's wall clock, in reporting order.
EXCLUDED_PARTS = ['label_load', 'model_load', 'iframe_scan', 'disk_write', 'ground_truth']


class CountQuery(CustomQuery):
    """Seiden's CustomQuery with the per-frame object count as the score (as in Seiden's LimitQuery)."""

    def score(self, target_dnn_output):
        return len(target_dnn_output)


QUERIES = {'presence': CustomQuery, 'count': CountQuery}


class Clock:
    """Time accumulated by the wrappers during one clip's run."""

    def __init__(self):
        # Excluded seconds by part, and the measured seconds of the query phases used to isolate y_true's conversion.
        self.excluded = dict.fromkeys(EXCLUDED_PARTS, 0.0)
        self.phases = {'finish_index_building': 0.0, 'propagate': 0.0}
        # Anchors of the current query (set once the bandit finishes); None charges every y_true object to Seiden.
        self.anchors = None


CLOCK = Clock()


def timed(fn, record):
    """Wrap fn so its duration is added through record(seconds) without changing its arguments or result."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Time the original call and account for it even if it raises.
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            record(time.perf_counter() - start)
    return wrapper


def exclude(part):
    """A record function adding to an excluded part of the current clip."""
    def record(seconds):
        CLOCK.excluded[part] += seconds
    return record


def finish_index_building(original):
    """Wrap CustomQuery.finish_index_building: time it and remember the anchors it leaves."""
    @functools.wraps(original)
    def wrapper(self):
        # Time the bandit phase, then record the anchors for the y_true exclusion.
        start = time.perf_counter()
        try:
            return original(self)
        finally:
            CLOCK.phases['finish_index_building'] += time.perf_counter() - start
            CLOCK.anchors = {int(r) for r in self.index.reps}
    return wrapper


def ground_truth_object(original):
    """Wrap DNNOutputCacheFloat.__init__: exclude the time of y_true objects for frames that are not anchors."""
    @functools.wraps(original)
    def wrapper(self, target_dnn_cache, scoring_fn, idx):
        # Build the object as Seiden does; charge it only if propagation reads it (an anchor) or anchors are unknown.
        start = time.perf_counter()
        try:
            return original(self, target_dnn_cache, scoring_fn, idx)
        finally:
            if CLOCK.anchors is not None and int(idx) not in CLOCK.anchors:
                CLOCK.excluded['ground_truth'] += time.perf_counter() - start
    return wrapper


def select_open_rep(original):
    """Seiden's select_rep, called only with segments that still have an unlabeled frame (see the module docstring)."""
    @functools.wraps(original)
    def wrapper(self, cluster_dict, rep_indices):
        # Segments (start, end) whose frames are not all anchors yet; the dict values are shared, so updates still apply.
        open_clusters = OrderedDict((k, v) for k, v in cluster_dict.items() if len(v['members']) < k[1] - k[0] + 1)
        return original(self, open_clusters, rep_indices)
    return wrapper


# Each patch: (owner, attribute, original, wrapped); installed at import and removable for test_run.py's comparisons.
PATCHES = {
    'timing': [
        (MAB.LabelDataset, '__init__', MAB.LabelDataset.__init__,
         timed(MAB.LabelDataset.__init__, exclude('label_load'))),
        (EKO_mab, 'get_pretrained_embedding_dnn', EKO_mab.get_pretrained_embedding_dnn,
         timed(EKO_mab.get_pretrained_embedding_dnn, exclude('model_load'))),
        (MAB.PYAV_wrapper, 'get_iframe_indices', MAB.PYAV_wrapper.get_iframe_indices,
         timed(MAB.PYAV_wrapper.get_iframe_indices, exclude('iframe_scan'))),
        (np, 'save', np.save, timed(np.save, exclude('disk_write'))),
        (TASTI.DNNOutputCacheFloat, '__init__', TASTI.DNNOutputCacheFloat.__init__,
         ground_truth_object(TASTI.DNNOutputCacheFloat.__init__)),
        (CustomQuery, 'finish_index_building', CustomQuery.finish_index_building,
         finish_index_building(CustomQuery.finish_index_building)),
        (PROPAGATE_CLASS, 'propagate', PROPAGATE_CLASS.propagate,
         timed(PROPAGATE_CLASS.propagate, lambda s: CLOCK.phases.__setitem__('propagate', CLOCK.phases['propagate'] + s))),
    ],
    'guard': [(EKO_mab, 'select_rep', EKO_mab.select_rep, select_open_rep(EKO_mab.select_rep))],
}


# Which patch groups are installed; run_clip derives Seiden's time only while the timing wrappers are installed.
INSTALLED = {'timing': False, 'guard': False}


def set_patches(timing: bool = True, guard: bool = True):
    """Install or remove the timing wrappers and the select_rep fix."""
    # Each group sets its attributes to the wrapped or the original function.
    for group, enabled in (('timing', timing), ('guard', guard)):
        for owner, name, original, wrapped in PATCHES[group]:
            setattr(owner, name, wrapped if enabled else original)
        INSTALLED[group] = enabled


set_patches()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--queries', nargs='+', default=list(QUERIES))
    parser.add_argument('--denominators', nargs='+', type=int, default=DENOMINATORS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    return parser.parse_args()


def clip_budget(frames: int, k: int) -> int:
    """Seiden's nb_buckets for a clip at budget fraction 1/k (Seiden needs at least the first and last frame)."""
    return max(2, round(frames / k))


def run_clip(dataset: str, clip: dict, query: str, budget: int, seed: int) -> dict:
    """Build Seiden's index on one clip and run one query; returns the answer, anchors, and time split."""
    global CLOCK
    # Seed every random source Seiden draws from (I-frame subsets, bandit choices) and reset the clip's clock.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    CLOCK = Clock()
    name = f"{dataset}/{clip['dir']}"
    # Only len(images) is used; the target DNN is an identity over the cached labels.
    images = np.zeros((clip['frames'], 1, 1, 3), dtype=np.uint8)
    started = time.perf_counter()
    index = EKO_mab(EKOPSConfig(name, category='car', nb_buckets=budget), images, str(VIDEO_DATA / name / 'video.mp4'))
    index.init()
    query_started = time.perf_counter()
    y_pred, _ = QUERIES[query](index).execute()
    finished = time.perf_counter()
    wall = finished - started
    # Without the timing wrappers (test_run.py's reference runs) only the answer is meaningful.
    if not INSTALLED['timing']:
        return dict(y_pred=np.asarray(y_pred, dtype=float), anchors=sorted({int(r) for r in index.reps}))
    # CustomQuery.execute's own work after bandit and propagation is the y_true conversion.
    conversion = (finished - query_started) - CLOCK.phases['finish_index_building'] - CLOCK.phases['propagate']
    assert conversion > -1e-3, f'{name}: query phases exceed the query time'
    CLOCK.excluded['ground_truth'] += max(0.0, conversion)
    excluded = sum(CLOCK.excluded.values())
    assert 0 <= excluded < wall, f'{name}: excluded {excluded:.3f}s of {wall:.3f}s'
    anchors = sorted({int(r) for r in index.reps})
    assert len(anchors) == budget, f'{name}: {len(anchors)} anchors for a budget of {budget}'
    assert len(y_pred) == clip['frames'], f'{name}: {len(y_pred)} answers for {clip["frames"]} frames'
    return dict(y_pred=np.asarray(y_pred, dtype=float), anchors=anchors, wall_s=wall, seiden_s=wall - excluded,
                excluded_s=dict(CLOCK.excluded))


def run_split(dataset: str, manifest: dict, query: str, k: int, seed: int) -> dict:
    """Run Seiden on every clip of the split at budget fraction 1/k and stitch the answers in manifest order."""
    clips = [run_clip(dataset, c, query, clip_budget(c['frames'], k), seed) for c in manifest['clips']]
    detector_frames = sum(len(c['anchors']) for c in clips)
    seiden_s = sum(c['seiden_s'] for c in clips)
    detector_s = detector_frames * manifest['detector_ms_per_frame'] / 1000
    return dict(dataset=dataset, query=query, denominator=k, fraction=1 / k, seed=seed, frames=manifest['frames'],
                clips=len(clips), detector_frames=detector_frames, detector_ms_per_frame=manifest['detector_ms_per_frame'],
                seiden_s=seiden_s, detector_s=detector_s, total_s=seiden_s + detector_s,
                fps=manifest['frames'] / (seiden_s + detector_s), wall_s=sum(c['wall_s'] for c in clips),
                excluded_s={p: sum(c['excluded_s'][p] for c in clips) for p in EXCLUDED_PARTS},
                y_pred=np.concatenate([c['y_pred'] for c in clips]).tolist())


def main():
    args = parse_args()
    manifest = json.loads((VIDEO_DATA / args.dataset / 'manifest.json').read_text())
    out = RESULTS / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    # One run per budget fraction, seed, and query.
    for k in args.denominators:
        for seed in args.seeds:
            for query in args.queries:
                result = run_split(args.dataset, manifest, query, k, seed)
                (out / f'{query}_f{k}_s{seed}.json').write_text(json.dumps(result) + '\n')
                print(f"{args.dataset} {query} 1/{k} s{seed}: {result['detector_frames']} detector frames, Seiden "
                      f"{result['seiden_s']:.1f}s (excluded {sum(result['excluded_s'].values()):.1f}s) + detector "
                      f"{result['detector_s']:.1f}s -> {result['fps']:.1f} FPS", flush=True)


if __name__ == '__main__':
    main()
