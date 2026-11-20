"""Checks that run.py's wrappers leave Seiden's answers unchanged and that its stitching and time accounting are sound.

Runs in this repository's container on the prepared inputs (B3D1 `jnc0` and CalDoT1 `caldot1-y05`):
  SEIDEN_DATA=/data/cwkt/data/polyis-cache/seiden docker compose run --rm -T --entrypoint python seiden \
      -m pytest -p no:cacheprovider tetris_comparison/test_run.py -q
"""
import json

import numpy as np
import pytest

import run

# A B3D clip (frame count not a multiple of 100, so its bandit has a short tail segment) and a CalDoT clip (900 frames,
# no tail segment, so Seiden's original select_rep never recurses on it).
B3D, CALDOT = 'jnc0', 'caldot1-y05'


def manifest(dataset: str) -> dict:
    """The prepared manifest of a dataset; fails clearly when the inputs are missing."""
    path = run.VIDEO_DATA / dataset / 'manifest.json'
    assert path.exists(), f'{path} missing; run tetris/measurements/r7o3_seiden_prepare.py first'
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def restore_patches():
    """Every test leaves run.py with both patch groups installed, as its runs use them."""
    yield
    run.set_patches(timing=True, guard=True)


def test_clip_budget_rounds_and_keeps_two_anchors():
    # Nearest integer (half to even), never fewer than Seiden's first and last frame.
    assert run.clip_budget(900, 8) == 112
    assert run.clip_budget(1075, 4) == 269
    assert run.clip_budget(900, 12) == 75
    assert run.clip_budget(10, 32) == 2


@pytest.mark.parametrize('query', ['presence', 'count'])
def test_timing_wrappers_do_not_change_answers(query):
    # Same clip, budget, and seed with and without the timing wrappers (the select_rep fix stays on in both).
    clip = manifest(B3D)['clips'][0]
    budget = run.clip_budget(clip['frames'], 32)
    run.set_patches(timing=True, guard=True)
    timed = run.run_clip(B3D, clip, query, budget, 0)
    run.set_patches(timing=False, guard=True)
    reference = run.run_clip(B3D, clip, query, budget, 0)
    assert timed['anchors'] == reference['anchors']
    assert np.array_equal(timed['y_pred'], reference['y_pred'])


@pytest.mark.parametrize('query', ['presence', 'count'])
def test_select_rep_fix_does_not_change_answers_where_seiden_completes(query):
    # Seiden's original select_rep against the fix, without timing wrappers, on a clip where the original completes.
    clip = manifest(CALDOT)['clips'][0]
    budget = run.clip_budget(clip['frames'], 8)
    run.set_patches(timing=False, guard=True)
    fixed = run.run_clip(CALDOT, clip, query, budget, 0)
    run.set_patches(timing=False, guard=False)
    original = run.run_clip(CALDOT, clip, query, budget, 0)
    assert fixed['anchors'] == original['anchors']
    assert np.array_equal(fixed['y_pred'], original['y_pred'])


def test_runs_are_deterministic_for_a_seed():
    # Two runs with the same seed give the same anchors and answers.
    clip = manifest(B3D)['clips'][1]
    budget = run.clip_budget(clip['frames'], 16)
    first, second = (run.run_clip(B3D, clip, 'count', budget, 0) for _ in range(2))
    assert first['anchors'] == second['anchors']
    assert np.array_equal(first['y_pred'], second['y_pred'])


def test_time_accounting():
    # Every excluded part is timed, non-negative, and the parts plus Seiden's time make up the wall clock.
    clip = manifest(B3D)['clips'][0]
    result = run.run_clip(B3D, clip, 'count', run.clip_budget(clip['frames'], 32), 0)
    excluded = result['excluded_s']
    assert list(excluded) == run.EXCLUDED_PARTS
    assert all(seconds >= 0 for seconds in excluded.values())
    # The label file, the unused model, the I-frame scan, and ground truth run on every clip.
    assert all(excluded[p] > 0 for p in ('label_load', 'model_load', 'iframe_scan', 'ground_truth'))
    assert result['seiden_s'] > 0
    assert result['seiden_s'] + sum(excluded.values()) == pytest.approx(result['wall_s'])


def test_split_stitches_clips_in_manifest_order():
    # The split's answer is each clip's answer, in manifest order, and its detector frames are the clips' anchors.
    m = manifest(B3D)
    split = run.run_split(B3D, m, 'count', 32, 0)
    clips = [run.run_clip(B3D, c, 'count', run.clip_budget(c['frames'], 32), 0) for c in m['clips']]
    assert len(split['y_pred']) == m['frames'] == sum(c['frames'] for c in m['clips'])
    assert np.array_equal(np.asarray(split['y_pred']), np.concatenate([c['y_pred'] for c in clips]))
    assert split['detector_frames'] == sum(run.clip_budget(c['frames'], 32) for c in m['clips'])
    assert split['detector_frames'] == sum(len(c['anchors']) for c in clips)
    # The cost combines Seiden's time with the charged detector time at the manifest's rate.
    assert split['detector_s'] == pytest.approx(split['detector_frames'] * m['detector_ms_per_frame'] / 1000)
    assert split['fps'] == pytest.approx(m['frames'] / (split['seiden_s'] + split['detector_s']))
