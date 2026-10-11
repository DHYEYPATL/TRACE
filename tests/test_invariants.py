"""
Tests for Core System Invariants (Section 10.5 #2).
1. Obstructed camera never yields SAFE.
2. MISSING never fires with visibility < 0.85 (Positive-evidence rule).
3. No alert fires on a single frame.
4. Privacy masking blurs people before crop storage.
"""

import numpy as np
import pytest

from trace.logic.fsm import AssetFSM, FSMConfig, Obs, S
from trace.platform.evidence import EvidenceWriter


@pytest.fixture
def default_fsm():
    cfg = FSMConfig(
        move_frac=0.10,
        present=0.70,
        vis_floor=0.85,
        missing_wait_s=15.0,
        swap_s=10.0,
        moved_votes=5,
        returned_hold_s=5.0,
        half_life_s=30.0,
    )
    return AssetFSM(cfg)


def test_invariant_obstructed_camera_never_safe(default_fsm):
    # Camera is blind / obstructed
    obs = Obs(
        t=1.0,
        present_score=1.0,
        drift_frac=0.0,
        visibility=1.0,
        empty_match=0.0,
        camera_ok=False,  # Camera failing
    )
    state = default_fsm.step(obs)
    # Must NOT be SAFE!
    assert state in (S.OCCLUDED, S.UNKNOWN)
    assert state != S.SAFE


def test_invariant_missing_suppressed_under_occlusion(default_fsm):
    # Asset is covered by a standing person (visibility 0.40)
    # and region matches empty background
    obs = Obs(
        t=10.0,
        present_score=0.1,
        drift_frac=0.0,
        visibility=0.40,  # Below visibility floor of 0.85
        empty_match=0.95,
        camera_ok=True,
    )
    state = default_fsm.step(obs)
    # Must be OCCLUDED, NOT MISSING!
    assert state != S.MISSING
    assert state in (S.OCCLUDED, S.UNKNOWN)


def test_invariant_no_alert_on_single_frame(default_fsm):
    # Single transient spike frame with high drift
    obs = Obs(
        t=1.0,
        present_score=0.9,
        drift_frac=0.45,  # High drift
        visibility=0.95,
        empty_match=0.1,
        camera_ok=True,
    )
    state = default_fsm.step(obs)
    # Single frame should NOT trigger MOVED (requires 5 votes)
    assert state != S.MOVED


def test_privacy_blurring_on_person_pixels():
    # Create test image with red object and blue person
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = (0, 0, 255)  # Red left side
    img[:, 50:] = (255, 0, 0)  # Blue right side

    person_mask = np.zeros((100, 100), dtype=np.uint8)
    person_mask[:, 50:] = 255  # Person on right side

    masked_img = EvidenceWriter.apply_privacy_mask(img, person_mask)

    # Left side (no person) should remain sharp/identical
    assert np.array_equal(masked_img[:, :40], img[:, :40])

    # Right side (person) should be blurred (variance lower or pixels smoothed)
    assert not np.array_equal(masked_img[:, 50:], img[:, 50:])
