"""
Tests for Finite State Machine, Hysteresis, and Doubt Engine (Section 10.5 #3).
"""

import pytest

from trace.logic.belief import DoubtEngine
from trace.logic.fsm import AssetFSM, FSMConfig, Obs, S


def test_doubt_engine_exponential_decay():
    engine = DoubtEngine(half_life_s=20.0)
    engine.reset_sighted(0.0)
    assert engine.belief == 1.0

    # Step forward 20 seconds unseen while visible/blind
    b1 = engine.step(current_t=20.0, is_verified=False, is_occluded=False, camera_ok=True)
    # Exactly one half-life -> belief ~ 0.50
    assert 0.48 <= b1 <= 0.52
    assert engine.is_unknown() is False

    # Step forward another 2 seconds
    b2 = engine.step(current_t=22.0, is_verified=False, is_occluded=False, camera_ok=True)
    assert b2 < 0.50
    assert engine.is_unknown() is True

    # Sighting resets belief to 1.0
    b3 = engine.step(current_t=23.0, is_verified=True, is_occluded=False, camera_ok=True)
    assert b3 == 1.0
    assert engine.is_unknown() is False


def test_fsm_moved_votes_and_returned_hysteresis():
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
    fsm = AssetFSM(cfg)

    # 4 frames of drift > threshold
    for i in range(1, 5):
        s = fsm.step(Obs(t=float(i), present_score=0.9, drift_frac=0.25, visibility=0.95, empty_match=0.1, camera_ok=True))
        assert s == S.SAFE

    # 5th frame triggers MOVED!
    s5 = fsm.step(Obs(t=5.0, present_score=0.9, drift_frac=0.25, visibility=0.95, empty_match=0.1, camera_ok=True))
    assert s5 == S.MOVED

    # Now move asset back within half the threshold (drift <= 0.05)
    s_back = fsm.step(Obs(t=6.0, present_score=0.9, drift_frac=0.03, visibility=0.95, empty_match=0.1, camera_ok=True))
    assert s_back == S.RETURNED

    # Stay stable in RETURNED for 4 seconds (< 5.0s hold)
    s_hold = fsm.step(Obs(t=9.5, present_score=0.9, drift_frac=0.03, visibility=0.95, empty_match=0.1, camera_ok=True))
    assert s_hold == S.RETURNED

    # At 5.0 seconds in RETURNED -> restores to SAFE!
    s_safe = fsm.step(Obs(t=11.5, present_score=0.9, drift_frac=0.03, visibility=0.95, empty_match=0.1, camera_ok=True))
    assert s_safe == S.SAFE


def test_fsm_swapped_identity_mismatch():
    cfg = FSMConfig(swap_s=2.0)
    fsm = AssetFSM(cfg)

    # 1. Single spurious tag reading does NOT immediately trigger SWAPPED (Finding 5)
    s1 = fsm.step(Obs(
        t=1.0,
        present_score=0.8,
        drift_frac=0.0,
        visibility=0.95,
        empty_match=0.1,
        camera_ok=True,
        tag_matched=False,
        tag_status="MISMATCH",
    ))
    assert s1 != S.SWAPPED

    # 2. Sustained tag mismatch across the swap window confirms SWAPPED
    s2 = fsm.step(Obs(
        t=4.0,
        present_score=0.8,
        drift_frac=0.0,
        visibility=0.95,
        empty_match=0.1,
        camera_ok=True,
        tag_matched=False,
        tag_status="MISMATCH",
    ))
    assert s2 == S.SWAPPED
