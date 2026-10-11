"""
Tests for Policy Layer (Section 10.5 #4).
Tests authorized moves, deduplication, cooldowns, and business hour schedules.
"""

import time
import pytest

from trace.config import BusinessHoursConfig
from trace.logic.policy import PolicyEngine


@pytest.fixture
def policy_engine():
    bh = BusinessHoursConfig(
        days=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        start="09:00",
        end="18:00"
    )
    return PolicyEngine(bh)


def test_authorized_move_suppresses_alerts(policy_engine):
    asset_id = "printer-01"
    # Authorize move for 10 minutes
    policy_engine.authorize_move(asset_id, duration_minutes=10, actor="Branch Manager")

    now = time.time()
    # Transition to MOVED during active authorized window
    alert = policy_engine.evaluate_transition(
        asset_id=asset_id,
        asset_class="printer",
        criticality=2,
        policy_mode="fixed",
        prev_state="SAFE",
        new_state="MOVED",
        confidence=0.95,
        current_t=now + 60.0 # 1 minute into window
    )
    # Alert must be suppressed!
    assert alert is None


def test_alert_deduplication_and_cooldown(policy_engine):
    asset_id = "safe-02"
    now = time.time()

    # 1. First alert triggers full alert payload
    alert1 = policy_engine.evaluate_transition(
        asset_id=asset_id,
        asset_class="safe",
        criticality=3,
        policy_mode="fixed",
        prev_state="SAFE",
        new_state="MOVED",
        confidence=0.94,
        current_t=now
    )
    assert alert1 is not None
    assert alert1["is_repeat"] is False
    assert alert1["suppress_push"] is False

    # Record dispatch
    incident_id = "inc-001"
    policy_engine.record_alert_dispatched(asset_id, "MOVED", incident_id, now)

    # 2. Duplicate alert within 5 minutes (e.g. 60 seconds later)
    alert2 = policy_engine.evaluate_transition(
        asset_id=asset_id,
        asset_class="safe",
        criticality=3,
        policy_mode="fixed",
        prev_state="MOVED",
        new_state="MOVED",
        confidence=0.96,
        current_t=now + 60.0
    )
    # Folds into incident counter and suppresses redundant push
    assert alert2 is not None
    assert alert2["is_repeat"] is True
    assert alert2["incident_id"] == incident_id
    assert alert2["suppress_push"] is True
