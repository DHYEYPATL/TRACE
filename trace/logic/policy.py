"""
Policy Engine module.
Enforces business hours, asset mobility rules, authorized-move windows,
deduplication, alert cooldowns, and escalation (Section 6.4).
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional, Tuple

from trace.config import BusinessHoursConfig


class PolicyEngine:
    """
    Decides whether an FSM state transition elevates to an operator security alert,
    assigns severity, and manages authorized moving workflows.
    """

    def __init__(self, business_hours: BusinessHoursConfig):
        self.bh_cfg = business_hours
        # Active authorized moves: {asset_id: {"ends_at": float, "actor": str, "duration_m": int}}
        self.authorized_moves: Dict[str, Dict[str, Any]] = {}
        # Cooldown & deduplication tracker: {(asset_id, event_type): {"last_alert_t": float, "incident_id": str}}
        self.alert_history: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def is_business_hours(self, dt: Optional[datetime.datetime] = None) -> bool:
        """Determines if the current moment falls within configured branch business hours."""
        if dt is None:
            dt = datetime.datetime.now()

        day_abbr = dt.strftime("%a")
        if day_abbr not in self.bh_cfg.days:
            return False

        cur_time_str = dt.strftime("%H:%M")
        return self.bh_cfg.start <= cur_time_str <= self.bh_cfg.end

    def authorize_move(self, asset_id: str, duration_minutes: int, actor: str = "Staff Operator") -> Dict[str, Any]:
        """
        Grants temporary suppression of MOVED/MISSING alerts for an authorized relocation.
        """
        now = time.time()
        ends_at = now + (duration_minutes * 60.0)
        self.authorized_moves[asset_id] = {
            "starts_at": now,
            "ends_at": ends_at,
            "actor": actor,
            "duration_m": duration_minutes,
        }
        return {
            "ok": True,
            "asset_id": asset_id,
            "ends_at": ends_at,
            "message": f"Move authorized for {duration_minutes} minutes by {actor}."
        }

    def is_move_authorized(self, asset_id: str, current_t: float) -> bool:
        """Checks if an authorized-move window is active for this asset."""
        if asset_id in self.authorized_moves:
            auth = self.authorized_moves[asset_id]
            if current_t <= auth["ends_at"]:
                return True
            else:
                del self.authorized_moves[asset_id]
        return False

    def evaluate_transition(
        self,
        asset_id: str,
        asset_class: str,
        criticality: int,
        policy_mode: str, # 'fixed' or 'mobile'
        prev_state: str,
        new_state: str,
        confidence: float,
        current_t: float
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates state transition against operational policies.
        Returns alert payload dict if an alert should be raised, else None.
        """
        # Non-alert states
        if new_state in ("SAFE", "OCCLUDED", "RETURNED"):
            return None

        # Check authorized move suppression
        if self.is_move_authorized(asset_id, current_t):
            if new_state in ("MOVED", "MISSING"):
                # Suppressed by authorized move policy
                return None

        in_bh = self.is_business_hours()

        # Mobile asset rules
        if policy_mode == "mobile":
            if new_state == "MOVED" and in_bh:
                # Logged only during business hours, no push alert
                return None

        # Determine severity
        severity = "low"
        if new_state in ("SWAPPED", "MISSING", "TAMPERED", "CAMERA_OBSTRUCTED", "SYSTEM_OFFLINE"):
            severity = "high"
        elif new_state == "MOVED":
            severity = "high" if (criticality >= 3 or not in_bh) else "medium"
        elif new_state == "STATE_CHANGED":
            severity = "high" if criticality >= 3 else "medium"
        elif new_state == "NEW_OBJECT":
            severity = "medium"
        elif new_state == "UNKNOWN":
            severity = "low"

        # Selective prediction decision bands (Section 3.3 #5)
        # >= 0.90: Auto-alert (immediate push)
        # 0.60 to 0.89: Needs review queue (no push, operator confirmation required)
        # < 0.60: Log only
        if confidence >= 0.90:
            tier = "AUTO_ALERT"
            suppress_push = False
        elif confidence >= 0.60:
            tier = "NEEDS_REVIEW"
            suppress_push = True
        else:
            tier = "LOG_ONLY"
            suppress_push = True

        # Deduplication and 5-minute Cooldown
        key = (asset_id, new_state)
        history_entry = self.alert_history.get(key)
        cooldown_window = 300.0 # 5 minutes

        is_repeat = False
        incident_id = None

        if history_entry:
            time_since = current_t - history_entry["last_alert_t"]
            if time_since < cooldown_window:
                is_repeat = True
                incident_id = history_entry["incident_id"]

        if is_repeat:
            # Folds into existing incident counter without duplicate push alert
            return {
                "asset_id": asset_id,
                "type": new_state,
                "severity": severity,
                "confidence": confidence,
                "tier": tier,
                "is_repeat": True,
                "incident_id": incident_id,
                "suppress_push": True
            }

        # New distinct incident
        return {
            "asset_id": asset_id,
            "type": new_state,
            "severity": severity,
            "confidence": confidence,
            "tier": tier,
            "is_repeat": False,
            "incident_id": None,
            "suppress_push": suppress_push
        }

    def record_alert_dispatched(self, asset_id: str, event_type: str, incident_id: str, current_t: float) -> None:
        """Records an emitted alert for deduplication and cooldown tracking."""
        self.alert_history[(asset_id, event_type)] = {
            "last_alert_t": current_t,
            "incident_id": incident_id,
        }
