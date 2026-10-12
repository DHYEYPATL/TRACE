"""
Finite State Machine module for asset states and transitions.
Implements the 10 states, override logic, temporal voting, and exit hysteresis
matching Section 6, Table 6.2, and System Design Image 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from trace.logic.belief import DoubtEngine


class S(str, Enum):
    SAFE = "SAFE"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"
    MOVED = "MOVED"
    SWAPPED = "SWAPPED"
    STATE_CHANGED = "STATE_CHANGED"
    MISSING = "MISSING"
    NEW_OBJECT = "NEW_OBJECT"
    TAMPERED = "TAMPERED"
    RETURNED = "RETURNED"
    CAMERA_OBSTRUCTED = "CAMERA_OBSTRUCTED"
    SYSTEM_OFFLINE = "SYSTEM_OFFLINE"


@dataclass
class FSMConfig:
    move_frac: float = 0.10
    present: float = 0.70
    vis_floor: float = 0.85
    missing_wait_s: float = 15.0
    swap_s: float = 10.0
    moved_votes: int = 5
    returned_hold_s: float = 5.0
    half_life_s: float = 30.0


@dataclass
class Obs:
    """One verification observation of an asset."""
    t: float                       # Monotonic seconds
    present_score: float           # Fused match score 0..1
    drift_frac: float              # Centroid drift / baseline diagonal
    visibility: float              # Region visibility 0..1
    empty_match: float             # SSIM against empty_roi baseline 0..1
    camera_ok: bool                # Camera health status
    found_elsewhere: bool = False  # Relocator sighting
    state_label: str = "CLOSED"    # State classifier output (OPEN / CLOSED)
    state_conf: float = 1.0        # State classifier confidence
    tag_matched: bool = True       # ArUco / QR tag identity verification
    tag_status: str = "NONE"       # "MATCH", "MISMATCH", "UNREADABLE", "NONE"


class AssetFSM:
    """
    Finite state machine governing lifecycle of a single asset.
    Guarantees:
    - Blind camera never yields SAFE.
    - MISSING and SWAPPED require visibility >= 0.85 and healthy camera (Positive-evidence rule).
    - No alert fires on single-frame transient noise.
    - Clean hysteresis with RETURNED bridge before restoring SAFE.
    """

    def __init__(self, cfg: FSMConfig):
        self.cfg = cfg
        self.state: S = S.UNKNOWN
        self.doubt_engine = DoubtEngine(half_life_s=cfg.half_life_s)

        self.votes: Dict[str, int] = {}
        self.since: Dict[str, float] = {}
        self.returned_entry_time: Optional[float] = None
        self.occluded_exit_start: Optional[float] = None

    @property
    def belief(self) -> float:
        return self.doubt_engine.belief

    def hold(
        self,
        name: str,
        condition: bool,
        t: float,
        n: Optional[int] = None,
        secs: Optional[float] = None
    ) -> bool:
        """
        Temporal voting and persistence counter.
        Returns True only if condition holds for >= n frames OR >= secs seconds.
        """
        if not condition:
            self.votes[name] = 0
            self.since.pop(name, None)
            return False

        self.votes[name] = self.votes.get(name, 0) + 1
        self.since.setdefault(name, t)

        frame_satisfied = (n is not None and self.votes[name] >= n)
        time_satisfied = (secs is not None and (t - self.since[name]) >= secs)

        return frame_satisfied or time_satisfied

    def step(self, o: Obs) -> S:
        c = self.cfg

        # 1. Evaluate visibility & verification
        visible = o.camera_ok and (o.visibility >= c.vis_floor)
        verified = visible and (o.present_score >= c.present) and o.tag_matched
        is_occluded = o.camera_ok and (o.visibility < c.vis_floor)

        # 2. Step the Doubt Engine
        self.doubt_engine.step(
            current_t=o.t,
            is_verified=verified,
            is_occluded=is_occluded,
            camera_ok=o.camera_ok,
        )

        # 3. Temporal condition trackers
        moved = self.hold(
            "moved",
            visible and (verified or o.found_elsewhere) and (o.drift_frac > c.move_frac),
            o.t,
            n=c.moved_votes,
        )

        # Tag substitution requires confirmed MISMATCH held across swap_s window.
        # An unreadable tag does not trigger substitution immediately.
        swapped = self.hold(
            "swapped_tag",
            visible and (o.tag_status == "MISMATCH" or (not o.tag_matched and o.tag_status != "UNREADABLE")) and (o.present_score >= 0.4),
            o.t,
            secs=c.swap_s,
        ) or self.hold(
            "swapped",
            visible and (o.empty_match < 0.85) and (o.present_score < c.present) and not o.found_elsewhere,
            o.t,
            secs=c.swap_s,
        )

        missing = self.hold(
            "missing",
            visible and (o.empty_match >= 0.85) and not o.found_elsewhere,
            o.t,
            secs=c.missing_wait_s,
        )

        state_changed = self.hold(
            "state_changed",
            visible and (o.state_conf >= 0.85) and (o.state_label == "OPEN"),
            o.t,
            n=5,
        )

        # 4. State transition arbitration (Top-down priority)

        # INVARIANT 1: Camera failure overrides all assets. Blind camera never shows SAFE.
        if not o.camera_ok:
            self.returned_entry_time = None
            if self.doubt_engine.is_unknown():
                self.state = S.UNKNOWN
            else:
                self.state = S.OCCLUDED
            return self.state

        # INVARIANT 2: Positive-evidence rule for alerts
        if missing:
            self.returned_entry_time = None
            self.state = S.MISSING
            return self.state

        if swapped:
            self.returned_entry_time = None
            self.state = S.SWAPPED
            return self.state

        if moved:
            self.returned_entry_time = None
            self.state = S.MOVED
            return self.state

        if state_changed:
            self.returned_entry_time = None
            self.state = S.STATE_CHANGED
            return self.state

        # 5. Recovery & Return Hysteresis from an alert state
        if self.state in (S.MOVED, S.MISSING, S.SWAPPED, S.STATE_CHANGED):
            # Recovery condition:
            # - For MOVED: must return within half the threshold
            # - For others: verified sighting
            recovery_cond = verified and (o.drift_frac <= (c.move_frac / 2.0))
            if recovery_cond:
                self.state = S.RETURNED
                self.returned_entry_time = o.t
            return self.state

        if self.state == S.RETURNED:
            # Must remain stable for returned_hold_s (5s) before returning to SAFE
            if verified and (o.drift_frac <= c.move_frac):
                if self.returned_entry_time and (o.t - self.returned_entry_time >= c.returned_hold_s):
                    self.state = S.SAFE
                    self.returned_entry_time = None
            else:
                self.returned_entry_time = o.t
            return self.state

        # 6. Occlusion vs Safe vs Unknown
        if not visible:
            # Under occlusion, exit timer reset
            self.occluded_exit_start = None
            if self.doubt_engine.is_unknown():
                self.state = S.UNKNOWN
            else:
                self.state = S.OCCLUDED
            return self.state

        # Region is visible
        if self.state == S.OCCLUDED:
            # Exit hysteresis: visibility >= 0.90 for 2.0s
            if o.visibility >= 0.90:
                if self.occluded_exit_start is None:
                    self.occluded_exit_start = o.t
                elif (o.t - self.occluded_exit_start) >= 2.0:
                    self.state = S.SAFE if verified else S.UNKNOWN
                    self.occluded_exit_start = None
            else:
                self.occluded_exit_start = None
            return self.state

        if verified:
            self.state = S.SAFE
        elif self.doubt_engine.is_unknown():
            self.state = S.UNKNOWN

        return self.state
