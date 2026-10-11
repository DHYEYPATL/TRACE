"""
The Doubt Engine (Belief Confidence Model).
Implements continuous confidence decay for unseen assets (Section 6.3).
Guarantees the system never guesses and never remains silent.
"""

from __future__ import annotations

import math
from typing import Optional


class DoubtEngine:
    """
    Computes real-time belief confidence C(t) in [0, 1] for an asset.
    Decays exponentially over time when the asset is not verifiably observed.
    """

    def __init__(self, half_life_s: float = 30.0):
        self.half_life_s = max(1.0, float(half_life_s))
        self.decay_lambda = math.log(2.0) / self.half_life_s
        self.belief: float = 1.0
        self.last_seen_t: Optional[float] = None
        self.last_step_t: Optional[float] = None

    def reset_sighted(self, current_t: float) -> float:
        """Called upon verified visual confirmation. Resets confidence to 1.0."""
        self.belief = 1.0
        self.last_seen_t = current_t
        self.last_step_t = current_t
        return self.belief

    def step(
        self,
        current_t: float,
        is_verified: bool,
        is_occluded: bool,
        camera_ok: bool
    ) -> float:
        """
        Steps belief forward in time according to observation conditions.
        """
        if self.last_step_t is None:
            self.last_step_t = current_t
            self.last_seen_t = current_t
            return self.belief

        dt = max(0.0, current_t - self.last_step_t)
        self.last_step_t = current_t

        if is_verified and camera_ok and not is_occluded:
            # Verified sighting resets confidence to maximum
            self.belief = 1.0
            self.last_seen_t = current_t
            return self.belief

        # If merely occluded by staff (expected normal branch activity), decay at 0.25x speed
        # If camera is blind, obstructed, or asset missing, decay at full 1.0x speed
        rate_mult = 0.25 if (is_occluded and camera_ok) else 1.0
        self.belief *= math.exp(-self.decay_lambda * rate_mult * dt)
        self.belief = max(0.0, min(1.0, self.belief))

        return self.belief

    def is_unknown(self) -> bool:
        """Below 0.5 confidence, the asset enters UNKNOWN state."""
        return self.belief < 0.50
