"""Long-term identity persistence layer for AssetWatch AI.

Each registered asset owns an IdentityState that survives track-ID recycling,
short occlusions, and full exits from the frame.  The gallery (persistent profile
samples from SQLite) is the ground truth; the rolling buffer is the short-term
memory that catches fast re-entries before a full gallery lookup is needed.

Online enrichment: while an asset is PRESENT with high confidence, new diverse
embeddings are appended to the gallery automatically (up to GALLERY_MAX samples).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Configurable thresholds (all overridable via env vars).
import os
ENRICH_MIN_CONF  = float(os.getenv("ASSETWATCH_ENRICH_MIN_CONF",  "0.85"))
ENRICH_DIVERSITY = float(os.getenv("ASSETWATCH_ENRICH_DIVERSITY", "0.08"))
GALLERY_MAX      = int(  os.getenv("ASSETWATCH_GALLERY_MAX",      "10"))


@dataclass
class IdentityState:
    asset_id: str
    # Current or last known BoT-SORT track ID.
    last_track_id: int | None = None
    # Monotonic time of the last confirmed sighting.
    last_seen_mono: float = field(default_factory=time.monotonic)
    # Last bounding box [x1,y1,x2,y2].
    last_box: list[float] | None = None
    # Rolling short-term embeddings (last 6 sightings, RAM only).
    rolling_embeddings: deque = field(default_factory=lambda: deque(maxlen=6))
    # Current composite similarity score.
    confidence: float = 0.0
    # Consecutive inference cycles with a confirmed match.
    consecutive_matches: int = 0
    # Monotonic time when the asset was first not found.
    lost_since: float | None = None
    # True when the most recent recovery came from the gallery, not rolling buffer.
    recovered_from_gallery: bool = False
    # Pending enrichment embedding (set when a new diverse view was observed).
    pending_enrich_embedding: list[float] | None = None
    # Coverage: number of distinct gallery samples (approximates viewpoint diversity).
    gallery_size: int = 0


class IdentityCache:
    """In-memory identity registry — one entry per registered asset."""

    def __init__(self) -> None:
        self._states: dict[str, IdentityState] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure(self, asset_id: str) -> IdentityState:
        if asset_id not in self._states:
            self._states[asset_id] = IdentityState(asset_id=asset_id)
        return self._states[asset_id]

    def remove(self, asset_id: str) -> None:
        self._states.pop(asset_id, None)

    def get(self, asset_id: str) -> IdentityState | None:
        return self._states.get(asset_id)

    def all_states(self) -> list[IdentityState]:
        return list(self._states.values())

    # ------------------------------------------------------------------
    # Match / lost / recovered
    # ------------------------------------------------------------------

    def on_matched(self, asset_id: str, track_id: int | None,
                   box: list[float], embedding: list[float], score: float) -> None:
        s = self.ensure(asset_id)
        s.last_track_id = track_id
        s.last_seen_mono = time.monotonic()
        s.last_box = box
        if embedding:
            s.rolling_embeddings.append(embedding)
        s.confidence = score
        s.consecutive_matches += 1
        s.lost_since = None
        s.recovered_from_gallery = False

    def on_lost(self, asset_id: str) -> None:
        s = self._states.get(asset_id)
        if s is not None and s.lost_since is None:
            s.lost_since = time.monotonic()
            s.consecutive_matches = 0

    def on_recovered(self, asset_id: str, track_id: int | None,
                     box: list[float], embedding: list[float],
                     score: float, from_gallery: bool) -> None:
        s = self.ensure(asset_id)
        s.last_track_id = track_id
        s.last_seen_mono = time.monotonic()
        s.last_box = box
        if embedding:
            s.rolling_embeddings.append(embedding)
        s.confidence = score
        s.consecutive_matches = 1
        s.lost_since = None
        s.recovered_from_gallery = from_gallery

    # ------------------------------------------------------------------
    # Online gallery enrichment
    # ------------------------------------------------------------------

    def check_enrich(self, asset_id: str, embedding: list[float],
                     gallery_samples: list[dict],
                     score: float, box: list[float]) -> bool:
        """Return True and set pending_enrich_embedding when the current observation
        is a good candidate to add to the gallery.

        Criteria:
        - Confidence >= ENRICH_MIN_CONF (only enrich on high-quality matches).
        - Box area >= 40×40 px (not a tiny crop).
        - Cosine distance to every existing gallery embedding > ENRICH_DIVERSITY.
        - Gallery is below GALLERY_MAX OR the new embedding is more diverse than
          the worst existing sample (replacement policy).
        """
        s = self._states.get(asset_id)
        if s is None or score < ENRICH_MIN_CONF:
            return False
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w < 40 or h < 40:
            return False
        if not embedding:
            return False

        new_emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(new_emb)
        if norm < 1e-6:
            return False
        new_emb = new_emb / norm

        existing_embs: list[np.ndarray] = []
        for sample in gallery_samples:
            e = np.asarray(sample.get("embedding", []), dtype=np.float32)
            if e.size == 0 or e.shape != new_emb.shape:
                continue
            n = np.linalg.norm(e)
            if n > 1e-6:
                existing_embs.append(e / n)

        if not existing_embs:
            # No gallery yet — always accept.
            s.pending_enrich_embedding = embedding
            return True

        # Min cosine distance to nearest gallery neighbour (1 - cosine similarity).
        min_dist = min(
            1.0 - float(np.clip(np.dot(new_emb, ge), -1, 1))
            for ge in existing_embs
        )
        if min_dist <= ENRICH_DIVERSITY:
            # Too similar to an existing sample — skip.
            return False

        s.pending_enrich_embedding = embedding
        return True

    def pop_pending_enrich(self, asset_id: str) -> list[float] | None:
        s = self._states.get(asset_id)
        if s is None:
            return None
        emb = s.pending_enrich_embedding
        s.pending_enrich_embedding = None
        return emb

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def seconds_since_seen(self, asset_id: str) -> float:
        s = self._states.get(asset_id)
        if s is None:
            return float("inf")
        return time.monotonic() - s.last_seen_mono

    def is_lost(self, asset_id: str) -> bool:
        s = self._states.get(asset_id)
        return s is not None and s.lost_since is not None

    def lost_asset_ids(self) -> list[str]:
        return [s.asset_id for s in self._states.values() if s.lost_since is not None]

    # ------------------------------------------------------------------
    # Location plausibility
    # ------------------------------------------------------------------

    @staticmethod
    def plausible_location(last_box: list[float] | None, candidate_box: list[float],
                           elapsed_s: float, frame_w: int, frame_h: int,
                           max_speed_fraction: float = 0.55) -> bool:
        if last_box is None or elapsed_s > 30.0:
            return True
        diag = max(1.0, (frame_w ** 2 + frame_h ** 2) ** 0.5)
        acx = (last_box[0] + last_box[2]) / 2
        acy = (last_box[1] + last_box[3]) / 2
        ccx = (candidate_box[0] + candidate_box[2]) / 2
        ccy = (candidate_box[1] + candidate_box[3]) / 2
        dist_frac = ((ccx - acx) ** 2 + (ccy - acy) ** 2) ** 0.5 / diag
        return dist_frac <= max_speed_fraction * max(elapsed_s, 0.1)
