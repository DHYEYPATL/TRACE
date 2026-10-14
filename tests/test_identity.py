"""Tests for identity cache, enrichment logic, and monitoring mode."""
from __future__ import annotations

import time
import os
from pathlib import Path

import numpy as np
import pytest

from assetwatch.identity import IdentityCache, ENRICH_MIN_CONF, ENRICH_DIVERSITY, GALLERY_MAX
from assetwatch.service import _mode_is_idle, _is_active_hours
from assetwatch.storage import Store


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _norm_emb(v: list[float]) -> list[float]:
    a = np.asarray(v, dtype=np.float32)
    return (a / (np.linalg.norm(a) + 1e-9)).tolist()


def _sample_emb(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


def _big_box():
    return [0.0, 0.0, 200.0, 200.0]


def _profile_with(n: int, dim: int = 8) -> list[dict]:
    return [{"embedding": _sample_emb(i, dim), "histogram": [], "orb_b64": "", "orb_rows": 0}
            for i in range(n)]


# ------------------------------------------------------------------
# IdentityCache lifecycle
# ------------------------------------------------------------------

def test_ensure_creates_state():
    cache = IdentityCache()
    s = cache.ensure("ABC")
    assert s.asset_id == "ABC"
    assert s.consecutive_matches == 0
    assert s.lost_since is None


def test_on_matched_updates_state():
    cache = IdentityCache()
    emb   = _sample_emb(0)
    cache.on_matched("ABC", track_id=7, box=[0,0,50,50], embedding=emb, score=0.9)
    s = cache.get("ABC")
    assert s.last_track_id == 7
    assert s.confidence == 0.9
    assert s.consecutive_matches == 1
    assert s.lost_since is None
    assert list(s.rolling_embeddings)[-1] == emb


def test_on_lost_sets_lost_since():
    cache = IdentityCache()
    cache.ensure("ABC")
    cache.on_lost("ABC")
    s = cache.get("ABC")
    assert s.lost_since is not None
    assert s.consecutive_matches == 0


def test_on_lost_idempotent():
    cache = IdentityCache()
    cache.ensure("ABC")
    cache.on_lost("ABC")
    t1 = cache.get("ABC").lost_since
    time.sleep(0.01)
    cache.on_lost("ABC")  # should not overwrite lost_since
    assert cache.get("ABC").lost_since == t1


def test_on_recovered_clears_lost():
    cache = IdentityCache()
    cache.ensure("ABC")
    cache.on_lost("ABC")
    cache.on_recovered("ABC", track_id=5, box=[0,0,50,50],
                       embedding=_sample_emb(1), score=0.88, from_gallery=True)
    s = cache.get("ABC")
    assert s.lost_since is None
    assert s.recovered_from_gallery is True
    assert s.consecutive_matches == 1


def test_remove_deletes_state():
    cache = IdentityCache()
    cache.ensure("X")
    cache.remove("X")
    assert cache.get("X") is None


def test_is_lost_query():
    cache = IdentityCache()
    cache.ensure("A")
    assert not cache.is_lost("A")
    cache.on_lost("A")
    assert cache.is_lost("A")


def test_lost_asset_ids():
    cache = IdentityCache()
    cache.ensure("A"); cache.ensure("B"); cache.ensure("C")
    cache.on_lost("B")
    assert cache.lost_asset_ids() == ["B"]


def test_rolling_buffer_capped_at_six():
    cache = IdentityCache()
    for i in range(10):
        cache.on_matched("A", None, [0,0,10,10], _sample_emb(i), 0.9)
    assert len(cache.get("A").rolling_embeddings) == 6


# ------------------------------------------------------------------
# Enrichment logic
# ------------------------------------------------------------------

def test_enrichment_rejects_low_confidence():
    cache    = IdentityCache()
    cache.ensure("A")
    gallery  = _profile_with(1)
    accepted = cache.check_enrich("A", _sample_emb(99), gallery,
                                   score=ENRICH_MIN_CONF - 0.01, box=_big_box())
    assert not accepted


def test_enrichment_rejects_tiny_box():
    cache   = IdentityCache()
    cache.ensure("A")
    gallery = _profile_with(1)
    accepted = cache.check_enrich("A", _sample_emb(99), gallery,
                                   score=ENRICH_MIN_CONF + 0.05,
                                   box=[0.0, 0.0, 30.0, 30.0])  # < 40×40
    assert not accepted


def test_enrichment_rejects_duplicate_embedding():
    cache   = IdentityCache()
    cache.ensure("A")
    emb     = _sample_emb(7)
    gallery = [{"embedding": emb, "histogram": [], "orb_b64": "", "orb_rows": 0}]
    # Same embedding → cosine distance ≈ 0 → below diversity threshold.
    accepted = cache.check_enrich("A", emb, gallery,
                                   score=ENRICH_MIN_CONF + 0.05, box=_big_box())
    assert not accepted


def test_enrichment_accepts_diverse_embedding():
    cache   = IdentityCache()
    cache.ensure("A")
    existing = _sample_emb(1, dim=512)
    # Create an orthogonal embedding (very different direction).
    orth = np.zeros(512, dtype=np.float32)
    orth[256] = 1.0
    orth_emb = orth.tolist()
    gallery  = [{"embedding": existing, "histogram": [], "orb_b64": "", "orb_rows": 0}]
    accepted = cache.check_enrich("A", orth_emb, gallery,
                                   score=ENRICH_MIN_CONF + 0.05, box=_big_box())
    assert accepted
    pending = cache.pop_pending_enrich("A")
    assert pending == orth_emb


def test_enrichment_pop_clears_pending():
    cache   = IdentityCache()
    cache.ensure("A")
    orth = np.zeros(512, dtype=np.float32); orth[100] = 1.0
    gallery = [{"embedding": _sample_emb(1, dim=512), "histogram": [], "orb_b64": "", "orb_rows": 0}]
    cache.check_enrich("A", orth.tolist(), gallery, score=0.9, box=_big_box())
    cache.pop_pending_enrich("A")
    assert cache.pop_pending_enrich("A") is None


def test_enrichment_accepts_empty_gallery():
    cache   = IdentityCache()
    cache.ensure("A")
    accepted = cache.check_enrich("A", _sample_emb(5), [],
                                   score=ENRICH_MIN_CONF + 0.05, box=_big_box())
    assert accepted


# ------------------------------------------------------------------
# Plausible location
# ------------------------------------------------------------------

def test_plausible_location_nearby():
    assert IdentityCache.plausible_location(
        [100, 100, 200, 200], [105, 105, 205, 205],
        elapsed_s=1.0, frame_w=1280, frame_h=720)


def test_plausible_location_teleport_rejected():
    # Jump from top-left to bottom-right in 0.1 s — physically impossible.
    assert not IdentityCache.plausible_location(
        [0, 0, 10, 10], [1270, 710, 1280, 720],
        elapsed_s=0.1, frame_w=1280, frame_h=720)


def test_plausible_location_no_prior_always_true():
    assert IdentityCache.plausible_location(
        None, [500, 300, 600, 400], elapsed_s=0.1, frame_w=1280, frame_h=720)


def test_plausible_location_old_sighting_always_true():
    # After 30 s the object can be anywhere.
    assert IdentityCache.plausible_location(
        [0, 0, 10, 10], [1270, 710, 1280, 720],
        elapsed_s=35.0, frame_w=1280, frame_h=720)


# ------------------------------------------------------------------
# Monitoring mode
# ------------------------------------------------------------------

def test_mode_always_never_idle(monkeypatch):
    import assetwatch.service as svc
    monkeypatch.setattr(svc, "_MODE", "always")
    assert not svc._mode_is_idle()


def test_mode_night_always_idle(monkeypatch):
    import assetwatch.service as svc
    monkeypatch.setattr(svc, "_MODE", "night")
    assert svc._mode_is_idle()


def test_set_mode_persists_to_settings(tmp_path: Path):
    store   = Store(tmp_path / "db.sqlite3")
    service_mod = __import__("assetwatch.service", fromlist=["CameraService"])
    svc = service_mod.CameraService(store, tmp_path)
    svc.set_mode("day")
    assert store.get_setting("mode") == "day"
    svc.set_mode("always")
    assert store.get_setting("mode") == "always"


def test_set_mode_rejects_invalid(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    svc   = __import__("assetwatch.service", fromlist=["CameraService"]).CameraService(store, tmp_path)
    with pytest.raises(ValueError):
        svc.set_mode("invalid_mode")


# ------------------------------------------------------------------
# Hard-delete cleans live state in service
# ------------------------------------------------------------------

def test_hard_delete_cleans_service_state(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    svc   = __import__("assetwatch.service", fromlist=["CameraService"]).CameraService(store, tmp_path)
    asset = store.create_asset("Phone", "custom", "detector",
                               {"version":1,"samples":[]}, [0,0,50,50])
    aid   = asset["asset_id"]
    # Manually seed some live state.
    svc.asset_last_seen_mono[aid] = time.monotonic()
    svc.asset_visible_boxes[aid]  = [0, 0, 50, 50]
    svc._identity_cache.ensure(aid)
    svc.hard_delete_asset(aid)
    # All in-memory state must be gone.
    assert aid not in svc.asset_last_seen_mono
    assert aid not in svc.asset_visible_boxes
    assert svc._identity_cache.get(aid) is None
    assert store.get_asset(aid, active_only=False) is None
