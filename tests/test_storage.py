import json
from pathlib import Path

from assetwatch.storage import Store, GALLERY_MAX


def profile():
    return {"version": 1, "samples": [{"embedding": [1.0, 0.0], "histogram": [1.0], "orb_b64": "", "orb_rows": 0}]}


def test_create_asset_roundtrip_and_archive(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("Blue bottle", "bottle", "detector", profile(), [1, 2, 50, 60])
    assert asset["name"] == "Blue bottle"
    assert asset["status"] == "PRESENT"
    assert store.get_asset(asset["asset_id"])["profile"]["samples"][0]["embedding"] == [1.0, 0.0]
    events = store.list_events()
    assert len(events) == 1 and events[0]["event_type"] == "REGISTERED"
    assert store.set_status(asset["asset_id"], "MOVED", "Blue bottle moved") is True
    assert store.set_status(asset["asset_id"], "MOVED", "Blue bottle moved") is False
    assert store.deactivate_asset(asset["asset_id"]) is True
    assert store.get_asset(asset["asset_id"]) is None
    assert store.get_asset(asset["asset_id"], active_only=False)["active"] is False


def test_add_reference_keeps_bounded_profile(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("Laptop", "laptop", "detector", profile(), [0, 0, 20, 20])
    store.update_profile(asset["asset_id"], {"version": 1, "samples": [1, 2, 3]})
    loaded = store.get_asset(asset["asset_id"])
    assert loaded["profile"]["samples"] == [1, 2, 3]


def test_gallery_max_is_ten():
    """Gallery limit must be 10 (raised from 5 for viewpoint robustness)."""
    assert GALLERY_MAX == 10


def test_hard_delete_removes_asset_and_events(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("Key", "custom", "detector", profile(), [0, 0, 10, 10])
    aid   = asset["asset_id"]
    store.set_status(aid, "MOVED", "Key moved")
    assert len(store.list_events()) >= 1
    # Hard delete should remove everything.
    assert store.hard_delete_asset(aid) is True
    assert store.get_asset(aid, active_only=False) is None
    # Events gone too.
    remaining = [e for e in store.list_events() if e["asset_id"] == aid]
    assert remaining == []
    # Second call returns False (not found).
    assert store.hard_delete_asset(aid) is False


def test_hard_delete_removes_crop_files(tmp_path: Path):
    store   = Store(tmp_path / "db.sqlite3")
    ref_dir = tmp_path / "references"
    asset   = store.create_asset("Pen", "custom", "detector", profile(), [0, 0, 5, 5])
    aid     = asset["asset_id"]
    crop_dir = ref_dir / aid
    crop_dir.mkdir(parents=True)
    (crop_dir / "reference_01.jpg").write_bytes(b"fake")
    assert crop_dir.exists()
    store.hard_delete_asset(aid, reference_dir=ref_dir)
    assert not crop_dir.exists()


def test_settings_persist(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    store.set_setting("mode", "night")
    assert store.get_setting("mode") == "night"
    store.set_setting("mode", "day")
    assert store.get_setting("mode") == "day"
    assert store.get_setting("missing_key", "default") == "default"


def test_silent_profile_update_no_event(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("Watch", "custom", "detector", profile(), [0, 0, 50, 50])
    initial_event_count = len(store.list_events())
    store.update_profile(asset["asset_id"], {"version": 1, "samples": [1, 2]}, silent=True)
    # Silent update must NOT add a REFERENCE_ADDED event.
    assert len(store.list_events()) == initial_event_count


def test_reference_update_adds_event(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("Watch", "custom", "detector", profile(), [0, 0, 50, 50])
    initial_event_count = len(store.list_events())
    store.update_profile(asset["asset_id"], {"version": 1, "samples": [1, 2]}, silent=False)
    events = store.list_events()
    assert len(events) == initial_event_count + 1
    assert events[0]["event_type"] == "REFERENCE_ADDED"
