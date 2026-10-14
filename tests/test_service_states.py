import time
from pathlib import Path

from assetwatch.service import CameraService
from assetwatch.storage import Store


def profile():
    return {"version": 1, "samples": [{"embedding": [1.0, 0.0], "histogram": [1.0], "orb_b64": "", "orb_rows": 0}]}


def test_camera_unhealthy_never_transitions_an_item_to_missing(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("tool", "custom", "csrt", profile(), [10, 10, 100, 100])
    service = CameraService(store, tmp_path, missing_after_s=5)
    service.asset_last_seen_mono[asset["asset_id"]] = time.monotonic() - 100
    service._update_asset_states([], 1280, 720, camera_healthy=False)
    assert store.get_asset(asset["asset_id"])["status"] == "PRESENT"


def test_long_absence_emits_missing_event_only_while_camera_healthy(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("tool", "custom", "csrt", profile(), [10, 10, 100, 100])
    service = CameraService(store, tmp_path, missing_after_s=5)
    service.asset_last_seen_mono[asset["asset_id"]] = time.monotonic() - 20
    service._update_asset_states([], 1280, 720, camera_healthy=True)
    assert store.get_asset(asset["asset_id"])["status"] == "MISSING"
    assert any(e["event_type"] == "MISSING" for e in store.list_events())


def test_large_position_change_marks_item_moved(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    asset = store.create_asset("tool", "custom", "detector", profile(), [10, 10, 80, 80])
    service = CameraService(store, tmp_path, missing_after_s=5)
    service._update_asset_states([{"asset_id": asset["asset_id"], "bbox": [540, 330, 610, 400], "similarity": 0.88}],
                                 1280, 720, camera_healthy=True)
    assert store.get_asset(asset["asset_id"])["status"] == "MOVED"
    assert any(e["event_type"] == "MOVED" for e in store.list_events())
