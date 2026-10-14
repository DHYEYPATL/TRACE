"""SQLite persistence layer for AssetWatch AI."""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Maximum gallery samples per asset (raised from 5 for viewpoint robustness).
GALLERY_MAX = int(__import__("os").getenv("ASSETWATCH_GALLERY_MAX", "10"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id    TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT 'custom',
                    tracker_mode TEXT NOT NULL DEFAULT 'appearance',
                    profile_json TEXT NOT NULL,
                    anchor_box_json TEXT,
                    last_box_json TEXT,
                    status      TEXT NOT NULL DEFAULT 'PRESENT',
                    last_seen   TEXT NOT NULL,
                    last_similarity REAL,
                    active      INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id    TEXT,
                    event_type  TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assets_active   ON assets(active, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_created  ON events(created_at DESC);
            """)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        for key in ("anchor_box_json", "last_box_json"):
            out_key = "anchor_box" if key.startswith("anchor") else "last_box"
            raw = item.pop(key, None)
            item[out_key] = json.loads(raw) if raw else None
        item["active"] = bool(item["active"])
        return item

    # ------------------------------------------------------------------
    # Asset CRUD
    # ------------------------------------------------------------------

    def create_asset(self, name: str, category: str, tracker_mode: str,
                     profile: dict, anchor_box: list[float],
                     similarity: float | None = None) -> dict:
        asset_id = uuid.uuid4().hex[:10].upper()
        now = utc_now()
        with self._connect() as conn:
            conn.execute("""INSERT INTO assets
                (asset_id,name,category,tracker_mode,profile_json,anchor_box_json,last_box_json,
                 status,last_seen,last_similarity,active,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?)""",
                (asset_id, name.strip(), category.strip() or "custom", tracker_mode,
                 json.dumps(profile, separators=(",", ":")), json.dumps(anchor_box),
                 json.dumps(anchor_box), "PRESENT", now, similarity, now, now))
            conn.execute(
                "INSERT INTO events(asset_id,event_type,message,created_at) VALUES(?,?,?,?)",
                (asset_id, "REGISTERED", f"Registered {name.strip()}", now))
            row = conn.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        result = self._decode(row)
        assert result is not None
        return result

    def get_asset(self, asset_id: str, active_only: bool = True) -> dict | None:
        with self._connect() as conn:
            sql = "SELECT * FROM assets WHERE asset_id=?" + (" AND active=1" if active_only else "")
            row = conn.execute(sql, (asset_id,)).fetchone()
        return self._decode(row)

    def list_assets(self, active_only: bool = True) -> list[dict]:
        with self._connect() as conn:
            sql = ("SELECT * FROM assets" +
                   (" WHERE active=1" if active_only else "") +
                   " ORDER BY created_at DESC")
            rows = conn.execute(sql).fetchall()
        return [self._decode(row) for row in rows]  # type: ignore[list-item]

    def update_profile(self, asset_id: str, profile: dict,
                       silent: bool = False) -> None:
        """Persist profile_json.  Pass silent=True for enrichment writes (no event)."""
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE assets SET profile_json=?, updated_at=? WHERE asset_id=? AND active=1",
                (json.dumps(profile, separators=(",", ":")), now, asset_id))
            if not silent:
                conn.execute(
                    "INSERT INTO events(asset_id,event_type,message,created_at) VALUES(?,?,?,?)",
                    (asset_id, "REFERENCE_ADDED", "Added another appearance reference", now))

    def observe(self, asset_id: str, box: list[float],
                similarity: float | None = None) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE assets SET last_box_json=?,last_seen=?,last_similarity=?,updated_at=?"
                " WHERE asset_id=? AND active=1",
                (json.dumps([float(x) for x in box]), now, similarity, now, asset_id))

    def set_status(self, asset_id: str, status: str,
                   message: str | None = None) -> bool:
        now = utc_now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status,name FROM assets WHERE asset_id=? AND active=1",
                (asset_id,)).fetchone()
            if row is None or row["status"] == status:
                return False
            conn.execute(
                "UPDATE assets SET status=?,updated_at=? WHERE asset_id=?",
                (status, now, asset_id))
            if message:
                conn.execute(
                    "INSERT INTO events(asset_id,event_type,message,created_at) VALUES(?,?,?,?)",
                    (asset_id, status, message, now))
            return True

    # ------------------------------------------------------------------
    # Soft archive (keeps history)
    # ------------------------------------------------------------------

    def deactivate_asset(self, asset_id: str) -> bool:
        now = utc_now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM assets WHERE asset_id=? AND active=1",
                (asset_id,)).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE assets SET active=0,updated_at=? WHERE asset_id=?",
                (now, asset_id))
            conn.execute(
                "INSERT INTO events(asset_id,event_type,message,created_at) VALUES(?,?,?,?)",
                (asset_id, "ARCHIVED", f"Archived {row['name']}", now))
            return True

    # ------------------------------------------------------------------
    # Hard delete (full purge — no leftovers)
    # ------------------------------------------------------------------

    def hard_delete_asset(self, asset_id: str,
                          reference_dir: Path | None = None) -> bool:
        """Permanently remove an asset, its events, and optionally its crop files."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM assets WHERE asset_id=?",
                (asset_id,)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM events WHERE asset_id=?", (asset_id,))
            conn.execute("DELETE FROM assets WHERE asset_id=?", (asset_id,))
        # Remove crop images from disk.
        if reference_dir is not None:
            crop_dir = Path(reference_dir) / asset_id
            if crop_dir.exists():
                shutil.rmtree(crop_dir, ignore_errors=True)
        return True

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event(self, asset_id: str | None, event_type: str,
                  message: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            recent = conn.execute(
                "SELECT created_at FROM events"
                " WHERE asset_id IS ? AND event_type=? AND message=?"
                " ORDER BY event_id DESC LIMIT 1",
                (asset_id, event_type, message)).fetchone()
            if recent:
                age = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(recent["created_at"])).total_seconds()
                if age < 8:
                    return
            conn.execute(
                "INSERT INTO events(asset_id,event_type,message,created_at) VALUES(?,?,?,?)",
                (asset_id, event_type, message, now))

    def list_events(self, limit: int = 40) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.*, a.name AS asset_name FROM events e
                   LEFT JOIN assets a ON a.asset_id=e.asset_id
                   ORDER BY e.event_id DESC LIMIT ?""",
                (limit,)).fetchall()
        return [dict(row) for row in rows]

    def acknowledge_event(self, event_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE events SET acknowledged=1 WHERE event_id=?",
                (event_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Settings (monitoring mode, intro flag, etc.)
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key,value) VALUES(?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))
