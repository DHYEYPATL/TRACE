"""
Database module for TRACE.
Manages SQLite in WAL mode, implements the 13 required tables from Section 7.1,
and installs append-only triggers for the immutable cryptographic ledger.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 1. cameras: one row per feed
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ONLINE',
    last_heartbeat REAL DEFAULT 0,
    reference_frame_path TEXT,
    landmarks_json TEXT
);

-- 2. assets: registration output
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    class TEXT NOT NULL,
    criticality INTEGER NOT NULL DEFAULT 1,
    policy_mode TEXT NOT NULL DEFAULT 'fixed',
    camera_id TEXT NOT NULL,
    region_polygon TEXT NOT NULL, -- JSON array of points [[x,y],...]
    zone_polygon TEXT NOT NULL,   -- JSON array of points [[x,y],...]
    tag_id INTEGER,
    present_threshold REAL NOT NULL DEFAULT 0.70,
    active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (camera_id) REFERENCES cameras (id)
);

-- 3. baselines: several rows per asset (median, empty, variant)
CREATE TABLE IF NOT EXISTS baselines (
    id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    kind TEXT NOT NULL, -- 'median', 'empty', 'variant'
    image_path TEXT NOT NULL,
    embedding_blob BLOB,
    keypoints_blob BLOB,
    lighting_label TEXT DEFAULT 'default',
    created_at REAL NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets (id)
);

-- 4. states: current state, one row per asset
CREATE TABLE IF NOT EXISTS states (
    asset_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'SAFE',
    belief REAL NOT NULL DEFAULT 1.0,
    confidence REAL NOT NULL DEFAULT 1.0,
    last_verified_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets (id)
);

-- 5. incidents: folds repeats of events for the same asset
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    asset_id TEXT,
    opened_at REAL NOT NULL,
    closed_at REAL,
    status TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'
    counter INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (asset_id) REFERENCES assets (id)
);

-- 6. events: every incident trigger
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    asset_id TEXT,
    camera_id TEXT NOT NULL,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'NEW',
    detected_at REAL NOT NULL,
    received_at REAL NOT NULL,
    before_path TEXT,
    after_path TEXT,
    payload_json TEXT,
    incident_id TEXT,
    FOREIGN KEY (asset_id) REFERENCES assets (id),
    FOREIGN KEY (incident_id) REFERENCES incidents (id)
);

-- 6b. foreign_objects: unexpected objects detected in zones
CREATE TABLE IF NOT EXISTS foreign_objects (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    incident_id TEXT
);

-- 7. ledger: immutable append-only hash chain
CREATE TABLE IF NOT EXISTS ledger (
    idx INTEGER PRIMARY KEY AUTOINCREMENT,
    body TEXT NOT NULL,
    hash TEXT NOT NULL,
    sig TEXT NOT NULL
);

-- Triggers protecting ledger from updates or deletes
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
BEGIN
    SELECT RAISE(ABORT, 'ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
BEGIN
    SELECT RAISE(ABORT, 'ledger is append-only');
END;

-- 8. anchors: off-box proof of the head hash
CREATE TABLE IF NOT EXISTS anchors (
    id TEXT PRIMARY KEY,
    ledger_idx INTEGER NOT NULL,
    head_hash TEXT NOT NULL,
    channel TEXT NOT NULL,
    anchored_at REAL NOT NULL,
    ref TEXT
);

-- 9. actions: operator decisions
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL, -- ack, resolve, false_positive, authorize, escalate
    note TEXT,
    at REAL NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events (id)
);

-- 10. authorizations: authorized-move windows
CREATE TABLE IF NOT EXISTS authorizations (
    id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    starts_at REAL NOT NULL,
    ends_at REAL NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets (id)
);

-- 11. policies: versioned rules
CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    asset_id TEXT,
    yaml TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- 12. heartbeats: monitoring feeds
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    at REAL NOT NULL,
    queue_depth INTEGER NOT NULL DEFAULT 0,
    fps REAL NOT NULL DEFAULT 0.0,
    model_version TEXT NOT NULL
);

-- 13. users: operator accounts
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'operator' -- admin, operator, viewer
);
"""


class ManagedConnection:
    """Wraps sqlite3.Connection so that exiting the context manager automatically commits/rolls back AND closes the connection."""
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._conn.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class DatabaseManager:
    """Manages SQLite connections with thread safety, WAL configuration, and leak-free lifecycle."""

    def __init__(self, db_path: str = "data/trace.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> ManagedConnection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return ManagedConnection(conn)

    def init_db(self) -> None:
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)

            # Migration: Ensure incidents and events allow NULL asset_id (camera faults & foreign objects)
            cur = conn.execute("PRAGMA table_info(incidents)")
            for col in cur.fetchall():
                if col[1] == "asset_id" and col[3] == 1:
                    conn.execute("PRAGMA foreign_keys = OFF;")
                    conn.execute("""
                        CREATE TABLE incidents_migrated (
                            id TEXT PRIMARY KEY,
                            asset_id TEXT,
                            opened_at REAL NOT NULL,
                            closed_at REAL,
                            status TEXT NOT NULL DEFAULT 'OPEN',
                            counter INTEGER NOT NULL DEFAULT 1,
                            FOREIGN KEY (asset_id) REFERENCES assets (id)
                        );
                    """)
                    conn.execute("INSERT INTO incidents_migrated SELECT id, asset_id, opened_at, closed_at, status, counter FROM incidents;")
                    conn.execute("DROP TABLE incidents;")
                    conn.execute("ALTER TABLE incidents_migrated RENAME TO incidents;")
                    conn.execute("PRAGMA foreign_keys = ON;")
                    break

            cur = conn.execute("PRAGMA table_info(events)")
            for col in cur.fetchall():
                if col[1] == "asset_id" and col[3] == 1:
                    conn.execute("PRAGMA foreign_keys = OFF;")
                    conn.execute("""
                        CREATE TABLE events_migrated (
                            id TEXT PRIMARY KEY,
                            asset_id TEXT,
                            camera_id TEXT NOT NULL,
                            type TEXT NOT NULL,
                            severity TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            status TEXT NOT NULL DEFAULT 'NEW',
                            detected_at REAL NOT NULL,
                            received_at REAL NOT NULL,
                            before_path TEXT,
                            after_path TEXT,
                            payload_json TEXT,
                            incident_id TEXT,
                            FOREIGN KEY (asset_id) REFERENCES assets (id),
                            FOREIGN KEY (incident_id) REFERENCES incidents (id)
                        );
                    """)
                    conn.execute("INSERT INTO events_migrated SELECT id, asset_id, camera_id, type, severity, confidence, status, detected_at, received_at, before_path, after_path, payload_json, incident_id FROM events;")
                    conn.execute("DROP TABLE events;")
                    conn.execute("ALTER TABLE events_migrated RENAME TO events;")
                    conn.execute("PRAGMA foreign_keys = ON;")
                    break

            # Seed default admin user if not exists
            cur = conn.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO users (id, name, role) VALUES (?, ?, ?)",
                    ("usr-admin", "Security Operations Admin", "admin")
                )
            conn.commit()


_GLOBAL_DB: Optional[DatabaseManager] = None


def get_db(db_path: str = "data/trace.db") -> DatabaseManager:
    global _GLOBAL_DB
    if _GLOBAL_DB is None or str(_GLOBAL_DB.db_path) != db_path:
        _GLOBAL_DB = DatabaseManager(db_path)
    return _GLOBAL_DB
