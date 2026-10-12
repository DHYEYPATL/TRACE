"""
Database Integrity and Crash Recovery Tests for TRACE.
Tests WAL persistence, foreign key enforcement, append-only triggers,
transaction rollbacks, and state preservation across process restarts.
"""

import json
import sqlite3
import time
from pathlib import Path
import pytest

from trace.config import AppConfig
from trace.pipeline import TracePipeline
from trace.platform.db import DatabaseManager, SCHEMA_SQL


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_trace.db"


@pytest.fixture
def db_mgr(db_path):
    return DatabaseManager(str(db_path))


def test_foreign_key_enforcement(db_mgr):
    """Ensure assets cannot reference a non-existent camera ID."""
    with db_mgr.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO assets (
                    id, name, class, camera_id, region_polygon, zone_polygon
                ) VALUES ('asset-orphan', 'Orphan', 'safe', 'cam-does-not-exist', '[]', '[]')
                """
            )


def test_append_only_trigger_blocks_updates(db_mgr):
    """Ensure database triggers strictly prohibit modifications to ledger records."""
    with db_mgr.get_connection() as conn:
        conn.execute("INSERT INTO ledger (body, hash, sig) VALUES ('body1', 'hash1', 'sig1')")
        conn.commit()

        with pytest.raises(sqlite3.DatabaseError) as excinfo:
            conn.execute("UPDATE ledger SET body = 'modified' WHERE idx = 1")
        assert "ledger is append-only" in str(excinfo.value)


def test_append_only_trigger_blocks_deletes(db_mgr):
    """Ensure database triggers strictly prohibit deletion of ledger records."""
    with db_mgr.get_connection() as conn:
        conn.execute("INSERT INTO ledger (body, hash, sig) VALUES ('body1', 'hash1', 'sig1')")
        conn.commit()

        with pytest.raises(sqlite3.DatabaseError) as excinfo:
            conn.execute("DELETE FROM ledger WHERE idx = 1")
        assert "ledger is append-only" in str(excinfo.value)


def test_transaction_rollback_preserves_consistency(db_mgr):
    """An interrupted multi-table write rolls back completely without partial state."""
    with db_mgr.get_connection() as conn:
        conn.execute("INSERT INTO cameras (id, name, source_uri) VALUES ('cam-1', 'Cam 1', 'synthetic')")
        conn.commit()

    conn = db_mgr.get_connection()
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            INSERT INTO assets (
                id, name, class, camera_id, region_polygon, zone_polygon
            ) VALUES ('asset-1', 'Asset 1', 'safe', 'cam-1', '[]', '[]')
            """
        )
        # Simulate interruption / exception before states record is written
        raise RuntimeError("Power outage simulation")
    except RuntimeError:
        conn.rollback()
    finally:
        conn.close()

    # Verify asset-1 was NOT committed
    with db_mgr.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM assets WHERE id = 'asset-1'")
        assert cur.fetchone()[0] == 0


def test_restart_preserves_consistent_state_and_history(db_path):
    """Simulates application restart: verifies state and incident history are preserved."""
    mgr1 = DatabaseManager(str(db_path))
    now = time.time()
    with mgr1.get_connection() as conn:
        conn.execute("INSERT INTO cameras (id, name, source_uri) VALUES ('cam-1', 'Cam 1', 'synthetic')")
        conn.execute(
            """
            INSERT INTO assets (
                id, name, class, camera_id, region_polygon, zone_polygon
            ) VALUES ('safe-02', 'Vault Safe', 'safe', 'cam-1', '[[0,0]]', '[[0,0]]')
            """
        )
        conn.execute(
            """
            INSERT INTO states (
                asset_id, state, belief, confidence, last_verified_at, updated_at
            ) VALUES ('safe-02', 'MOVED', 0.85, 0.92, ?, ?)
            """,
            (now, now)
        )
        conn.execute(
            """
            INSERT INTO incidents (id, asset_id, opened_at, status, counter)
            VALUES ('inc-1', 'safe-02', ?, 'OPEN', 1)
            """,
            (now,)
        )
        conn.execute(
            """
            INSERT INTO events (
                id, asset_id, camera_id, type, severity, confidence, detected_at, received_at, before_path, after_path, incident_id
            ) VALUES ('ev-1', 'safe-02', 'cam-1', 'MOVED', 'high', 0.92, ?, ?, 'data/evidence/before.jpg', 'data/evidence/after.jpg', 'inc-1')
            """,
            (now, now)
        )
        conn.commit()

    # Simulate complete shutdown and re-open
    mgr2 = DatabaseManager(str(db_path))
    with mgr2.get_connection() as conn:
        cur_state = conn.execute("SELECT state, belief FROM states WHERE asset_id = 'safe-02'").fetchone()
        assert cur_state[0] == "MOVED"
        assert cur_state[1] == 0.85

        cur_inc = conn.execute("SELECT status, counter FROM incidents WHERE id = 'inc-1'").fetchone()
        assert cur_inc[0] == "OPEN"
        assert cur_inc[1] == 1

        cur_ev = conn.execute("SELECT before_path, after_path FROM events WHERE id = 'ev-1'").fetchone()
        assert cur_ev[0] == "data/evidence/before.jpg"
        assert cur_ev[1] == "data/evidence/after.jpg"
