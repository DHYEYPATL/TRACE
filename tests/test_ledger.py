"""
Tests for the Cryptographic Evidence Ledger (Section 10.5 #1).
Verifies Ed25519 signatures, SHA-256 hash chaining, and tamper detection.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
import pytest

from trace.platform.db import SCHEMA_SQL
from trace.platform.ledger import LedgerManager, GENESIS


@pytest.fixture
def test_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    return conn


@pytest.fixture
def ledger_mgr(tmp_path):
    key_file = tmp_path / "test_key.hex"
    return LedgerManager(str(key_file))


def test_ledger_append_and_verify(test_db, ledger_mgr):
    event1 = {"id": "ev-1", "type": "MOVED", "asset_id": "safe-02", "confidence": 0.95}
    h1 = ledger_mgr.append(test_db, event1)["hash"]

    event2 = {"id": "ev-2", "type": "MISSING", "asset_id": "laptop-03", "confidence": 0.92}
    h2 = ledger_mgr.append(test_db, event2)["hash"]

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is True
    assert res["count"] == 2
    assert res["head"] == h2


def test_ledger_detects_body_tampering(test_db, ledger_mgr):
    event = {"id": "ev-1", "type": "MOVED", "asset_id": "safe-02", "confidence": 0.95}
    ledger_mgr.append(test_db, event)

    # Simulate malicious insider dropping triggers and modifying body
    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    cur = test_db.cursor()
    cur.execute("SELECT idx, body FROM ledger WHERE idx = 1")
    row = cur.fetchone()
    body_obj = json.loads(row[1])
    body_obj["event"]["confidence"] = 0.001  # Tamper confidence
    test_db.execute("UPDATE ledger SET body = ? WHERE idx = 1", (json.dumps(body_obj),))
    test_db.commit()

    # Run verification
    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 1
    assert "Body hash mismatch" in res["why"]


def test_ledger_detects_deleted_entry(test_db, ledger_mgr):
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "SAFE"})
    ledger_mgr.append(test_db, {"id": "ev-2", "type": "MOVED"})
    ledger_mgr.append(test_db, {"id": "ev-3", "type": "RETURNED"})

    # Drop trigger and delete middle entry
    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_delete")
    test_db.execute("DELETE FROM ledger WHERE idx = 2")
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 3
    assert "Hash chain broken" in res["why"]


def test_ledger_anchor_detection(test_db, ledger_mgr):
    h1 = ledger_mgr.append(test_db, {"id": "ev-1"})["hash"]
    h2 = ledger_mgr.append(test_db, {"id": "ev-2"})["hash"]

    # Valid anchor
    anchors = [(1, h1), (2, h2)]
    check_ok = ledger_mgr.check_anchors(test_db, anchors)
    assert check_ok["ok"] is True

    # Bad anchor (e.g. truncated tail)
    bad_anchors = [(2, "0" * 64)]
    check_bad = ledger_mgr.check_anchors(test_db, bad_anchors)
    assert check_bad["ok"] is False
    assert check_bad["bad_anchor_idx"] == 2
