"""
Comprehensive Tests for the Cryptographic Evidence Ledger.
Validates Ed25519 digital signatures, SHA-256 hash chaining,
canonical JSON serialization, and all 10 tamper scenarios.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
import nacl.signing
import pytest

from trace.platform.db import SCHEMA_SQL
from trace.platform.ledger import LedgerManager, GENESIS, canon, sha256_bytes


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
    """1. Legitimate untampered history verifies cleanly."""
    event1 = {"id": "ev-1", "type": "MOVED", "asset_id": "safe-02", "confidence": 0.95}
    h1 = ledger_mgr.append(test_db, event1)["hash"]

    event2 = {"id": "ev-2", "type": "MISSING", "asset_id": "laptop-03", "confidence": 0.92}
    h2 = ledger_mgr.append(test_db, event2)["hash"]

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is True
    assert res["count"] == 2
    assert res["head"] == h2


def test_ledger_detects_body_tampering(test_db, ledger_mgr):
    """2. Modified event payload inside record body is caught."""
    event = {"id": "ev-1", "type": "MOVED", "asset_id": "safe-02", "confidence": 0.95}
    ledger_mgr.append(test_db, event)

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    cur = test_db.cursor()
    cur.execute("SELECT idx, body FROM ledger WHERE idx = 1")
    row = cur.fetchone()
    body_obj = json.loads(row[1])
    body_obj["event"]["confidence"] = 0.001  # Tamper confidence
    test_db.execute("UPDATE ledger SET body = ? WHERE idx = 1", (json.dumps(body_obj),))
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 1
    assert "Body hash mismatch" in res["why"]


def test_ledger_detects_deleted_middle_entry(test_db, ledger_mgr):
    """3. Deleted middle entry breaks the cryptographic hash chain."""
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "SAFE"})
    ledger_mgr.append(test_db, {"id": "ev-2", "type": "MOVED"})
    ledger_mgr.append(test_db, {"id": "ev-3", "type": "RETURNED"})

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_delete")
    test_db.execute("DELETE FROM ledger WHERE idx = 2")
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 3
    assert "Hash chain broken" in res["why"]


def test_ledger_detects_reordered_entries(test_db, ledger_mgr):
    """4. Reordered entries break the hash link sequence."""
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "SAFE"})
    ledger_mgr.append(test_db, {"id": "ev-2", "type": "MOVED"})

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    cur = test_db.cursor()
    cur.execute("SELECT body, hash, sig FROM ledger WHERE idx = 1")
    r1 = cur.fetchone()
    cur.execute("SELECT body, hash, sig FROM ledger WHERE idx = 2")
    r2 = cur.fetchone()

    # Swap rows
    cur.execute("UPDATE ledger SET body = ?, hash = ?, sig = ? WHERE idx = 1", r2)
    cur.execute("UPDATE ledger SET body = ?, hash = ?, sig = ? WHERE idx = 2", r1)
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert "Hash chain broken" in res["why"]


def test_ledger_detects_corrupt_signature(test_db, ledger_mgr):
    """5. Altered or forged Ed25519 signature fails verification."""
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "MOVED"})

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    # Replace signature with forged hex
    fake_sig = "ff" * 64
    test_db.execute("UPDATE ledger SET sig = ? WHERE idx = 1", (fake_sig,))
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 1
    assert "Cryptographic signature verification failed" in res["why"]


def test_ledger_detects_wrong_public_key(test_db, ledger_mgr):
    """6. Signature verification against an untrusted public key fails."""
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "MOVED"})

    # Generate different untrusted keypair
    untrusted_signing_key = nacl.signing.SigningKey.generate()
    untrusted_verify_key = untrusted_signing_key.verify_key

    res = ledger_mgr.verify(test_db, verify_disk_crops=False, trusted_verify_key=untrusted_verify_key)
    assert res["ok"] is False
    assert "Cryptographic signature verification failed" in res["why"]


def test_ledger_detects_corrupt_json_body(test_db, ledger_mgr):
    """7. Malformed non-canonical JSON is flagged."""
    ledger_mgr.append(test_db, {"id": "ev-1", "type": "MOVED"})

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    test_db.execute("UPDATE ledger SET body = '{invalid_json...' WHERE idx = 1")
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 1


def test_ledger_detects_modified_crop_on_disk(test_db, ledger_mgr, tmp_path):
    """8. Alteration of crop image on disk triggers evidence mismatch."""
    crop_file = tmp_path / "crop.jpg"
    crop_file.write_bytes(b"initial_jpeg_bytes_version_1")

    import hashlib
    initial_h = hashlib.sha256(b"initial_jpeg_bytes_version_1").hexdigest()

    event = {
        "id": "ev-crop-1",
        "type": "MOVED",
        "evidence": {"before": str(crop_file).replace("\\", "/")}
    }
    ledger_mgr.append(test_db, event, crop_hashes={"before": initial_h})

    # Verify untampered crop passes
    res_ok = ledger_mgr.verify(test_db, verify_disk_crops=True)
    assert res_ok["ok"] is True

    # Tamper crop bytes on disk
    crop_file.write_bytes(b"tampered_bytes_injected_by_attacker")

    res_bad = ledger_mgr.verify(test_db, verify_disk_crops=True)
    assert res_bad["ok"] is False
    assert "modified on disk" in res_bad["why"]


def test_ledger_anchor_detection(test_db, ledger_mgr):
    """9. Valid external checkpoints verify cleanly; tail truncation is detected."""
    h1 = ledger_mgr.append(test_db, {"id": "ev-1"})["hash"]
    h2 = ledger_mgr.append(test_db, {"id": "ev-2"})["hash"]

    # Valid anchors
    anchors = [(1, h1), (2, h2)]
    check_ok = ledger_mgr.check_anchors(test_db, anchors)
    assert check_ok["ok"] is True

    # Bad anchor (e.g. truncated history or altered block)
    bad_anchors = [(2, "0" * 64)]
    check_bad = ledger_mgr.check_anchors(test_db, bad_anchors)
    assert check_bad["ok"] is False
    assert check_bad["bad_anchor_idx"] == 2


def test_ledger_detects_broken_prev_link(test_db, ledger_mgr):
    """10. Broken previous-hash pointer with recalculated body hash is caught."""
    h1 = ledger_mgr.append(test_db, {"id": "ev-1"})["hash"]
    ledger_mgr.append(test_db, {"id": "ev-2"})

    test_db.execute("DROP TRIGGER IF EXISTS ledger_no_update")
    # For entry 2, point 'prev' to a forged hash and re-hash/sign to bypass check #1
    body_obj = {"event": {"id": "ev-2"}, "crops": {}, "prev": "deadbeef" * 8, "ts_ns": 12345}
    b_bytes = canon(body_obj)
    b_hash = sha256_bytes(b_bytes)
    sig = ledger_mgr.signing_key.sign(b_hash.encode("utf-8")).signature.hex()

    test_db.execute("UPDATE ledger SET body = ?, hash = ?, sig = ? WHERE idx = 2", (b_bytes.decode("utf-8"), b_hash, sig))
    test_db.commit()

    res = ledger_mgr.verify(test_db, verify_disk_crops=False)
    assert res["ok"] is False
    assert res["bad_idx"] == 2
    assert "Hash chain broken" in res["why"]
