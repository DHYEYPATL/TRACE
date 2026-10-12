"""
Cryptographic Evidence Ledger module.
Implements Ed25519 signing, SHA-256 hash chaining, off-box anchor proofs,
and strict tamper detection matching Section 10.2 of the Blueprint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import nacl.encoding
from nacl.signing import SigningKey, VerifyKey

GENESIS = "0" * 64


def canon(obj: Any) -> bytes:
    """Canonical JSON encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    """Computes hexadecimal SHA-256 digest."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(filepath: Path | str) -> Optional[str]:
    """Computes SHA-256 digest of a disk file."""
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class LedgerManager:
    """
    Manages the cryptographic ledger, handles signing keys, and performs audit verification.
    """

    def __init__(self, key_path: str = "data/signing_key.hex"):
        self.key_path = Path(key_path)
        self.signing_key, self.verify_key = self._load_or_create_keys()

    def _load_or_create_keys(self) -> Tuple[SigningKey, VerifyKey]:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if self.key_path.exists():
            hex_data = self.key_path.read_text(encoding="utf-8").strip()
            sk = SigningKey(bytes.fromhex(hex_data))
        else:
            sk = SigningKey.generate()
            self.key_path.write_text(sk.encode(encoder=nacl.encoding.HexEncoder).decode("utf-8"))
        return sk, sk.verify_key

    def append(
        self,
        db: sqlite3.Connection,
        event: Dict[str, Any],
        crop_hashes: Optional[Dict[str, str]] = None,
        commit: bool = True,
    ) -> Dict[str, Any]:
        """
        Appends an event to the ledger with SHA-256 hash chaining and Ed25519 signature.
        Supports atomic caller transactions when commit=False.
        """
        if crop_hashes is None:
            crop_hashes = {}

        cur = db.cursor()
        cur.execute("SELECT idx, hash FROM ledger ORDER BY idx DESC LIMIT 1")
        row = cur.fetchone()
        prev = row[1] if row else GENESIS
        next_idx = (row[0] + 1) if row else 1

        body_dict = {
            "event": event,
            "crops": crop_hashes,
            "prev": prev,
            "ts_ns": time.time_ns(),
        }
        body_bytes = canon(body_dict)
        h = sha256_bytes(body_bytes)
        signed = self.signing_key.sign(h.encode("utf-8"))
        sig_hex = signed.signature.hex()

        cur.execute(
            "INSERT INTO ledger (body, hash, sig) VALUES (?, ?, ?)",
            (body_bytes.decode("utf-8"), h, sig_hex),
        )
        if commit:
            db.commit()

        return {
            "idx": next_idx,
            "hash": h,
            "prev": prev,
            "sig": sig_hex,
            "verify_key": self.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode("utf-8")
        }

    def verify(
        self,
        db: sqlite3.Connection,
        verify_disk_crops: bool = True,
        trusted_verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        """
        Verifies the full integrity of the cryptographic chain.
        Detects: body tampering, chain breakage, forged signatures, and crop modifications.
        """
        prev = GENESIS
        cur = db.cursor()
        cur.execute("SELECT idx, body, hash, sig FROM ledger ORDER BY idx ASC")
        rows = cur.fetchall()

        if not rows:
            return {"ok": True, "count": 0, "head": GENESIS, "message": "Ledger is empty (genesis)"}

        vkey = trusted_verify_key if trusted_verify_key is not None else self.verify_key

        for idx, body_str, h, sig_hex in rows:
            body_bytes = body_str.encode("utf-8")
            # 1. Body hash integrity
            if sha256_bytes(body_bytes) != h:
                return {
                    "ok": False,
                    "bad_idx": idx,
                    "why": f"Body hash mismatch at index {idx}. Data has been altered!"
                }

            # 2. Chain linkage
            try:
                body_obj = json.loads(body_str)
            except Exception as e:
                return {"ok": False, "bad_idx": idx, "why": f"Corrupt JSON body at index {idx}: {e}"}

            if body_obj.get("prev") != prev:
                return {
                    "ok": False,
                    "bad_idx": idx,
                    "why": f"Hash chain broken at index {idx}! Prev was {body_obj.get('prev')}, expected {prev}"
                }

            # 3. Cryptographic signature verification
            try:
                vkey.verify(h.encode("utf-8"), bytes.fromhex(sig_hex))
            except Exception as e:
                return {
                    "ok": False,
                    "bad_idx": idx,
                    "why": f"Cryptographic signature verification failed at index {idx}: {e}"
                }

            # 4. Evidence crop verification
            if verify_disk_crops and "crops" in body_obj:
                for crop_label, expected_hash in body_obj["crops"].items():
                    # check event payload for file path
                    ev = body_obj.get("event", {})
                    ev_evidence = ev.get("evidence", {})
                    c_path = ev_evidence.get(crop_label)
                    if c_path and Path(c_path).exists():
                        actual_h = sha256_file(c_path)
                        if actual_h != expected_hash:
                            return {
                                "ok": False,
                                "bad_idx": idx,
                                "why": f"Crop image '{crop_label}' modified on disk! Hash mismatch: expected {expected_hash}, got {actual_h}"
                            }

            prev = h

        return {
            "ok": True,
            "count": len(rows),
            "head": prev,
            "message": f"Ledger verified successfully ({len(rows)} entries checked, zero tampering detected)"
        }

    def check_anchors(self, db: sqlite3.Connection, anchors: List[Tuple[int, str]]) -> Dict[str, Any]:
        """Validates that off-box published checkpoints match current ledger state."""
        cur = db.cursor()
        for idx, expected_head in anchors:
            cur.execute("SELECT hash FROM ledger WHERE idx = ?", (idx,))
            row = cur.fetchone()
            if not row or row[0] != expected_head:
                return {
                    "ok": False,
                    "bad_anchor_idx": idx,
                    "why": f"Off-box anchor mismatch at index {idx}! Local: {row[0] if row else 'None'}, Anchored: {expected_head}"
                }
        return {"ok": True, "anchors_verified": len(anchors)}

    def inject_tamper_for_demo(self, db: sqlite3.Connection, target_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Demo simulation helper: temporarily disables SQLite triggers to tamper a ledger entry,
        allowing the operator to prove live that the tamper is immediately caught by 'Verify Ledger'.
        """
        cur = db.cursor()
        if target_idx is None:
            cur.execute("SELECT idx, body FROM ledger ORDER BY idx DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return {"ok": False, "why": "No ledger entries to tamper"}
            target_idx, old_body = row[0], row[1]
        else:
            cur.execute("SELECT body FROM ledger WHERE idx = ?", (target_idx,))
            row = cur.fetchone()
            if not row:
                return {"ok": False, "why": f"Entry {target_idx} not found"}
            old_body = row[0]

        try:
            body_obj = json.loads(old_body)
            # Alter event confidence or details
            if "event" in body_obj:
                body_obj["event"]["confidence"] = 0.0001
                body_obj["event"]["tampered_by_insider"] = True
            tampered_body = json.dumps(body_obj)

            # Drop triggers temporarily to simulate root database access
            cur.execute("DROP TRIGGER IF EXISTS ledger_no_update")
            cur.execute("DROP TRIGGER IF EXISTS ledger_no_delete")
            cur.execute("UPDATE ledger SET body = ? WHERE idx = ?", (tampered_body, target_idx))

            # Re-install triggers
            cur.execute("""
            CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            """)
            cur.execute("""
            CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            """)
            db.commit()
            return {"ok": True, "tampered_idx": target_idx, "message": f"Tampered entry #{target_idx} body"}
        except Exception as e:
            db.rollback()
            return {"ok": False, "why": str(e)}
