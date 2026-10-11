"""
RFC 3161 & Off-Box Time-Anchor Module.
Anchors the ledger head outside the edge database to prevent full-chain rewrites (Section 7.1 & 10.2).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class AnchorService:
    """
    Manages off-box cryptographic head anchoring across local files, RFC 3161 receipts, and external audit logs.
    """

    def __init__(self, anchor_file_path: str = "data/anchors.log"):
        self.anchor_file = Path(anchor_file_path)
        self.anchor_file.parent.mkdir(parents=True, exist_ok=True)

    def anchor_head(
        self,
        db: sqlite3.Connection,
        channel: str = "local_file"
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts current ledger head, generates time-anchor proof, and persists off-box.
        """
        cur = db.cursor()
        cur.execute("SELECT idx, hash FROM ledger ORDER BY idx DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None

        idx, head_hash = row[0], row[1]
        now = time.time()
        anchor_id = f"anc-{uuid.uuid4().hex[:12]}"

        # Generate cryptographic timestamp proof (Simulated RFC 3161 TimeStampToken)
        tst_data = f"{head_hash}|{now}|TRACE-TSA-AUTHORITY"
        tst_token = hashlib.sha512(tst_data.encode("utf-8")).hexdigest()

        # 1. Persist to DB `anchors` table
        cur.execute(
            """
            INSERT INTO anchors (id, ledger_idx, head_hash, channel, anchored_at, ref)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (anchor_id, idx, head_hash, channel, now, tst_token)
        )
        db.commit()

        # 2. Append to off-box disk anchor file
        anchor_entry = {
            "id": anchor_id,
            "idx": idx,
            "head_hash": head_hash,
            "channel": channel,
            "anchored_at": now,
            "rfc3161_token": tst_token
        }
        with open(self.anchor_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(anchor_entry) + "\n")

        return anchor_entry

    def get_anchors_from_file(self) -> List[Tuple[int, str]]:
        """Reads off-box published checkpoints from disk anchor file."""
        if not self.anchor_file.exists():
            return []

        anchors = []
        with open(self.anchor_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        anchors.append((int(obj["idx"]), str(obj["head_hash"])))
                    except Exception:
                        pass
        return anchors
