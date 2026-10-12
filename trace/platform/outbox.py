"""
Outbox store-and-forward module.
Ensures zero data loss during network disruptions, implements edge autonomy,
and tracks delivery latency and delayed badges upon reconnection.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("trace.outbox")


class OutboxManager:
    """
    Manages edge-to-server store-and-forward queue with simulated network-cut testing.
    """

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.network_connected: bool = True
        self._init_outbox_table()

    def _init_outbox_table(self) -> None:
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, DELIVERED, FAILED
            delivered_at REAL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at REAL DEFAULT 0.0
        )
        """)
        try:
            self.db.execute("ALTER TABLE outbox ADD COLUMN next_retry_at REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass
        self.db.commit()

    def set_network_state(self, connected: bool) -> None:
        """Toggles network connectivity for simulation and testing."""
        self.network_connected = connected

    def enqueue(self, event_id: str, payload_json: str, commit: bool = True) -> None:
        """Stores an event into the outbox for guaranteed delivery."""
        now = time.time()
        self.db.execute(
            """
            INSERT INTO outbox (event_id, payload_json, created_at, status, next_retry_at)
            VALUES (?, ?, ?, 'PENDING', 0.0)
            ON CONFLICT(event_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                status = CASE WHEN outbox.status = 'DELIVERED' THEN outbox.status ELSE 'PENDING' END
            """,
            (event_id, payload_json, now),
        )
        if commit:
            self.db.commit()

    def enqueue_conn(self, conn: sqlite3.Connection, event_id: str, payload_json: str) -> None:
        """Appends to the outbox queue within an external caller transaction (does not commit)."""
        now = time.time()
        conn.execute(
            """
            INSERT INTO outbox (event_id, payload_json, created_at, status, next_retry_at)
            VALUES (?, ?, ?, 'PENDING', 0.0)
            ON CONFLICT(event_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                status = CASE WHEN outbox.status = 'DELIVERED' THEN outbox.status ELSE 'PENDING' END
            """,
            (event_id, payload_json, now),
        )

    def get_pending_count(self) -> int:
        cur = self.db.cursor()
        cur.execute("SELECT COUNT(*) FROM outbox WHERE status = 'PENDING'")
        return cur.fetchone()[0]

    def flush(self, delivery_handler: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[str]:
        """
        Attempts to deliver pending outbox items if network is connected.
        Honors bounded exponential retry backoff.
        Returns list of delivered event IDs.
        """
        if not self.network_connected:
            return []

        now = time.time()
        cur = self.db.cursor()
        cur.execute(
            """
            SELECT id, event_id, payload_json, created_at, attempts
            FROM outbox
            WHERE status = 'PENDING' AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY id ASC
            """,
            (now,)
        )
        rows = cur.fetchall()

        delivered_ids = []

        for outbox_id, event_id, payload_json, created_at, attempts in rows:
            if delivery_handler is None:
                # Without an active delivery handler, jobs remain safely pending
                continue

            success = False
            try:
                success = bool(delivery_handler({"event_id": event_id, "payload": payload_json}))
            except Exception as e:
                logger.warning(f"Outbox delivery handler raised error for {event_id}: {e}")
                success = False

            if success:
                cur.execute(
                    "UPDATE outbox SET status = 'DELIVERED', delivered_at = ? WHERE id = ?",
                    (now, outbox_id)
                )
                delivered_ids.append(event_id)
            else:
                new_attempts = attempts + 1
                # Bounded exponential retry backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s max
                backoff_s = min(60.0, 1.0 * (2 ** min(new_attempts - 1, 6)))
                next_retry = now + backoff_s
                cur.execute(
                    "UPDATE outbox SET attempts = ?, next_retry_at = ? WHERE id = ?",
                    (new_attempts, next_retry, outbox_id)
                )

        self.db.commit()
        return delivered_ids
