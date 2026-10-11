"""
WebSocket Manager for real-time frontend streaming.
Broadcasts asset state updates, health metrics, and alert triggers to active browser dashboards.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List
from fastapi import WebSocket

logger = logging.getLogger("trace.websocket")


class ConnectionManager:
    """Manages active browser WebSocket subscriptions."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected (Total: {len(self.active_connections)})")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected (Total: {len(self.active_connections)})")

    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        async with self._lock:
            dead_connections = []
            msg_str = json.dumps(data)
            for conn in self.active_connections:
                try:
                    await conn.send_text(msg_str)
                except Exception:
                    dead_connections.append(conn)

            for d in dead_connections:
                if d in self.active_connections:
                    self.active_connections.remove(d)
