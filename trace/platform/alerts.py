"""
Alert router and notification dispatcher.
Routes alerts to Telegram, ntfy, and on-screen WebSocket clients based on policy and severity.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional
import requests

from trace.config import AlertConfig

logger = logging.getLogger("trace.alerts")


class AlertRouter:
    """Dispatches notifications across configured external and local channels."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.websocket_subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def register_ws_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self.websocket_subscribers.append(cb)

    def dispatch(self, event: Dict[str, Any]) -> None:
        """
        Dispatches an event asynchronously to prevent blocking the vision loop.
        """
        threading.Thread(target=self._dispatch_sync, args=(event,), daemon=True).start()

    def _dispatch_sync(self, event: Dict[str, Any]) -> None:
        severity = event.get("severity", "medium").lower()
        ev_type = event.get("type", "UNKNOWN")
        asset_id = event.get("asset_id", "Unknown Asset")
        conf = event.get("confidence", 0.0)

        # 1. Local WebSocket broadcast (always)
        msg = {
            "kind": "alert",
            "event": event,
            "banner_text": f"[{severity.upper()}] {asset_id}: {ev_type} (conf {conf:.2f})"
        }
        for cb in self.websocket_subscribers:
            try:
                cb(msg)
            except Exception as e:
                logger.warning(f"Error in websocket alert callback: {e}")

        # 2. External channels for medium and high severity
        if severity in ("medium", "high"):
            self._send_telegram(event)
            self._send_ntfy(event)

    def _send_telegram(self, event: Dict[str, Any]) -> None:
        tg_cfg = self.cfg.telegram
        if not tg_cfg.enabled or not tg_cfg.bot_token or not tg_cfg.chat_id:
            return

        ev_type = event.get("type")
        asset_id = event.get("asset_id")
        conf = event.get("confidence", 0.0)
        sev = event.get("severity", "MEDIUM").upper()

        text = (
            f"🚨 *TRACE SECURITY ALERT* 🚨\n"
            f"*Severity:* {sev}\n"
            f"*Asset:* `{asset_id}`\n"
            f"*Event:* *{ev_type}*\n"
            f"*Confidence:* `{conf:.2f}`\n"
            f"*Time:* `{event.get('detected_at')}`\n\n"
            f"Review incident in TRACE Operations Console."
        )

        url = f"https://api.telegram.org/bot{tg_cfg.bot_token}/sendMessage"
        payload = {
            "chat_id": tg_cfg.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload, timeout=4.0)
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")

    def _send_ntfy(self, event: Dict[str, Any]) -> None:
        ntfy_cfg = self.cfg.ntfy
        if not ntfy_cfg.enabled or not ntfy_cfg.topic:
            return

        sev = event.get("severity", "medium").lower()
        prio = 5 if sev == "high" else 3
        url = f"https://ntfy.sh/{ntfy_cfg.topic}"

        title = f"TRACE Alert: {event.get('asset_id')} {event.get('type')}"
        msg = f"Severity: {sev.upper()} | Confidence: {event.get('confidence', 0.0):.2f}"

        try:
            requests.post(
                url,
                data=msg.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": str(prio),
                    "Tags": "warning,shield"
                },
                timeout=4.0
            )
        except Exception as e:
            logger.error(f"Failed to deliver ntfy alert: {e}")
