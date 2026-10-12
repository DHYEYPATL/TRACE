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

    def broadcast_ws(self, event: Dict[str, Any]) -> None:
        """Broadcasts alert locally to connected WebSocket subscribers."""
        severity = event.get("severity", "medium").lower()
        ev_type = event.get("type", "UNKNOWN")
        asset_id = event.get("asset_id", "Unknown Asset")
        conf = event.get("confidence", 0.0)

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

    def send_external(self, event: Dict[str, Any]) -> bool:
        """
        Delivers notification to external channels (Telegram, ntfy).
        Returns True if and only if at least one external channel is configured AND all configured channels succeeded.
        Never reports True if no external channel is configured or if delivery fails.
        """
        severity = event.get("severity", "medium").lower()
        if severity not in ("medium", "high"):
            return False

        has_telegram = bool(self.cfg.telegram.enabled and self.cfg.telegram.bot_token and self.cfg.telegram.chat_id)
        has_ntfy = bool(self.cfg.ntfy.enabled and self.cfg.ntfy.topic)

        if not has_telegram and not has_ntfy:
            logger.debug("No external notification channels (Telegram, ntfy) configured.")
            return False

        tg_ok = self._send_telegram(event) if has_telegram else True
        ntfy_ok = self._send_ntfy(event) if has_ntfy else True
        return tg_ok and ntfy_ok

    def dispatch(self, event: Dict[str, Any]) -> bool:
        """
        Dispatches both local websocket broadcast and external channels.
        Returns True if external delivery succeeded.
        """
        self.broadcast_ws(event)
        return self.send_external(event)

    def _send_telegram(self, event: Dict[str, Any]) -> bool:
        tg_cfg = self.cfg.telegram
        if not tg_cfg.enabled or not tg_cfg.bot_token or not tg_cfg.chat_id:
            return False

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
            resp = requests.post(url, json=payload, timeout=4.0)
            if resp.status_code >= 400:
                logger.error(f"Failed to deliver Telegram alert: HTTP {resp.status_code}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")
            return False

    def _send_ntfy(self, event: Dict[str, Any]) -> bool:
        ntfy_cfg = self.cfg.ntfy
        if not ntfy_cfg.enabled or not ntfy_cfg.topic:
            return False

        sev = event.get("severity", "medium").lower()
        prio = 5 if sev == "high" else 3
        url = f"https://ntfy.sh/{ntfy_cfg.topic}"

        title = f"TRACE Alert: {event.get('asset_id')} {event.get('type')}"
        msg = f"Severity: {sev.upper()} | Confidence: {event.get('confidence', 0.0):.2f}"

        try:
            resp = requests.post(
                url,
                data=msg.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": str(prio),
                    "Tags": "warning,shield"
                },
                timeout=4.0
            )
            if resp.status_code >= 400:
                logger.error(f"Failed to deliver ntfy alert: HTTP {resp.status_code}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to deliver ntfy alert: {e}")
            return False
