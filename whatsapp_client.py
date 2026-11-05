"""
WhatsApp Agent Platform Client
Direct integration with the official WhatsApp Agent Platform API (https://api.whatsapp.com/agent/v1).
Supports long-poll updates, instant read receipts, typing indicators, and outbound message delivery.
"""

import os
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("WhatsAppClient")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [WA-Client] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

API_BASE_URL = "https://api.whatsapp.com/agent/v1"

def _load_env():
    """Loads environment variables from local .env or fallback locations."""
    search_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", "videos automation", ".env")
    ]
    for env_path in search_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

_load_env()


class WhatsAppAgentClient:
    def __init__(self, token: Optional[str] = None):
        self.token = (token or os.environ.get("WA_AGENT_TOKEN", "")).strip()
        if not self.token:
            raise ValueError(
                "WhatsApp Agent Token is missing. Provide it via constructor or set WA_AGENT_TOKEN in .env"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "WhatsAppDigitalTwin/1.0"
        })

    def poll_updates(
        self,
        offset: Optional[int] = None,
        limit: int = 50,
        timeout: int = 15
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Long-polls for inbound messages and receipts from WhatsApp.
        - offset: integer pointer. If None, starts from current head.
        - Returns: (list_of_messages, next_offset)
        """
        params: Dict[str, Any] = {
            "limit": min(max(1, limit), 100),
            "timeout": min(max(0, timeout), 25)
        }
        if offset is not None:
            params["offset"] = offset

        try:
            resp = self.session.get(f"{API_BASE_URL}/updates", params=params, timeout=timeout + 10)
            if resp.status_code == 204:
                # Normal long-poll timeout with zero updates
                return [], offset

            if resp.status_code == 200:
                data = resp.json()
                next_offset = data.get("next_offset", offset)
                extracted_messages = []

                for entry in data.get("entry", []):
                    for change in entry.get("changes", []):
                        val = change.get("value", {})
                        contacts_map = {
                            c.get("wa_id"): c.get("profile", {}).get("name", "Friend")
                            for c in val.get("contacts", [])
                            if "wa_id" in c
                        }
                        for msg in val.get("messages", []):
                            sender_id = msg.get("from")
                            sender_name = contacts_map.get(sender_id, "Friend")
                            msg["_sender_name"] = sender_name
                            extracted_messages.append(msg)

                return extracted_messages, next_offset

            if resp.status_code == 429:
                logger.warning("WhatsApp API rate limit (429) hit. Backing off for 5 seconds...")
                time.sleep(5)
                return [], offset

            logger.error(f"Updates endpoint returned {resp.status_code}: {resp.text[:200]}")
            time.sleep(2)
            return [], offset

        except requests.exceptions.Timeout:
            return [], offset
        except Exception as e:
            logger.error(f"Error polling WhatsApp updates: {e}")
            time.sleep(3)
            return [], offset

    def mark_as_read(self, message_id: str, show_typing: bool = True) -> bool:
        """
        Marks an inbound message as read and activates the real-time typing indicator.
        """
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        if show_typing:
            payload["typing_indicator"] = {"type": "text"}

        try:
            resp = self.session.post(
                f"{API_BASE_URL}/statuses",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to send read/typing status for {message_id}: {e}")
            return False

    def send_text_message(
        self,
        to_user: str,
        text: str,
        reply_to_wamid: Optional[str] = None
    ) -> Optional[str]:
        """
        Sends an outbound text message to a WhatsApp user.
        - to_user: Recipient user ID in format 'user:<id>'
        - text: Message content string
        - reply_to_wamid: Optional WhatsApp message id to quote in reply
        - Returns: server-assigned message id (wamid) or None on error
        """
        if not to_user or not text:
            return None

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": to_user,
            "type": "text",
            "text": {"body": text}
        }
        if reply_to_wamid:
            payload["context"] = {"message_id": reply_to_wamid}

        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{API_BASE_URL}/messages",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    wamid = data.get("messages", [{}])[0].get("id")
                    return wamid

                if resp.status_code == 429:
                    backoff = (attempt + 1) * 2
                    logger.warning(f"Rate limited while sending message. Backing off {backoff}s...")
                    time.sleep(backoff)
                    continue

                if resp.status_code == 503:
                    time.sleep(1)
                    continue

                logger.error(f"Send message failed [{resp.status_code}]: {resp.text[:200]}")
                break

            except Exception as e:
                logger.error(f"Send message error (attempt {attempt+1}): {e}")
                time.sleep(1)

        return None
