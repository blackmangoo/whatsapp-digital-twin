"""
WhatsApp Digital Twin - Live Agent Daemon
Connects your WhatsApp Business account via the official WhatsApp Agent Platform API.
Processes live incoming messages in real-time, displays typing indicators, queries
Supabase pgvector RAG + local Qwen 2.5 7B LoRA model, and sends authentic multi-bubble replies.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
from typing import Optional

from bot import WhatsAppTwinAgent
from whatsapp_client import WhatsAppAgentClient

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("WhatsAppLiveBot")

OFFSET_FILE = os.path.join(os.path.dirname(__file__), "last_offset.json")


def load_last_offset() -> Optional[int]:
    """Loads the last recorded event offset from disk to prevent duplicate replies."""
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("offset")
        except Exception as e:
            logger.warning(f"Could not read offset file: {e}")
    return None


def save_last_offset(offset: Optional[int]):
    """Persists the latest event offset."""
    if offset is None:
        return
    try:
        with open(OFFSET_FILE, "w", encoding="utf-8") as f:
            json.dump({"offset": offset, "updated_at": time.time()}, f)
    except Exception as e:
        logger.warning(f"Could not save offset file: {e}")


class LiveWhatsAppDaemon:
    def __init__(self, use_rag: bool = True, force_head: bool = True):
        logger.info("Initializing WhatsApp Digital Twin live agent...")
        self.client = WhatsAppAgentClient()
        self.agent = WhatsAppTwinAgent(use_rag=use_rag)
        self.running = True

        # Determine starting offset
        saved_offset = load_last_offset()
        if force_head or saved_offset is None:
            logger.info("Starting from HEAD (will only process brand-new incoming messages).")
            self.current_offset = None
        else:
            logger.info(f"Resuming from saved offset: {saved_offset}")
            self.current_offset = saved_offset

        # Register termination signal handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info("\nReceived exit signal. Shutting down WhatsApp Live Bot gracefully...")
        self.running = False

    def run(self):
        logger.info("=" * 65)
        logger.info(" 🟢 WHATSAPP LIVE DIGITAL TWIN IS RUNNING")
        logger.info(" 📱 Connected to WhatsApp Agent Platform API")
        logger.info(f" 🧠 RAG Grounding (Supabase pgvector): {'ACTIVE' if self.agent.use_rag else 'DISABLED'}")
        logger.info(f" 🤖 LLM Server Target: {self.agent.model_name}")
        logger.info(" Send a message to your WhatsApp Business number to start chatting!")
        logger.info(" Press Ctrl+C at any time to stop.")
        logger.info("=" * 65)

        consecutive_errors = 0

        while self.running:
            try:
                # 1. Long-poll WhatsApp for updates (15s timeout)
                messages, next_offset = self.client.poll_updates(
                    offset=self.current_offset,
                    limit=20,
                    timeout=15
                )

                # Persist updated offset
                if next_offset is not None and next_offset != self.current_offset:
                    self.current_offset = next_offset
                    save_last_offset(self.current_offset)

                consecutive_errors = 0

                # 2. Process incoming messages
                for msg in messages:
                    self._process_inbound_message(msg)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in main polling loop: {e}")
                time.sleep(min(15, 2 * consecutive_errors))

        logger.info("WhatsApp Live Bot stopped cleanly.")

    def _process_inbound_message(self, msg: dict):
        sender_id = msg.get("from")
        wamid = msg.get("id")
        msg_type = msg.get("type")
        sender_name = msg.get("_sender_name", "Friend")

        if not sender_id or not wamid:
            return

        # Handle non-text incoming types
        if msg_type != "text":
            logger.info(f"[{sender_name}] Sent non-text message ({msg_type}).")
            self.client.mark_as_read(wamid, show_typing=True)
            self.client.send_text_message(
                to_user=sender_id,
                text="Media check kr kay btata hun.",
                reply_to_wamid=wamid
            )
            return

        body = msg.get("text", {}).get("body", "").strip()
        if not body:
            return

        logger.info(f"\n📩 [INCOMING] From: {sender_name} ({sender_id})")
        logger.info(f"   Body: \"{body}\"")

        # 1. Immediate Read Receipt + Typing Indicator
        # This shows blue ticks and 'typing...' on the sender's phone immediately
        self.client.mark_as_read(wamid, show_typing=True)

        # 2. Generate Reply via Digital Twin (RAG + Qwen 2.5 7B LoRA)
        bubbles, simulated_delay = self.agent.generate_reply(sender_id, body)

        # 3. Realistic human typing latency
        # Cap delay between 1.5s and 5.0s for responsive live chat
        actual_delay = min(5.0, max(1.5, simulated_delay))
        logger.info(f"   Simulating typing for {actual_delay:.1f}s...")
        time.sleep(actual_delay)

        # 4. Dispatch multi-bubble burst response
        logger.info(f"📤 [OUTGOING] Sending {len(bubbles)} bubble(s) to {sender_name}:")
        for i, bubble in enumerate(bubbles):
            # First bubble quotes context message id; follow-up bubbles send as normal bursts
            reply_ctx = wamid if i == 0 else None
            sent_wamid = self.client.send_text_message(
                to_user=sender_id,
                text=bubble,
                reply_to_wamid=reply_ctx
            )
            logger.info(f"   └── Bubble {i+1}: \"{bubble}\" (wamid: {sent_wamid or 'sent'})")
            if i < len(bubbles) - 1:
                # Brief natural pause between message bubbles
                time.sleep(1.2)


def main():
    parser = argparse.ArgumentParser(description="Live WhatsApp Digital Twin Bot Daemon")
    parser.add_argument("--no-rag", action="store_true", help="Disable Supabase pgvector RAG")
    parser.add_argument("--from-start", action="store_true", help="Process from offset 0 instead of HEAD")
    args = parser.parse_args()

    daemon = LiveWhatsAppDaemon(
        use_rag=not args.no_rag,
        force_head=not args.from_start
    )
    daemon.run()


if __name__ == "__main__":
    main()
