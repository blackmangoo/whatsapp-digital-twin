"""
WhatsApp Digital Twin - Local Agent Engine
Connects to local Ollama instance, handles short-term memory,
calculates realistic human typing delay, and splits responses into natural WhatsApp bubbles.
"""

import time
import random
import re
import json
import sqlite3
import argparse
from typing import List, Dict, Tuple
import urllib.request
import urllib.error
from rag_engine import DigitalTwinRAG

LLM_API_BASE = "http://127.0.0.1:8080"
DEFAULT_MODEL = "qwen2.5-7b-instruct"

# Sensitive keywords where the bot should not auto-commit or make decisions
SENSITIVE_TRIGGERS = [
    "password", "otp", "pin", "credit card", "bank", "account number",
    "send money", "paisa", "transfer", "emergency", "hospital"
]

class MemoryStore:
    """Manages short-term conversation context per sender in SQLite."""
    def __init__(self, db_path: str = "chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_message(self, sender_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (sender_id, role, content) VALUES (?, ?, ?)",
                (sender_id, role, content)
            )

    def get_recent_history(self, sender_id: str, limit: int = 8) -> List[Dict[str, str]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE sender_id = ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (sender_id, limit)
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def clear_history(self, sender_id: str):
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE sender_id = ?", (sender_id,))


class WhatsAppTwinAgent:
    def __init__(
        self,
        model_name: str = None,
        system_prompt: str = None,
        db_path: str = "chat_history.db",
        use_rag: bool = True
    ):
        self.memory = MemoryStore(db_path)
        self.use_rag = use_rag
        self.rag = DigitalTwinRAG() if use_rag else None
        self.model_name = model_name or self._detect_model_name()
        self.fallback_prompt = system_prompt or (
            "You are Ammar Akbar chatting on WhatsApp. Talk naturally in your authentic bilingual voice, "
            "matching your casual tone, Roman Urdu/Hinglish phrasing, slang, sentence length, and capitalization. "
            "Never sound like an AI assistant. Never write formal paragraphs."
        )

    def _detect_model_name(self) -> str:
        """Auto-detects the currently loaded model name from the LLM server."""
        try:
            req = urllib.request.Request(f"{LLM_API_BASE}/v1/models", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
                if "models" in data and len(data["models"]) > 0:
                    return data["models"][0]["name"]
        except Exception:
            pass
        return "qwen2.5-7b-instruct"

    def is_sensitive(self, text: str) -> bool:
        """Flags messages that touch sensitive or high-risk topics."""
        lower = text.lower()
        return any(trigger in lower for trigger in SENSITIVE_TRIGGERS)

    def generate_reply(self, sender_id: str, incoming_message: str) -> Tuple[List[str], float]:
        """
        1. Checks safety / emergency keywords.
        2. Retrieves RAG entities & memories from Supabase pgvector.
        3. Retrieves recent conversational context.
        4. Queries local LLM server.
        5. Splits output into multi-message bubbles.
        6. Computes realistic human typing delay.
        """
        # Safety check
        if self.is_sensitive(incoming_message):
            print(f"[Warning] Sensitive message detected from {sender_id}. Auto-reply safe fallback.")
            return [
                "bhai thora busy hun abhi, thori der me call ya msg karta hun."
            ], 3.0

        # Update user message in memory
        self.memory.add_message(sender_id, "user", incoming_message)

        # Dynamic RAG Grounded System Prompt
        if self.use_rag and self.rag:
            active_system_prompt = self.rag.build_grounded_system_prompt(incoming_message)
        else:
            active_system_prompt = self.fallback_prompt

        # Build payload for OpenAI-compatible llama-server / Ollama
        history = self.memory.get_recent_history(sender_id, limit=8)
        messages = [{"role": "system", "content": active_system_prompt}] + history

        raw_reply = None

        # 1. Try OpenAI-compatible endpoint (llama-server /v1/chat/completions)
        openai_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 128,
            "stream": False
        }

        try:
            req = urllib.request.Request(
                f"{LLM_API_BASE}/v1/chat/completions",
                data=json.dumps(openai_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_reply = data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            # Check if server expects Ollama /api/chat
            if e.code == 404:
                ollama_payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.15}
                }
                try:
                    req = urllib.request.Request(
                        f"{LLM_API_BASE}/api/chat",
                        data=json.dumps(ollama_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        raw_reply = data["message"]["content"].strip()
                except Exception as inner_e:
                    print(f"[Error] Failed to connect to LLM server on {LLM_API_BASE}: {inner_e}")
                    return ["(Error: LLM server not responding)"], 0.0
            else:
                print(f"[Error] HTTP {e.code} from LLM server: {e}")
                return ["(Error: LLM server returned error)"], 0.0
        except Exception as e:
            print(f"[Error] Request failed: {e}")
            return ["(Error: LLM server timeout or connection failure)"], 0.0

        # Save assistant reply to memory
        self.memory.add_message(sender_id, "assistant", raw_reply)

        # Split reply into natural WhatsApp bubbles
        bubbles = self.split_into_whatsapp_bubbles(raw_reply)

        # Compute realistic human typing delay
        total_chars = sum(len(b) for b in bubbles)
        # Average typing speed: ~25-35 chars/sec + thinking time
        simulated_delay = min(12.0, max(2.0, (total_chars * 0.05) + random.uniform(1.0, 2.5)))

        return bubbles, simulated_delay

    @staticmethod
    def split_into_whatsapp_bubbles(text: str) -> List[str]:
        """
        Splits a single paragraph into 2-3 separate WhatsApp messages
        so it feels like a real person typing in bursts.
        """
        # If the model already separated ideas by newlines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines

        # Otherwise, split on sentence terminators (. ! ?) if text is long
        if len(text) > 80:
            parts = re.split(r'(?<=[.!?])\s+', text)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]

        return [text]


# ----------------- CLI INTERACTIVE TEST -----------------
def interactive_cli():
    parser = argparse.ArgumentParser(description="Test your WhatsApp Digital Twin locally")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model tag")
    parser.add_argument("--sender", type=str, default="friend_test", help="Test sender ID")
    parser.add_argument("--no-rag", action="store_true", help="Disable Supabase pgvector RAG")
    args = parser.parse_args()

    use_rag = not args.no_rag
    agent = WhatsAppTwinAgent(model_name=args.model, use_rag=use_rag)

    print("\n" + "="*55)
    print(f" WhatsApp Twin CLI Tester [Model: {args.model}]")
    print(f" RAG Grounding (Supabase pgvector): {'ENABLED' if use_rag else 'DISABLED'}")
    print(" Type a message as your friend. Press Ctrl+C or 'exit' to quit.")
    print(" Type '/clear' to reset conversation history.")
    print("="*55 + "\n")

    while True:
        try:
            user_input = input("Friend: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input.lower() == "/clear":
                agent.memory.clear_history(args.sender)
                print("[Memory cleared for this chat]\n")
                continue

            bubbles, delay = agent.generate_reply(args.sender, user_input)

            print(f"*(Simulating typing for {delay:.1f}s...)*")
            time.sleep(min(delay, 2.0)) # In CLI test, cap wait time to 2s for convenience

            print("You (Twin):")
            for bubble in bubbles:
                print(f"  └── \"{bubble}\"")
            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    interactive_cli()
