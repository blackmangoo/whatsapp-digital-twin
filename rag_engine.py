"""
Advanced RAG Engine for WhatsApp Digital Twin
Connects local embedding engine (all-MiniLM-L6-v2) with Supabase pgvector.
Retrieves entity facts, past chat context, and assembles dynamic grounded context.
"""

import os
import json
import urllib.request
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# Load environment variables from .env if present
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

SUPABASE_TOKEN = os.environ.get("SUPABASE_ACCESS_TOKEN", "")
PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF", "ttutisutchtdarbvybzx")
SQL_ENDPOINT = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"

class DigitalTwinRAG:
    _embedder = None

    def __init__(self):
        if DigitalTwinRAG._embedder is None:
            # Shared singleton model to save RAM and initialization time
            DigitalTwinRAG._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedder = DigitalTwinRAG._embedder

    def _query_supabase(self, sql: str) -> List[Dict]:
        try:
            req = urllib.request.Request(
                SQL_ENDPOINT,
                data=json.dumps({"query": sql}).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {SUPABASE_TOKEN}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[RAG Warning] Failed to query Supabase: {e}")
            return []

    def retrieve_context(self, incoming_message: str) -> Dict[str, List]:
        """
        Runs semantic search across entities and historical chat memories.
        """
        emb = self.embedder.encode(incoming_message).tolist()

        # 1. Search Entities
        entity_sql = f"SELECT name, relationship, summary, similarity FROM search_entities('{emb}'::vector, 0.25, 2);"
        matched_entities = self._query_supabase(entity_sql)

        # 2. Search Memories
        memory_sql = f"SELECT sender, content, msg_time, similarity FROM search_memories('{emb}'::vector, 0.28, 3);"
        matched_memories = self._query_supabase(memory_sql)

        return {
            "entities": matched_entities,
            "memories": matched_memories
        }

    def build_grounded_system_prompt(self, incoming_message: str) -> str:
        """
        Dynamically constructs an enriched system prompt with retrieved context.
        """
        retrieved = self.retrieve_context(incoming_message)
        entities = retrieved.get("entities", [])
        memories = retrieved.get("memories", [])

        sections = [
            "You are Ammar Akbar chatting on WhatsApp with your close friend.",
            "Talk naturally in your authentic bilingual voice, matching your exact casual tone, "
            "Roman Urdu/Hinglish phrasing, slang, sentence length, and capitalization.",
            "Never sound like a formal AI assistant. Avoid generic filler like 'Hm' or 'Kia hua' when you can share a real opinion or answer with context."
        ]

        if entities:
            sections.append("\n### RELEVANT PEOPLE & ENTITY KNOWLEDGE (Truth Grounding):")
            for ent in entities:
                sections.append(f"- **{ent['name']}** ({ent['relationship']}): {ent['summary']}")

        if memories:
            sections.append("\n### RELEVANT PAST CHAT MEMORIES:")
            for m in memories:
                # Format memory cleanly
                clean_content = m['content'].replace('\n', ' | ')
                sections.append(f"- {clean_content}")

        sections.append("\nNow reply naturally as Ammar Akbar based on what your friend is saying.")

        return "\n".join(sections)


if __name__ == "__main__":
    rag = DigitalTwinRAG()
    test_q = "ammar tujy pata rai kon h? rai say meri larai h ya dosti?"
    print(f"Test Query: {test_q}\n")
    prompt = rag.build_grounded_system_prompt(test_q)
    print("--- Dynamic Grounded Prompt ---")
    print(prompt)
