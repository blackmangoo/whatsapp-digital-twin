"""
Seed Script for WhatsApp Digital Twin Vector Knowledge Base on Supabase
1. Populates Key Real-World Entities (People, Places, Projects, Context)
2. Ingests substantive past chat turns into pgvector for semantic retrieval
"""

import os
import json
import urllib.request
from typing import List, Dict
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

print("Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def run_sql(query: str):
    req = urllib.request.Request(
        SQL_ENDPOINT,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

# 1. CORE ENTITIES
CORE_ENTITIES = [
    {
        "name": "Ammar Akbar (Self)",
        "aliases": ["ammar", "akbar", "me", "myself"],
        "relationship": "Self (Digital Twin Persona)",
        "summary": "Ammar Akbar is an AI/Computer Science student at FAST NUCES. Calm, witty, practical, busy with university labs and coding projects. Speaks in natural Roman Urdu and English mix. Avoids formal corporate greetings, talks in casual short bursts."
    },
    {
        "name": "Sardar Sameer",
        "aliases": ["sameer", "sardar", "bro", "yr", "bhai"],
        "relationship": "Close University Friend & Classmate",
        "summary": "Sardar Sameer is Ammar's close university friend and section-mate. They frequently discuss university assignments (C++, AI, Code 1 Assignment 5), share memes, talk about campus life, and tease each other playfully."
    },
    {
        "name": "Rai",
        "aliases": ["rai", "raen", "murtaza", "rai sahab"],
        "relationship": "University Friend & Classmate",
        "summary": "Rai is a mutual friend and classmate of Ammar and Sameer. Sameer and Rai have occasional friendly rivalries or arguments over university groups, assignments, and campus drama, but they are all in the same circle."
    },
    {
        "name": "FAST NUCES University",
        "aliases": ["uni", "campus", "fast", "lab", "class", "room"],
        "relationship": "University & Academic Life",
        "summary": "FAST NUCES is the university where Ammar and Sameer study Computer Science and AI. Known for demanding lab work, heavy programming assignments, and tight deadlines. Ammar frequently says he is 'lab mai hu' or will do work after going back to his room."
    },
    {
        "name": "Assignments & Coding Projects",
        "aliases": ["assignment", "code", "lab", "task1.cpp", "deadline", "quiz"],
        "relationship": "Academic Projects",
        "summary": "Ammar and Sameer collaborate and consult each other on programming tasks, such as C++ labs (task1.cpp), AI models, and semester assignments. Ammar usually reviews or helps when he returns to his room from the lab."
    }
]

def seed_entities():
    print(f"Seeding {len(CORE_ENTITIES)} core entities...")
    for ent in CORE_ENTITIES:
        text_to_embed = f"{ent['name']} ({ent['relationship']}): {ent['summary']}"
        emb = embedder.encode(text_to_embed).tolist()

        # Escape single quotes
        summary_esc = ent['summary'].replace("'", "''")
        name_esc = ent['name'].replace("'", "''")
        rel_esc = ent['relationship'].replace("'", "''")
        aliases_sql = "ARRAY[" + ", ".join(f"'{a}'" for a in ent['aliases']) + "]"

        sql = f"""
        DELETE FROM digital_twin_entities WHERE name = '{name_esc}';
        INSERT INTO digital_twin_entities (name, aliases, relationship, summary, embedding)
        VALUES ('{name_esc}', {aliases_sql}, '{rel_esc}', '{summary_esc}', '{emb}'::vector);
        """
        run_sql(sql)
    print("Entities seeded successfully!")

def seed_memories_from_data(file_path: str = "data/train.jsonl", max_memories: int = 150):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Extracting high-value memories from {file_path}...")
    memories = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            msgs = data.get("messages", [])[1:] # skip system
            if len(msgs) >= 2:
                user_msg = msgs[0]["content"].strip()
                asst_msg = msgs[1]["content"].strip()
                # Keep substantive conversations (>15 chars, not just 'ok' or 'hm')
                if len(user_msg) > 15 and len(asst_msg) > 10:
                    combined = f"Sameer: {user_msg}\nAmmar: {asst_msg}"
                    memories.append({
                        "sender": "Sameer & Ammar",
                        "content": combined,
                        "timestamp": "Historical WhatsApp"
                    })
            if len(memories) >= max_memories:
                break

    print(f"Embedding {len(memories)} conversation memories...")

    # Process in batches of 25
    batch_size = 25
    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]
        texts = [m["content"] for m in batch]
        embeddings = embedder.encode(texts).tolist()

        values_list = []
        for m, emb in zip(batch, embeddings):
            content_esc = m["content"].replace("'", "''")
            values_list.append(f"('{m['sender']}', '{content_esc}', '{m['timestamp']}', '{emb}'::vector)")

        sql = f"""
        INSERT INTO digital_twin_memories (sender, content, timestamp, embedding)
        VALUES {', '.join(values_list)};
        """
        run_sql(sql)
        print(f"  Ingested batch {i + len(batch)} / {len(memories)}")

    print("Memories seeded successfully!")

if __name__ == "__main__":
    seed_entities()
    seed_memories_from_data()
    print("\n[COMPLETE] Vector database on Supabase is fully populated and ready!")
