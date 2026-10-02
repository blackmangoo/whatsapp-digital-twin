# WhatsApp Digital Twin AI: Fine-Tuning (QLoRA) + Advanced RAG (Supabase pgvector)

An end-to-end AI engineering architecture to create an authentic bilingual (English + Roman Urdu / Hinglish) digital twin on WhatsApp chat data.

Combines **parameter-efficient fine-tuning (QLoRA on Qwen 2.5 7B)** for style and conversational cadence with **advanced vector retrieval (Supabase `pgvector`)** for entity knowledge, relationship memory, and truth grounding.

---

## Architecture Overview

```
                                [ Incoming Message ]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 1. Safety & Sensitive Data Guardrail  │
                     │    Filters passwords, OTPs, finance   │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 2. Advanced Hybrid RAG Engine         │
                     │    • all-MiniLM-L6-v2 local embedding │
                     │    • Supabase pgvector (HNSW Index)   │
                     │    • Entity knowledge (People/Places) │
                     │    • Episodic chat history recall     │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 3. Fine-Tuned Model (Qwen 2.5 7B)     │
                     │    • Base: Qwen 2.5 7B Instruct       │
                     │    • LoRA Adapter (4-bit QLoRA)       │
                     │    • Generates in authentic voice     │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ 4. Human-like Post-Processing         │
                     │    • Multi-bubble message burst split │
                     │    • Dynamic typing latency simulator │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                             [ WhatsApp / CLI Output ]
```

---

## Why Fine-Tuning Alone Fails (The Engineering Insight)

1. **The Casual Chat Probability Trap:** In casual messaging, people mostly text 2-word acknowledgments (*"Hm"*, *"Ok"*, *"Kia hua"*, *"Haan"*). Fine-tuning solely on chat logs trains the model to default to low-information filler.
2. **The Factual Blind Spot:** Neural network weights cannot reliably memorize arbitrary relational facts (e.g., *"Who is Rai? Are we friends or fighting?"*).
3. **The Solution:** Decouple **Tone** from **Memory**:
   * **Fine-Tuning (LoRA):** Teaches the model *how* you talk (slang, Roman Urdu phrasing, capitalization, sentence rhythm).
   * **Vector RAG (Supabase pgvector):** Grounds the model with *what* you know (entities, friends, shared history, university projects).

---

## Tech Stack & Components

* **Base LLM:** Qwen 2.5 7B Instruct (top open-weight model for multilingual & romanized code-switching).
* **Fine-Tuning:** Unsloth QLoRA 4-bit (`rank=16`, `alpha=32`, 2 epochs on Kaggle T4 GPUs).
* **Local Inference:** `llama.cpp` Vulkan engine with dynamic runtime LoRA loading (`--lora lora_model.gguf`).
* **Vector Database:** Supabase PostgreSQL with `pgvector` extension and HNSW cosine similarity indexing.
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~5ms CPU inference).
* **Agent Engine:** Python (`bot.py`) with SQLite conversational context memory, burst message splitter, and typing latency calculator.

---

## Repository Structure

```
├── .gitignore                   # Excludes raw chats, models, and personal data
├── Modelfile                    # Ollama configuration
├── bot.py                       # Main agent engine with RAG integration & CLI tester
├── kaggle_train_unsloth.py      # Unsloth fine-tuning script for Kaggle/Colab
├── preprocess.py                # Chat parser, burst merger & ChatML dataset generator
├── rag_engine.py                # Supabase pgvector retrieval & dynamic prompt assembler
├── run_server.bat               # Automated local LLM server launcher
├── seed_rag.py                  # Embeds entities & historical memories into Supabase
├── whatsapp_twin_training.ipynb # Kaggle notebook for 1-click training
└── README.md                    # System documentation
```

---

## Quickstart Guide

### 1. Data Preprocessing
Export your 1-on-1 chat from WhatsApp ("Without Media"):
```bash
# Inspect detected sender names
python preprocess.py --input "my_chat.txt" --inspect_only

# Generate clean, burst-merged ChatML dataset
python preprocess.py --input "my_chat.txt" --my_name "Your Name"
```

### 2. Fine-Tuning (Kaggle or Google Colab)
1. Upload `data/train.jsonl` to Kaggle.
2. Open `whatsapp_twin_training.ipynb`, select **GPU T4 x 2**, turn **Internet ON**.
3. Run all cells. It trains in ~15 minutes and exports `lora_model.gguf` (~78 MB).
4. Download `lora_model.gguf` to this folder.

### 3. Setup Supabase Vector Database
Set your Supabase token in environment variables, then run:
```bash
python seed_rag.py
```
This sets up `digital_twin_entities` and `digital_twin_memories` with HNSW vector indexes and custom RPC search functions on Supabase.

### 4. Start Local Model Server
Run the automated launcher:
```cmd
.\run_server.bat
```
* Automatically fetches and caches the base Qwen 2.5 7B Instruct GGUF.
* Dynamically binds your fine-tuned `lora_model.gguf`.
* Listens on `http://127.0.0.1:8080`.

### 5. Chat with Your Digital Twin
In another terminal:
```bash
python bot.py
```
* With RAG enabled (default), it semantically retrieves real facts and past conversations.
* To compare without RAG, run: `python bot.py --no-rag`.

---

## Key Hurdles Faced & How They Were Solved

1. **WhatsApp Message "Bursts":** Real humans don't text in clean, single paragraphs; they send 3–5 rapid bursts.
   * *Solution:* Developed a 90-second time-window aggregation algorithm in `preprocess.py` to merge rapid messages into authentic conversational turns.
2. **Kaggle 20 GB Disk Limit with Merged GGUFs:** Merging 7B weights and quantizing full models requires ~30.5 GB of temporary disk space.
   * *Solution:* Decoupled the LoRA adapter export using `save_method="lora"`, producing a lightweight 78 MB GGUF adapter and dynamically attaching it at inference via `llama-server`.
3. **Ollama Deprecation of `ADAPTER` Directive:** Recent Ollama builds deprecated runtime LoRA adapters.
   * *Solution:* Integrated native `llama-server` directly into the local architecture with OpenAI-compatible API endpoints, maintaining zero cloud dependency.
4. **Conversational "Amnesia":** Standard fine-tuning failed on niche relationship questions (*"Who is Rai?"*).
   * *Solution:* Implemented hybrid RAG with Supabase `pgvector`, retrieving entity definitions and past chat snippets on the fly before generating replies.

---

## Author
Built by [Ammar Akbar](https://github.com/blackmangoo).
