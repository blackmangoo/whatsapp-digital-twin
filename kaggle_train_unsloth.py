"""
WhatsApp Digital Twin - Qwen 2.5 7B Instruct Fine-Tuning with Unsloth
Environment: Kaggle GPU (T4 x 2 or P100) or Google Colab T4
Outputs: Direct GGUF quantized model ready for Ollama
"""

# 1. Install Unsloth & dependencies (Run in Kaggle terminal or top cell)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.29" trl peft accelerate bitsandbytes

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ----------------- CONFIGURATION -----------------
MAX_SEQ_LENGTH = 2048        # 2048 is ideal for chat history
DTYPE = None                 # None for auto detection (Float16 for T4, Bfloat16 for Ampere+)
LOAD_IN_4BIT = True          # 4-bit quantization saves massive VRAM

# Base Model: Qwen 2.5 7B Instruct (bnb 4-bit)
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# Auto-detect train.jsonl path (works whether placed locally or in /kaggle/input/)
def find_dataset_file(filename="train.jsonl"):
    if os.path.exists(filename):
        return filename
    # Search /kaggle/input/
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return filename

TRAIN_FILE = find_dataset_file("train.jsonl")
VAL_FILE = find_dataset_file("val.jsonl")

# ----------------- 1. LOAD MODEL & TOKENIZER -----------------
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply Qwen 2.5 ChatML template to tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

# ----------------- 2. CONFIGURE LORA ADAPTERS -----------------
print("Setting up QLoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                         # LoRA Rank (16 or 32 is optimal for tone mimicry)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,                # Typically 2 * r
    lora_dropout=0,               # Unsloth supports 0 for maximum speed & memory optimization
    bias="none",
    use_gradient_checkpointing="unsloth", # Saves ~70% VRAM
    random_state=3407,
)

# ----------------- 3. FORMAT DATASET -----------------
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

print(f"Loading dataset from {TRAIN_FILE}...")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
dataset = dataset.map(formatting_prompts_func, batched=True)

# ----------------- 4. TRAINER SETUP -----------------
training_args = TrainingArguments(
    per_device_train_batch_size=2,       # Small batch size to avoid OOM
    gradient_accumulation_steps=4,       # Effective batch size = 2 * 4 = 8
    warmup_steps=10,
    num_train_epochs=2,                  # 1-2 epochs is ideal. 3+ risks overfitting on old memories
    learning_rate=2e-4,                  # Standard LoRA learning rate
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,                       # Keep False for conversational multi-turn data
    args=training_args,
)

# ----------------- 5. TRAIN -----------------
print("Starting training...")
trainer_stats = trainer.train()
print("Training complete!")

# ----------------- 6. QUICK TEST INFERENCE -----------------
print("\n--- Testing Model on a Sample Message ---")
FastLanguageModel.for_inference(model)

sample_messages = [
    {"role": "system", "content": "You are chatting on WhatsApp. Talk naturally in your authentic bilingual voice."},
    {"role": "user", "content": "kya scene hai bhai? free ho?"}
]

inputs = tokenizer.apply_chat_template(
    sample_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
print("Generated Reply:\n", tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))

# ----------------- 7. EXPORT LORA GGUF ADAPTER (FOR OLLAMA) -----------------
print("\n--- Exporting LoRA Adapter to GGUF format for Ollama ---")
# Bypass Kaggle disk preflight check (LoRA adapter is only ~50MB, so it easily fits)
os.environ["UNSLOTH_DISK_PREFLIGHT"] = "0"
LORA_GGUF_DIR = "/kaggle/working/qwen_whatsapp_lora_gguf"

model.save_pretrained_gguf(
    LORA_GGUF_DIR,
    tokenizer,
    save_method="lora",
)

# Also save standard PEFT adapter as backup
model.save_pretrained("/kaggle/working/qwen_whatsapp_lora")
tokenizer.save_pretrained("/kaggle/working/qwen_whatsapp_lora")

print(f"\n[DONE] LoRA adapter exported to {LORA_GGUF_DIR}/")
print("Download the adapter .gguf file (~50MB) to your local PC to use with Ollama!")
