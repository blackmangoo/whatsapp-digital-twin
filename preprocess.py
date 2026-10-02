import re
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Regex patterns for different WhatsApp export formats
# 1. Android: "12/04/2024, 14:32 - Sender: Message" or "12/04/24, 2:32 pm - Sender: Message"
ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\s*-\s*([^:]+?):\s*(.*)$"
)

# 2. iOS: "[12/04/2024, 14:32:10] Sender: Message" or "[12/04/24, 2:32:10 PM] Sender: Message"
IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\]\s*([^:]+?):\s*(.*)$"
)

# Common noise strings to filter out
IGNORE_SUBSTRINGS = [
    "omitted",
    "this message was deleted",
    "you deleted this message",
    "messages and calls are end-to-end encrypted",
    "missed voice call",
    "missed video call",
    "poll:",
    "contact card",
    "live location shared",
]

# Regex to strip all invisible Unicode control characters that WhatsApp inserts
UNICODE_NOISE = re.compile(r"[‎‏‪-‮ \xa0﻿]")

def parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse various timestamp formats from WhatsApp exports."""
    # Clean up whitespace and non-breaking spaces
    date_str = UNICODE_NOISE.sub(" ", date_str).strip()
    time_str = UNICODE_NOISE.sub(" ", time_str).strip()

    # Standardize time format
    full_str = f"{date_str} {time_str}"

    # Try different format patterns
    date_patterns = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%y %I:%M:%S %p",
        "%d/%m/%y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in date_patterns:
        try:
            return datetime.strptime(full_str, fmt)
        except ValueError:
            continue
    return None

def is_system_or_noise(text: str) -> bool:
    """Check if message is media placeholder or WhatsApp system notification."""
    text_lower = text.lower()
    for noise in IGNORE_SUBSTRINGS:
        if noise.lower() in text_lower:
            return True
    return False

def parse_whatsapp_file(file_path: str) -> List[Dict]:
    """Parses raw WhatsApp .txt export into a list of message objects."""
    messages = []
    current_msg = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Strip hidden LTR/RTL Unicode characters and non-breaking spaces
            line_clean = UNICODE_NOISE.sub(" ", line).strip()
            if not line_clean:
                continue

            # Match Android or iOS pattern
            match = ANDROID_PATTERN.match(line_clean) or IOS_PATTERN.match(line_clean)

            if match:
                date_str, time_str, sender, content = match.groups()
                content = content.strip()

                # Check for noise or omitted media
                if is_system_or_noise(content):
                    current_msg = None
                    continue

                ts = parse_timestamp(date_str, time_str)

                current_msg = {
                    "timestamp": ts,
                    "sender": sender.strip(),
                    "content": content
                }
                messages.append(current_msg)
            else:
                # Continuation of previous message (multiline message)
                if current_msg and line_clean:
                    if not is_system_or_noise(line_clean):
                        current_msg["content"] += "\n" + line_clean

    print(f"Total raw parsed messages: {len(messages)}")
    return messages

def aggregate_bursts(messages: List[Dict], max_gap_seconds: int = 120) -> List[Dict]:
    """
    Groups consecutive messages from the same sender sent within max_gap_seconds
    into a single turn. People on WhatsApp send 3-5 rapid messages instead of one paragraph.
    """
    if not messages:
        return []

    aggregated = []
    current_burst = {
        "sender": messages[0]["sender"],
        "messages": [messages[0]["content"]],
        "start_time": messages[0]["timestamp"],
        "end_time": messages[0]["timestamp"]
    }

    for msg in messages[1:]:
        same_sender = (msg["sender"] == current_burst["sender"])
        within_time = True

        if msg["timestamp"] and current_burst["end_time"]:
            time_diff = (msg["timestamp"] - current_burst["end_time"]).total_seconds()
            within_time = (0 <= time_diff <= max_gap_seconds)

        if same_sender and within_time:
            current_burst["messages"].append(msg["content"])
            if msg["timestamp"]:
                current_burst["end_time"] = msg["timestamp"]
        else:
            # Finalize previous burst
            aggregated.append({
                "sender": current_burst["sender"],
                "content": "\n".join(current_burst["messages"]),
                "timestamp": current_burst["start_time"]
            })
            current_burst = {
                "sender": msg["sender"],
                "messages": [msg["content"]],
                "start_time": msg["timestamp"],
                "end_time": msg["timestamp"]
            }

    # Add final burst
    if current_burst:
        aggregated.append({
            "sender": current_burst["sender"],
            "content": "\n".join(current_burst["messages"]),
            "timestamp": current_burst["start_time"]
        })

    print(f"Aggregated into {len(aggregated)} conversational turns (bursts merged).")
    return aggregated

def create_training_conversations(
    bursts: List[Dict],
    my_name: str,
    system_prompt: str,
    window_turns: int = 6
) -> List[Dict]:
    """
    Creates multi-turn ChatML conversations using a sliding window.
    Only creates examples where the final response is generated by 'my_name' (assistant).
    """
    training_data = []

    for i in range(len(bursts)):
        if bursts[i]["sender"] != my_name:
            continue

        # Target assistant response is bursts[i]
        start_idx = max(0, i - window_turns + 1)
        history_slice = bursts[start_idx : i + 1]

        # Build clean dialogue with strictly alternating turns
        dialogue = []
        for turn in history_slice:
            role = "assistant" if turn["sender"] == my_name else "user"
            content = turn["content"].strip()
            if not content:
                continue
            if dialogue and dialogue[-1]["role"] == role:
                # Merge consecutive turns from the same sender
                dialogue[-1]["content"] += "\n" + content
            else:
                dialogue.append({"role": role, "content": content})

        # Ensure conversation starts with user and ends with assistant
        while dialogue and dialogue[0]["role"] != "user":
            dialogue.pop(0)

        if len(dialogue) >= 2 and dialogue[-1]["role"] == "assistant":
            messages = [{"role": "system", "content": system_prompt}] + dialogue
            training_data.append({"messages": messages})

    print(f"Generated {len(training_data)} training conversation samples.")
    return training_data

def inspect_senders(file_path: str):
    """Utility to quickly list all senders found in the chat export."""
    raw = parse_whatsapp_file(file_path)
    senders = {}
    for m in raw:
        s = m["sender"]
        senders[s] = senders.get(s, 0) + 1
    print("\n--- Detected Senders ---")
    for s, count in sorted(senders.items(), key=lambda x: x[1], reverse=True):
        print(f"'{s}': {count} messages")
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WhatsApp chat export for Qwen 2.5 fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Path to exported WhatsApp .txt file")
    parser.add_argument("--my_name", type=str, default=None, help="Your exact sender name as shown in WhatsApp")
    parser.add_argument("--output_train", type=str, default="data/train.jsonl", help="Output path for training JSONL")
    parser.add_argument("--output_val", type=str, default="data/val.jsonl", help="Output path for validation JSONL")
    parser.add_argument("--inspect_only", action="store_true", help="Only list senders and exit")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio (default: 0.05)")
    parser.add_argument("--max_gap", type=int, default=120, help="Max seconds between messages to merge into one burst")

    args = parser.parse_args()

    if args.inspect_only or not args.my_name:
        inspect_senders(args.input)
        if not args.my_name:
            print("Please run again with --my_name '<Your Exact Name From List Above>'")
            exit(0)

    raw_msgs = parse_whatsapp_file(args.input)
    bursts = aggregate_bursts(raw_msgs, max_gap_seconds=args.max_gap)

    sys_prompt = (
        f"You are chatting on WhatsApp as {args.my_name}. "
        "Talk naturally in your authentic bilingual voice, matching your exact casual tone, "
        "Roman Urdu/Hinglish phrasing, slang, sentence length, and capitalization."
    )

    samples = create_training_conversations(bursts, my_name=args.my_name, system_prompt=sys_prompt)

    # Split train and val
    val_count = max(1, int(len(samples) * args.val_ratio)) if len(samples) > 20 else 0
    train_samples = samples[:-val_count] if val_count > 0 else samples
    val_samples = samples[-val_count:] if val_count > 0 else []

    # Save to JSONL
    import os
    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)

    with open(args.output_train, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if val_samples:
        os.makedirs(os.path.dirname(args.output_val) or ".", exist_ok=True)
        with open(args.output_val, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[Done] Saved {len(train_samples)} training samples to {args.output_train}")
    if val_samples:
        print(f"[Done] Saved {len(val_samples)} validation samples to {args.output_val}")
