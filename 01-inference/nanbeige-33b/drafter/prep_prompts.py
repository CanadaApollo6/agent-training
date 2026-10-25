"""Prompts for Nanbeige4.3-33B draft-head training data: the head learns to guess what *this* model writes next, so
the responses come from Nanbeige itself (gen_responses.py); only the prompts come from public sets. Mix aimed at our
use (agent coding first): code 50%, chat 30%, math 20%. The TB2 pilot's own trajectories are added at training time.

    uv run --with datasets prep_prompts.py [scale]   # scale 1.0 = 36K prompts -> prompts.jsonl

The first run used a third of each source (prompts-12k.jsonl, shuffled) to fit a ~$25 generation budget.
"""
import json
import random
import sys

from datasets import load_dataset

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
random.seed(0)
SOURCES = [  # (dataset, split, count, function row -> first user message)
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", "train", 10000, lambda r: r["instruction"]),
    ("nvidia/OpenCodeInstruct", "train", 8000, lambda r: r["input"]),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", 11000, lambda r: r["messages"][0]["content"]),
    ("AI-MO/NuminaMath-CoT", "train", 7000, lambda r: r["problem"]),
]
seen = set()
with open("prompts.jsonl", "w") as out:
    for name, split, n, get in SOURCES:
        n = int(n * SCALE)
        ds = load_dataset(name, split=split, streaming=True).shuffle(seed=0, buffer_size=50000)
        kept = 0
        for row in ds:
            text = (get(row) or "").strip()
            if not 20 <= len(text) <= 12000 or text in seen:
                continue
            seen.add(text)
            out.write(json.dumps({"id": f"{name.split('/')[-1]}-{kept}", "source": name,
                                  "messages": [{"role": "user", "content": text}]}) + "\n")
            kept += 1
            if kept >= n:
                break
        print(name, kept, flush=True)
