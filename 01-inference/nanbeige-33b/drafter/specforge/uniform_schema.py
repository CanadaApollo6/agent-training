"""Rewrite SpecForge rows so every row has the same Arrow schema, shuffled into one training file.

SpecForge reads data with load_dataset("json"), which takes its schema from the first block of rows and casts the
other blocks to it. Agent windows carry `tools` and `tool_calls` whose argument objects differ per tool, so mixing them
with plain responses broke the full run (CastError, 2026-10-26). SpecForge's parser accepts both fields as JSON
strings, so store them that way, give every row a `tools` field and every message the same keys, and put a row that
sets every key first: a first block where a key is always null types it as null, and later strings won't cast to null.

Checked: 451 agent windows and 300 responses give identical input_ids and loss masks before and after.

    python uniform_schema.py in.jsonl [in2.jsonl ...] > out.jsonl
"""
import json
import random
import sys

KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        row = json.loads(line)
        msgs = []
        for m in row["conversations"]:
            out = {k: m.get(k) for k in KEYS}
            out["content"] = out["content"] or ""
            out["tool_calls"] = json.dumps(m["tool_calls"]) if m.get("tool_calls") else None
            msgs.append(out)
        rows.append({"id": str(row["id"]), "conversations": msgs, "tools": json.dumps(row.get("tools") or [])})

random.Random(0).shuffle(rows)
full = next(i for i, r in enumerate(rows) if all(any(m[k] is not None for m in r["conversations"]) for k in KEYS))
rows[0], rows[full] = rows[full], rows[0]
for r in rows:
    print(json.dumps(r))
