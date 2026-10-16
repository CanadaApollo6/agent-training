"""Turn our agent traces (plus a little general text) into token windows for the Flash-Next router job.

Each trace is rendered with Flash-Next's own chat template, tokenized, and cut into 8K-token windows. Tasks are split
by name: 80% go to "calib" (picks which experts to keep), 20% to "heldout" (scores the pruned model). "control" is
general web and Wikipedia text, to see which experts our traces never need.

    uv run --with transformers==5.20.1 --with datasets --with numpy python -I prep_tokens.py \
        <tokenizer dir> <06-agents/harness-evals/outputs> <out dir>
"""
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np
from transformers import AutoTokenizer

WIN = 8192
CAPS = {"calib": 1_500_000, "heldout": 250_000}
CONTROL = [  # (dataset, config, tokens)
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 120_000),
    ("wikimedia/wikipedia", "20231101.zh", 40_000),
    ("wikimedia/wikipedia", "20231101.es", 40_000),
    ("wikimedia/wikipedia", "20231101.ru", 40_000),
]


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def to_messages(trace):
    tools, msgs = None, []
    for node in trace["nodes"]:
        if node.get("tools") and tools is None:
            tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                     for t in node["tools"]]
        m = node.get("message")
        if not m:
            continue
        out = {"role": m["role"], "content": text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", c)
                args = f.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
            out["tool_calls"] = calls
        if m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        msgs.append(out)
    # the template takes one system message, first; harnesses send two up front and sometimes inject more later
    head = [m for m in msgs[:2] if m["role"] == "system"]
    rest = [dict(m, role="user") if m["role"] == "system" else m for m in msgs[len(head):]]
    if head:
        rest.insert(0, {"role": "system", "content": "\n\n".join(m["content"] for m in head)})
    return rest, tools


def windows(ids):
    return [ids[i:i + WIN] for i in range(0, len(ids) - 1023, WIN)]


def main(tok_dir, traces_dir, out_dir):
    tok = AutoTokenizer.from_pretrained(tok_dir)
    os.makedirs(out_dir, exist_ok=True)
    by_split = {"calib": [], "heldout": []}
    for path in sorted(glob.glob(f"{traces_dir}/*/traces.jsonl")):
        run = os.path.basename(os.path.dirname(path))
        for line in open(path):
            row = json.loads(line)
            name = row["task"]["data"].get("name") or str(row["task"]["data"].get("idx"))
            split = "heldout" if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 5 == 0 else "calib"
            for trace in row.get("traces", []):
                by_split[split].append((run, name, trace))
    stats = {}
    for split, items in by_split.items():
        # round-robin over runs so the token cap samples every benchmark and model, not just the first files
        random.Random(0).shuffle(items)
        queues = {}
        for it in items:
            queues.setdefault(it[0], []).append(it)
        items = [q[i] for i in range(max(map(len, queues.values()))) for q in queues.values() if i < len(q)]
        wins, n_fail = [], 0
        for run, name, trace in items:
            msgs, tools = to_messages(trace)
            if not any(m["role"] == "user" for m in msgs):
                n_fail += 1
                continue
            try:
                text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
            except Exception:
                n_fail += 1
                continue
            for w in windows(tok(text, add_special_tokens=False)["input_ids"]):
                wins.append((run, w))
            if sum(len(w) for _, w in wins) >= CAPS[split]:
                break
        save(out_dir, split, wins)
        stats[split] = {"windows": len(wins), "tokens": sum(len(w) for _, w in wins), "render_failures": n_fail,
                        "runs": sorted({r for r, _ in wins})}
    from datasets import load_dataset
    wins = []
    for ds, cfg, cap in CONTROL:
        buf = []
        for row in load_dataset(ds, cfg, split="train", streaming=True):
            buf += tok(row["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(buf) >= cap:
                break
        wins += [(f"{ds}:{cfg}", w) for w in windows(buf)]
    save(out_dir, "control", wins)
    stats["control"] = {"windows": len(wins), "tokens": sum(len(w) for _, w in wins)}
    json.dump(stats, open(f"{out_dir}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def save(out_dir, split, wins):
    ids = np.concatenate([np.array(w, dtype=np.int32) for _, w in wins])
    lens = np.array([len(w) for _, w in wins], dtype=np.int32)
    np.savez(f"{out_dir}/{split}.npz", ids=ids, lens=lens, src=np.array([r for r, _ in wins]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
