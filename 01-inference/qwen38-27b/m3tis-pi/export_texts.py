"""Texts for the pi adapter build's checks, taken from our own pilot runs (06-agents/harness-evals, run_q27pi.sh).

- kl.jsonl: whole conversations (messages + tools) for the KL check, 16 from the adapter arm (thinking off) and 16
  from the base-at-medium arm (thinking kept), one per task and attempt where possible, longest first.
- turns.jsonl: 12 mid-run agent turns from the adapter arm (the prefix up to one of its replies, prompts under 12K
  tokens) for the speed check: the next turn is generated with thinking off.

    python3 export_texts.py      # writes pod/kl.jsonl and pod/turns.jsonl next to this file
"""
import json
import random
import sys
from pathlib import Path

HE = Path(__file__).resolve().parents[3] / "06-agents/harness-evals"
sys.path.insert(0, str(HE))
from compare_builds import OUT  # noqa: E402
from length_replay import openai_message, openai_tools  # noqa: E402

POD = Path(__file__).parent / "pod"


def traces(arm):
    for d in sorted(OUT.glob(f"terminal-bench-2--q27pi-{arm}-a*--pi--*")):
        for line in open(d / "traces.jsonl"):
            r = json.loads(line)
            if not r.get("traces"):
                continue
            t = r["traces"][0]
            task = t["task"]["data"]["name"].split("/")[-1]
            if task == "qemu-startup" or not t["nodes"]:
                continue
            if any(isinstance(n["message"].get("content"), list) and any(p.get("type") != "text" for p in
                   n["message"]["content"]) for n in t["nodes"]):
                continue  # images
            yield task, t


def conversation(t, thinking):
    # the longest path: the last node and its parents
    nodes, i = [], len(t["nodes"]) - 1
    while i is not None and i >= 0:
        nodes.append(t["nodes"][i]["message"])
        i = t["nodes"][i].get("parent")
    return [openai_message(m, thinking) for m in reversed(nodes)]


def main():
    rng = random.Random(0)
    POD.mkdir(exist_ok=True)
    rows = []
    for arm, thinking in (("ad", False), ("med", True)):
        seen, picked = set(), []
        cands = list(traces(arm))
        rng.shuffle(cands)
        for task, t in cands:
            if task in seen:
                continue
            seen.add(task)
            picked.append({"set": f"agent-{arm}", "task": task, "thinking": thinking,
                           "messages": conversation(t, thinking), "tools": openai_tools(t["tools"])})
            if len(picked) == 16:
                break
        rows += picked
    with open(POD / "kl.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    turns = []
    for task, t in traces("ad"):
        for call in t["calls"]:
            if "node" not in call or call["usage"]["prompt_tokens"] > 12000 or call["node"] < 4:
                continue
            node = t["nodes"][call["node"]]
            prefix, i = [], node.get("parent")
            while i is not None and i >= 0:
                prefix.append(t["nodes"][i]["message"])
                i = t["nodes"][i].get("parent")
            turns.append({"task": task, "prompt_tokens": call["usage"]["prompt_tokens"],
                          "messages": [openai_message(m, False) for m in reversed(prefix)],
                          "tools": openai_tools(node.get("tools") or t["tools"])})
    rng.shuffle(turns)
    seen, picked = set(), []
    for r in turns:
        if r["task"] not in seen:
            seen.add(r["task"])
            picked.append(r)
        if len(picked) == 12:
            break
    with open(POD / "turns.jsonl", "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"kl.jsonl {len(rows)} conversations, turns.jsonl {len(picked)} turns "
          f"(prompt tokens {sorted(r['prompt_tokens'] for r in picked)})")


if __name__ == "__main__":
    main()
