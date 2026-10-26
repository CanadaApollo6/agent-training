"""Pilot agent runs (vf-eval traces.jsonl, pi harness) -> SpecForge rows of Nanbeige's own tool-using turns.

A run is a chain of messages, often far past the 8K training length, so each run is cut into windows: the system
prompt and task, then as many whole turns (an assistant message and the tool results after it) as fit in ~7K
tokens. Tool results over 6K characters keep their head and tail. Tool calls are stored flat; the template wants
OpenAI shape with arguments as a mapping.

    python3 traj_to_specforge.py out.jsonl outputs/primeintellect/*nanbeige*/traces.jsonl
"""
import json
import sys

BUDGET, TOOL_MAX = 7000 * 3.5, 6000  # characters, ~3.5 a token


def tool_spec(t):
    return {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}


def call(c):
    f = c.get("function", c)
    args = f.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"input": args}
    return {"id": c.get("id", ""), "type": "function", "function": {"name": f["name"], "arguments": args}}


def clean(m):
    m = dict(m)
    if isinstance(m.get("content"), list):
        m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
    if m["role"] == "tool" and len(m.get("content") or "") > TOOL_MAX:
        c = m["content"]
        m["content"] = c[:TOOL_MAX // 2] + "\n...\n" + c[-TOOL_MAX // 2:]
    if m.get("tool_calls"):
        m["tool_calls"] = [call(c) for c in m["tool_calls"]]
    return m


def size(m):
    return len(json.dumps(m, ensure_ascii=False))


dst, srcs = sys.argv[1], sys.argv[2:]
rows = 0
with open(dst, "w") as out:
    for src in srcs:
        for line in open(src):
            r = json.loads(line)
            for t in r["traces"]:
                msgs = [clean(n["message"]) for n in t["nodes"]]
                tools = [tool_spec(x) for x in (t["nodes"][0].get("tools") or t.get("tools") or [])]
                head = [m for m in msgs[:2] if m["role"] in ("system", "user")]
                turns, cur = [], []
                for m in msgs[len(head):]:
                    if m["role"] == "assistant" and cur:
                        turns.append(cur)
                        cur = []
                    cur.append(m)
                turns.append(cur)
                base, win, k = sum(map(size, head)), [], 0
                for turn in turns + [None]:
                    s = sum(map(size, turn)) if turn else 0
                    if win and (turn is None or base + sum(sum(map(size, w)) for w in win) + s > BUDGET):
                        conv = head + [m for w in win for m in w]
                        out.write(json.dumps({"id": f"{t['id']}-{k}", "conversations": conv, "tools": tools}) + "\n")
                        rows, k, win = rows + 1, k + 1, []
                    if turn and turn[0]["role"] == "assistant":
                        win.append(turn)
print(f"{rows} windows -> {dst}")
