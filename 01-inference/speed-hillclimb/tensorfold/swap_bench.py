"""Two long conversations taking turns on one server: python3 swap_bench.py <port> [A tokens] [B tokens] [rounds].
Prints each turn's time to first token; greedy, so two servers' answers can be compared for equality."""
import json
import random
import sys
import time
import urllib.request

port = sys.argv[1]
sizes = [int(sys.argv[2]) if len(sys.argv) > 2 else 40000, int(sys.argv[3]) if len(sys.argv) > 3 else 20000]
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


def document(seed, n):
    rng, lines, facts = random.Random(seed), [], {}
    while len(" ".join(lines)) / 4.2 < n:
        k, v = f"item-{len(lines)}", rng.randint(1000, 9999)
        facts[k] = v
        lines.append(f"Record {k}: {' '.join(rng.choice(words) for _ in range(12))}; code {v}.")
    return "\n".join(lines), facts, rng


def ask(messages):
    body = {"model": "x", "max_tokens": 24, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, first, out, usage = time.perf_counter(), None, "", None
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            piece = (c.get("delta") or {}).get("content") or ""
            if piece:
                first = first or time.perf_counter()
                out += piece
    return usage["prompt_tokens"], round(first - t0, 3), out.strip()


convs = []
for seed, n in enumerate(sizes):
    text, facts, rng = document(seed, n)
    convs.append({"messages": [{"role": "system", "content": "Answer with the number only."},
                               {"role": "user", "content": text}], "facts": facts, "rng": rng})
for r in range(rounds):
    for name, c in zip("AB", convs):
        probe = c["rng"].choice(list(c["facts"]))
        c["messages"].append({"role": "user", "content": f"What is the code for record {probe}?"})
        tokens, ttft, answer = ask(c["messages"])
        c["messages"].append({"role": "assistant", "content": answer})
        print(json.dumps({"turn": f"{name}{r + 1}", "prompt_tokens": tokens, "ttft_s": ttft, "answer": answer,
                          "expected": c["facts"][probe]}), flush=True)
