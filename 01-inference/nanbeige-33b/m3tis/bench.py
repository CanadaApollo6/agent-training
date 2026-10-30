"""The release card's numbers, from a running server (./m3tis-serve.sh): reply speed by prompt and sampling, first
token, and long prompts' read time. A warm-up, then the median of 3; the speeds are the server's own decode timing,
read from the `done` line it logs for each request (start the server with its output in SERVER_LOG).
Usage: SERVER_LOG=serve.log python3 bench.py [port] [out.json] [long_prompt.txt ...]."""

import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
OUT = sys.argv[2] if len(sys.argv) > 2 else None
LONG = [Path(p) for p in sys.argv[3:]]
BASE = f"http://127.0.0.1:{PORT}/v1"
LOG = Path(os.environ["SERVER_LOG"])
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CASES = {
    "code": {"messages": [{"role": "user", "content": "Write a Python function that computes the Fibonacci sequence "
                                                      "and explain it."}], "max_tokens": 512},
    "chat": {"messages": [{"role": "user", "content": "Explain how matrix multiplication uses a GPU in plain English, "
                                                      "then give a small numerical example."}], "max_tokens": 512},
    "tool call": {"messages": [{"role": "user", "content": "What's the weather in Paris and in Tokyo?"}],
                  "tools": TOOLS, "max_tokens": 512},
}
MODES = {"greedy": {"temperature": 0}, "nanbeige default (T 1.0, top_p 0.95)": {}}


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps({"model": "m", **body}).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    time.sleep(0.2)
    line = [x for x in LOG.read_text().splitlines() if "] done " in x][-1]
    out["done"] = {k: v for k, v in re.findall(r"(\S+?)=(\S+)", line)}
    return out


def num(r, key):
    return float(r["done"][key])


def first_token(body):
    """Seconds to the first streamed delta of any kind (thinking, text or a call)."""
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps({"model": "m", **body, "stream": True}).encode(),
                                 {"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:]).get("choices", []):
                    if any(ch.get("delta", {}).get(k) for k in ("content", "reasoning_content", "tool_calls")):
                        return time.perf_counter() - t0
    return None


def other():  # moves the stream off the last prompt, so the next read starts cold
    post("/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0})


rows = []
for case, body in CASES.items():
    for mode, sampling in MODES.items():
        post("/chat/completions", {**body, **sampling, "seed": 0})
        runs = [post("/chat/completions", {**body, **sampling, "seed": i}) for i in (1, 2, 3)]
        tps = [num(r, "tok/s") for r in runs]
        rows.append({"case": case, "mode": mode, "tok_s": statistics.median(tps), "all": [round(x, 1) for x in tps],
                     "tokens": [r["usage"]["completion_tokens"] for r in runs],
                     "finish": [r["choices"][0]["finish_reason"] for r in runs],
                     "accepted": [r["done"].get("accepted") for r in runs]})
        print(json.dumps(rows[-1]), flush=True)

ttft = []
for i in range(4):
    other()
    ttft.append(first_token({**CASES["code"], "temperature": 0, "max_tokens": 16}))
rows.append({"case": "first token, code prompt, cold", "s": statistics.median(ttft[1:]), "all": ttft[1:]})
print(json.dumps(rows[-1]), flush=True)

for path in LONG:
    text = path.read_text()
    body = {"messages": [{"role": "user", "content": text + "\n\nSummarize what this code does in five bullet points."}],
            "max_tokens": 400, "temperature": 0}
    runs = []
    for i in range(3):
        other()
        runs.append(post("/chat/completions", {**body, "seed": i}))
    runs = runs[1:]
    rows.append({"case": f"long prompt {path.name}", "prompt_tokens": runs[0]["usage"]["prompt_tokens"],
                 "read_s": statistics.median(num(r, "ttft") for r in runs),
                 "tok_s": statistics.median(num(r, "tok/s") for r in runs),
                 "all_read_s": [num(r, "ttft") for r in runs]})
    print(json.dumps(rows[-1]), flush=True)
    t0 = time.perf_counter()
    nxt = post("/chat/completions", {**body, "messages": body["messages"] + [
        {"role": "assistant", "content": runs[-1]["choices"][0]["message"].get("content") or ""},
        {"role": "user", "content": "Now name the riskiest function and why, in one sentence."}], "max_tokens": 64})
    rows.append({"case": f"next turn after {path.name}", "read_s": num(nxt, "ttft"),
                 "cached_tokens": nxt["done"].get("cached"),
                 "wall_s": time.perf_counter() - t0})
    print(json.dumps(rows[-1]), flush=True)

if OUT:
    Path(OUT).write_text(json.dumps(rows, indent=1) + "\n")
