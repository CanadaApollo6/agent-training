"""The release card's numbers, from a running server (./m3tis-serve.sh): reply speed by prompt and sampling, first
token, and a long prompt's read time. A warm-up, then the median of 3; the speeds are the server's own decode timing.
Usage: python3 bench.py [port] [out.json]."""

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
OUT = sys.argv[2] if len(sys.argv) > 2 else None
BASE = f"http://127.0.0.1:{PORT}/v1"
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
MODES = {"greedy": {"temperature": 0}, "liquid default (T 0.1, top_k 50)": {}, "T 1.0": {"temperature": 1.0}}


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps({"model": "m", **body}).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


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
        tps = [r["tensorfold"]["tokens_per_second"] for r in runs]
        rows.append({"case": case, "mode": mode, "tok_s": statistics.median(tps), "all": [round(x) for x in tps],
                     "tokens": [r["usage"]["completion_tokens"] for r in runs],
                     "finish": [r["choices"][0]["finish_reason"] for r in runs]})
        print(json.dumps(rows[-1]), flush=True)

ttft = []
for i in range(4):
    other()
    ttft.append(first_token({**CASES["code"], "temperature": 0, "max_tokens": 16}))
rows.append({"case": "first token, code prompt, cold", "s": statistics.median(ttft[1:]), "all": ttft[1:]})
print(json.dumps(rows[-1]), flush=True)

ids = json.loads((Path(__file__).resolve().parent.parent / "long_prompt.json").read_text())["long"]
for mode, sampling in (("greedy", {"temperature": 0}), ("liquid default (T 0.1, top_k 50)", {})):
    runs = []
    for i in range(4):
        other()
        runs.append(post("/completions", {"prompt": ids, "max_tokens": 600, "seed": i, **sampling}))
    runs = runs[1:]
    rows.append({"case": f"long prompt ({len(ids)} tokens)", "mode": mode,
                 "read_s": statistics.median(r["tensorfold"]["prefill_seconds"] for r in runs),
                 "tok_s": statistics.median(r["tensorfold"]["tokens_per_second"] for r in runs),
                 "all": [round(r["tensorfold"]["tokens_per_second"]) for r in runs]})
    print(json.dumps(rows[-1]), flush=True)

if OUT:
    Path(OUT).write_text(json.dumps(rows, indent=1) + "\n")
