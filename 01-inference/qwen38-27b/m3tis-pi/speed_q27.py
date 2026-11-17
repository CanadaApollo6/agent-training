"""Reply speed and draft acceptance of a running M3TiS server (m3tis-serve.sh --no-thinking), for comparing the
Vontra weights with the pi adapter build on the same card and drafter.

Cases: the release card's code and chat prompts (a warm-up, then 3 runs each), and 12 mid-run agent turns from our
pi pilot (pod/turns.jsonl, one run each). Each runs greedy and at T 0.8 (the adapter arm's setting). Speeds and
acceptance come from the server's own `done` log line, as in nanbeige-33b/m3tis/bench.py.

    SERVER_LOG=serve.log python3 speed_q27.py 8080 out.json turns.jsonl
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PORT, OUT, TURNS = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BASE = f"http://127.0.0.1:{PORT}/v1"
LOG = Path(os.environ["SERVER_LOG"])
CASES = {
    "code": [{"role": "user", "content": "Write a Python function that computes the Fibonacci sequence and explain it."}],
    "chat": [{"role": "user", "content": "Explain how matrix multiplication uses a GPU in plain English, then give a "
                                         "small numerical example."}],
}
MODES = {"greedy": {"temperature": 0}, "T 0.8": {"temperature": 0.8}}


def post(body):
    n = sum("] done " in x for x in LOG.read_text().splitlines())
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps({"model": "m", **body}).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    for _ in range(50):
        lines = [x for x in LOG.read_text().splitlines() if "] done " in x]
        if len(lines) > n:
            break
        time.sleep(0.1)
    out["done"] = dict(re.findall(r"(\S+?)=(\S+)", lines[-1]))
    return out


def summary(runs):
    def f(r, k):
        try:
            return float(r["done"][k].rstrip("%x"))
        except (KeyError, ValueError):
            return None
    tps = [f(r, "tok/s") for r in runs]
    acc = [a for a in (f(r, "accepted") for r in runs) if a is not None]
    return {"tok_s": statistics.median(tps), "all": [round(x, 1) for x in tps],
            "accepted": statistics.mean(acc) if acc else None,
            "tokens": [r["usage"]["completion_tokens"] for r in runs],
            "finish": [r["choices"][0]["finish_reason"] for r in runs],
            "raw_done": runs[-1]["done"]}


rows = []
turns = [json.loads(x) for x in open(TURNS)]
for mode, sampling in MODES.items():
    for case, messages in CASES.items():
        body = {"messages": messages, "max_tokens": 512, **sampling}
        post({**body, "seed": 0})
        rows.append({"case": case, "mode": mode, **summary([post({**body, "seed": i}) for i in (1, 2, 3)])})
        print(json.dumps(rows[-1]), flush=True)
    runs = [post({"messages": t["messages"], "tools": t["tools"], "max_tokens": 512, "seed": 1, **sampling})
            for t in turns]
    rows.append({"case": f"agent turns ({len(turns)})", "mode": mode, **summary(runs),
                 "prompt_tokens": [r["usage"]["prompt_tokens"] for r in runs]})
    print(json.dumps(rows[-1]), flush=True)

OUT.write_text(json.dumps(rows, indent=1) + "\n")
print("SPEED_DONE")
