"""Reply speed and draft acceptance of a running M3TiS server (m3tis-serve.sh --no-thinking), for comparing the
Vontra weights with the pi adapter build on the same card and drafter.

Cases: the release card's code and chat prompts (a warm-up, then 3 runs each), and 12 mid-run agent turns from our
pi pilot (pod/turns.jsonl, one run each). Each runs greedy and at T 0.8 (the adapter arm's setting). Speed is timed
from the stream; acceptance comes from the server's `done` log line (tokens per draft round, accepted/drafted).

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
    """Stream one reply. Decode speed is timed here (the release build's log has no timing): tokens after the first,
    over the time from the first streamed delta to the last. Tokens, rounds and accepted drafts come from the
    server's `done` line."""
    n = sum("] done " in x for x in LOG.read_text().splitlines())
    req = urllib.request.Request(BASE + "/chat/completions",
                                 json.dumps({"model": "m", **body, "stream": True}).encode(),
                                 {"content-type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                d = ch.get("delta", {})
                if any(d.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                    last = time.perf_counter()
                    first = first or last
                finish = ch.get("finish_reason") or finish
    for _ in range(50):
        lines = [x for x in LOG.read_text().splitlines() if "] done " in x]
        if len(lines) > n:
            break
        time.sleep(0.1)
    done = dict(re.findall(r"(\S+?)=(\S+)", lines[-1]))
    tokens, rounds = int(done["tokens"]), int(done.get("rounds", 0))
    acc, drafted = (int(x) for x in done.get("accepted", "0/0").split("/"))
    if first is None:
        first = last = t0
    return {"ttft": first - t0, "tok_s": (tokens - 1) / (last - first) if last > first else None, "tokens": tokens,
            "per_round": tokens / rounds if rounds else None, "accept_rate": acc / drafted if drafted else None,
            "finish": finish, "done": done}


def summary(runs):
    def med(k):
        xs = [r[k] for r in runs if r[k] is not None]
        return statistics.median(xs) if xs else None
    return {"tok_s": med("tok_s"), "all": [round(r["tok_s"] or 0, 1) for r in runs],
            "tokens_per_round": med("per_round"), "accept_rate": med("accept_rate"), "ttft_s": med("ttft"),
            "tokens": [r["tokens"] for r in runs], "finish": [r["finish"] for r in runs]}


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
                 "prompt_tokens": [int(r["done"]["prompt"]) for r in runs]})
    print(json.dumps(rows[-1]), flush=True)

OUT.write_text(json.dumps(rows, indent=1) + "\n")
print("SPEED_DONE")
