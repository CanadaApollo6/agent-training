"""probe.py BASE MODEL FIXTURE POS...: the served greedy next token after fixture ids[:POS+1]."""
import json
import sys
import urllib.request

base, model, fixture = sys.argv[1:4]
ids = json.load(open(fixture))["ids"]
for pos in map(int, sys.argv[4:]):
    body = {"model": model, "prompt": ids[: pos + 1], "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    c = out["choices"][0]
    print(json.dumps({"pos": pos, "next_id": ids[pos + 1], "text": c.get("text")}), flush=True)
