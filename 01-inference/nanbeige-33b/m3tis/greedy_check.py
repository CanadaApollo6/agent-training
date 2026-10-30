"""Greedy tokens from a running server vs the exllamav3 reference's 128 (the P1 prompt): prints the first position
where they split, or 128 if none. Usage: python3 greedy_check.py port p1-ids.json oracle.json out.json"""

import json
import sys
import urllib.request

port, ids, oracle, out = sys.argv[1], *sys.argv[2:5]
ids, want = json.load(open(ids)), json.load(open(oracle))
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(dict(
    model="m", prompt=ids, max_tokens=128, temperature=0, return_token_ids=True, ignore_eos=True)).encode(),
    {"content-type": "application/json"})
got = json.load(urllib.request.urlopen(req, timeout=600))["tensorfold"]["token_ids"]
split = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
json.dump(dict(equal_prefix=split, got=got, want=want), open(out, "w"))
print(split)
