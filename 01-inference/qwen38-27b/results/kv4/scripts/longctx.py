"""Long-context served receipt: for each fixture depth, the prompt up to the planted answer, greedy, drafted then plain.

Records prompt tokens, cached tokens, time to first token, decode tok/s, whether the reply starts with the code,
and whether the drafted and plain replies are equal. LONG_PLAIN=1 skips the drafted run. Usage: longctx.py BASE MODEL FIXTURE OUT [depth ...]
"""
import json
import os
import sys
import time
import urllib.request


def run(base, model, prompt, tokens, draft):
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    text, times, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                if c.get("text"):
                    text.append(c["text"])
                    times.append(time.perf_counter())
    n = usage.get("completion_tokens", 0)
    decode = (n - 1) / (times[-1] - times[0]) if len(times) > 1 and n > 1 else None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"draft": draft, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
            "completion_tokens": n, "ttft_s": round(times[0] - sent, 3) if times else None,
            "decode_tps": round(decode, 1) if decode else None, "text": "".join(text)}


def main():
    base, model, fixture, out = sys.argv[1:5]
    want = {int(d) for d in sys.argv[5:]}
    fx = json.load(open(fixture))
    tokens = 128
    rows = []
    for d in fx["depths"]:
        if want and d["at"] not in want:
            continue
        prompt = fx["ids"][: d["answer"]]
        b = run(base, model, prompt, tokens, False)
        a = b if os.environ.get("LONG_PLAIN") else run(base, model, prompt, tokens, True)
        row = {"depth": d["at"], "prompt_len": len(prompt), "code": d["code"],
               "recall": a["text"].strip().startswith(d["code"]), "drafted_equals_plain": None if a is b else a["text"] == b["text"],
               "drafted": {k: v for k, v in a.items() if k != "text"}, "plain": {k: v for k, v in b.items() if k != "text"},
               "reply_head": a["text"][:40]}
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump({"fixture": fx["name"], "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
