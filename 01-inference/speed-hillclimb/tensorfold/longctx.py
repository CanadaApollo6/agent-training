"""Send a long prompt: python3 longctx.py <port> <approx tokens>. Prints prefill time, decode rate and a recall check."""
import json, sys, time, urllib.request, random
random.seed(0)
n = int(sys.argv[2])
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
facts, lines = {}, []
while len(" ".join(lines)) / 4.2 < n:
    k = f"item-{len(lines)}"
    v = random.randint(1000, 9999)
    facts[k] = v
    lines.append(f"Record {k}: {' '.join(random.choice(words) for _ in range(12))}; code {v}.")
probe = random.choice(list(facts))
text = "\n".join(lines) + f"\n\nWhat is the code for record {probe}? Answer with the number only."
body = {"model": "ornith", "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.perf_counter(); first = None; out = ""; usage = None; times = []
for raw in urllib.request.urlopen(req, timeout=3600):
    line = raw.decode().strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        continue
    d = json.loads(line[5:])
    if d.get("usage"):
        usage = d["usage"]
    for c in d.get("choices", []):
        piece = (c.get("delta") or {}).get("content") or ""
        if piece:
            now = time.perf_counter(); first = first or now; times.append(now); out += piece
print(json.dumps({"prompt_tokens": usage and usage["prompt_tokens"], "prefill_s": round(first - t0, 2),
                  "prefill_tps": round(usage["prompt_tokens"] / (first - t0)), "answer": out.strip()[:40],
                  "expected": facts[probe]}))
