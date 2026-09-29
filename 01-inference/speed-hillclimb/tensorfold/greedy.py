"""Greedy outputs for the exactness gate: python3 greedy.py <port> <out.json>."""
import json, sys, time, urllib.request

PROMPTS = [
    ("chat", "Prove that there are infinitely many primes, then list the first 20 of them."),
    ("chat", "Write a Python function that parses ISO 8601 durations like P3DT4H5M, with tests."),
    ("completion", "The history of the transistor begins"),
    ("chat", "A train leaves at 3:40 pm going 72 km/h; another leaves the same station at 4:05 pm going 90 km/h. When does the second catch the first? Think step by step."),
]
out = []
for kind, text in PROMPTS:
    body = {"model": "ornith", "max_tokens": 1024, "temperature": 0, "ignore_eos": True, "logprobs": False}
    if kind == "chat":
        url = f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": text}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = f"http://127.0.0.1:{sys.argv[1]}/v1/completions"
        body["prompt"] = text
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})))
    dt = time.perf_counter() - t
    c = r["choices"][0]
    txt = c.get("text") if kind == "completion" else (c["message"].get("reasoning_content") or "") + "\n<<>>\n" + (c["message"].get("content") or "")
    out.append({"prompt": text, "text": txt, "tokens": r["usage"]["completion_tokens"], "seconds": round(dt, 3)})
    print(kind, r["usage"]["completion_tokens"], "tokens", round(r["usage"]["completion_tokens"] / dt, 1), "tok/s end to end")
json.dump(out, open(sys.argv[2], "w"), indent=1)
