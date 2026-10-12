import json, sys, time, urllib.request
url = sys.argv[1]
def req(body):
    r = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; n = 0
    with urllib.request.urlopen(r) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            if first is None: first = time.perf_counter() - t0
            n += 1
    return first, time.perf_counter() - t0, n
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
for label, prompt, mt in [("14 tok, 1 out", "Write a short Python function that computes the Fibonacci sequence and explain it.", 1),
                          ("14 tok, 8 out", "Write a short Python function that computes the Fibonacci sequence and explain it.", 8),
                          ("~1000 tok, 1 out", "word " * 1000, 1), ("~4000 tok, 1 out", "word " * 4000, 1)]:
    for rep in range(3):
        f, t, n = req({"model": model, "prompt": prompt, "max_tokens": mt, "temperature": 0, "stream": True})
        print(f"{label:18s} rep {rep}: first {f*1000:7.1f} ms, total {t*1000:7.1f} ms, chunks {n}")
