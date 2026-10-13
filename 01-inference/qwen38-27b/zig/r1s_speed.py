import json, sys, time, urllib.request
url = sys.argv[1]
qs = ["Write a Python function that merges two sorted lists, with tests.", "Explain how a GPU's memory bandwidth limits LLM decoding.", "Write a short story about a lighthouse keeper."]
for q in qs:
    body = {"model": "r1s", "messages": [{"role": "user", "content": q}], "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=600)); dt = time.time() - t
    n = d["usage"]["completion_tokens"]
    print(f"{n} tokens in {dt:.2f}s = {n/dt:.0f} tok/s ({q[:40]})")
