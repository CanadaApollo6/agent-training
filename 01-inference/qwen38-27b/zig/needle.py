import json, sys, time, urllib.request, random
url = sys.argv[1]; n = int(sys.argv[2])
random.seed(0)
words = "the river crossed a quiet valley where farmers grew wheat and barley under a pale sky while birds sang".split()
filler = []
for i in range(int(n / 1.3)):  # ~1.3 words a token is a guess; usage reports the real count
    filler.append(random.choice(words))
mid = len(filler) // 2
filler[mid:mid] = "Remember this: the secret code for the vault is PURPLE-HARBOR-7429.".split()
text = " ".join(filler)
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": text + "\n\nWhat is the secret code for the vault? Reply with the code only."}],
        "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); out = json.load(urllib.request.urlopen(r, timeout=900)); t = time.perf_counter() - t0
print(f"prompt tokens {out['usage']['prompt_tokens']}, {t:.1f} s, {out['usage']['prompt_tokens']/t:.0f} tok/s, reply: {out['choices'][0]['message']['content']!r}")
