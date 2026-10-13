# replies of N concurrent greedy requests vs each alone (and draft:false), the CONTRIBUTING contract
import json, sys, urllib.request, concurrent.futures as cf, hashlib
url = sys.argv[1]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
prompts = ["Write a Python function that merges two sorted lists.", "Explain how a GPU's memory bandwidth limits LLM decoding.", "List five facts about octopuses."]
def ask(p, draft=True):
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 160, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft: body["draft"] = False
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]["content"]
h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
solo = [ask(p) for p in prompts]
plain = [ask(p, draft=False) for p in prompts]
with cf.ThreadPoolExecutor(len(prompts)) as ex: together = list(ex.map(ask, prompts))
for i, p in enumerate(prompts):
    print(f"prompt {i}: solo {h(solo[i])} draft-off {h(plain[i])} concurrent {h(together[i])}", "SAME" if solo[i] == plain[i] == together[i] else "DIFF")
