# a ~TARGET-token prompt (TensorFold's Zig sources) greedy with drafts on and off: same reply, decode speed of each
import json, sys, glob, time, urllib.request, hashlib
url, target = sys.argv[1], int(sys.argv[2])
root = sys.argv[3]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
body, files = "", sorted(glob.glob(root + "/**/*.zig", recursive=True))
for f in files:
    body += f"\n// FILE {f[len(root):]}\n" + open(f).read()
    if len(body) > target * 3.2: break
prompt = "Here is some Zig source code:\n" + body + "\n\nExplain in detail what the last file does."
def ask(draft, n):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft: b["draft"] = False
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); j = json.load(urllib.request.urlopen(r, timeout=3600)); return j, time.time() - t
for draft in (False, True):
    one, t1 = ask(draft, 1)
    full, tn = ask(draft, 256)
    n = full["usage"]["completion_tokens"]
    print(f"draft {'on ' if draft else 'off'}: prompt {full['usage']['prompt_tokens']} tokens, prefill {t1:.1f}s, decode {(n - 1) / (tn - t1):.0f} tok/s, reply {hashlib.sha256(full['choices'][0]['message']['content'].encode()).hexdigest()[:12]}", flush=True)
