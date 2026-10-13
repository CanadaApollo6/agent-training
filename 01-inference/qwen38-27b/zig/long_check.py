# a ~TARGET-token prompt (TensorFold's Zig sources) greedy, streamed, with drafts off and on: same reply, decode speed of each
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
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft: b["draft"] = False
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0, times, out, usage = time.time(), [], "", None
    for raw in urllib.request.urlopen(r, timeout=3600):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage
        for c in d.get("choices", []):
            piece = (c.get("delta") or {}).get("content") or ""
            if piece: times.append(time.time()); out += piece
    return out, usage, times[0] - t0, times[-1] - times[0]
# decode rate from the first streamed token to the last (prefill excluded); the Python port ignores draft:false
for draft in (False, True):
    out, usage, pre, dec = ask(draft, 256)
    n = usage["completion_tokens"]
    print(f"draft {'on ' if draft else 'off'}: prompt {usage['prompt_tokens']} tokens, prefill {pre:.1f}s, decode {(n - 1) / dec:.0f} tok/s, reply {hashlib.sha256(out.encode()).hexdigest()[:12]}", flush=True)
