"""Checks the LFM2 server (tensorfold-native serve) end to end: chat, tool calls, an agent loop's turn reuse, streaming,
sampling and a long prompt. Usage: python3 server_check.py [port]. Prints one line a check and exits 1 on a failure."""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8897
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
failed = []


def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps({"model": "m", **body, "stream": stream}).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r), time.time() - t0
        events = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return events, time.time() - t0


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failed.append(name)


def tf(d):
    t = d["tensorfold"]
    return f"{d['usage']['completion_tokens']} tokens at {t['tokens_per_second']:.0f} tok/s, prefill {t['prefill_seconds']:.3f}s"


# 1. tool call
ask = [{"role": "user", "content": "What's the weather in Paris and in Tokyo right now, in celsius?"}]
d, _ = post({"messages": ask, "tools": TOOLS, "max_tokens": 2000})
m = d["choices"][0]["message"]
calls = m.get("tool_calls") or []
check("tool calls", len(calls) == 2 and d["choices"][0]["finish_reason"] == "tool_calls",
      f"{[c['function']['arguments'] for c in calls]} ({tf(d)})")

# 2. the agent loop's next turn: the tool results appended. Resumed (the stream holds turn 1) vs cold (after another
# prompt), the reply must be the same tokens.
turn2 = ask + [{"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content"),
                "tool_calls": calls}]
for c, temp in zip(calls, ["18", "24"]):
    turn2.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"temp_c": int(temp), "sky": "clear"})})
warm, _ = post({"messages": turn2, "tools": TOOLS, "max_tokens": 1500})
post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 20})
cold, _ = post({"messages": turn2, "tools": TOOLS, "max_tokens": 1500})
check("turn 2 resumed == cold", warm["tensorfold"]["token_sha"] == cold["tensorfold"]["token_sha"],
      f"resumed prefill {warm['tensorfold']['prefill_seconds']:.3f}s vs cold {cold['tensorfold']['prefill_seconds']:.3f}s; "
      f"answer {warm['choices'][0]['message']['content'][:80]!r}")

# 3. streaming: the same reply as the whole one, calls arriving as deltas
events, _ = post({"messages": ask, "tools": TOOLS, "max_tokens": 2000}, stream=True)
names = [tc["function"].get("name") for e in events for ch in e.get("choices", []) for tc in
         (ch.get("delta", {}).get("tool_calls") or []) if tc.get("function", {}).get("name")]
content = "".join(ch.get("delta", {}).get("content") or "" for e in events for ch in e.get("choices", []))
check("streamed tool calls", names == ["get_weather", "get_weather"] and "tool_call" not in content,
      f"names {names}, content {content[:60]!r}")

# 4. plain code reply, greedy and drafted; then sampled
code = [{"role": "user", "content": "Write a Python function that parses an ISO 8601 date string without "
                                    "using datetime, with tests."}]
d, _ = post({"messages": code, "max_tokens": 1500})
check("greedy code", d["usage"]["completion_tokens"] > 100, tf(d))
d, _ = post({"messages": code, "max_tokens": 1500, "temperature": 0.7, "top_p": 0.95, "seed": 1})
check("sampled code", d["usage"]["completion_tokens"] > 100, tf(d))

# 5. a long prompt: the rewrite task of results/drafting.md (long_prompt.json: 14.7K chat-templated token ids)
ids = json.load(open(__file__.rsplit("/", 1)[0] + "/long_prompt.json"))["long"]
req = urllib.request.Request(URL.replace("chat/completions", "completions"),
                             json.dumps({"model": "m", "prompt": ids, "max_tokens": 600}).encode(),
                             {"content-type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=600))
check("long prompt", d["usage"]["completion_tokens"] > 50,
      f"prompt {d['usage']['prompt_tokens']} tokens, {tf(d)}, "
      f"{d['usage']['prompt_tokens'] / d['tensorfold']['prefill_seconds']:.0f} tok/s prefill")

# 6. a long agent history's next turn: only the new tokens prefill (the 14.7K prompt as a user turn, then a follow-up)
req = urllib.request.Request(URL.replace("v1/chat/completions", "detokenize"), json.dumps({"model": "m", "tokens": ids}).encode(),
                             {"content-type": "application/json"})
text = json.load(urllib.request.urlopen(req))["prompt"]
user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
hist = [{"role": "user", "content": user}]
d1, _ = post({"messages": hist, "max_tokens": 400})
hist.append({"role": "assistant", "content": d1["choices"][0]["message"]["content"] or "",
             "reasoning_content": d1["choices"][0]["message"].get("reasoning_content")})
hist.append({"role": "user", "content": "Now add type hints to it."})
d2, _ = post({"messages": hist, "max_tokens": 200})
post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 20})
d3, _ = post({"messages": hist, "max_tokens": 200})
check("long history turn 2", d2["tensorfold"]["prefill_seconds"] < 0.5 and
      d2["tensorfold"]["token_sha"] == d3["tensorfold"]["token_sha"],
      f"turn 1 prefill {d1['tensorfold']['prefill_seconds']:.2f}s for {d1['usage']['prompt_tokens']} tokens; "
      f"turn 2 prefill {d2['tensorfold']['prefill_seconds']:.3f}s for {d2['usage']['prompt_tokens']} tokens "
      f"(cold {d3['tensorfold']['prefill_seconds']:.2f}s, same reply: {d2['tensorfold']['token_sha'] == d3['tensorfold']['token_sha']}), {tf(d2)}")

sys.exit(1 if failed else 0)
