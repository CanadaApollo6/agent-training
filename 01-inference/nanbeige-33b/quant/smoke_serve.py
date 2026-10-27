"""Smoke test for serve_exl3.py before a TB2 pilot: one streamed tool-call turn (the reply must parse into a bash
call), then N concurrent long-ish replies to measure decode speed per stream under load. Stdlib only.

    python smoke_serve.py [--url http://127.0.0.1:8000] [--concurrency 7] [--tokens 1500]
"""
import argparse
import json
import threading
import time
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
                   "required": ["command"]}}}]


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = {"reasoning": "", "content": "", "tool_calls": [], "usage": None, "first": None}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                out["usage"] = c["usage"]
            for ch in c.get("choices", []):
                d = ch["delta"]
                if d.get("reasoning") and out["first"] is None:
                    out["first"] = time.time() - t0
                out["reasoning"] += d.get("reasoning") or ""
                out["content"] += d.get("content") or ""
                out["tool_calls"] += d.get("tool_calls") or []
                out["finish"] = ch.get("finish_reason") or out.get("finish")
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=7)
    ap.add_argument("--tokens", type=int, default=1500)
    a = ap.parse_args()

    r = post(a.url, {"model": "nanbeige-exl3-a1", "stream": True, "stream_options": {"include_usage": True},
                     "tools": TOOLS, "messages": [
                         {"role": "system", "content": "You are a coding agent in a Linux sandbox."},
                         {"role": "user", "content": "How many CPU cores does this machine have? Use the bash tool."}]})
    calls = r["tool_calls"]
    ok = calls and calls[0]["function"]["name"] == "bash" and "command" in json.loads(calls[0]["function"]["arguments"])
    print(f"TOOL_TURN {'OK' if ok else 'BAD'} finish={r.get('finish')} reasoning={len(r['reasoning'])} chars "
          f"content={r['content'][:80]!r} calls={json.dumps(calls)[:200]} usage={r['usage']}")

    results = [None] * a.concurrency

    def one(i):
        results[i] = post(a.url, {"model": "nanbeige-exl3-a1", "stream": True, "max_tokens": a.tokens,
                                  "stream_options": {"include_usage": True}, "messages": [
                                      {"role": "user", "content": f"Write a detailed essay (#{i}) on the history of "
                                                                  "compilers, with code examples."}]})

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = [x["usage"]["completion_tokens"] for x in results]
    per = [n / x["seconds"] for n, x in zip(toks, results)]
    print(f"LOAD concurrency={a.concurrency} tokens={sum(toks)} wall={wall:.0f}s aggregate={sum(toks) / wall:.0f} tok/s "
          f"per-stream min/mean={min(per):.1f}/{sum(per) / len(per):.1f} tok/s")


if __name__ == "__main__":
    main()
