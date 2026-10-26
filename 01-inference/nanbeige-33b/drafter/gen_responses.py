"""On a pod with Nanbeige4.3-33B on vLLM (06-agents/harness-evals/serve_nanbeige_pod.sh): write the model's own
responses, thinking included, for every prompt. Same sampling as the agent runs (the server's T 1.0, top_p 0.95) so the
draft head trains on the text it will see. Resumable: prompts already in the output are skipped. Each pod takes the
prompts with index % N == K. PORT (default 8000) picks the server, for one server a GPU.

    python3 gen_responses.py prompts.jsonl out.jsonl <model name> [K N] [concurrency] [max tokens]
"""
import asyncio
import json
import os
import sys

import httpx

src, dst, model = sys.argv[1:4]
K, N = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
CONC = int(sys.argv[6]) if len(sys.argv) > 6 else 256
MAX = int(sys.argv[7]) if len(sys.argv) > 7 else 4096


async def main():
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(dst)}
    except FileNotFoundError:
        pass
    todo = [p for i, l in enumerate(open(src)) if i % N == K and (p := json.loads(l))["id"] not in done]
    print(f"{len(todo)} to go, {len(done)} done", flush=True)
    sem, out, n = asyncio.Semaphore(CONC), open(dst, "a"), 0
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{os.environ.get('PORT', 8000)}/v1", timeout=3600,
                                 limits=httpx.Limits(max_connections=CONC)) as client:
        async def one(p):
            nonlocal n
            async with sem:
                try:
                    r = (await client.post("/chat/completions", json={
                        "model": model, "messages": p["messages"], "max_tokens": MAX})).json()
                    c = r["choices"][0]
                except Exception as e:  # keep going; a rerun picks it up
                    print("fail", p["id"], repr(e)[:200], flush=True)
                    return
                m = c["message"]
                out.write(json.dumps({**p, "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
                                      "content": m.get("content") or "", "finish": c["finish_reason"],
                                      "usage": r["usage"]}) + "\n")
                n += 1
                if n % 500 == 0:
                    out.flush()
                    print(n, flush=True)
        await asyncio.gather(*(one(p) for p in todo))
    out.close()


asyncio.run(main())
