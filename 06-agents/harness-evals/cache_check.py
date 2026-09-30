"""Does the live serving path give the same tokens as a cold prefill? A bit-exact check of R1 on real agent turns.

In an eval, most turns resume from state the server kept: a prompt prefix held on the GPU, or a conversation parked
in RAM and copied back (the swap). The engine is built so that none of this changes a bit of the output. This checks
that on R1 itself:

    live   drive 6 conversations turn by turn through the running server (``serve_r2_local.sh`` with R1), round-robin
           so they evict each other and come back through the swap. Greedy for 3 of them, seeded sampling for the
           other 3; each reply's token ids are kept.
    cold   stop the server, load R1 in-process, and generate every one of those prompts again from an empty cache.

The two must match token for token. Starting points are R1's recorded eval conversations (system prompt, task, tools);
tool results are the recorded ones, in order, so the conversations stay realistic.

    uv run python cache_check.py live                  # harness-evals env, server up on :8000
    cd 01-inference/envs/tensorfold && uv run python ../../../06-agents/harness-evals/cache_check.py cold
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from length_replay import MODEL_DIR, RESULTS, openai_message, openai_tools, post, traces  # noqa: E402

LIVE = RESULTS / "cache-check-live.json"
SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}


def starts(n, min_turns):
    """The first n R1 conversations from distinct tasks with at least min_turns turns: opening messages, tools and
    the recorded tool results in order."""
    seen, out = set(), []
    for attempt, task, t in traces("r1-c4s"):
        msgs = [n["message"] for n in t["nodes"]]
        if task in seen or sum(m["role"] == "assistant" for m in msgs) < min_turns:
            continue
        if any(isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"]) for m in msgs):
            continue
        seen.add(task)
        opening = [m for m in msgs[:2] if m["role"] in ("system", "user")]
        out.append({"task": task, "attempt": attempt, "opening": opening, "tools": t["tools"],
                    "results": [m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                                for m in msgs if m["role"] == "tool"]})
        if len(out) == n:
            break
    return out


def cmd_live(args):
    convs = starts(args.conversations, args.turns)
    for i, c in enumerate(convs):
        c["greedy"] = i < len(convs) // 2
        c["messages"] = [openai_message(m) for m in c["opening"]]
        c["used"], c["requests"], c["done"] = 0, [], False
    t0 = time.time()
    for turn in range(args.turns):
        for i, c in enumerate(convs):
            if c["done"]:
                continue
            body = {"model": args.model, "messages": c["messages"], "tools": openai_tools(c["tools"]),
                    "max_tokens": args.max_tokens, "return_token_ids": True}
            body |= {"temperature": 0} if c["greedy"] else SAMPLING | {"seed": 1000 * i + turn}
            out = post(args.port, body, 900)
            msg, stats = out["choices"][0]["message"], out["tensorfold"]
            c["requests"].append({"body": body, "prompt_tokens": out["usage"]["prompt_tokens"],
                                  "token_ids": stats["token_ids"], "stats": {k: v for k, v in stats.items()
                                                                              if k != "token_ids"},
                                  "finish": out["choices"][0]["finish_reason"]})
            print(f"{time.time() - t0:5.0f} s  conv {i} turn {turn}: prompt {out['usage']['prompt_tokens']}, "
                  f"reply {len(stats['token_ids'])}, cached {stats.get('cached')}, "
                  f"swapped in {stats.get('swapped_in')}", flush=True)
            reply = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                reply["reasoning_content"] = msg["reasoning_content"]
            if msg.get("tool_calls"):
                reply["tool_calls"] = msg["tool_calls"]
            c["messages"] = c["messages"] + [reply]
            if not msg.get("tool_calls"):
                c["done"] = True
                continue
            for call in msg["tool_calls"]:
                result = c["results"][c["used"] % len(c["results"])][:args.result_chars]
                c["used"] += 1
                c["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": result})
    RESULTS.mkdir(exist_ok=True)
    LIVE.write_text(json.dumps([{k: c[k] for k in ("task", "attempt", "greedy", "requests")} for c in convs]))
    n = sum(len(c["requests"]) for c in convs)
    swapped = sum(bool(r["stats"].get("swapped_in")) for c in convs for r in c["requests"])
    print(f"wrote {LIVE}: {n} requests, {swapped} resumed from the RAM swap")


def cmd_cold(args):
    import os

    import torch
    from tokenizers import Tokenizer

    from tensorfold.cuda import capacity
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe import cuda_engine
    from tensorfold.families.qwen3_5_moe.cuda.engine import KEEP
    from tensorfold.server.tools import active_tool_specs
    from tensorfold.cuda.streams import PrefixCache

    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    convs = json.loads(LIVE.read_text())
    template, tok = ChatTemplate(MODEL_DIR), Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    engine = cuda_engine(MODEL_DIR, context=args.context)
    same = total = 0
    rows = []
    for i, c in enumerate(convs):
        for turn, r in enumerate(c["requests"]):
            body = r["body"]
            text = template.render(body["messages"], tools=active_tool_specs(body["tools"], None), enable_thinking=True)
            prompt = tok.encode(text, add_special_tokens=False).ids
            engine.cache, engine.swap = PrefixCache(KEEP), None          # cold: nothing kept, nothing parked
            sampling = None if c["greedy"] else Sampling(body["seed"], SAMPLING["temperature"], SAMPLING["top_k"],
                                                         SAMPLING["top_p"])
            out = []
            engine.generate(prompt, body["max_tokens"], sampling, lambda t: out.extend(t) and False)
            live = r["token_ids"]
            first = next((j for j, (a, b) in enumerate(zip(live, out)) if a != b), None)
            ok = len(prompt) == r["prompt_tokens"] and first is None and len(out) >= len(live)
            same += ok
            total += 1
            rows.append({"conv": i, "turn": turn, "greedy": c["greedy"], "prompt_same": len(prompt) == r["prompt_tokens"],
                         "live_len": len(live), "cold_len": len(out), "first_diff": first,
                         "cached": r["stats"].get("cached"), "swapped_in": r["stats"].get("swapped_in"), "ok": ok})
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.synchronize()
    (RESULTS / "cache-check-cold.json").write_text(json.dumps(rows, indent=1))
    print(f"{same} of {total} replies identical token for token")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live")
    p.add_argument("--conversations", type=int, default=6)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--result-chars", type=int, default=3000)
    p.add_argument("--model", default="ornith35b-r1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("cold")
    p.add_argument("--context", type=int, default=32768)
    args = ap.parse_args()
    {"live": cmd_live, "cold": cmd_cold}[args.cmd](args)


if __name__ == "__main__":
    main()
