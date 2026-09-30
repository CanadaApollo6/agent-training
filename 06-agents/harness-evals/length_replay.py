"""Is a build wordier per turn given the same context? Replay recorded agent turns through a served build.

Whole-task token counts mix two things: how much a build writes per turn, and how many turns its run takes (a wrong
step early costs many turns later). Replaying fixes the context: each prefix is a recorded conversation up to one of
the model's turns, sent again as the same chat request, and only the next turn is generated.

    prompts   render the recorded prefixes with the model's chat template, earlier turns' thinking kept or dropped,
              and match the token counts each server recorded: did both engines see the same prompts?
    replay    send prefixes from --from builds to the server at --port, --samples times each, and compare each
              reply's length with the recorded turn at that prefix

Run it in the TensorFold env (it renders with TensorFold's own template code):

    cd 01-inference/envs/tensorfold && uv run python ../../../06-agents/harness-evals/length_replay.py prompts
"""
import argparse
import concurrent.futures as cf
import json
import math
import random
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_builds import OUT  # noqa: E402

MODEL_DIR = Path.home() / "models/ornith/r1"
RESULTS = Path(__file__).parent / "results"


def traces(build):
    """(attempt, task, trace) for every trace of a build, qemu-startup dropped."""
    for d in sorted(OUT.glob(f"terminal-bench-2--ornith35b-{build}-a*--pi--*")):
        attempt = int(d.name.split(f"{build}-a")[1].split("--")[0])
        for line in open(d / "traces.jsonl"):
            r = json.loads(line)
            if not r.get("traces"):
                continue
            t = r["traces"][0]
            task = t["task"]["data"]["name"].split("/")[-1]
            if task != "qemu-startup":
                yield attempt, task, t


def openai_message(m, thinking=True):
    """A recorded message as the chat request sent it (tool calls nested under ``function``)."""
    out = {k: v for k, v in m.items() if k in ("role", "content", "name", "tool_call_id")}
    if m.get("reasoning_content") and thinking:
        out["reasoning_content"] = m["reasoning_content"]
    if m.get("tool_calls"):
        out["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
    if m["role"] == "assistant" and out.get("content") is None:
        out["content"] = ""
    return out


def openai_tools(tools):
    return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}}
            for t in tools]


def turns(build):
    """Each recorded model turn: its prefix messages, tools, and what the server reported for it."""
    for attempt, task, t in traces(build):
        for i, call in enumerate(c for c in t["calls"] if "node" in c):     # errored calls have no reply
            node = t["nodes"][call["node"]]
            if any(isinstance(n["message"].get("content"), list) and any(p.get("type") != "text" for p in
                   n["message"]["content"]) for n in t["nodes"][:call["node"]]):
                break                                                         # images: a text server refuses them
            yield {"build": build, "attempt": attempt, "task": task, "turn": i,
                   "messages": [n["message"] for n in t["nodes"][:call["node"]]],
                   "tools": node.get("tools") or t["tools"], "reply": node["message"],
                   "prompt_tokens": call["usage"]["prompt_tokens"],
                   "completion_tokens": call["usage"]["completion_tokens"], "finish": call["finish_reason"]}


def cmd_prompts(args):
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    template, tok = ChatTemplate(MODEL_DIR), Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rng = random.Random(0)
    for build in args.builds:
        rows = [r for r in turns(build) if r["turn"] > 0]
        diffs = {"kept": [], "dropped": []}
        for r in rng.sample(rows, min(args.n, len(rows))):
            for variant, thinking in (("kept", True), ("dropped", False)):
                text = template.render([openai_message(m, thinking) for m in r["messages"]],
                                       tools=openai_tools(r["tools"]), enable_thinking=True)
                diffs[variant].append(len(tok.encode(text, add_special_tokens=False).ids) - r["prompt_tokens"])
        print(f"{build}: rendered minus recorded prompt tokens over {len(diffs['kept'])} turns")
        for variant, d in diffs.items():
            exact = sum(x == 0 for x in d)
            print(f"  thinking {variant:7s}: median {st.median(d):+.0f}, range {min(d):+d} to {max(d):+d}, "
                  f"exact on {exact}")


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def cmd_replay(args):
    rng = random.Random(args.seed)
    picked = []
    for build in args.builds:
        by_task = {}
        for r in turns(build):
            if r["prompt_tokens"] <= args.max_prompt:
                by_task.setdefault(r["task"], []).append(r)
        for task, rows in sorted(by_task.items()):         # the same number of turns from every task
            picked += rng.sample(rows, min(args.per_task, len(rows)))
    jobs = [(r, s) for r in picked for s in range(args.samples)]
    rng.shuffle(jobs)
    print(f"{len(picked)} prefixes x {args.samples} samples = {len(jobs)} requests, {args.workers} at a time",
          flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"replay-{args.label}.jsonl"
    t0 = time.time()

    def run(job):
        r, s = job
        body = {"model": args.model, "messages": [openai_message(m) for m in r["messages"]],
                "tools": openai_tools(r["tools"]), "max_tokens": 16384}
        out = post(args.port, body, args.timeout)
        choice = out["choices"][0]
        return {k: r[k] for k in ("build", "attempt", "task", "turn", "prompt_tokens", "completion_tokens",
                                  "finish")} | {
            "sample": s, "replay_tokens": out["usage"]["completion_tokens"],
            "replay_prompt_tokens": out["usage"]["prompt_tokens"], "replay_finish": choice["finish_reason"],
            "replay_tools": len(choice["message"].get("tool_calls") or [])}

    with open(path, "w") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {path}")
    summarize(path)


def summarize(path):
    """Per prefix: the mean replayed length against the recorded one; geometric mean over prefixes, bootstrap CI."""
    rows = [json.loads(line) for line in open(path)]
    groups = {}
    for r in rows:
        groups.setdefault((r["build"], r["attempt"], r["task"], r["turn"]), []).append(r)
    rng = random.Random(0)
    for build in sorted({k[0] for k in groups}):
        logs, tasks = [], {}
        for key, g in groups.items():
            if key[0] != build:
                continue
            rec, rep = g[0]["completion_tokens"], st.mean(x["replay_tokens"] for x in g)
            logs.append(math.log((rep + 1) / (rec + 1)))
            tasks.setdefault(key[2], []).append(logs[-1])
        bs = sorted(math.exp(st.mean(rng.choices(logs, k=len(logs)))) for _ in range(20000))
        rec_all = sum(g[0]["completion_tokens"] for k, g in groups.items() if k[0] == build)
        rep_all = sum(st.mean(x["replay_tokens"] for x in g) for k, g in groups.items() if k[0] == build)
        print(f"prefixes from {build}: replayed / recorded length per turn {math.exp(st.mean(logs)):.2f} "
              f"(CI {bs[500]:.2f}-{bs[19500]:.2f}) over {len(logs)} turns; total tokens {rep_all / rec_all:.2f}; "
              f"longer on {sum(x > 0 for x in logs)} of {len(logs)}")
        for task, v in sorted(tasks.items()):
            print(f"    {task:30s} {math.exp(st.mean(v)):.2f} ({len(v)} turns)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--builds", nargs="+", default=["q8", "r1-c4s"])
    p.add_argument("--n", type=int, default=60)
    p = sub.add_parser("replay")
    p.add_argument("--builds", nargs="+", default=["q8", "r1-c4s"], help="builds whose recorded turns to replay")
    p.add_argument("--label", required=True)
    p.add_argument("--model", default="ornith35b-r1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-prompt", type=int, default=32000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("summary")
    p.add_argument("path")
    args = ap.parse_args()
    {"prompts": cmd_prompts, "replay": cmd_replay, "summary": lambda a: summarize(a.path)}[args.cmd](args)


if __name__ == "__main__":
    main()
