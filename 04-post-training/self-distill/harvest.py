"""Harvest self-distillation SFT data from verifiers eval traces: Ornith 35B's own solved runs, rendered exactly as the
serving engine rendered them, with the loss on what the model sampled.

    ../../01-inference/envs/tensorfold/.venv/bin/python harvest.py --out data/all.jsonl

Each trace stores the conversation as a tree of nodes (Claude Code branches when it compacts or rewrites context) and
each model call names the node its prompt ended at. Every call's prompt is rendered with the model's chat template the
way TensorFold renders it (its normalize_messages and tool-argument normalization, json.dumps tojson); the call's
completion is the text between that prompt and the same path rendered through the sampled reply. Consecutive calls
whose prompt extends the previous prompt + completion merge into one sample, so a linear pi or prime_agent run is a
single sample with loss on every turn the model wrote.

Checks, per call: the rendered prompt must be a prefix of the rendered reply path, and on TensorFold runs (which report
exact prompt sizes) its token count must equal the call's prompt_tokens. Runs whose calls fail either check are
reported and dropped.

Held out (never harvested): the 20 TB2 tasks in ../../06-agents/harness-evals/tasks.txt, the per-attempt metric, and
every DeepSWE task, the cross-type check.
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVALS = ROOT / "06-agents/harness-evals"
sys.path.insert(0, str(ROOT / "01-inference/tools/TensorFold/src"))
from tensorfold.server.errors import RequestError  # noqa: E402
from tensorfold.server.messages import (_normalize_tool_call_arguments, late_system_role,  # noqa: E402
                                        normalize_messages)

MODEL = Path.home() / "models/ornith/r1s"
# runs served by TensorFold report the exact rendered prompt size; llama.cpp reports uncached tokens, vLLM its own render
TENSORFOLD = ("ornith35b-r1s", "ornith35b-r1-", "ornith35b-r2")


class Template:
    """The model's chat template rendered as TensorFold's ChatTemplate does (tensorfold/cuda/server.py)."""

    def __init__(self, model_dir: Path):
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string((model_dir / "chat_template.jinja").read_text())
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        self.late_system = late_system_role(
            lambda m: self.template.render(**self.specials, messages=m, add_generation_prompt=False))

    def render(self, messages: list, tools: list | None, generate: bool) -> str:
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system))
        return self.template.render(**self.specials, messages=messages, tools=tools or None,
                                    add_generation_prompt=generate, enable_thinking=True)


def chat_message(m: dict, messages_api: bool) -> dict:
    """A trace message as the chat-completions request carried it. Messages API runs (Claude Code) went through
    tensorfold/server/anthropic.py, which joins text blocks with a blank line where the chat path concatenates them."""

    out = {k: v for k, v in m.items() if v is not None and k != "tool_calls"}
    if messages_api and isinstance(out.get("content"), list):
        out["content"] = "\n\n".join(p.get("text", "") for p in out["content"] if p.get("text"))
    # it also marks failed tool results "Error: ", from an is_error flag the trace drops; Claude Code sets the flag
    # on a nonzero Bash exit, whose result opens "Exit code N"
    if messages_api and out.get("role") == "tool" and str(out.get("content", "")).startswith("Exit code "):
        out["content"] = "Error: " + out["content"]
    if m.get("tool_calls"):
        out["tool_calls"] = [{"id": c.get("id"), "type": "function",
                              "function": {"name": c["name"], "arguments": c.get("arguments", "{}")}}
                             if "function" not in c else c for c in m["tool_calls"]]
    return out


def chat_tools(tools: list | None) -> list | None:
    if not tools:
        return None
    return [t if "function" in t else {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
            for t in tools]


def held_out() -> set[str]:
    names = set((EVALS / "tasks.txt").read_text().split())
    for f in (EVALS / "deepswe_tasks.txt",):
        names |= {n.split("/")[-1] for n in f.read_text().split()}
    return names


def reward(t: dict) -> float:
    return sum((v or {}).get("score", 0) * (v or {}).get("weight", 1) for v in (t.get("rewards") or {}).values())


def harvest(trace: dict, tmpl: Template, tok: Tokenizer, exact: bool):
    """(samples, problems) for one solved trace. A sample: text, loss character spans, call count."""

    nodes = trace["nodes"]
    def path(i):
        out = []
        while i is not None:
            out.append(i)
            i = nodes[i].get("parent")
        return out[::-1]

    def conversation(i):
        """The messages and tools of the request ending at node i. A conversation's root carries its tools; side
        calls (prime_agent's /refine review gate) start a new root without them and went out with none."""
        ids = path(i)
        tools = next((nodes[j]["tools"] for j in reversed(ids) if nodes[j].get("tools")), None)
        return [chat_message(nodes[j]["message"], messages_api) for j in ids], chat_tools(tools)

    def render(i, generate):
        return tmpl.render(*conversation(i), generate)

    messages_api = any(c.get("endpoint") == "/v1/messages" for c in trace["calls"])
    samples, problems = [], []
    for call in trace["calls"]:
        if "node" not in call:                            # a failed call (429, provider error) the harness retried
            continue
        r = call["node"]                                  # the reply this call sampled; its parent ends the prompt
        p = nodes[r].get("parent")
        if not nodes[r].get("sampled") or (nodes[r]["message"] or {}).get("role") != "assistant" or p is None:
            problems.append(f"call at node {r}: not a sampled assistant reply")
            continue
        prompt, full = render(p, True), render(r, False)
        if not full.startswith(prompt):
            problems.append(f"call at node {p}: prompt is not a prefix of the reply path")
            continue
        completion = full[len(prompt):]
        completion = completion[:-1] if completion.endswith("<|im_end|>\n") else completion   # the model stops at <|im_end|>
        if exact:
            n = len(tok.encode(prompt, add_special_tokens=False).ids)
            want = (call.get("usage") or {}).get("prompt_tokens")
            if want and n != want:
                problems.append(f"call at node {p}: rendered {n} tokens, served {want}")
        # continue the sample this prompt extends (the main conversation resumes after a side call), else start one
        cur = next((x for x in reversed(samples) if prompt.startswith(x["text"])), None)
        if cur:
            cur["text"] = prompt
        else:
            cur = {"text": prompt, "loss": [], "calls": 0}
            samples.append(cur)
        cur["loss"].append([len(cur["text"]), len(cur["text"]) + len(completion)])
        cur["text"] += completion
        cur["calls"] += 1
        cur["last"] = r
    for x in samples:                       # as the template saw them, for trainers that render themselves
        messages, tools = conversation(x.pop("last"))
        x["messages"] = _normalize_tool_call_arguments(normalize_messages(messages, late_system=tmpl.late_system))
        x["tools"] = tools
    return samples, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(EVALS / "outputs/**/traces.jsonl")])
    ap.add_argument("--model", nargs="+", default=["ornith35b"],
                    help="label prefixes of the runs to harvest (hosted model ids keep their slash: deepseek/...)")
    ap.add_argument("--out", type=Path, default=HERE / "data/all.jsonl")
    a = ap.parse_args()
    tmpl, tok = Template(MODEL), Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    skip = held_out()
    files = sorted({f for g in a.runs for f in glob.glob(g, recursive=True)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    with a.out.open("w") as out:
        for f in files:
            run = Path(f).parent.name
            # <env>--<label>--<harness>--<hash>; a label's slash (hosted model ids) is written as "--" too
            parts = run.split("--")
            label, harness = "/".join(parts[1:-2]), parts[-2]
            if any(s in f for s in ("/broken", "/contended", "/lessons/")) or not label.startswith(tuple(a.model)):
                continue
            for line in open(f):
                r = json.loads(line)
                if not r.get("traces"):
                    continue
                t = r["traces"][0]
                task = r["task"]["data"]["name"].split("/")[-1]
                if reward(t) <= 0:
                    continue
                stats["solved"] += 1
                if task in skip:
                    stats["held_out"] += 1
                    continue
                try:
                    samples, problems = harvest(t, tmpl, tok, label.startswith(TENSORFOLD))
                except jinja2.exceptions.TemplateError as exc:
                    samples, problems = [], [f"template: {exc}"]
                except RequestError as exc:              # image inputs (hosted teachers); Ornith serves text only
                    samples, problems = [], [f"request: {exc}"]
                if problems:
                    stats["dropped"] += 1
                    print(f"DROP {run} {task}: {len(problems)} problems, first: {problems[0]}")
                    continue
                stats["kept"] += 1
                for k, s in enumerate(samples):
                    ids = tok.encode(s["text"], add_special_tokens=False)
                    loss_tokens = sum(1 for (a0, a1) in ids.offsets if any(lo <= a0 < hi for lo, hi in s["loss"]))
                    out.write(json.dumps({"task": task, "harness": harness, "run": run, "rollout": r["id"],
                                          "segment": k, "calls": s["calls"], "tokens": len(ids.ids),
                                          "loss_tokens": loss_tokens, "text": s["text"], "loss": s["loss"],
                                          "messages": s["messages"], "tools": s["tools"]}) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main()
