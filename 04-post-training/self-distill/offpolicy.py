"""How off-policy is teacher data for Ornith? Score harvested samples under a model served by vLLM and compare the
log-probability it gives the written tokens of teacher rollouts with the same for its own rollouts.

Each sample is sent as token ids (the Ornith render harvest.py made) to /v1/completions with prompt_logprobs=0, which
returns the log-probability of every prompt token. Only loss positions count, the tokens the agent wrote, split into
reasoning (before </think> in a turn) and the rest (answer text and tool calls).

    python offpolicy.py score --samples /tmp/teacher.jsonl data/round2.jsonl --own-limit 80 --url http://127.0.0.1:8200
    python offpolicy.py report

Per-sample results append to --out (default results/offpolicy.jsonl).
"""
import argparse
import collections
import json
import random
import statistics
import sys
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODEL = Path.home() / "models/ornith/r1s"
SURPRISE = -5.0     # a written token the model gives under e^-5 (0.7%) counts as surprising


def source(r: dict) -> str:
    label = "/".join(r["run"].split("--")[1:-2])
    return label if "/" in label else "own:" + label.split("-dg-")[0]


def reasoning_mask(r: dict, offsets: list) -> list:
    """Per token: None outside the loss, True in a turn's reasoning, False after its </think>."""
    spans = []
    for a, b in r["loss"]:
        end = r["text"].find("</think>", a, b)
        spans.append((a, b, end if end >= 0 else a))     # no </think> in the turn: nothing counts as reasoning
    out = []
    for a0, _ in offsets:
        tag = None
        for a, b, end in spans:
            if a <= a0 < b:
                tag = a0 < end
                break
        out.append(tag)
    return out


def score(args):
    OUT = args.out
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    rows = []
    for f in args.samples:
        rs = [json.loads(line) for line in open(f)]
        if all(source(r).startswith("own:") for r in rs) and args.own_limit:
            random.Random(0).shuffle(rs)
            rs = rs[:args.own_limit]
        rows += rs
    done = {(json.loads(line)["rollout"], json.loads(line)["segment"]) for line in open(OUT)} if OUT.exists() else set()
    OUT.parent.mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        if (r["rollout"], r["segment"]) in done:
            continue
        enc = tok.encode(r["text"], add_special_tokens=False)
        body = json.dumps({"model": args.model, "prompt": enc.ids, "max_tokens": 1, "prompt_logprobs": 0,
                           "temperature": 0}).encode()
        req = urllib.request.Request(args.url + "/v1/completions", body, {"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(req, timeout=3600))
        plp = res["choices"][0]["prompt_logprobs"]                   # [None, {id: {logprob, ...}}, ...]
        lps = [None] + [d[str(t)]["logprob"] for d, t in zip(plp[1:], enc.ids[1:])]
        mask = reasoning_mask(r, enc.offsets)
        by = {"reasoning": [], "action": []}
        for lp, m in zip(lps, mask):
            if m is not None and lp is not None:
                by["reasoning" if m else "action"].append(lp)
        rec = {"source": source(r), "task": r["task"], "rollout": r["rollout"], "segment": r["segment"],
               "tokens": len(enc.ids)}
        for k, v in by.items():
            rec[k] = {"n": len(v), "sum": sum(v), "surprising": sum(x < SURPRISE for x in v),
                      "median": statistics.median(v) if v else None}
        with OUT.open("a") as out:
            out.write(json.dumps(rec) + "\n")
        print(f"{i + 1}/{len(rows)} {rec['source']:28} {r['task'][:24]:24} {len(enc.ids):6} tok  "
              f"reasoning {rec['reasoning']['sum'] / max(1, rec['reasoning']['n']):6.3f}  "
              f"action {rec['action']['sum'] / max(1, rec['action']['n']):6.3f}", flush=True)


def report(args):
    OUT = args.out
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, 0]))
    per_sample = collections.defaultdict(list)
    for line in open(OUT):
        r = json.loads(line)
        for k in ("reasoning", "action"):
            a = agg[r["source"]][k]
            a[0] += r[k]["n"]; a[1] += r[k]["sum"]; a[2] += r[k]["surprising"]
        n = r["reasoning"]["n"] + r["action"]["n"]
        if n:
            per_sample[r["source"]].append((r["reasoning"]["sum"] + r["action"]["sum"]) / n)
    print(f"{'source':30} {'samples':>7} {'tokens':>9} {'logprob/tok':>11} {'ppl':>6} {'reason':>7} {'action':>7} "
          f"{'surprising':>10}")
    for s in sorted(agg):
        a = agg[s]
        n = a["reasoning"][0] + a["action"][0]
        tot = a["reasoning"][1] + a["action"][1]
        sur = a["reasoning"][2] + a["action"][2]
        print(f"{s:30} {len(per_sample[s]):7} {n:9} {tot / n:11.3f} {2.718281828 ** (-tot / n):6.2f} "
              f"{a['reasoning'][1] / max(1, a['reasoning'][0]):7.3f} {a['action'][1] / max(1, a['action'][0]):7.3f} "
              f"{sur / n:10.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--samples", nargs="+", required=True)
    s.add_argument("--own-limit", type=int, default=80, help="random own samples to score per file of own runs")
    s.add_argument("--url", default="http://127.0.0.1:8200")
    s.add_argument("--model", default="ornith")
    r = sub.add_parser("report")
    for p in (s, r):
        p.add_argument("--out", type=Path, default=HERE / "results/offpolicy.jsonl")
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
