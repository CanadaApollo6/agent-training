"""Share of bf16 weight elements that differ between two checkpoints of the same model, overall and by kind.

NeMo-DCR (root README) measured 0.6-1.2% of bf16 elements changing per GRPO step across six models. A pilot whose
weights barely differ from the start after all its steps had updates too small to survive bf16 rounding, so it
can't have learned, whatever the rewards say.

    python weights_moved.py BASE_DIR TUNED_DIR [--out moved.json]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

KINDS = [("experts", r"\.mlp\.experts\."), ("shared_expert", r"shared_expert"), ("router", r"\.mlp\.gate\.weight$"),
         ("attention", r"self_attn|linear_attn"), ("embed_head", r"embed_tokens|lm_head"), ("vision", r"visual"),
         ("norm", r"norm")]


def kind(name):
    return next((k for k, rx in KINDS if re.search(rx, name)), "other")


def index(d):
    files = json.loads((Path(d) / "model.safetensors.index.json").read_text())["weight_map"]
    return {n: Path(d) / f for n, f in files.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tuned")
    ap.add_argument("--out")
    a = ap.parse_args()
    bi, ti = index(a.base), index(a.tuned)
    common = sorted(set(bi) & set(ti))
    tot, chg = collections.Counter(), collections.Counter()
    handles = {}

    def get(path, name):
        if path not in handles:
            handles[path] = safe_open(str(path), "pt")
        return handles[path].get_tensor(name)

    for n in common:
        b, t = get(bi[n], n), get(ti[n], n)
        if b.shape != t.shape:
            continue
        k = kind(n)
        tot[k] += b.numel()
        chg[k] += int((b.to(torch.bfloat16).view(torch.int16) != t.to(torch.bfloat16).view(torch.int16)).sum())
    res = {k: {"elements": tot[k], "changed": chg[k], "share": chg[k] / tot[k]} for k in tot}
    all_t, all_c = sum(tot.values()), sum(chg.values())
    res["all"] = {"elements": all_t, "changed": all_c, "share": all_c / all_t}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["elements"]):
        print(f"{k:14s} {v['share'] * 100:7.3f}% of {v['elements'] / 1e9:.2f}B changed")
    print(f"only in base: {len(set(bi) - set(ti))}, only in tuned: {len(set(ti) - set(bi))}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
