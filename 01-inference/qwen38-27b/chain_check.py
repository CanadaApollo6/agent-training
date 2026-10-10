"""Which verify widths give a chain's rows the bits of serial decoding (Qwen3.8-27B, TensorFold CUDA)?

Prefills one prompt, decodes N tokens serially (keeping each step's logits), then from the same prefix runs one
verify window holding the first W of those tokens as a chain, for each W, and compares every row's logits to the
serial step's. Prints the first differing row and the max abs logit difference per width. Env toggles (QMM_SWAP=0,
TENSORFOLD_KV_BITS) narrow down which kernel is responsible.

Usage (from envs/tensorfold): uv run python ../../qwen38-27b/chain_check.py [--widths 1,2,4,8,9,12,16]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def snapshot(repo: str) -> Path:
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    return sorted(base.iterdir())[-1]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,2,3,4,6,8,9,12,16")
    ap.add_argument("--context", type=int, default=8192)
    a = ap.parse_args()
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()
    from tensorfold.cuda import capacity
    capacity.available_bytes = lambda torch_mod: 200 * capacity.GIB
    from tensorfold.cuda.kernels import qmm
    if os.environ.get("QMM_SWAP") == "0":
        qmm.SWAP = False
    if os.environ.get("QMM_SEG_ROWS"):
        qmm.SEG_ROWS = int(os.environ["QMM_SEG_ROWS"])
    from tokenizers import Tokenizer
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
    from tensorfold.families.qwen3_5.cuda.decode import prefill, clone_state, _tokens
    from tensorfold.families.qwen3_5.cuda.forward import tree_forward, commit

    model_dir = snapshot("Vontra/Qwen3.8-27B-MLX-4bit")
    eng = Qwen27Engine(model_dir, None, max_rows=16, context=a.context, context_explicit=True)
    tok, template = Tokenizer.from_file(str(model_dir / "tokenizer.json")), ChatTemplate(model_dir)
    text = template.render([{"role": "user", "content": "Write a Python function that merges two sorted lists."}],
                           tools=None, enable_thinking=True, extra={"reasoning_effort": "medium"})
    ids = tok.encode(text, add_special_tokens=False).ids
    widths = [int(x) for x in a.widths.split(",")]
    n = max(widths)
    base, pending = prefill(eng.w, ids, None, None, limit=eng.context_window)

    st = clone_state(base)
    seq, serial = [pending], []
    for _ in range(n):
        logits, record = tree_forward(eng.w, _tokens([seq[-1]], eng.w.norm.device), [-1], st)
        serial.append(logits[0].float().clone())
        commit(st, record, [0])
        seq.append(int(logits[0].argmax()))

    if os.environ.get("BISECT"):                         # first op whose row-0 output differs: width 1 vs BISECT
        import tensorfold.families.qwen3_5.cuda.forward as fw
        log: list = []
        def wrap(mod, name):
            fn = getattr(mod, name)
            def inner(*args, **kw):
                out = fn(*args, **kw)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for j, o in enumerate(outs):
                    if isinstance(o, torch.Tensor) and o.dim() >= 1 and o.shape[0] >= 1:
                        log.append((f"{mod.__name__.split('.')[-1]}.{name}[{j}]", o[0].float().clone()))
                return out
            setattr(mod, name, inner)
        for name in ("add_rmsnorm", "gdn_pre", "gated_norm", "attn_prep", "gate_mul", "swiglu", "embedding"):
            wrap(fw.glue, name)
        wrap(fw.deltanet, "tree")
        wrap(fw.tree_attention, "attention")
        for name in ("_mm", "_mm_many", "_row_mm"):
            wrap(fw, name)
        runs = []
        for wd in (1, int(os.environ["BISECT"])):
            log.clear()
            tree_forward(eng.w, _tokens(seq[:wd], eng.w.norm.device), list(range(-1, wd - 1)), clone_state(base))
            runs.append(list(log))
        for i, ((n1, a1), (n2, a2)) in enumerate(zip(*runs)):
            if a1.shape != a2.shape or not torch.equal(a1, a2):
                d = (a1 - a2).abs().max().item() if a1.shape == a2.shape else float("nan")
                print(f"first difference: op #{i} {n1} / {n2}, max |d| {d:.4g} (of {len(runs[0])} ops)", flush=True)
                for j in range(max(0, i - 3), min(i + 3, len(runs[0]))):
                    print("  ", j, runs[0][j][0], tuple(runs[0][j][1].shape))
                break
        else:
            print("no difference in row 0", flush=True)
        return

    for wd in widths:
        st = clone_state(base)
        logits, _ = tree_forward(eng.w, _tokens(seq[:wd], eng.w.norm.device), list(range(-1, wd - 1)), st)
        diffs = [(logits[r].float() - serial[r]).abs().max().item() for r in range(wd)]
        bad = [r for r, d in enumerate(diffs) if d != 0]
        print(f"width {wd:2d}: {'exact' if not bad else f'rows differ {bad[:8]}, max |dlogit| {max(diffs):.4g}'}",
              flush=True)


if __name__ == "__main__":
    main()
