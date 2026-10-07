"""Next-token predictions of Qwen3.8-27B (TensorFold, MLX 4-bit) deep into a real document, for comparing KV caches.

usage: TENSORFOLD_KV_BITS=8|16 python kv_kl.py OUT.pt      -> log-probs for SCORE tokens at each depth
       python kv_kl.py --compare A.pt B.pt                  -> KL(A || B), top-1 agreement, perplexity, per depth
The document: this repo's READMEs, then TensorFold's Python source (code and prose), same tokens every run.
Scoring runs the decode path (tree_forward + commit, 8 tokens a step), so it reads the cache the way drafting does.
"""
import os
import sys
from pathlib import Path

import torch

DEPTHS = (1024, 16384, 49152)
SCORE = 256
STEP = 8


def document() -> str:
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for p in root.rglob("README.md") if "envs" not in p.parts and "tools" not in p.parts)
    files += sorted((root / "01-inference/tools/TensorFold/src/tensorfold").rglob("*.py"))
    return "\n\n".join(f"# file: {p.relative_to(root)}\n{p.read_text(errors='ignore')}" for p in files)


def run(out: str) -> None:
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    from tensorfold.cuda.kernels import kvq
    from tensorfold.families.qwen3_5.cuda.decode import _tokens, prefill
    from tensorfold.families.qwen3_5.cuda.forward import commit, tree_forward
    from tensorfold.families.qwen3_5.cuda.weights import load

    d = Path(snapshot_download("Vontra/Qwen3.8-27B-MLX-4bit", local_files_only=True))
    need = DEPTHS[-1] + SCORE + 1
    frozen = Path(__file__).resolve().parent / "results/kv_kl_tokens.pt"
    if not frozen.exists():
        ids = Tokenizer.from_file(str(d / "tokenizer.json")).encode(document()).ids[:need]
        torch.save(torch.tensor(ids, dtype=torch.int32), frozen)
    ids = torch.load(frozen).tolist()
    assert len(ids) >= need, f"document has {len(ids)} tokens, need {need}"
    w = load(d, tiled=True)
    rows, st = {}, None
    with torch.no_grad():
        for depth in DEPTHS:
            st, _ = prefill(w, ids[:depth], None, state=st, limit=need)
            got = []
            for p in range(depth, depth + SCORE, STEP):
                logits, record = tree_forward(w, _tokens(ids[p:p + STEP], w.norm.device), list(range(-1, STEP - 1)), st)
                commit(st, record, list(range(STEP)))
                got.append(logits.float().log_softmax(-1).half().cpu())
            rows[depth] = torch.cat(got)
            print(f"depth {depth}: scored {SCORE} tokens, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
    torch.save({"bits": kvq.BITS, "rows": rows, "targets": {k: torch.tensor(ids[k + 1:k + SCORE + 1]) for k in DEPTHS}}, out)


def compare(a: str, b: str) -> None:
    A, B = torch.load(a), torch.load(b)
    if any(not torch.equal(A["targets"][k], B["targets"][k]) for k in A["targets"]):
        raise SystemExit("the two runs read different documents")
    print(f"KL({A['bits']}-bit cache || {B['bits']}-bit cache), log-probs over the full vocabulary")
    print("| Depth | KL mean | KL max | Top-1 agree | Perplexity A | Perplexity B |")
    print("|---|---|---|---|---|---|")
    for k in A["rows"]:
        la, lb, t = A["rows"][k].float(), B["rows"][k].float(), A["targets"][k]
        kl = (la.exp() * (la - lb)).sum(-1)
        agree = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
        ppa = la.gather(1, t[:, None]).mean().neg().exp().item()
        ppb = lb.gather(1, t[:, None]).mean().neg().exp().item()
        print(f"| {k:,} | {kl.mean().item():.4f} | {kl.max().item():.3f} | {agree:.1%} | {ppa:.3f} | {ppb:.3f} |")


if __name__ == "__main__":
    compare(sys.argv[2], sys.argv[3]) if sys.argv[1] == "--compare" else run(sys.argv[1])
