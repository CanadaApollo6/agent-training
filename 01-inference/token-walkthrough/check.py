"""Part 2 checker: runs mine.py against the model's own RMSNorm and MLP on real inputs.

    uv run 01-inference/token-walkthrough/check.py

"exact" means bit-for-bit. "rounding" means a different but valid order of bf16 operations: correct math.
"WRONG" means the math differs.
"""
import sys

import torch

import mine
from skeleton import args, forward, load


def compare(name, fn, ref_fn, x):
    try:
        with torch.inference_mode():
            out, ref = fn(x), ref_fn(x)
    except NotImplementedError:
        print(f"{name:<9} not written yet")
        return False
    if out.shape != ref.shape:
        print(f"{name:<9} WRONG shape: {tuple(out.shape)}, expected {tuple(ref.shape)}")
        return False
    if out.dtype != ref.dtype:
        print(f"{name:<9} WRONG dtype: {out.dtype}, expected {ref.dtype}")
        return False
    diff = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    if torch.equal(out, ref):
        verdict = "exact"
    elif diff <= 2e-2 * scale:  # a few bf16 roundings (bf16 keeps ~3 significant digits)
        verdict = "rounding"
    else:
        verdict = "WRONG"
    print(f"{name:<9} {verdict:<9} max difference {diff:.3g} (largest value {scale:.3g})")
    return verdict != "WRONG"


def main():
    tok, model = load()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    stream, _ = forward(model, ids)
    layer = model.model.layers[0]
    x = stream[5]  # real residual-stream vectors: all prompt tokens after layer 5, [1, T, 1024]
    ok = compare("rmsnorm", lambda v: mine.rmsnorm(v, layer.input_layernorm.weight), layer.input_layernorm, x)
    xn = layer.post_attention_layernorm(x)
    m = layer.mlp
    ok &= compare("mlp", lambda v: mine.mlp(v, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight), m, xn)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
