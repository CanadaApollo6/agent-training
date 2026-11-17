"""Build the pi adapter's 4-bit model in M3TiS #1's weight format: Vontra's MLX 4-bit (affine, group 64) checkpoint
with only the LoRA-touched tensors replaced.

The harness-hub LoRA touches 256 linear weights: gate/up/down in all 64 layers and q/k/v/o in the 16 full-attention
layers. For each, the merged weight is the base BF16 weight plus (alpha/r)·B·A, computed in fp32 and rounded to BF16,
the same as PEFT's merge. It is quantized with MLX's own mx.quantize (CPU) and replaces that weight's
weight/scales/biases. Every other tensor, and the shard layout and metadata, stay byte-identical to Vontra's.

Recipe check first: mx.quantize on the *base* weight must reproduce Vontra's tensors for the first --check weights:
at least 99% of groups with the same scale and bias, and the same error within 1%. If it doesn't, the script stops, since the new tensors wouldn't be quantized the way
the rest were.

    python build_mlx4.py --base ~/qwen27 --lora ~/pi-lora --vontra ~/vontra --out ~/q27pi-mlx4
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP, BITS = 64, 4


def to_mx(t):  # torch bf16 -> mx bf16 (exact through fp32)
    return mx.array(t.float().numpy()).astype(mx.bfloat16)


def from_mx(a, dtype):
    if dtype == torch.uint32:
        return torch.from_numpy(np.array(a, copy=True).view(np.int32)).view(torch.uint32)
    return torch.from_numpy(np.array(a.astype(mx.float32), copy=True)).to(dtype)


def quantize(w):
    wq, s, b = mx.quantize(to_mx(w), group_size=GROUP, bits=BITS)
    return from_mx(wq, torch.uint32), from_mx(s, torch.bfloat16), from_mx(b, torch.bfloat16)


def dequant(wq, s, b):
    """MLX affine 4-bit: 8 values per uint32, first value in the low bits."""
    q = wq.view(torch.int32)
    nib = torch.stack([(q >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(q.shape[0], -1).float()
    return (nib.view(q.shape[0], -1, GROUP) * s.float()[..., None] + b.float()[..., None]).reshape(q.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--vontra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=int, default=12, help="weights whose base quantization must match Vontra's")
    args = ap.parse_args()

    cfg = json.loads((args.lora / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = {}
    with safe_open(args.lora / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            mod, ab = k.removeprefix("base_model.model.model.language_model.").rsplit(".lora_", 1)
            lora.setdefault(mod, {})[ab[0]] = f.get_tensor(k).float()
    print(f"LoRA: {len(lora)} weights, scale {scale}")

    base_map = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    base_files = {}

    def base_weight(mod):
        name = f"model.language_model.{mod}.weight"
        fn = base_map[name]
        if fn not in base_files:
            base_files[fn] = safe_open(args.base / fn, "pt")
        return base_files[fn].get_tensor(name)

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.vontra.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p, args.out / p.name)

    checked, replaced, stats, recipe = 0, 0, [], []
    t0 = time.time()
    for shard in sorted(args.vontra.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for k in [k for k in tensors if k.endswith(".scales")]:
            mod = k.removeprefix("language_model.model.").removesuffix(".scales")
            if mod not in lora:
                continue
            w = base_weight(mod)
            pre = k.removesuffix(".scales")
            if checked < args.check:
                # CPU and Metal kernels may round a rare group differently, so allow a few in a thousand packed words
                wq, s, b = quantize(w)
                vq, vs, vb = tensors[pre + ".weight"], tensors[pre + ".scales"], tensors[pre + ".biases"]
                words = (wq == vq).float().mean().item()
                groups = ((s == vs) & (b == vb)).float().mean().item()
                ours = (dequant(wq, s, b) - w.float()).norm() / w.float().norm()
                theirs = (dequant(vq, vs, vb) - w.float()).norm() / w.float().norm()
                print(f"check {mod}: packed words equal {words:.6f}, groups equal {groups:.6f}, "
                      f"error ours {ours:.4e} vs Vontra {theirs:.4e}", flush=True)
                recipe.append({"weight": mod, "words_equal": words, "groups_equal": groups,
                               "err_ours": ours.item(), "err_vontra": theirs.item()})
                # Measured on the pod: 98.8% of words and 99.93% of groups equal, same error to 4 digits. The words that
                # differ hold one value on a rounding boundary (CPU vs Metal arithmetic), so the gate is on groups
                # (scale and bias) and error; a calibrated quant such as DWQ would show a lower Vontra error.
                if groups < 0.99 or abs(ours - theirs) > 0.01 * theirs:
                    raise SystemExit(f"recipe check failed on {mod}: Vontra's tensors were not made by plain "
                                     "mx.quantize")
                checked += 1
            merged = (w.float() + scale * (lora[mod]["B"] @ lora[mod]["A"])).to(torch.bfloat16)
            wq, s, b = quantize(merged)
            delta = (merged.float() - w.float()).norm() / w.float().norm()
            err = (dequant(wq, s, b) - merged.float()).norm() / merged.float().norm()
            stats.append({"weight": mod, "delta_rel": delta.item(), "quant_err_rel": err.item()})
            tensors[pre + ".weight"], tensors[pre + ".scales"], tensors[pre + ".biases"] = wq, s, b
            replaced += 1
        save_file(tensors, args.out / shard.name, metadata=meta)
        print(f"{shard.name}: done, {replaced} replaced so far, {time.time() - t0:.0f} s", flush=True)
    if replaced != len(lora):
        raise SystemExit(f"replaced {replaced} weights, LoRA has {len(lora)}")
    (args.out / "M3TIS_BUILD.json").write_text(json.dumps({
        "base": "Qwen/Qwen3.8-27B", "lora": "harness-hub/Qwen3.8-27B-pi-adapted", "quant_source": "Vontra/Qwen3.8-27B-MLX-4bit",
        "lora_scale": scale, "replaced": replaced, "recipe_check": recipe, "weights": stats}, indent=1))
    d = sorted(x["delta_rel"] for x in stats)
    e = sorted(x["quant_err_rel"] for x in stats)
    print(f"recipe check: {checked} weights, worst group match {min(r['groups_equal'] for r in recipe):.6f}")
    print(f"LoRA change, relative norm: median {d[len(d) // 2]:.2e}, max {d[-1]:.2e}")
    print(f"4-bit error, relative norm: median {e[len(e) // 2]:.2e}, max {e[-1]:.2e}")
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
