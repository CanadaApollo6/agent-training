"""Re-quantize parts of the always-on path from bf16 with the imatrix-weighted search, in MLX format.

Same code as the KL harness's 'q<bits>' source, so the tensors written here are the ones it scored. The output holds
MLX names (language_model.model.layers.N....{weight,scales,biases}); make_build.py swaps them into a checkpoint.

    python export_always_on.py --bf16 ~/models/bf16 --imatrix imatrix.gguf --parts attention linear_attention \
        shared_expert --bits 4 --out always-on-q4.safetensors
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kl_harness import PARTS, dense_importance, hf_name, layer_of, part, qe

NAMES = {"attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
         "linear_attention": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"),
         "shared_expert": ("gate_proj", "up_proj", "down_proj")}
MODULE = {"attention": "self_attn", "linear_attention": "linear_attn", "shared_expert": "mlp.shared_expert"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--parts", nargs="+", choices=list(NAMES), required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert set(args.parts) <= set(PARTS)

    bf16 = Path(args.bf16)
    where = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"]
    imp = dense_importance(args.imatrix)
    out = {}
    for layer in range(qe.LAYERS):
        for p in args.parts:
            for proj in NAMES[p]:
                mlx = f"language_model.model.layers.{layer}.{MODULE[p]}.{proj}"
                name = hf_name(mlx)
                if name not in where:                   # full attention on every 4th layer, linear on the rest
                    continue
                assert part(mlx) == p and layer_of(mlx) == layer
                with safe_open(bf16 / where[name], "pt", device="cuda") as f:
                    w = f.get_tensor(name)
                (qw, s, b), _ = qe.quantize(w[None].cpu(), imp[(layer, w.shape[1])][None], args.bits)
                out |= {mlx + ".weight": qw[0], mlx + ".scales": s[0], mlx + ".biases": b[0]}
        print(f"layer {layer}: {len(out) // 3} matrices", flush=True)
    save_file({k: v.contiguous() for k, v in out.items()}, args.out,
              metadata={"parts": ",".join(args.parts), "bits": str(args.bits), "source": "imatrix search, kl_harness"})
    print(f"wrote {len(out) // 3} matrices to {args.out}")


if __name__ == "__main__":
    main()
