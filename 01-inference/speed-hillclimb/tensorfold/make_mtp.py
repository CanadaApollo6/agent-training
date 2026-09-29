"""Build TensorFold's mtp-4bit.safetensors for Ornith 1.5 35B-A3B from the bf16 checkpoint.

The MLX 4-bit conversion drops the MTP layer, and TensorFold drafts with it. This quantizes the bf16 MTP layer the
way the MLX conversion quantized the rest:
- linears at 4-bit, groups of 64; routers at 8-bit
- experts stacked as switch_mlp
- RMSNorm weights plus 1 (MLX stores 1 + w; Qwen3.5 checkpoints store w)

Usage: python make_mtp.py <bf16 dir with the shard holding mtp.*> <MLX model dir>
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GS = 64


def quantize(w: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    """MLX affine quantization of (..., k): uint32 words packed low nibble first, bf16 scales and biases per group."""

    bins = (1 << bits) - 1
    g = w.float().reshape(*w.shape[:-1], -1, GS)
    hi, lo = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    # like mlx: pick the larger-magnitude edge and make it land exactly on a level
    below = lo.abs() > hi.abs()
    scales = ((hi - lo) / bins).clamp_min(1e-7)
    scales = torch.where(below, scales, -scales)
    edge = torch.where(below, lo, hi)
    q0 = torch.round(edge / scales)
    scales = torch.where(q0 != 0, edge / q0, scales)
    biases = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = torch.clamp(torch.round((g - biases) / scales), 0, bins).to(torch.int64)
    per = 32 // bits
    q = q.reshape(*w.shape[:-1], -1, per)
    shifts = torch.arange(per, device=w.device, dtype=torch.int64) * bits
    words = (q << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).view(torch.uint32)
    return {"weight": words.contiguous(), "scales": scales.squeeze(-1).to(torch.bfloat16).contiguous(),
            "biases": biases.squeeze(-1).to(torch.bfloat16).contiguous()}


def main(bf16_dir: str, mlx_dir: str) -> None:
    index = json.loads((Path(bf16_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith("mtp.")})
    raw = {}
    for shard in shards:
        with safe_open(str(Path(bf16_dir) / shard), "pt", device="cuda") as f:
            raw.update({k: f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")})
    config = json.loads((Path(mlx_dir) / "config.json").read_text())
    experts = config.get("text_config", config)["num_experts"]

    out: dict[str, torch.Tensor] = {}

    def put(name: str, bits: int) -> None:
        for part, t in quantize(raw.pop(name + ".weight"), bits).items():
            out[f"{name}.{part}"] = t

    p = "mtp.layers.0."
    for name in ("mtp.fc", *(p + f"self_attn.{x}_proj" for x in "qkvo"),
                 *(p + f"mlp.shared_expert.{x}_proj" for x in ("gate", "up", "down"))):
        put(name, 4)
    for name in (p + "mlp.gate", p + "mlp.shared_expert_gate"):
        put(name, 8)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = torch.stack([raw.pop(f"{p}mlp.experts.{e}.{proj}.weight") for e in range(experts)])
        for part, t in quantize(stacked, 4).items():
            out[f"{p}mlp.switch_mlp.{proj}.{part}"] = t
    for name in [k for k in raw if "norm" in k]:
        out[name] = (raw.pop(name).float() + 1.0).to(torch.bfloat16)
    if raw:
        raise ValueError(f"unconverted MTP tensors: {sorted(raw)[:5]}")

    path = Path(mlx_dir) / "mtp-4bit.safetensors"
    save_file({k: v.cpu() for k, v in out.items()}, str(path), metadata={"format": "mlx"})
    print(f"{path}: {len(out)} tensors, {sum(v.numel() * v.element_size() for v in out.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:])
