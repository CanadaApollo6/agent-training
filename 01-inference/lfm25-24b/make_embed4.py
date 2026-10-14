"""The tied embedding (and head) at 4 bits, group 64, from the bf16 checkpoint, for the Zig engine's affine-4 kernels.

The MLX build stores model.embed_tokens at 6 bits; the engine's embed-row and dense kernels read 4-bit tiles only.
Writes ~/models/lfm25-24b/mlx4/embed4.safetensors (outside the index: the loader and ref.py --embed4 pick it up).
MLX's affine rule: the group's larger-magnitude edge lands exactly on a code; scales and biases rounded to bf16 first.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path.home() / "models/lfm25-24b"


def quantize4(w, group=64):
    rows, cols = w.shape
    g = w.float().reshape(rows, cols // group, group)
    hi, lo = g.amax(-1), g.amin(-1)
    n = 15.0
    side = lo.abs() > hi.abs()
    scale = ((hi - lo) / n).clamp_min(1e-7)
    scale = torch.where(side, scale, -scale)
    edge = torch.where(side, lo, hi)
    q0 = (edge / scale).round()
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    scale, bias = scale.bfloat16(), bias.bfloat16()
    codes = ((g - bias.float()[..., None]) / scale.float()[..., None]).round().clamp(0, 15).to(torch.int64)
    codes = codes.reshape(rows, cols // 8, 8)
    words = (codes << (4 * torch.arange(8, device=w.device))).sum(-1).to(torch.int64)
    err = (codes.reshape(rows, cols // group, group).float() * scale.float()[..., None] + bias.float()[..., None] - g)
    return words.to(torch.uint32 if hasattr(torch, "uint32") else torch.int64), scale, bias, err


def main():
    idx = json.loads((ROOT / "bf16/model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(str(ROOT / "bf16" / idx[key]), "pt", device="cuda") as f:
        w = f.get_tensor(key)
    out = {}
    parts = [quantize4(c) for c in w.split(16000)]
    words = torch.cat([p[0].to(torch.int64) for p in parts]).cpu().numpy().astype(np.uint32)
    scales = torch.cat([p[1] for p in parts]).view(torch.int16).cpu().numpy().view(np.uint16)
    biases = torch.cat([p[2] for p in parts]).view(torch.int16).cpu().numpy().view(np.uint16)
    rel = sum(float(p[3].pow(2).sum()) for p in parts) / float(w.float().pow(2).sum())
    print(f"embed {tuple(w.shape)}: relative squared error {rel:.3e}")
    # safetensors.numpy has no bf16: write the bytes as uint16 and patch the header's dtype after
    out["model.embed_tokens.weight"] = words
    out["model.embed_tokens.scales"] = scales
    out["model.embed_tokens.biases"] = biases
    path = ROOT / "mlx4/embed4.safetensors"
    save_file(out, str(path))
    raw = bytearray(path.read_bytes())
    n = int.from_bytes(raw[:8], "little")
    hdr = json.loads(raw[8 : 8 + n])
    for k in ("model.embed_tokens.scales", "model.embed_tokens.biases"):
        hdr[k]["dtype"] = "BF16"
    text = json.dumps(hdr, separators=(",", ":")).encode()
    text += b" " * (n - len(text))
    assert len(text) == n
    raw[8 : 8 + n] = text
    path.write_bytes(bytes(raw))
    print(f"wrote {path} ({path.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
