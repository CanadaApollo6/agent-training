"""Re-quantize Ornith 1.5 35B-A3B's routed experts to 2 or 3 bits for TensorFold's low-bit expert kernels.

Everything but the routed experts comes from the MLX 4-bit checkpoint unchanged: the always-on path, the routers,
the shared expert (kept at 4 bits), the MTP side file. The routed experts are quantized again from bf16 in the same
MLX affine format (w = s * q + b, bf16 s and b per group of 64), at the recipe's width per projection.

Scales and biases come from a search rather than plain round-to-nearest:
- An error weight per input column comes from an importance matrix, the mean squared activation each expert's
  column saw on calibration text (bartowski's imatrix, the one build A used). Columns that see big inputs matter more.
- Per group, a sweep of range settings. For each, round to the levels, then refit s and b by weighted least squares,
  round them to bf16 and round the levels again. The lowest weighted error wins.
- `--rtn` skips the search: MLX's own min/max rounding, for comparison.

Usage: python quantize_experts.py --bf16 DIR --base MLX4_DIR --imatrix FILE --out DIR --gate-up 2 --down 3 [--rtn]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GS = 64
LAYERS = 40
EXPERTS = 256
BASE_PREFIX = "language_model.model.layers.{}.mlp.switch_mlp.{}"
BF16_PREFIX = "model.language_model.layers.{}.mlp.experts.{}"
IMATRIX = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}


def rtn(g: torch.Tensor, bits: int):
    """MLX's affine rounding of groups g [..., GS]: the larger-magnitude edge lands exactly on a level."""

    bins = (1 << bits) - 1
    hi, lo = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    below = lo.abs() > hi.abs()
    s = ((hi - lo) / bins).clamp_min(1e-7)
    s = torch.where(below, s, -s)
    edge = torch.where(below, lo, hi)
    q0 = torch.round(edge / s)
    s = torch.where(q0 != 0, edge / q0, s)
    b = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    return bf16(s), bf16(b)


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def levels(g: torch.Tensor, s: torch.Tensor, b: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.clamp(torch.round((g - b) / s), 0, (1 << bits) - 1)


def error(g, w, s, b, q):
    return (w * (s * q + b - g) ** 2).sum(-1, keepdim=True)


def search(g: torch.Tensor, w: torch.Tensor, bits: int, steps: int = 24, refits: int = 2):
    """Groups g [..., GS] fp32, error weights w (broadcastable) -> bf16-exact s, b [..., 1] minimizing weighted error."""

    bins = (1 << bits) - 1
    w = w.expand_as(g)
    best_s, best_b = rtn(g, bits)
    best = error(g, w, best_s, best_b, levels(g, best_s, best_b, bits))
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)
    sw = w.sum(-1, keepdim=True)
    sx = (w * g).sum(-1, keepdim=True)
    for i in range(steps + 1):
        # the levels spread over the range a little narrower or wider than min..max (clipping trades for resolution)
        iscale = (bins + (i - steps / 2) * (1.0 / steps) * bins * 0.5) / span
        q = torch.clamp(torch.round(iscale * (g - lo)), 0, bins)
        for _ in range(refits):
            sq, sqq, sqx = (w * q).sum(-1, keepdim=True), (w * q * q).sum(-1, keepdim=True), (w * q * g).sum(-1, keepdim=True)
            det = sw * sqq - sq * sq
            ok = det > 1e-12 * sw * sqq
            s = bf16(torch.where(ok, (sw * sqx - sq * sx) / det.clamp_min(1e-30), best_s))
            b = bf16(torch.where(ok, (sqq * sx - sq * sqx) / det.clamp_min(1e-30), best_b))
            s = torch.where(s.abs() < 1e-8, best_s, s)
            q = levels(g, s, b, bits)
            err = error(g, w, s, b, q)
            better = err < best
            best = torch.where(better, err, best)
            best_s = torch.where(better, s, best_s)
            best_b = torch.where(better, b, best_b)
    return best_s, best_b


def words(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Levels [..., K] -> MLX words [..., K * bits / 32] (int32 bits of uint32): an LSB-first bitstream, bits bytes
    per 8 values."""

    v = (q.to(torch.int64).reshape(*q.shape[:-1], -1, 8) << (bits * torch.arange(8, device=q.device))).sum(-1)
    b = ((v[..., None] >> (8 * torch.arange(bits, device=q.device))) & 0xFF).to(torch.uint8)
    return b.reshape(*q.shape[:-1], -1).contiguous().view(torch.int32)


def quantize(w: torch.Tensor, imp: torch.Tensor | None, bits: int, chunk: int = 16):
    """Experts [E, N, K] bf16, importance [E, K] -> MLX (words, scales, biases), weighted errors (search, plain)."""

    e, n, k = w.shape
    out_w = torch.empty((e, n, k * bits // 32), dtype=torch.int32)
    out_s = torch.empty((e, n, k // GS), dtype=torch.bfloat16)
    out_b = torch.empty_like(out_s)
    errs = torch.zeros(3, dtype=torch.float64)
    for e0 in range(0, e, chunk):
        g = w[e0:e0 + chunk].cuda().float().reshape(-1, n, k // GS, GS)
        wt = (imp[e0:e0 + chunk].cuda().reshape(-1, 1, k // GS, GS) if imp is not None
              else torch.ones_like(g[:, :1]))
        s0, b0 = rtn(g, bits)
        s, b = (s0, b0) if imp is None else search(g, wt, bits)
        q = levels(g, s, b, bits)
        wx = wt.expand_as(g)
        errs[0] += error(g, wx, s, b, q).sum().double().item()
        errs[1] += error(g, wx, s0, b0, levels(g, s0, b0, bits)).sum().double().item()
        errs[2] += (wx * g * g).sum().double().item()
        out_w[e0:e0 + chunk] = words(q.reshape(-1, n, k), bits).cpu()
        out_s[e0:e0 + chunk] = s[..., 0].to(torch.bfloat16).cpu()
        out_b[e0:e0 + chunk] = b[..., 0].to(torch.bfloat16).cpu()
    return (out_w, out_s, out_b), errs


def importance(path: str) -> dict[str, torch.Tensor]:
    """imatrix GGUF -> {tensor name: mean squared input [experts, K]}; experts the calibration never routed to take
    the layer's mean. Normalized to mean 1 with a small floor, so no column is ignored outright."""

    from gguf import GGUFReader

    t = {x.name: torch.from_numpy(x.data.copy()) for x in GGUFReader(path).tensors}
    out = {}
    for name in t:
        if not name.endswith(".in_sum2") or "_exps" not in name:
            continue
        base = name[:-len(".in_sum2")]
        s2, c = t[name].float(), t[base + ".counts"].float().reshape(-1, 1)
        m = s2 / c.clamp_min(1)
        seen = (c > 0).squeeze(-1)
        m[~seen] = m[seen].mean(0)
        m = m / m.mean(-1, keepdim=True).clamp_min(1e-12)
        out[base] = m + 0.05
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--base", required=True, help="the MLX 4-bit checkpoint (with mtp-4bit.safetensors)")
    ap.add_argument("--imatrix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-up", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--down", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--rtn", action="store_true", help="MLX's plain rounding, no importance-weighted search")
    ap.add_argument("--layers", type=int, default=LAYERS)
    a = ap.parse_args()
    if not a.rtn and not a.imatrix:
        ap.error("the search needs --imatrix (or pass --rtn)")
    bf, base, out = Path(a.bf16), Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bits = {"gate_proj": a.gate_up, "up_proj": a.gate_up, "down_proj": a.down}
    imp = None if a.rtn else importance(a.imatrix)

    base_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    bf_map = json.loads((bf / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map, report = {}, {}

    # everything but the routed experts, as the MLX 4-bit checkpoint has it
    keep = {}
    for shard in sorted(set(base_map.values())):
        with safe_open(base / shard, "pt") as f:
            for key in f.keys():
                if ".switch_mlp." not in key:
                    keep[key] = f.get_tensor(key)
    save_file(keep, out / "model-base.safetensors", metadata={"format": "mlx"})
    weight_map.update({key: "model-base.safetensors" for key in keep})
    del keep

    shard, n = {}, 0
    for i in range(a.layers):
        key = BF16_PREFIX.format(i, "gate_up_proj")
        with safe_open(bf / bf_map[key], "pt") as f:
            gate_up = f.get_tensor(key)
        key = BF16_PREFIX.format(i, "down_proj")
        with safe_open(bf / bf_map[key], "pt") as f:
            down = f.get_tensor(key)
        ni = gate_up.shape[1] // 2
        for proj, w in (("gate_proj", gate_up[:, :ni]), ("up_proj", gate_up[:, ni:]), ("down_proj", down)):
            m = None if imp is None else imp[f"blk.{i}.{IMATRIX[proj]}.weight"]
            (qw, s, b), errs = quantize(w, m, bits[proj])
            name = BASE_PREFIX.format(i, proj)
            shard.update({name + ".weight": qw.view(torch.uint32), name + ".scales": s, name + ".biases": b})
            report[name] = {"bits": bits[proj], "error": float(errs[0] / errs[2]), "error_rtn": float(errs[1] / errs[2])}
            print(f"layer {i:2d} {proj:9s} {bits[proj]}-bit  weighted rel. error {errs[0] / errs[2]:.5f}"
                  f" (plain rounding {errs[1] / errs[2]:.5f})", flush=True)
        del gate_up, down
        if (i + 1) % 5 == 0 or i + 1 == a.layers:
            n += 1
            fname = f"model-experts-{n:02d}.safetensors"
            save_file(shard, out / fname, metadata={"format": "mlx"})
            weight_map.update({key: fname for key in shard})
            shard = {}

    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    cfg = json.loads((base / "config.json").read_text())
    for field in ("quantization", "quantization_config"):
        if field in cfg:
            for i in range(a.layers):
                for proj in bits:
                    cfg[field][BASE_PREFIX.format(i, proj)] = {"group_size": GS, "bits": bits[proj]}
    cfg["tensorfold_recipe"] = {"routed_gate_up_bits": a.gate_up, "routed_down_bits": a.down,
                                "method": "mlx-rtn" if a.rtn else "imatrix-weighted affine search",
                                "imatrix": None if a.rtn else Path(a.imatrix).name}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in base.iterdir():
        if f.is_file() and not f.name.startswith("model") and f.name != "config.json":
            shutil.copy2(f, out / f.name)
    (out / "quantize-report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
