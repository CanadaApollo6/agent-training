"""The rest of R1s from a fine-tuned bf16: every tensor that R1s takes unchanged from the MLX 4-bit base.

R1s = the MLX 4-bit checkpoint, with routed experts re-quantized to 3 bits (quantize_experts.py) and attention, linear
attention and the shared expert re-quantized to 4 bits by the imatrix search (export_always_on.py). The remaining
tensors (embeddings, LM head, routers, shared-expert gates, norms, DeltaNet's conv, A_log and dt_bias) still come from
the base checkpoint, which was made from the original weights. After a full fine-tune they must come from the new
weights, the way the MLX conversion made them: MLX's round-to-nearest at the base's per-module width, or bf16 with
mlx_lm's sanitize transforms.

    python requant_rest.py --bf16 FT_DIR --base ~/models/ornith/mlx4 --out rest.safetensors
    python requant_rest.py --bf16 ORIGINAL_BF16 --base ~/models/ornith/mlx4 --verify

--verify derives the tensors from the original bf16 and compares them with the base: every one should match exactly,
which shows the transforms are the ones the base was made with. The output (MLX names) goes to make_build.py --replace
together with the experts and the always-on export.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-inference/speed-hillclimb/tensorfold"))
import quantize_experts as qe  # noqa: E402

GS = 64
# re-made by quantize_experts.py (routed experts) and export_always_on.py (attention, linear attention, shared expert)
ELSEWHERE = (".switch_mlp.", ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.o_proj.",
             ".linear_attn.in_proj_", ".linear_attn.out_proj.", ".mlp.shared_expert.")


def hf_name(mlx: str) -> str:
    if mlx == "language_model.lm_head.weight":
        return "lm_head.weight"
    assert mlx.startswith("language_model.model."), mlx
    return "model.language_model." + mlx[len("language_model.model."):]


def module_quant(config: dict, module: str) -> dict:
    q = config["quantization"]
    return q.get(module, q)


def round_away(x: torch.Tensor) -> torch.Tensor:
    """Metal's round(): halves away from zero (torch.round sends them to even)."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def rtn_mlx(w: torch.Tensor, bits: int, rows: int = 4096):
    """MLX's affine round-to-nearest of a matrix [N, K] -> (words, scales, biases), as its Metal kernel computes them
    (the base was converted on a Mac): ties round away from zero, and levels use the fp32 scale, not the stored bf16."""
    n, k = w.shape
    bins = (1 << bits) - 1
    out_w = torch.empty((n, k * bits // 32), dtype=torch.int32)
    out_s = torch.empty((n, k // GS), dtype=torch.bfloat16)
    out_b = torch.empty_like(out_s)
    for r in range(0, n, rows):
        g = w[r:r + rows].float().reshape(-1, k // GS, GS)
        hi, lo = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
        s = ((hi - lo) / bins).clamp_min(1e-7)
        below = lo.abs() > hi.abs()
        s = torch.where(below, s, -s)
        edge = torch.where(below, lo, hi)
        q0 = round_away(edge / s)
        s = torch.where(q0 == 0, s, edge / q0)
        b = torch.where(q0 == 0, torch.zeros_like(edge), edge)
        q = torch.clamp(round_away((g - b) / s), 0, bins)
        out_w[r:r + rows] = qe.words(q.reshape(-1, k), bits)
        out_s[r:r + rows], out_b[r:r + rows] = s[..., 0].to(torch.bfloat16), b[..., 0].to(torch.bfloat16)
    return out_w, out_s, out_b


# RMSNorms that scale by (1 + w) in HF, which mlx_lm stores as 1 + w; the gated DeltaNet norm scales by w in both
SHIFTED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight",
           "model.norm.weight")


def sanitize(mlx: str, t: torch.Tensor) -> torch.Tensor:
    """mlx_lm's transforms for qwen3_5 unquantized tensors (checked against the base by --verify)."""
    if mlx.endswith("conv1d.weight") and t.shape[-1] != 1:
        return t.moveaxis(2, 1).contiguous()     # [C, 1, k] -> [C, k, 1]
    if mlx.endswith(SHIFTED):
        return (t.float() + 1.0).to(t.dtype)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--base", default=str(Path.home() / "models/ornith/mlx4"))
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    bf16, base = Path(a.bf16).expanduser(), Path(a.base).expanduser()
    config = json.loads((base / "config.json").read_text())
    base_where = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    bf_where = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"]

    def get(where, root, name):
        with safe_open(root / where[name], "pt") as f:
            return f.get_tensor(name)

    rest = sorted(k for k in base_where if not any(s in k for s in ELSEWHERE))
    out, report = {}, collections.Counter()
    for k in rest:
        if k.endswith((".scales", ".biases")):
            continue
        module = k.removesuffix(".weight")
        if f"{module}.scales" in base_where:
            q = module_quant(config, module)
            w, s, b = rtn_mlx(get(bf_where, bf16, hf_name(k)), q["bits"])
            new = {k: w.view(get(base_where, base, k).dtype), f"{module}.scales": s, f"{module}.biases": b}
        else:
            new = {k: sanitize(k, get(bf_where, bf16, hf_name(k))).to(get(base_where, base, k).dtype)}
        if a.verify:
            for name, t in new.items():
                ref = get(base_where, base, name)
                same = t.shape == ref.shape and torch.equal(t, ref)
                kind = name.split(".")[-2] + "." + name.split(".")[-1] if ".layers." in name else name
                report[(kind, same)] += 1
                if not same and report[(kind, False)] <= 2:
                    detail = (f"shapes {tuple(t.shape)} vs {tuple(ref.shape)}" if t.shape != ref.shape else
                              f"{(t != ref).float().mean().item():.4%} elements differ")
                    print(f"MISMATCH {name}: {detail}")
        out |= new
    if a.verify:
        for (kind, same), n in sorted(report.items()):
            print(f"{'ok  ' if same else 'DIFF'} {n:4d} {kind}")
        return
    save_file(out, a.out, metadata={"format": "mlx"})
    print(f"{len(out)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
