"""LFM2.5-24B-A2B teacher-forced reference, one layer on the GPU at a time.

Weights come from the bf16 checkpoint (the true model) or from an MLX affine build dequantized to bf16 (the same
numbers the Zig engine reads). Follows transformers' modeling_lfm2_moe.py: RMSNorm in fp32, sigmoid router whose
expert bias only picks experts, normalized top-k weights, short conv y = C * conv3(B * x).

  python ref.py --weights mlx4 --ids ids.json --out logits.npy        # fp16 logits [T, vocab]
  python ref.py --weights bf16 --text "Hello" --chat --greedy 32       # greedy continuation (no cache, slow)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path.home() / "models/lfm25-24b"
DEV = "cuda"


class Weights:
    def __init__(self, name):
        self.root = ROOT / name
        self.cfg = json.loads((self.root / "config.json").read_text())
        self.map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.mlx = "quantization" in self.cfg
        self.files = {}

    def raw(self, key):
        fn = self.map[key]
        if fn not in self.files:
            self.files[fn] = safe_open(str(self.root / fn), "pt", device=DEV)
        return self.files[fn].get_tensor(key)

    def has(self, key):
        return key in self.map

    def get(self, key):
        """bf16 (or fp32 for router/bias) weight for `key` (without the .weight suffix for linears)."""
        if self.has(key + ".scales"):
            return dequant(self.raw(key + ".weight"), self.raw(key + ".scales"), self.raw(key + ".biases"))
        return self.raw(key + ".weight") if self.has(key + ".weight") else self.raw(key)


def dequant(q, scales, biases, group=64):
    """MLX affine: values are a little-endian bitstream of `bits`-bit codes; w = scale * code + bias per group."""
    cols = scales.shape[-1] * group
    bits = q.shape[-1] * 32 // cols
    lead = q.shape[:-1]
    b = q.contiguous().view(torch.uint8).reshape(-1, q.shape[-1] * 4)
    if bits in (2, 4, 8):
        per = 8 // bits
        shifts = torch.arange(0, 8, bits, device=q.device, dtype=torch.uint8)
        codes = (b.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
        codes = codes.reshape(b.shape[0], -1)
    else:
        parts = []
        for c in b.split(4096):  # the bit-by-bit unpack is 32x the packed size, so go in row chunks
            bitsarr = ((c.unsqueeze(-1) >> torch.arange(8, device=q.device, dtype=torch.uint8)) & 1).reshape(c.shape[0], -1)
            bitsarr = bitsarr.reshape(c.shape[0], cols, bits).to(torch.int32)
            parts.append((bitsarr << torch.arange(bits, device=q.device, dtype=torch.int32)).sum(-1).to(torch.uint8))
        codes = torch.cat(parts)
    codes = codes.reshape(*lead, cols // group, group).float()
    w = codes * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return w.reshape(*lead, cols).to(torch.bfloat16)


def rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x, pos, theta):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]


def conv_layer(W, p, h):
    T = h.shape[0]
    bcx = h @ W.get(p + "conv.in_proj").T  # [T, 3H]
    B, C, x = bcx.chunk(3, -1)
    bx = (B * x).T.unsqueeze(0)  # [1, H, T]
    w = W.get(p + "conv.conv.weight")
    w = w.reshape(w.shape[0], -1)  # [H, 3] (bf16 is [H,1,3]; MLX is [H,3,1])
    y = F.conv1d(bx, w.unsqueeze(1), padding=w.shape[1] - 1, groups=w.shape[0])[:, :, :T]
    y = C * y[0].T
    return y @ W.get(p + "conv.out_proj").T


def attn_layer(W, p, h, cfg):
    T = h.shape[0]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh
    eps = cfg["norm_eps"]
    q = (h @ W.get(p + "self_attn.q_proj").T).view(T, nh, hd)
    k = (h @ W.get(p + "self_attn.k_proj").T).view(T, nkv, hd)
    v = (h @ W.get(p + "self_attn.v_proj").T).view(T, nkv, hd)
    q = rms(q, W.get(p + "self_attn.q_layernorm.weight"), eps)
    k = rms(k, W.get(p + "self_attn.k_layernorm.weight"), eps)
    pos = torch.arange(T, device=h.device)
    theta = cfg.get("rope_parameters", {}).get("rope_theta") or cfg.get("rope_theta")
    q, k = rope(q, pos, theta), rope(k, pos, theta)
    rep = nh // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
    o = o.transpose(0, 1).reshape(T, nh * hd)
    return o @ W.get(p + "self_attn.out_proj").T


def dense_mlp(W, p, h):
    names = ("w1", "w3", "w2") if W.has(p + "feed_forward.w1.weight") else ("gate_proj", "up_proj", "down_proj")
    g, u, d = (W.get(p + f"feed_forward.{n}") for n in names)
    return (F.silu(h @ g.T) * (h @ u.T)) @ d.T


def moe(W, p, h, cfg, route_log=None):
    k = cfg["num_experts_per_tok"]
    logits = h.float() @ W.get(p + "feed_forward.gate.weight").float().T  # router in fp32
    probs = logits.sigmoid()
    _, sel = torch.topk(probs + W.get(p + "feed_forward.expert_bias").float(), k, -1)
    wts = torch.gather(probs, 1, sel)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-6)
    wts = wts.to(h.dtype)
    if route_log is not None:
        route_log.append(sel.cpu())
    out = torch.zeros_like(h)
    for e in sel.unique().tolist():
        rows, slot = torch.where(sel == e)
        g, u, d = expert(W, p, e)
        xe = h[rows]
        ye = (F.silu(xe @ g.T) * (xe @ u.T)) @ d.T
        out.index_add_(0, rows, ye * wts[rows, slot, None])
    return out


_expert_cache = {}


def expert(W, p, e):
    if W.mlx:
        key = p
        if key not in _expert_cache:
            _expert_cache.clear()
            _expert_cache[key] = {
                n: (W.raw(p + f"feed_forward.switch_mlp.{n}.weight"), W.raw(p + f"feed_forward.switch_mlp.{n}.scales"),
                    W.raw(p + f"feed_forward.switch_mlp.{n}.biases"))
                for n in ("gate_proj", "up_proj", "down_proj")
            }
        c = _expert_cache[key]
        return tuple(dequant(c[n][0][e], c[n][1][e], c[n][2][e]) for n in ("gate_proj", "up_proj", "down_proj"))
    return tuple(W.get(p + f"feed_forward.experts.{e}.{n}") for n in ("w1", "w3", "w2"))


DUMP = []  # debugging: each norm's output row 0 (operator_norm, ffn_norm by layer, then the final norm)


@torch.no_grad()
def forward(W, ids, route_log=None):
    cfg = W.cfg
    eps = cfg["norm_eps"]
    if "emb" not in W.__dict__:
        W.emb = W.get("model.embed_tokens")
    emb = W.emb
    h = emb[torch.tensor(ids, device=DEV)]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        x = rms(h, W.get(p + "operator_norm.weight"), eps)
        DUMP.append(x[0].float().cpu())
        h = h + (attn_layer(W, p, x, cfg) if kind == "full_attention" else conv_layer(W, p, x))
        x = rms(h, W.get(p + "ffn_norm.weight"), eps)
        DUMP.append(x[0].float().cpu())
        h = h + (dense_mlp(W, p, x) if i < cfg["num_dense_layers"] else moe(W, p, x, cfg, route_log))
    h = rms(h, W.get("model.embedding_norm.weight"), eps)
    DUMP.append(h[0].float().cpu())
    return (h.float() @ emb.float().T)  # tied head, fp32 logits


def chat_ids(text):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "bf16"))
    return tok, tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="mlx4")
    ap.add_argument("--ids")
    ap.add_argument("--text")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--routes")
    ap.add_argument("--dump", help="each norm's row 0 (fp32 [norms, hidden]) to this .npy")
    ap.add_argument("--embed4", action="store_true", help="the tied table from mlx4/embed4.safetensors (the Zig engine's)")
    a = ap.parse_args()
    W = Weights(a.weights)
    if a.embed4:
        with safe_open(str(ROOT / "mlx4/embed4.safetensors"), "pt", device=DEV) as f:
            W.emb = dequant(*(f.get_tensor(f"model.embed_tokens.{n}") for n in ("weight", "scales", "biases")))
    if a.ids:
        ids = json.loads(Path(a.ids).read_text())
        tok = None
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(ROOT / "bf16"))
        ids = chat_ids(a.text)[1] if a.chat else tok(a.text)["input_ids"]
        ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    for _ in range(a.greedy):
        nxt = int(forward(W, ids)[-1].argmax())
        ids.append(nxt)
        if tok:
            print(tok.decode([nxt]), end="", flush=True)
    if a.greedy:
        print()
    if a.out or a.routes:
        log = [] if a.routes else None
        logits = forward(W, ids, log)
        if a.out:
            np.save(a.out, logits.half().cpu().numpy())
        if a.routes:
            torch.save(torch.stack(log), a.routes)  # [38 MoE layers, T, 4]
        print(f"saved {len(ids)} positions")
    if a.dump:
        DUMP.clear()
        forward(W, ids[:1])
        np.save(a.dump, torch.stack(DUMP).numpy())


if __name__ == "__main__":
    main()
