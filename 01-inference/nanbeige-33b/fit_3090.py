"""Paper estimate: does Nanbeige4.3-33B plus our DFlash2 draft fit on a 24 GB RTX 3090, how much context is left,
and how fast would a TensorFold port decode? No GPU is used. Shapes are from the model's config.json (rev c1093c5).

Speed model: decode reads every weight once per step, so step time = bytes read / usable bandwidth. A drafted round
costs one verify step (a few rows, slightly dearer than one) plus the draft's own pass, and yields `accept` tokens
(our held-out acceptance, sampled at the agent's settings). The constants come from our Qwen3.8-27B and R1s work on
this card; treat the results as +-25%.

    python3 fit_3090.py
"""
GB, GIB = 1e9, 2**30

# Nanbeige4.3-33B (LlamaForCausalLM)
layers, hidden, heads, kv_heads, head_dim, ffn, vocab = 52, 7168, 56, 8, 128, 22016, 166912
attn = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2  # q, o + k, v
body = layers * (attn + 3 * hidden * ffn)  # 30.72B; the embedding table stays in host RAM
head = vocab * hidden  # lm_head, 1.20B
kv_per_token = layers * 2 * kv_heads * head_dim  # values per token (k and v, all 52 layers are full attention)
draft = 1.9e9  # our 5-layer DFlash2 head's own weights (it shares the target's embedding and lm_head)

# 3090 constants (qwen27b-local-ceiling / tensorfold-zig-port notes)
usable = 24 * GIB - 2.0 * GB - 0.5 * GB  # the desktop holds ~1.7-2.0 GB; keep a margin
overhead = 1.5 * GIB  # CUDA context, scratch, the engine's pool reserve
bandwidth = 730 * GB  # ~88% of the measured ~830 GB/s ceiling for big matmuls
verify_cost = 1.15  # a 9-row verify step vs a 1-row step (bandwidth-bound)
draft_ms = 3.0  # Qwen 27B's drafter takes ~2.4 ms; ours is 7168 wide, not 5120, but stored at 4 bits
accept = {"code": 3.30, "chat": 2.71}  # held-out, temperature 1.0 (results/eval-cool-step570)

quants = [  # name, body bits per weight, head bits per weight
    ("MLX 4-bit g64 (stock)", 4.5, 4.5),
    ("EXL3 4.0", 4.0, 6.0),
    ("EXL3 3.5", 3.5, 6.0),
    ("EXL3 3.0", 3.0, 6.0),
]
kv_bytes = {"bf16": 2, "fp8": 1}


def main():
    print(f"body {body / 1e9:.2f}B params, head {head / 1e9:.2f}B, KV {kv_per_token * 2 / 2**10:.0f} KiB/token at bf16")
    print(f"usable VRAM {usable / GIB:.1f} GiB\n")
    print(f"{'quant':24} {'weights':>8} {'KV room':>8} {'ctx bf16':>9} {'ctx fp8':>8}   tok/s plain | code chat @2K | code chat @32K fp8")
    for name, bpw, head_bpw in quants:
        weights = body * bpw / 8 + head * head_bpw / 8
        room = usable - weights - draft * 4.5 / 8 - overhead
        ctx = {k: room / (kv_per_token * b) for k, b in kv_bytes.items()}

        def speeds(context):
            step = (weights + kv_per_token * context) / bandwidth  # fp8 cache
            rnd = step * verify_cost + draft_ms / 1e3
            return 1 / step, accept["code"] / rnd, accept["chat"] / rnd

        p, c2, h2 = speeds(2048)
        long = "   (no room)" if ctx["fp8"] < 32768 else "{:5.0f} {:4.0f}".format(*speeds(32768)[1:])
        print(f"{name:24} {weights / GIB:6.1f}Gi {room / GIB:6.1f}Gi {ctx['bf16'] / 1e3:7.0f}K {ctx['fp8'] / 1e3:7.0f}K"
              f"   {p:5.0f}       | {c2:4.0f} {h2:4.0f}       | {long}")


if __name__ == "__main__":
    main()
