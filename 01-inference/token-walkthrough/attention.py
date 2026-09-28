"""Part 3: what the 6 full-attention layers do for the last token, and what the KV cache costs.

1. How the last prompt token splits its attention over the prompt, per layer (averaged over the 8 heads).
2. The same split per head, for layers 4, 16 and 24.
3. The output gate: how far each head's result is let through (sigmoid(gate), 0 = closed, 1 = open).
4. What each position's value vector carries (its size), in layer 16.
5. Knock out one attention layer at a time and watch P(' Mah').
6. KV cache arithmetic, and decode with vs without the cache.

    uv run 01-inference/token-walkthrough/attention.py
"""
import sys
import time

sys.modules["fla"] = None

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

from skeleton import args  # noqa: E402


def row(label, values, fmt="{:>14.2f}"):
    return f"{label:<9}" + "".join(fmt.format(v) for v in values)


@torch.inference_mode()
def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    # Eager attention materializes the attention weights; the fused kernels never write them out.
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    names = [tok.decode(i) for i in ids[0]]
    header = f"{'':<9}" + "".join(f"{n!r:>14}" for n in names)
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    layers = model.model.layers

    # Capture each attention layer's input (to recompute its gate and values) alongside the weights.
    inputs = {}
    hooks = [layers[i].self_attn.register_forward_hook(
        lambda m, a, kw, o, i=i: inputs.__setitem__(i, kw["hidden_states"]), with_kwargs=True) for i in full]
    out = model(ids, output_attentions=True)
    for h in hooks:
        h.remove()
    weights = dict(zip(full, out.attentions))  # one [1, heads, T, T] per full-attention layer
    mah = tok(" Mah").input_ids[0]
    base = out.logits[0, -1].float().softmax(-1)[mah].item()

    print(f"1. Attention from {names[-1]!r}, averaged over {cfg.num_attention_heads} heads (each row sums to 1)")
    print(header)
    for i in full:
        print(row(f"layer {i + 1}", weights[i][0, :, -1].float().mean(0).tolist()))
    later = sum(weights[i][0, :, :, :].float().triu(1).sum().item() for i in full)
    print(f"   weight on later tokens, all layers and heads: {later}")

    for i in [3, 15, 23]:
        print(f"\n2. Layer {i + 1}, per head")
        print(header)
        for h, r in enumerate(weights[i][0, :, -1].float().tolist()):
            print(row(f"head {h}", r))

    print("\n3. Output gate at the last token: average of sigmoid(gate) per head (0 = closed, 1 = open)")
    print(f"{'':<9}" + "".join(f"{'head ' + str(h):>9}" for h in range(cfg.num_attention_heads)))
    hd = cfg.head_dim
    for i in full:
        attn = layers[i].self_attn
        q = attn.q_proj(inputs[i][0, -1]).view(-1, 2 * hd)
        gate = torch.sigmoid(q[:, hd:].float()).mean(-1)
        print(row(f"layer {i + 1}", gate.tolist(), "{:>9.2f}"))

    print("\n4. Layer 16: size (RMS) of each position's value vector, averaged over the 2 KV heads")
    attn = layers[15].self_attn
    v = attn.v_proj(inputs[15][0]).float().view(len(names), -1, hd)
    print(header)
    print(row("value", v.pow(2).mean(-1).sqrt().mean(-1).tolist()))

    print(f"\n5. Knock out one attention layer. Intact: P(' Mah') = {base:.1%}")
    for i in full:
        hook = layers[i].self_attn.register_forward_hook(lambda m, a, o: (torch.zeros_like(o[0]),) + tuple(o[1:]))
        lg = model(ids).logits[0, -1].float()
        hook.remove()
        rank = (lg > lg[mah]).sum().item() + 1
        print(f"   layer {i + 1:2d} removed: P(' Mah') {lg.softmax(-1)[mah].item():6.1%}  rank {rank:3d}  top {tok.decode(lg.argmax().item())!r}")

    per_token = 2 * cfg.num_key_value_heads * hd * 2 * len(full)  # K and V, bf16, 6 layers
    weight_bytes = 1504.788e6
    print(f"\n6. KV cache: 2 (K,V) x {cfg.num_key_value_heads} KV heads x {hd} x 2 bytes x {len(full)} layers = {per_token:,} bytes per token")
    print(f"   without KV-head sharing ({cfg.num_attention_heads} KV heads): {per_token * cfg.num_attention_heads // cfg.num_key_value_heads:,} bytes per token")
    for n in [1_000, 8_192, 32_768, 131_072, 262_144]:
        print(f"   {n:>7,} tokens: {n * per_token / 1e9:6.2f} GB")
    print(f"   cache read per decode step equals the 1.5 GB of weights at {weight_bytes / per_token:,.0f} tokens of context")
    if all(t != "full_attention" for t in cfg.layer_types[:1]):
        print(f"   if all 24 layers were full attention: {per_token * 4:,} bytes per token")

    for n_new, use_cache in [(64, True), (64, False), (512, True), (512, False)]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        gen = model.generate(ids, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, use_cache=use_cache)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"   generate {n_new} tokens, cache {'on ' if use_cache else 'off'}: {dt:6.2f} s  ({n_new / dt:5.1f} tok/s)  "
              f"text starts {tok.decode(gen[0, ids.shape[1]:ids.shape[1] + 8])!r}")


if __name__ == "__main__":
    main()
