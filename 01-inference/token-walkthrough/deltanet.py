"""Part 4: Gated DeltaNet, the fixed-size memory in 18 of the 24 layers.

1. The delta rule on a 2-number toy memory: store, re-store, overwrite, and collide.
2. What the cache actually holds after a short and a long prompt: DeltaNet state vs attention KV.
3. The learned gates on the Patrick prompt: how much each head keeps per token (as a half-life) and how hard it writes.
4. Pass-key recall: a 4-digit code hidden at the start of filler text. Full model vs attention limited to the last
   64 tokens, where only DeltaNet (plus the 4-token convolution) can carry the code across the gap.

    uv run 01-inference/token-walkthrough/deltanet.py
"""
import math
import random
import sys

sys.modules["fla"] = None

import torch  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

from skeleton import args  # noqa: E402

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def toy():
    print("1. The delta rule on a toy memory: keys have 2 numbers, values have 1")

    def show(label, S):
        print(f"   {label:<52} read Chiefs {S @ torch.tensor([1.0, 0.0]):5.2f}   read quarterback {S @ torch.tensor([0.0, 1.0]):5.2f}")

    chiefs, qb, mix = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]) / math.sqrt(2)

    def delta(S, k, v, beta=1.0, keep=1.0):
        S = keep * S
        return S + beta * (v - S @ k) * k      # write only the error: what should be there minus what is

    def plain(S, k, v):
        return S + v * k                        # linear attention without the delta rule: just pile it on

    S = torch.zeros(2)
    S = delta(S, chiefs, 5.0)
    S = delta(S, qb, 3.0)
    show("store Chiefs=5, quarterback=3", S)
    P = plain(plain(torch.zeros(2), chiefs, 5.0), qb, 3.0)
    show("store Chiefs=5 again, plain sum", plain(P, chiefs, 5.0))
    show("store Chiefs=5 again, delta rule", delta(S, chiefs, 5.0))
    show("update Chiefs=7, delta rule", delta(S, chiefs, 7.0))
    show("update Chiefs=7, delta rule, write strength 0.5", delta(S, chiefs, 7.0, beta=0.5))
    show("one step with keep 0.9 and nothing new written", 0.9 * S)
    S3 = delta(S, mix, 4.0)
    show("store a third key (half Chiefs, half qb)=4", S3)
    print(f"   {'':<52} read the third key {S3 @ mix:5.2f}")


def cache_bytes(cache):
    """Bytes held by the DeltaNet layers (memory matrix + convolution window) and by the attention layers (K and V)."""
    lin = full = 0
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            for t in [*layer.recurrent_states.values(), *layer.conv_states.values()]:
                lin += t.numel() * t.element_size()
        else:
            full += layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
    return lin, full


@torch.inference_mode()
def main():
    toy()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    layers = model.model.layers

    print("\n2. What the cache holds")
    d0 = layers[0].linear_attn
    print(f"   one DeltaNet layer: {cfg.linear_num_value_heads} heads x a {cfg.linear_key_head_dim} x {cfg.linear_value_head_dim} memory matrix,"
          f" plus the last {cfg.linear_conv_kernel_dim} tokens' inputs for the short convolution")
    for text in [args.prompt, FILLER * 80]:
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        out = model(ids, use_cache=True)
        lin, full = cache_bytes(out.past_key_values)
        print(f"   {ids.shape[1]:>5} tokens: DeltaNet state {lin / 1e6:6.2f} MB (18 layers), attention KV {full / 1e6:6.2f} MB (6 layers)")
    print(f"   the fixed DeltaNet state is as big as the attention KV cache of {lin / (full / ids.shape[1]):,.0f} tokens")
    del d0

    print("\n3. Gates on the Patrick prompt, last token. keep = fraction of memory kept per token; half-life in tokens")
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    inputs = {}
    hooks = [l.linear_attn.register_forward_hook(lambda m, a, kw, o, i=i: inputs.__setitem__(i, kw["hidden_states"]), with_kwargs=True)
             for i, l in enumerate(layers) if cfg.layer_types[i] == "linear_attention"]
    model(ids)
    for h in hooks:
        h.remove()
    all_half = []
    for i, x in sorted(inputs.items()):
        m = layers[i].linear_attn
        x = x[0, -1]
        g = -m.A_log.float().exp() * torch.nn.functional.softplus(m.in_proj_a(x).float() + m.dt_bias.float())
        keep = g.exp()
        half = (math.log(0.5) / g).clamp(max=1e7)
        beta = m.in_proj_b(x).float().sigmoid()
        all_half.append(half)
        hs = half.sort().values.tolist()
        print(f"   layer {i + 1:2d}  keep min {keep.min():.3f} max {keep.max():.4f}   half-life shortest {hs[0]:7.1f}  median {hs[len(hs) // 2]:8.1f}"
              f"  longest {hs[-1]:10.0f}   write strength median {beta.median():.2f} (min {beta.min():.2f}, max {beta.max():.2f})")
    h = torch.cat(all_half)
    bins = [(0, 2), (2, 10), (10, 100), (100, 1000), (1000, 1e4), (1e4, 1e9)]
    print("   all 288 heads by half-life: " + ", ".join(f"{lo:g}-{hi:g}: {((h >= lo) & (h < hi)).sum().item()}" for lo, hi in bins))

    print("\n4. Pass-key recall. 'Window' = attention sees only the last 64 tokens plus the first 4 (the sink).")
    print("   Beyond 64 tokens, only the DeltaNet layers (and the 4-token convolution) can carry the code forward.")
    window = {"W": None}
    orig = qwen.eager_attention_forward

    def windowed(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        if window["W"] is not None:
            T = query.shape[2]
            pos = torch.arange(T, device=query.device)
            gap = pos[:, None] - pos[None, :]
            hidden = ((gap < 0) | (gap >= window["W"])) & ~((pos[None, :] < 4) & (gap >= 0))
            attention_mask = torch.zeros(T, T, device=query.device, dtype=query.dtype)
            attention_mask = attention_mask.masked_fill(hidden, torch.finfo(query.dtype).min)[None, None]
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)

    qwen.eager_attention_forward = windowed
    rng = random.Random(0)
    codes = [str(rng.randint(1000, 9999)) for _ in range(10)]
    fill_tokens = len(tok(FILLER).input_ids)
    print(f"   {'filler after the code':<24}" + "".join(f"{w:>34}" for w in ["full attention", "window (DeltaNet carries it)"]))
    for reps in [0, 1, 2, 4, 16, 64, 256]:
        cells, sample = [], ""
        for W in [None, 64]:
            window["W"] = W
            ok, digits, probs = 0, 0, []
            for code in codes:
                prompt = ("There is an important pass key hidden inside a lot of irrelevant text. Find it and memorize it. "
                          f"The pass key is {code}. Remember it. {code} is the pass key. " + FILLER * reps
                          + "What is the pass key? The pass key is")
                p_ids = tok(prompt).input_ids
                a_ids = tok(prompt + " " + code).input_ids[len(p_ids):]
                lg = model(torch.tensor([p_ids + a_ids], device="cuda")).logits[0, len(p_ids) - 1:-1].float()
                pred = lg.argmax(-1).tolist()
                right = sum(pred[j] == a_ids[j] for j in range(1, len(a_ids)))  # a_ids[0] is the space
                digits += right
                ok += int(right == len(a_ids) - 1 and pred[0] == a_ids[0])
                probs.append(lg.softmax(-1)[torch.arange(len(a_ids)), torch.tensor(a_ids)].prod().item())
                if W and code == codes[0]:
                    sample = f"{code} -> {tok.decode(pred).strip()!r}"
            cells.append(f"{ok:2d}/10 codes, {digits:2d}/40 digits")
        print(f"   {reps * fill_tokens:>6,} tokens{'':<11}" + "".join(f"{c:>34}" for c in cells) + f"   e.g. {sample}")
    qwen.eager_attention_forward = orig


if __name__ == "__main__":
    main()
