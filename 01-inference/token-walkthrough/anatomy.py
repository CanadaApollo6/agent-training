"""Part 2 measurements behind CONCEPTS.md: what RMSNorm and the MLP do inside the real model.

    uv run 01-inference/token-walkthrough/anatomy.py

1. How big the residual stream gets before each layer, and what the norm turns it into.
2. The learned per-channel gains (1 + weight) of the norms.
3. How much each layer's mixer and MLP write into the stream at the last token.
4. Knock out one MLP (or one mixer) at a time and watch P(' Mah').
5. Inside the most important MLP: how many of the 3584 neurons fire, and which tokens the top ones push toward.
"""
import sys

import torch

from skeleton import args, forward, load

sys.argv = sys.argv[:1]


def silu(z):
    return z * torch.sigmoid(z)


@torch.inference_mode()
def main():
    tok, model = load()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    layers = model.model.layers
    kinds = model.config.layer_types
    E = model.model.embed_tokens.weight
    mah = tok(" Mah").input_ids[0]
    stream, logits = forward(model, ids)

    print("0. SiLU, the MLP's switch: silu(z) = z * sigmoid(z)")
    for z in [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]:
        print(f"   z = {z:5.1f}  ->  {silu(torch.tensor(z)).item():7.3f}")

    print("\n1. Size (RMS) of the last token's vector entering each layer, before and after input_layernorm")
    for i, layer in enumerate(layers):
        x = stream[i][0, -1].float()
        normed = layer.input_layernorm(stream[i])[0, -1].float()
        print(f"   layer {i + 1:2d}  in {x.pow(2).mean().sqrt():6.2f}   after norm {normed.pow(2).mean().sqrt():5.2f}")
    fin = model.model.norm(stream[-1])[0, -1].float()
    print(f"   final norm: in {stream[-1][0, -1].float().pow(2).mean().sqrt():.2f}, out {fin.pow(2).mean().sqrt():.2f}")

    print("\n2. Learned gains (1 + weight) per channel")
    for name, w in [("layer 1 input", layers[0].input_layernorm.weight),
                    ("layer 12 input", layers[11].input_layernorm.weight),
                    ("layer 24 post-mixer", layers[23].post_attention_layernorm.weight),
                    ("final norm", model.model.norm.weight)]:
        g = 1 + w.float()
        q = g.quantile(torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=g.device)).tolist()
        print(f"   {name:<20} min {q[0]:6.2f}  5% {q[1]:6.2f}  median {q[2]:6.2f}  95% {q[3]:6.2f}  max {q[4]:6.2f}")

    print("\n3. How much each part writes into the stream at the last token (RMS of what it adds)")
    captured = {}
    hooks = []
    for i, layer in enumerate(layers):
        mixer = layer.linear_attn if kinds[i] == "linear_attention" else layer.self_attn
        hooks.append(mixer.register_forward_hook(lambda m, a, o, i=i: captured.__setitem__(("mix", i), o[0] if isinstance(o, tuple) else o)))
        hooks.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: captured.__setitem__(("mlp", i), o)))
    model(ids)
    for h in hooks:
        h.remove()
    for i in range(len(layers)):
        mix = captured[("mix", i)][0, -1].float().pow(2).mean().sqrt()
        mlp = captured[("mlp", i)][0, -1].float().pow(2).mean().sqrt()
        kind = "DeltaNet " if kinds[i] == "linear_attention" else "attention"
        print(f"   layer {i + 1:2d} {kind}  mixer {mix:5.2f}   MLP {mlp:5.2f}")

    def p_mah(knock=None):
        hook = None
        if knock:
            part, i = knock
            mod = layers[i].mlp if part == "mlp" else (layers[i].linear_attn if kinds[i] == "linear_attention" else layers[i].self_attn)
            zero = (lambda m, a, o: torch.zeros_like(o)) if part == "mlp" else \
                   (lambda m, a, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else torch.zeros_like(o))
            hook = mod.register_forward_hook(zero)
        lg = model(ids).logits[0, -1].float()
        if hook:
            hook.remove()
        p = lg.softmax(-1)
        rank = (lg > lg[mah]).sum().item() + 1
        return p[mah].item(), rank, tok.decode(lg.argmax().item())

    base = p_mah()
    print(f"\n4. Knockouts. Intact model: P(' Mah') = {base[0]:.1%}, rank {base[1]}")
    worst = None
    for i in range(len(layers)):
        pm, rm, tm = p_mah(("mlp", i))
        px, rx, tx = p_mah(("mix", i))
        print(f"   layer {i + 1:2d}  no MLP: {pm:6.1%} rank {rm:5d} top {tm!r:<12}  no mixer: {px:6.1%} rank {rx:5d} top {tx!r}")
        if worst is None or pm < worst[1]:
            worst = (i, pm)

    i = worst[0]
    layer = layers[i]
    # Recompute the MLP's input at the last token: stream before layer i, plus the mixer's contribution, normed.
    mid = stream[i] + captured[("mix", i)]
    xn = layer.post_attention_layernorm(mid)[0, -1]
    m = layer.mlp
    gate = xn @ m.gate_proj.weight.T
    up = xn @ m.up_proj.weight.T
    h = (silu(gate.float()) * up.float())
    a = h.abs()
    print(f"\n5. Inside layer {i + 1}'s MLP at the last token (its knockout hurt most)")
    print(f"   gate values: {(gate > 0).float().mean():.0%} positive; "
          f"silu(gate) > 1 for {(silu(gate.float()) > 1).float().mean():.1%} of the 3584")
    s = a.sort(descending=True).values
    for frac in [0.5, 0.9]:
        k = int((s.cumsum(0) < frac * s.sum()).sum().item()) + 1
        print(f"   {k} neurons carry {frac:.0%} of the total activation")
    print(f"   neurons above 10% of the strongest: {(a > 0.1 * a.max()).sum().item()} of 3584")
    Wd = m.down_proj.weight.float()           # [1024, 3584]: column j is what neuron j writes
    gain = 1 + model.model.norm.weight.float()
    for j in a.topk(6).indices.tolist():
        write = h[j] * Wd[:, j]
        scores = (write * gain) @ E.float().T
        top = [tok.decode(t) for t in scores.topk(6).indices.tolist()]
        print(f"   neuron {j:4d}: activation {h[j].item():+7.2f}  pushes toward {top}  (' Mah' score {scores[mah].item():+.2f})")

    # Direct effect: the final norm divides the whole stream by one number, so the ' Mah' minus ' E' logit gap
    # splits exactly into one share per part that wrote into the stream.
    e_tok = tok(" E").input_ids[0]
    rms = stream[-1][0, -1].float().pow(2).mean().add(1e-6).sqrt()
    direction = gain * (E[mah].float() - E[e_tok].float()) / rms
    parts = [("embedding", stream[0][0, -1].float())]
    for k in range(len(layers)):
        parts += [(f"layer {k + 1} mixer", captured[("mix", k)][0, -1].float()),
                  (f"layer {k + 1} MLP", captured[("mlp", k)][0, -1].float())]
    shares = [(n, (v @ direction).item()) for n, v in parts]
    total = sum(v for _, v in shares)
    print(f"\n5b. Who writes the ' Mah' vs ' E' gap directly (total {total:+.2f} logits; model says {(logits[0, mah] - logits[0, e_tok]).item():+.2f})")
    for n, v in sorted(shares, key=lambda t: -abs(t[1]))[:10]:
        print(f"   {n:<18} {v:+6.2f}")
    mlp_sum = sum(v for n, v in shares if "MLP" in n)
    mix_sum = sum(v for n, v in shares if "mixer" in n)
    print(f"   all MLPs {mlp_sum:+.2f}, all mixers {mix_sum:+.2f}, embedding {shares[0][1]:+.2f}")

    mlp_bytes = sum(p.numel() * p.element_size() for p in layers[0].mlp.parameters())
    print(f"\n6. One MLP = 3 matrices = {mlp_bytes / 1e6:.1f} MB; 24 of them = {24 * mlp_bytes / 1e6:.1f} MB "
          f"of the 1,504.8 MB read per token ({24 * mlp_bytes / 1504.788e6:.1%})")


if __name__ == "__main__":
    main()
