"""Part 1 of the token walkthrough: the whole model as a five-line loop.

Text becomes token ids, each id picks one row of the embedding table, 24 layers each add something to that
vector, and the final vector is scored against every row of the same table to pick the next token. The layers
are black boxes here; later parts open them. The hand-written loop must match the model's own logits exactly,
which proves the skeleton is the whole story.

    uv run 01-inference/token-walkthrough/skeleton.py
"""
import argparse
import sys

sys.modules["fla"] = None  # pure-PyTorch DeltaNet: slower, but every step is readable Python

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--prompt", default="The Kansas City Chiefs quarterback is Patrick")
parser.add_argument("--new-tokens", type=int, default=6)
args = parser.parse_args()


def load():
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    return tok, model


def layer_inputs(model, h):
    """Rotary position info for the full-attention layers (Part 3 explains it). Text uses 3 identical rows."""
    positions = torch.arange(h.shape[1], device=h.device).view(1, 1, -1).expand(3, 1, -1)
    return model.model.rotary_emb(h, positions)


@torch.inference_mode()
def forward(model, ids):
    """The entire model. Returns the residual stream after every layer and the last position's logits.

    Scoring all T positions (not just the last) matches the model's own forward bit for bit. In bf16, a
    different matmul shape sums in a different order and the logits differ by up to ~0.06.
    """
    E = model.model.embed_tokens.weight          # [248320, 1024]: one row per token in the vocabulary
    h = E[ids]                                   # look up one row per token: [1, T, 1024]
    rope = layer_inputs(model, h)
    stream = [h]
    for layer in model.model.layers:             # 24 layers, each reads h and adds its result back into it
        h = layer(h, position_embeddings=rope)
        stream.append(h)
    h = model.model.norm(h)                      # rescale the final vector (RMSNorm, Part 2)
    logits = h @ E.T                             # score every position against every row of the same table
    return stream, logits[:, -1]                 # [1, 248320]: only the last position predicts the next token


def mb(n):
    return f"{n / 1e6:8.3f} MB" if n >= 1e5 else f"{n / 1e3:8.1f} KB"


def main():
    tok, model = load()
    E = model.model.embed_tokens.weight
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

    print("1. Text -> token ids (the tokenizer is a fixed lookup, not part of the network)")
    for i in ids[0].tolist():
        print(f"   {i:>7}  {tok.decode(i)!r}")
    print(f"   vocabulary: {E.shape[0]:,} tokens\n")

    last = ids[0, -1].item()
    row = E[last]
    print(f"2. Token id -> vector: row {last} of the embedding table")
    print(f"   table {tuple(E.shape)} in bf16 = {mb(E.numel() * 2)}; one row = {row.numel()} numbers = {mb(row.numel() * 2)}")
    print(f"   first 8 numbers of {tok.decode(last)!r}: {[round(v, 4) for v in row[:8].float().tolist()]}\n")

    stream, logits = forward(model, ids)
    with torch.inference_mode():
        ref = model(ids).logits[:, -1]
    print("3. The hand-written loop vs the model's own forward")
    print(f"   logits identical: {torch.equal(logits, ref)}  (shape {tuple(logits.shape)}: one score per vocabulary token)\n")

    probs = logits[0].float().softmax(-1)
    top = probs.topk(8)
    print("4. Scores -> probabilities (softmax) -> next token")
    for p, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   {p:6.1%}  {tok.decode(t)!r}")

    # Decode: append the winner and run the whole loop again. No cache yet, so every step recomputes the prompt.
    out = ids
    for _ in range(args.new_tokens):
        _, lg = forward(model, out)
        out = torch.cat([out, lg.argmax(-1, keepdim=True)], dim=1)
    print(f"\n   greedy continuation: {args.prompt!r} + {tok.decode(out[0, ids.shape[1]:])!r}\n")

    # What one decode step reads from memory, stage by stage (the 1,505 MB from decode-gap, taken apart).
    layers = model.model.layers
    kinds = model.config.layer_types
    size = lambda mod: sum(p.numel() * p.element_size() for p in mod.parameters())  # noqa: E731
    lin = [size(l) for l, k in zip(layers, kinds) if k == "linear_attention"]
    full = [size(l) for l, k in zip(layers, kinds) if k == "full_attention"]
    stages = [
        ("embedding lookup (1 row)", row.numel() * 2),
        (f"{len(lin)} DeltaNet layers x {mb(lin[0]).strip()}", sum(lin)),
        (f"{len(full)} full-attention layers x {mb(full[0]).strip()}", sum(full)),
        ("final norm", size(model.model.norm)),
        ("LM head (every row of the table)", E.numel() * 2),
    ]
    total = sum(b for _, b in stages)
    print("5. Weight bytes read per decoded token")
    for name, b in stages:
        print(f"   {name:<42}{mb(b)}  {b / total:6.1%}")
    print(f"   {'total':<42}{mb(total)}")


if __name__ == "__main__":
    main()
