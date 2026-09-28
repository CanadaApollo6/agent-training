"""Part 3: where does each full-attention layer look from the last token?

Prints, for each of the 6 full-attention layers, how the last prompt token splits its attention over the prompt
(averaged over the 8 heads), then the per-head split for the final layer.

    uv run 01-inference/token-walkthrough/attention.py
"""
import sys

sys.modules["fla"] = None

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

from skeleton import args  # noqa: E402


def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    # Eager attention materializes the attention weights; the fused kernels never write them out.
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    names = [tok.decode(i) for i in ids[0]]
    with torch.inference_mode():
        weights = model(ids, output_attentions=True).attentions  # one [1, heads, T, T] per full-attention layer
    full = [i + 1 for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
    print(f"attention from {names[-1]!r}, averaged over 8 heads")
    print(f"{'layer':<7}" + "".join(f"{n!r:>14}" for n in names))
    for layer, w in zip(full, weights):
        print(f"{layer:<7}" + "".join(f"{v:>14.2f}" for v in w[0, :, -1].float().mean(0).tolist()))
    print(f"\nlayer {full[-1]}, per head")
    for h, row in enumerate(weights[-1][0, :, -1].float().tolist()):
        print(f"head {h:<2}" + "".join(f"{v:>14.2f}" for v in row))


if __name__ == "__main__":
    main()
