"""Part 1 follow-up: read the model's mind after every layer (the "logit lens").

skeleton.py applies the final norm and LM head once, after layer 24. Nothing stops us applying them after any
layer: that shows which token the vector in the residual stream would pick if the model stopped there. Row 0
is "no layers at all": embedding lookup, then straight to the LM head.

    uv run 01-inference/token-walkthrough/lens.py
"""
import torch

from skeleton import args, forward, load


def main():
    tok, model = load()
    E = model.model.embed_tokens.weight
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    stream, logits = forward(model, ids)
    answer = logits.argmax().item()

    print(f"prompt {args.prompt!r}; the full model's answer is {tok.decode(answer)!r}")
    print(f"{'after layer':<12}{'type':<9}{'top 3 guesses':<52}{f'P({tok.decode(answer)!r})':>12}")
    with torch.inference_mode():
        for i, h in enumerate(stream):
            probs = (model.model.norm(h[:, -1]) @ E.T).float().softmax(-1)[0]
            top = probs.topk(3)
            guesses = "  ".join(f"{tok.decode(t)!r} {p:.0%}" for p, t in zip(top.values.tolist(), top.indices.tolist()))
            kind = "-" if i == 0 else {"linear_attention": "DeltaNet", "full_attention": "full"}[model.config.layer_types[i - 1]]
            print(f"{i:<12}{kind:<9}{guesses:<52}{probs[answer].item():>12.2%}")


if __name__ == "__main__":
    main()
