"""Does Ornith thinking in soft tokens keep its answers with fewer steps? Step 0 of the inner-link plan, no training.

While the model thinks, each step's input is normally the embedding of one sampled token. Soft thinking feeds the
probability-weighted mix of the top-k tokens' embeddings instead (Zhang et al. 2025, "Soft Thinking"), so a step can
carry more than one token's worth of the distribution. With ``--gumbel``, the mix is taken after Gumbel noise (Wu et
al. 2025 found plain soft thinking collapses toward greedy decoding). The answer after </think> is always sampled as
plain tokens, so only the thinking differs.

Every mode runs through the same batched loop, so ``discrete`` (the baseline) differs from the soft modes only in the
thinking-step inputs. Problems, sampling and grading are the reasoning probe's (20 level-5 MATH-500 problems).

    python soft_think.py --model ornith-ai/Ornith-1.5-9B --modes discrete soft gumbel --samples 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen38-27b"))
from reasoning_probe import SAMPLING, SUFFIX, grade, problems  # noqa: E402

OUT = Path(__file__).parent / "results"


def filtered(logits, temperature, top_k, top_p):
    """The probe's sampling distribution: temperature, then top-k, then top-p; (probs, ids) over the kept tokens."""
    vals, ids = (logits / temperature).topk(top_k, dim=-1)
    probs = vals.softmax(-1)
    keep = probs.cumsum(-1) - probs < top_p                    # the smallest prefix reaching top_p
    probs = probs * keep
    return probs / probs.sum(-1, keepdim=True), ids


@torch.no_grad()
def run(model, tok, rows, mode, args, gen):
    dev = model.device
    emb = model.get_input_embeddings().weight
    end_think = tok.convert_tokens_to_ids("</think>")
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    texts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for r in rows]
    tok.padding_side = "left"
    batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    b = len(rows)
    mask = batch.attention_mask
    out = model(input_ids=batch.input_ids, attention_mask=mask, use_cache=True)
    cache, logits = out.past_key_values, out.logits[:, -1].float()
    thinking = torch.ones(b, dtype=torch.bool, device=dev)
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    steps = torch.zeros(b, dtype=torch.long, device=dev)          # thinking steps
    answer_len = torch.zeros(b, dtype=torch.long, device=dev)
    low = torch.zeros(b, dtype=torch.long, device=dev)            # consecutive near-certain soft steps
    ended = ["" for _ in range(b)]                                # how thinking ended
    shadow = [[] for _ in range(b)]                               # each soft step's top token, to read the path
    answers = [[] for _ in range(b)]
    t0 = time.perf_counter()
    for step in range(args.cap):
        probs, ids = filtered(logits, SAMPLING["temperature"], SAMPLING["top_k"], SAMPLING["top_p"])
        sampled = ids.gather(1, torch.multinomial(probs, 1, generator=gen)).squeeze(1)
        nxt = emb[sampled]                                        # plain tokens: answers, and discrete thinking
        soft = thinking & (mode != "discrete")
        if soft.any():
            sp, sids = filtered(logits, args.soft_temperature, args.soft_k, args.soft_p)
            if mode == "gumbel":
                g = -torch.log(-torch.log(torch.rand(sp.shape, device=dev, generator=gen).clamp_min(1e-20)))
                sp = ((sp.clamp_min(1e-20).log() + g) / args.gumbel_tau).softmax(-1) * (sp > 0)
                sp = sp / sp.sum(-1, keepdim=True)
            mix = torch.einsum("bk,bkd->bd", sp.to(emb.dtype), emb[sids])
            top = sids.gather(1, sp.argmax(-1, keepdim=True)).squeeze(1)
            entropy = -(sp * sp.clamp_min(1e-20).log()).sum(-1)
            low = torch.where(soft & (entropy < args.cold_entropy), low + 1, torch.zeros_like(low))
            # thinking ends when the top token is </think>, or cold stop: the mix has been one token for a while
            stop_soft = soft & ((top == end_think) | (low >= args.cold_patience))
            sampled = torch.where(soft, torch.where(stop_soft, end_think, top), sampled)
            nxt = torch.where((soft & ~stop_soft)[:, None], mix, emb[sampled])
        forced = thinking & (steps >= args.think_cap)             # out of thinking room: close it
        sampled = torch.where(forced, end_think, sampled)
        nxt = torch.where(forced[:, None], emb[end_think][None], nxt)
        s_list, t_list, d_list = sampled.tolist(), thinking.tolist(), done.tolist()
        for i in range(b):
            if d_list[i]:
                continue
            if t_list[i]:
                shadow[i].append(s_list[i])
                if s_list[i] == end_think:
                    ended[i] = "cap" if forced[i] else "cold" if soft[i] and low[i] >= args.cold_patience else "model"
            else:
                answers[i].append(s_list[i])
        new_done = ~thinking & (torch.tensor([s in stops for s in s_list], device=dev) | (answer_len >= args.answer_cap))
        steps += thinking.long()
        answer_len += (~thinking).long()
        thinking = thinking & (sampled != end_think)
        done = done | new_done
        if done.all():
            break
        mask = torch.cat([mask, torch.ones(b, 1, dtype=mask.dtype, device=dev)], 1)
        out = model(inputs_embeds=nxt[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
        if step % 1000 == 0:
            print(f"   {mode}: step {step}, {int(thinking.sum())} thinking, {int(done.sum())} done, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
    results = []
    for i, row in enumerate(rows):
        answer = tok.decode(answers[i], skip_special_tokens=True)
        finish = "stop" if done[i] else "cap"
        g = grade(row, "</think>" + answer if not thinking[i] else "", finish)
        results.append(g | {"mode": mode, "think_steps": int(steps[i]), "answer_tokens": len(answers[i]),
                            "ended": ended[i] or "never", "finish": finish,
                            "path_head": tok.decode(shadow[i][:300]), "path_tail": tok.decode(shadow[i][-200:])})
    return results, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--modes", nargs="+", default=["discrete", "soft", "gumbel"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=16384, help="steps, thinking and answer together (the probe's cap)")
    ap.add_argument("--think-cap", type=int, default=14336)
    ap.add_argument("--answer-cap", type=int, default=2048)
    ap.add_argument("--soft-k", type=int, default=15)
    ap.add_argument("--soft-p", type=float, default=0.95)
    ap.add_argument("--soft-temperature", type=float, default=0.6)
    ap.add_argument("--gumbel-tau", type=float, default=0.5)
    ap.add_argument("--cold-entropy", type=float, default=0.01)
    ap.add_argument("--cold-patience", type=int, default=256)
    ap.add_argument("--label", default="ornith9b")
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = [r for r in problems(args.n, args.seed) for _ in range(args.samples)]
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        gen = torch.Generator(device=model.device).manual_seed(args.seed)
        results, secs = run(model, tok, rows, mode, args, gen)
        correct = sum(r["outcome"] == "correct" for r in results)
        steps = sorted(r["think_steps"] for r in results)
        summary = {"label": args.label, "mode": mode, "n": len(results), "correct": correct,
                   "median_think_steps": steps[len(steps) // 2], "mean_think_steps": sum(steps) / len(steps),
                   "ended": {k: sum(r["ended"] == k for r in results) for k in ("model", "cold", "cap", "never")},
                   "seconds": round(secs), "args": vars(args)}
        print(json.dumps({k: v for k, v in summary.items() if k != "args"}), flush=True)
        path = OUT / f"{args.label}-{mode}.json"
        path.write_text(json.dumps({"summary": summary, "results": results}, indent=1))


if __name__ == "__main__":
    main()
