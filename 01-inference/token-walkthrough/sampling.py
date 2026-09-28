"""Part 5: choosing the next token. Greedy vs sampling, temperature, truncation, majority vote, and batching.

1. The Patrick distribution under different temperatures.
2. Sample the Patrick prompt 200 times: what names come out?
3. Greedy vs sampled text on an open-ended prompt: repetition.
4. Arithmetic: greedy vs one sample vs majority vote of 16 samples vs "any of 16 right".
5. Decode throughput at batch 1, 16 and 64.

    uv run 01-inference/token-walkthrough/sampling.py
"""
import collections
import random
import re
import sys
import time

sys.modules["fla"] = None

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

from skeleton import args  # noqa: E402


def distinct_4grams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / max(1, len(grams))


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    logits = model(ids).logits[0, -1].float()

    print("1. Temperature: divide the logits by T before softmax")
    for T in [0.3, 0.7, 1.0, 1.5, 3.0]:
        p = (logits / T).softmax(-1)
        s = p.sort(descending=True)
        n90 = int((s.values.cumsum(0) < 0.9).sum().item()) + 1
        top = ", ".join(f"{tok.decode(i)!r} {v:.1%}" for v, i in zip(s.values[:4].tolist(), s.indices[:4].tolist()))
        print(f"   T={T:<4} P(' Mah') {p[tok(' Mah').input_ids[0]]:6.1%}   tokens needed for 90% of the probability: {n90:>6,}   top: {top}")

    print("\n2. Sample the Patrick prompt 200 times (T=1, no truncation), 3 more tokens each")
    names = collections.Counter()
    for _ in range(4):
        out = model.generate(ids.repeat(50, 1), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=3)
        for row in out[:, ids.shape[1]:]:
            names[tok.decode(row).split("\n")[0].strip()] += 1
    for name, n in names.most_common(12):
        print(f"   {n:4d}  Patrick {name}")
    print(f"   distinct continuations: {len(names)}")

    print("\n3. Open-ended prompt, 200 tokens: greedy vs sampled")
    prompt = "My favorite thing about the weekend is"
    pid = tok(prompt, return_tensors="pt").input_ids.cuda()
    for label, kw in [("greedy", dict(do_sample=False)),
                      ("T=0.7, top-p 0.9", dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=0)),
                      ("T=1.0, no truncation", dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)),
                      ("T=1.5, no truncation", dict(do_sample=True, temperature=1.5, top_p=1.0, top_k=0))]:
        torch.manual_seed(1)
        out = model.generate(pid, max_new_tokens=200, min_new_tokens=200, **kw)[0, pid.shape[1]:].tolist()
        text = tok.decode(out).replace("\n", " / ")
        print(f"   {label:<22} distinct 4-grams {distinct_4grams(out):5.0%}   {text[:230]!r} ... {text[-120:]!r}")

    print("\n4. Arithmetic, 30 problems per task: greedy vs one sample vs majority vote of 16 vs any of 16 right")
    rng = random.Random(0)

    def first_int(text):
        m = re.search(r"-?\d+", text)
        return int(m.group()) if m else None

    tasks = {
        "3-digit addition": lambda a=0, b=0: (lambda a, b: (f"{a} + {b} =", a + b))(rng.randint(100, 999), rng.randint(100, 999)),
        "2-digit multiplication": lambda: (lambda a, b: (f"{a} × {b} =", a * b))(rng.randint(12, 99), rng.randint(12, 99)),
        "word problem (a + b x c)": lambda: (lambda a, b, c: (
            f"Question: Sam has {a} apples. He buys {b} bags with {c} apples in each bag. "
            f"How many apples does Sam have now?\nAnswer: The answer is", a + b * c))(rng.randint(3, 40), rng.randint(2, 9), rng.randint(3, 12)),
    }
    k = 16
    for name, make in tasks.items():
        greedy_ok = one_ok = vote_ok = any_ok = 0
        for j in range(30):
            q, ans = make()
            qid = tok(q, return_tensors="pt").input_ids.cuda()
            if j == 0:
                p = model(qid).logits[0, -1].float().softmax(-1).topk(3)
                first = ", ".join(f"{tok.decode(i)!r} {v:.0%}" for v, i in zip(p.values.tolist(), p.indices.tolist()))
                greedy_text = tok.decode(model.generate(qid, do_sample=False, max_new_tokens=10)[0, qid.shape[1]:])
                print(f"   {name}: e.g. {q!r} -> greedy {greedy_text!r}; first-token odds {first}")
            g = first_int(tok.decode(model.generate(qid, do_sample=False, max_new_tokens=8)[0, qid.shape[1]:]))
            outs = model.generate(qid.repeat(k, 1), do_sample=True, temperature=0.8, top_p=1.0, top_k=0, max_new_tokens=8)
            answers = [first_int(tok.decode(o[qid.shape[1]:])) for o in outs]
            vote = collections.Counter(a for a in answers if a is not None).most_common(1)
            greedy_ok += g == ans
            one_ok += answers[0] == ans
            vote_ok += bool(vote) and vote[0][0] == ans
            any_ok += ans in answers
        print(f"   {'':<4}greedy {greedy_ok:2d}/30   one sample {one_ok:2d}/30   majority of {k} {vote_ok:2d}/30   any of {k} right {any_ok:2d}/30")

    print("\n5. Decode throughput: every sequence in a batch shares one read of the weights per step")
    for batch in [1, 16, 64]:
        x = pid.repeat(batch, 1)
        model.generate(x, do_sample=True, max_new_tokens=8, min_new_tokens=8)
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(x, do_sample=True, max_new_tokens=64, min_new_tokens=64)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"   batch {batch:3d}: {64 / dt:6.1f} tok/s per sequence, {batch * 64 / dt:7.1f} tok/s total")


if __name__ == "__main__":
    main()
