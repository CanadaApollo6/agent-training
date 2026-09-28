"""A first RL loop: GRPO on 3-digit addition with Qwen3.5-0.8B, measured before and during training.

Each step: 32 random problems "abc + def =", 8 sampled answers each (T=1). Reward 1 if the first number in the
answer is the sum, else 0. Advantage = reward minus the group's mean reward, so an answer is pushed up only for
beating its siblings. One on-policy gradient step, no KL penalty, full fine-tune with fp32 weights and bf16 compute.

Evaluation on held-out problems (never trained on): greedy accuracy, pass@k for k = 1..256 from 256 samples at
T=0.8, majority of 16, distinct answers among 32 samples, and P(first token is ' ?'). Tasks: 3-digit addition (the
training task), 4-digit addition, 2-digit multiplication, and a word problem.

    uv run 04-post-training/first-rl-loop/grpo.py
"""
import argparse
import collections
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--problems", type=int, default=32)
parser.add_argument("--group", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-6)
parser.add_argument("--beta", type=float, default=0.0, help="KL penalty toward the starting model (0 = none)")
parser.add_argument("--eval-at", default="0,5,20,100")
parser.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
args = parser.parse_args()

MAX_NEW = 8
BF16 = dict(device_type="cuda", dtype=torch.bfloat16)


def first_int(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def add3(rng):
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return f"{a} + {b} =", a + b


def add4(rng):
    a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
    return f"{a} + {b} =", a + b


def mul2(rng):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    return f"{a} × {b} =", a * b


def word(rng):
    a, b, c = rng.randint(3, 40), rng.randint(2, 9), rng.randint(3, 12)
    return (f"Question: Sam has {a} apples. He buys {b} bags with {c} apples in each bag. "
            f"How many apples does Sam have now?\nAnswer: The answer is", a + b * c)


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


KS = [1, 4, 16, 64, 256]


@torch.no_grad()
def evaluate(model, tok, sets, n=256):
    """Greedy, majority of 16, and pass@k for k up to 256, from n samples at T=0.8 per problem."""
    model.eval()
    report = {}
    q_id = tok(" ?").input_ids[0]
    for name, problems in sets.items():
        acc = collections.Counter()
        for q, ans in problems:
            ids = tok(q, return_tensors="pt").input_ids.cuda()
            with torch.autocast(**BF16):
                acc["p_question"] += model(ids).logits[0, -1].float().softmax(-1)[q_id].item()
                g = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW)
                s = model.generate(ids.repeat(n, 1), do_sample=True, temperature=0.8, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW)
            acc["greedy"] += first_int(tok.decode(g[0, ids.shape[1]:])) == ans
            answers = [first_int(t) for t in tok.batch_decode(s[:, ids.shape[1]:])]
            c = sum(a == ans for a in answers)
            for k in KS:
                acc[f"pass{k}"] += pass_at(n, c, k)
            vote = collections.Counter(a for a in answers[:16] if a is not None).most_common(1)
            acc["maj16"] += bool(vote) and vote[0][0] == ans
            acc["distinct32"] += len(set(answers[:32]))
        report[name] = {k: round(v / len(problems), 4) for k, v in acc.items()}
    model.train()
    return report


def show(step, report):
    print(f"\n   step {step}")
    for name, r in report.items():
        ks = "  ".join(f"@{k} {r[f'pass{k}']:5.1%}" for k in KS)
        print(f"   {name:<13} greedy {r['greedy']:5.1%}  maj@16 {r['maj16']:5.1%}  pass{ks}  "
              f"distinct {r['distinct32']:4.1f}/32  P(' ?') {r['p_question']:5.1%}")


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda()
    model.generation_config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    ref = None
    if args.beta > 0:
        ref = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    eos = {tok.eos_token_id, model.generation_config.pad_token_id} | set(
        model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])

    sets = {}
    for name, make, n, seed in [("add 3-digit", add3, 100, 101), ("add 4-digit", add4, 50, 102),
                                ("multiply", mul2, 50, 103), ("word problem", word, 50, 104)]:
        r = random.Random(seed)
        sets[name] = [make(r) for _ in range(n)]
    eval_at = {int(s) for s in args.eval_at.split(",")}
    log = {"args": vars(args), "evals": {}, "reward": [], "kl": []}
    train_rng = random.Random(0)

    for step in range(args.steps + 1):
        if step in eval_at:
            t = time.time()
            log["evals"][step] = evaluate(model, tok, sets)
            show(step, log["evals"][step])
            print(f"   (eval {time.time() - t:.0f}s)")
            Path(args.out).write_text(json.dumps(log, indent=1))
        if step == args.steps:
            break

        t = time.time()
        problems = [add3(train_rng) for _ in range(args.problems)]
        ids = tok([q for q, _ in problems], return_tensors="pt").input_ids.cuda()  # every "abc + def =" has the same length
        prompt = ids.repeat_interleave(args.group, 0)
        model.eval()
        with torch.no_grad(), torch.autocast(**BF16):
            seqs = model.generate(prompt, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  max_new_tokens=MAX_NEW, min_new_tokens=1)
        model.train()
        comp = seqs[:, ids.shape[1]:]
        rewards = torch.tensor([float(first_int(tok.decode(c)) == problems[i // args.group][1]) for i, c in enumerate(comp)],
                               device="cuda")
        grouped = rewards.view(args.problems, args.group)
        adv = (grouped - grouped.mean(1, keepdim=True)).view(-1)

        # Mask everything after the first end-of-sequence token (it is kept: stopping is a choice too).
        is_eos = torch.zeros_like(comp, dtype=torch.bool)
        for e in eos:
            is_eos |= comp == e
        after = (is_eos.cumsum(1) - is_eos.long()) > 0
        mask = (~after).float()

        if adv.abs().sum() > 0:
            # Score only the completion positions, 64 sequences at a time: the 248K-wide logits are the memory hog.
            opt.zero_grad(set_to_none=True)
            total = mask.sum()
            kl_sum = 0.0
            for j in range(0, seqs.shape[0], 64):
                with torch.autocast(**BF16):
                    logits = model(seqs[j:j + 64], logits_to_keep=MAX_NEW + 1).logits
                logits = logits[:, -comp.shape[1] - 1:-1].float()
                logp = -torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), comp[j:j + 64].reshape(-1),
                                                          reduction="none").view(logits.shape[:2])
                loss = -(adv[j:j + 64].unsqueeze(1) * logp * mask[j:j + 64]).sum() / total
                if ref is not None:
                    with torch.no_grad(), torch.autocast(**BF16):
                        rl = ref(seqs[j:j + 64], logits_to_keep=MAX_NEW + 1).logits[:, -comp.shape[1] - 1:-1].float()
                    ref_logp = -torch.nn.functional.cross_entropy(rl.reshape(-1, rl.shape[-1]), comp[j:j + 64].reshape(-1),
                                                                  reduction="none").view(rl.shape[:2])
                    d = ref_logp - logp
                    kl = (d.exp() - d - 1) * mask[j:j + 64]   # k3: >= 0, zero when policy and reference agree
                    loss = loss + args.beta * kl.sum() / total
                    kl_sum += kl.sum().item()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)  # free the 3.2 GB of gradients before the next rollout
        solved_any = (grouped.max(1).values > 0).float().mean().item()
        log["reward"].append(round(rewards.mean().item(), 4))
        log["kl"].append(round(kl_sum / mask.sum().item(), 5) if ref is not None and adv.abs().sum() > 0 else 0.0)
        if step % 5 == 0:
            print(f"   step {step:3d}  mean reward {rewards.mean():5.1%}  groups with any right {solved_any:5.1%}  "
                  f"KL {log['kl'][-1]:.4f}  ({time.time() - t:.1f}s)")

    Path(args.out).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
