"""Where the all-positions KL of base vs Vontra 4-bit comes from (kl.json: mean 1.46, p99 20 nats, while assistant
positions are 0.03). For a few texts: per-position KL, the share of the total from positions above 5 nats, where
they sit (role span, position in the text), and the tokens at the worst ones.

    python kl_outliers.py --base ~/qwen27 --vontra ~/vontra --texts ~/pod/kl.jsonl --out ~/out/kl-outliers.json
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from kl_q27 import dict_args, load_into, quantized


def main():
    ap = argparse.ArgumentParser()
    for a in ("base", "vontra", "texts", "out"):
        ap.add_argument("--" + a, type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--swap", action="store_true", help="BF16 on GPUs 2-3, 4-bit on 0-1")
    args = ap.parse_args()
    from transformers import Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.base)

    def load(gpus):
        mem = {g: "46GiB" if g in gpus else 0 for g in range(4)}
        return Qwen3_5ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto",
                                                               max_memory=mem).eval()
    r_model, q_model = (load((2, 3)), load((0, 1))) if args.swap else (load((0, 1)), load((2, 3)))
    load_into(q_model, quantized(args.vontra))
    start = tok.convert_tokens_to_ids("<|im_start|>")
    report = []
    rows = [json.loads(x) for x in open(args.texts)][:args.n]
    with torch.no_grad():
        for row in rows:
            ids = tok.apply_chat_template(dict_args(row["messages"]), tools=row["tools"], tokenize=True,
                                          enable_thinking=row["thinking"], return_dict=False)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            ids = list(ids)[:6144]
            role, roles = "?", []
            for i, t in enumerate(ids):
                if i > 0 and ids[i - 1] == start:
                    role = tok.decode([t])
                roles.append(role)
            x = torch.tensor([ids])
            lr = r_model(input_ids=x.to(r_model.device)).logits[0, :-1]
            lq = q_model(input_ids=x.to(q_model.device)).logits[0, :-1].to(lr.device)
            nxt = x[0, 1:].to(lr.device)
            nll_r, nll_q = [], []
            kl, ent_r, r_arg, q_arg, r_ptop, q_prtop = [], [], [], [], [], []
            for c in range(0, lr.shape[0], 512):
                a = torch.log_softmax(lr[c:c + 512].float(), -1)
                b = torch.log_softmax(lq[c:c + 512].float(), -1)
                kl.append((a.exp() * (a - b)).sum(-1).cpu())
                ent_r.append(-(a.exp() * a).sum(-1).cpu())
                ra = a.argmax(-1)
                r_arg.append(ra.cpu())
                q_arg.append(b.argmax(-1).cpu())
                r_ptop.append(a.gather(-1, ra[:, None])[:, 0].exp().cpu())
                q_prtop.append(b.gather(-1, ra[:, None])[:, 0].exp().cpu())
                nll_r.append(-a.gather(-1, nxt[c:c + 512, None])[:, 0].cpu())
                nll_q.append(-b.gather(-1, nxt[c:c + 512, None])[:, 0].cpu())
                del a, b
            kl, ent_r, r_arg, q_arg, r_ptop, q_prtop = (torch.cat(t) for t in (kl, ent_r, r_arg, q_arg, r_ptop,
                                                                                 q_prtop))
            r_max, q_max = lr.max().item(), lq.max().item()
            nll_r, nll_q = torch.cat(nll_r), torch.cat(nll_q)
            asst = torch.tensor([r == "assistant" for r in roles[:-1]])
            del lr, lq
            big = (kl > 5).nonzero().flatten().tolist()
            by_role = {}
            for i in big:
                by_role[roles[i]] = by_role.get(roles[i], 0) + 1
            worst = kl.topk(8).indices.tolist()
            rep = {"task": row["task"], "set": row["set"], "tokens": len(ids), "kl_mean": kl.mean().item(),
                   "share_from_gt5": kl[kl > 5].sum().item() / kl.sum().item(), "n_gt5": len(big),
                   "gt5_by_role": by_role, "first_gt5": big[:10],
                   "r_logit_max": r_max, "q_logit_max": q_max, "swap": args.swap,
                   "nll_r": nll_r.mean().item(), "nll_q": nll_q.mean().item(),
                   "nll_r_assistant": nll_r[asst].mean().item(), "nll_q_assistant": nll_q[asst].mean().item(),
                   "nll_r_other": nll_r[~asst].mean().item(), "nll_q_other": nll_q[~asst].mean().item(),
                   "worst": [{"pos": i, "role": roles[i], "kl": kl[i].item(), "entropy_r": ent_r[i].item(),
                              "context": tok.decode(ids[max(0, i - 12):i + 1]), "next": tok.decode([ids[i + 1]]),
                              "r_top": tok.decode([r_arg[i].item()]), "q_top": tok.decode([q_arg[i].item()]),
                              "r_p_top": r_ptop[i].item(), "q_p_rtop": q_prtop[i].item()}
                             for i in worst]}
            report.append(rep)
            print(json.dumps(rep, indent=1), flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("OUTLIERS_DONE")


if __name__ == "__main__":
    main()
