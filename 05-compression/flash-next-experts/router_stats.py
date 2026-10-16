"""Which Flash-Next experts do our agent traces use, and what happens if we drop half of them?

Pass 1 runs the calib windows through the model and counts, per layer and expert: how often the router picks it, the
routing weight it gets, its share of the full router softmax, and a REAP-style score (routing weight times the size of
the expert's output). From those it picks which experts to keep. Pass 2 runs the held-out and control windows with the
full model and with the dropped experts masked out of the router, and compares the next-token predictions: KL
divergence, top-1 agreement, and loss on the real next token. No weights are changed; masking only stops the router
from choosing the dropped experts.

    python router_stats.py <model dir> <data dir> <out dir>
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen4_exp import modeling_qwen4_exp as M

STATE = {"collect": None, "keep": None}  # collect: dict of per-layer stat tensors; keep: per-layer bool masks


def moe_forward(self, hidden_states):
    b, s, h = hidden_states.shape
    x = hidden_states.view(-1, h)
    shared = F.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    gate = self.gate
    logits = F.linear(x, gate.weight)
    if STATE["keep"] is not None:
        logits = logits.masked_fill(~STATE["keep"][self.layer_idx], float("-inf"))
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    top_w, top_i = torch.topk(probs, gate.top_k, dim=-1)
    if gate.norm_topk_prob:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w.to(logits.dtype)

    ex = self.experts
    flat_e = top_i.reshape(-1)
    order = flat_e.argsort()
    tok = order // gate.top_k
    w_sorted = top_w.reshape(-1)[order]
    counts = torch.bincount(flat_e, minlength=ex.num_experts)
    xs = x[tok]
    outs, start = [], 0
    st = STATE["collect"]
    if st is not None:
        st["probmass"][self.layer_idx] += probs.sum(0)
        st["count"][self.layer_idx] += counts.float()
    for e, n in enumerate(counts.tolist()):
        if n == 0:
            continue
        seg = slice(start, start + n)
        g, u = F.linear(xs[seg], ex.gate_up_proj[e]).chunk(2, dim=-1)
        y = F.linear(ex.act_fn(g) * u, ex.down_proj[e])
        outs.append(y * w_sorted[seg, None])
        start += n
    weighted = torch.cat(outs)
    if st is not None:  # REAP-style: routing weight times the expert's output size, i.e. the norm of each weighted output
        e_sorted = flat_e[order]
        st["wsum"][self.layer_idx].index_add_(0, e_sorted, w_sorted.float())
        st["reap"][self.layer_idx].index_add_(0, e_sorted, weighted.float().norm(dim=-1))
    out = torch.zeros_like(x)
    out.index_add_(0, tok, weighted.to(out.dtype))
    return (out + shared).reshape(b, s, h)


M.Qwen4ExpTextSparseMoeBlock.forward = moe_forward


def moe_blocks(model):
    blocks = [m for m in model.modules() if isinstance(m, M.Qwen4ExpTextSparseMoeBlock)]
    for i, m in enumerate(blocks):
        m.layer_idx = i
    return blocks


def new_stats(blocks, n_exp):
    """One row per layer, each on that layer's GPU."""
    return {k: [torch.zeros(n_exp, device=b.gate.weight.device) for b in blocks]
            for k in ("count", "wsum", "probmass", "reap")}


def add_stats(dst, src):
    for k in dst:
        dst[k] += torch.stack([row.cpu() for row in src[k]])


def windows(data_dir, split):
    """All windows of a split, or LIMIT_<SPLIT> of them spread evenly so every source is sampled."""
    d = np.load(f"{data_dir}/{split}.npz")
    ids, lens, src = d["ids"], d["lens"], d["src"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    limit = os.environ.get(f"LIMIT_{split.upper()}")
    pick = range(len(lens)) if not limit else sorted(set(np.linspace(0, len(lens) - 1, int(limit)).round().astype(int)))
    for i in pick:
        yield str(src[i]), torch.from_numpy(ids[offs[i]:offs[i + 1]].astype(np.int64))[None]


def hidden(model, ids, dev):
    return model.model(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0]


def compare(model, h0, hk, ids, slice_len=1024):
    """Sums over positions: KL(full || pruned), top-1 agreement, and next-token loss for both."""
    head = model.lm_head
    tgt = ids[0, 1:].to(h0.device)
    tot = {"n": 0, "kl": 0.0, "agree": 0, "nll_full": 0.0, "nll_pruned": 0.0}
    T = h0.shape[0] - 1
    for s in range(0, T, slice_len):
        e = min(s + slice_len, T)
        l0 = head(h0[s:e].to(head.weight.device)).float().log_softmax(-1)
        lk = head(hk[s:e].to(head.weight.device)).float().log_softmax(-1)
        t = tgt[s:e].to(l0.device)
        tot["n"] += e - s
        tot["kl"] += (l0.exp() * (l0 - lk)).sum(-1).sum().item()
        tot["agree"] += (l0.argmax(-1) == lk.argmax(-1)).sum().item()
        tot["nll_full"] += -l0.gather(1, t[:, None]).sum().item()
        tot["nll_pruned"] += -lk.gather(1, t[:, None]).sum().item()
    return tot


def keep_sets(calib, n_exp):
    """Per-layer boolean masks: same number kept in every layer."""
    sets = {}
    reap_mean = calib["reap"] / calib["count"].clamp(min=1)
    for name, score, frac in (("freq50", calib["wsum"], 0.5), ("reap50", reap_mean, 0.5),
                              ("freq25", calib["wsum"], 0.75)):
        k = int(n_exp * frac)
        top = score.topk(k, dim=-1).indices
        sets[name] = torch.zeros_like(score, dtype=torch.bool).scatter_(1, top, True)
    return sets


def main(model_dir, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/progress.log", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    n_gpu = torch.cuda.device_count()
    kw = {"dtype": torch.bfloat16}
    if n_gpu:
        mem = os.environ.get("GPU_MEM", "60GiB")
        kw.update(device_map="auto", max_memory={i: mem for i in range(n_gpu)})
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **kw).eval()
    dev = model.model.get_input_embeddings().weight.device
    blocks = moe_blocks(model)
    n_exp = blocks[0].experts.num_experts
    say(f"loaded in {time.time() - t0:.0f}s: {len(blocks)} MoE layers x {n_exp} experts, first device {dev}")

    torch.set_grad_enabled(False)
    # Pass 1: router statistics on the calibration split.
    calib = {k: torch.zeros(len(blocks), n_exp) for k in ("count", "wsum", "probmass", "reap")}
    t0, ntok = time.time(), 0
    for i, (src, ids) in enumerate(windows(data_dir, "calib")):
        STATE["collect"] = new_stats(blocks, n_exp)
        hidden(model, ids, dev)
        add_stats(calib, STATE["collect"])
        ntok += ids.shape[1]
        if i % 10 == 0:
            say(f"calib window {i}: {ntok} tokens, {ntok / (time.time() - t0):.0f} tok/s")
    STATE["collect"] = None
    np.savez(f"{out_dir}/calib_stats.npz", **{k: v.numpy() for k, v in calib.items()})
    say(f"pass 1 done: {ntok} tokens in {time.time() - t0:.0f}s")

    sets = keep_sets(calib, n_exp)
    np.savez(f"{out_dir}/keep_sets.npz", **{k: v.numpy() for k, v in sets.items()})
    plan = {"heldout": ["freq50", "reap50", "freq25"], "control": ["freq50"]}

    # Pass 2: full vs pruned on held-out and control; router stats for both splits come from the full runs.
    results = {}
    for split, names in plan.items():
        stats = {k: torch.zeros(len(blocks), n_exp) for k in ("count", "wsum", "probmass", "reap")}
        agg = {n: {"n": 0, "kl": 0.0, "agree": 0, "nll_full": 0.0, "nll_pruned": 0.0} for n in names}
        per_src = {}
        t0 = time.time()
        for i, (src, ids) in enumerate(windows(data_dir, split)):
            STATE["keep"] = None
            STATE["collect"] = new_stats(blocks, n_exp)
            h0 = hidden(model, ids, dev)
            add_stats(stats, STATE["collect"])
            STATE["collect"] = None
            for name in names:
                STATE["keep"] = [sets[name][j].to(b.gate.weight.device) for j, b in enumerate(blocks)]
                r = compare(model, h0, hidden(model, ids, dev), ids)
                for k in r:
                    agg[name][k] += r[k]
                ps = per_src.setdefault(f"{src}|{name}", {k: 0 for k in r})
                for k in r:
                    ps[k] += r[k]
            STATE["keep"] = None
            say(f"{split} window {i} ({src}): " + ", ".join(
                f"{n} KL {agg[n]['kl'] / agg[n]['n']:.4g} agree {agg[n]['agree'] / agg[n]['n']:.3f}" for n in names)
                + f", {time.time() - t0:.0f}s")
            results[split] = {"summary": {n: summarize(a) for n, a in agg.items()},
                              "by_source": {k: summarize(v) for k, v in per_src.items()}}
            json.dump(results, open(f"{out_dir}/results.json", "w"), indent=1)
        np.savez(f"{out_dir}/{split}_stats.npz", **{k: v.numpy() for k, v in stats.items()})
    say("done")


def summarize(a):
    n = max(a["n"], 1)
    return {"tokens": a["n"], "kl": a["kl"] / n, "top1_agree": a["agree"] / n,
            "ppl_full": math.exp(a["nll_full"] / n), "ppl_pruned": math.exp(a["nll_pruned"] / n)}


if __name__ == "__main__":
    main(*sys.argv[1:4])
