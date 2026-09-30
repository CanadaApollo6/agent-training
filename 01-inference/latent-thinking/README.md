# Latent thinking for Ornith

The goal: think in fewer steps by feeding the model something richer than one sampled token each step. The inner link
in [RecursiveMAS](https://arxiv.org/abs/2604.25917) maps a model's own last hidden state back into its input.

**The speed catch:** TensorFold's MTP drafts make plain tokens ~2.5× faster than serial steps. Latent steps can't be
drafted, so a latent step has to replace at least ~2.5 thinking tokens just to break even on speed.

Plan, cheapest first:
0. Soft thinking, no training (this page)
1. A learned link, trained locally from dumped hidden states
2. The full inner/outer-loop training through the frozen bf16 model on a pod

## Step 0: soft thinking on Ornith 1.5 9B (2026-09-30)

[`soft_think.py`](soft_think.py): Ornith 1.5 9B in bf16 (the dense sibling of the 35B, with the same 3:1 GDN/attention
layout) on a rented A100 80 GB, in transformers.
- The problems and grading are the reasoning probe's: 20 level-5 MATH-500 problems, 2 samples each. The cap is 16K
  steps, 14K of them for thinking.
- All modes share one batched loop. Only the input fed back during thinking differs, and the answer after
  `</think>` is always sampled as plain tokens.
- Left padding was checked: it moves the logits no more than batching a prompt with itself (0.41 vs 0.33).

Modes:
- **discrete:** sampled tokens (temperature 1.0, top-k 20, top-p 0.95), the baseline.
- **soft:** the probability-weighted mix of the top-15 token embeddings (temperature 0.6, top-p 0.95). Thinking ends
  when the top token is `</think>`, or after 256 near-certain steps (a "cold stop").
- **gumbel:** the same mix after Gumbel noise (τ 0.5), which Wu et al. (2025) proposed against soft thinking's
  collapse.

| Mode | Correct (of 40) | Median thinking steps | How thinking ended |
|---|---|---|---|
| discrete | 35 | 1,361 | 35 on its own, 5 at the cap |
| soft | 34 | 1,656 | 34 on its own, 6 cold stops |
| gumbel | 35 | 1,553 | 34 on its own, 5 cold stops, 1 at the cap |

- **Accuracy is unchanged.**
  - Soft vs discrete: 33 solved by both, 2 only by discrete, 1 only by soft.
  - Gumbel vs discrete: 34 solved by both, 1 each only by one mode.
- **Plain soft thinking collapses to greedy.** Both samples of every problem followed the same path, to the step. So
  its 40 runs are really 20. That matches Wu et al.
- **Gumbel shortens thinking a little, not reliably.**
  - On the 34 problems both solve, it was shorter on 21 (sign test p = 0.23).
  - The per-problem length ratio is 0.88 (95% CI 0.75–1.03).
  - Discrete's mean looks much longer only because of its 5 cap runaways. Gumbel instead had cold stops, forced
    endings that were right 3 times out of 5.
- **Verdict:** the hybrid GDN model takes blended input embeddings without trouble: no accuracy loss and no breakdown.
  But ~12% fewer steps is far from the ~2.5× a latent step needs to beat drafted tokens. The training-free route
  doesn't pay for decode.
  - A learned link (steps 1–2) would have to compress thinking by 2.5× or more. The paper's best case is 76% fewer
    tokens (about 4×), at 3 rounds, on dense models.
  - The 9B gets 35/40 here, so this problem set is near its ceiling. It can show length changes but not small
    accuracy gains.
- **Predictions (Claude, before running):**
  - Soft would be a bit shorter and lose some accuracy. Wrong on both: same accuracy, not shorter.
  - Gumbel would match accuracy with 10–20% fewer steps. Right, but the effect isn't significant.
- Cost: A100 80 GB for about 1.6 h, roughly $2.50. The three modes took 38, 28 and 35 minutes.
