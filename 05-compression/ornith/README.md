# Ornith 1.5 quantization ladder

Where does Ornith 1.5 9B stop reasoning well as its weights get rounded harder? This is the same probe as the
Qwen3.8-27B one in [01-inference/qwen38-27b](../../01-inference/qwen38-27b/): 20 MATH-500 level-5 problems (seed 0),
thinking on, temperature 1.0 / top_p 0.95 / top_k 20, a 16K-token cap, 8 at a time, 8-bit KV, graded with
math-verify on the text after `</think>`.

GGUF builds, served by PrismML's llama.cpp fork on the 3090 (`../probe_gguf.sh <gguf> <label>`), with a 10-minute
cooldown before each one. Q8_0 is Ornith's official build; the rest are bartowski's imatrix builds.

| Build | File | Avg bits/weight* | Correct | Wrong | Hit the cap | Loopy | Median tokens |
|---|---|---|---|---|---|---|---|
| Q8_0 | 9.79 GB | 8.5 | 16/20 | 1 | 3 | 0 | 4,157 |
| Q4_K_M | 5.91 GB | 5.1 | 16/20 | 1 | 3 | 0 | 4,097 |
| IQ3_XXS | 4.28 GB | 3.7 | 13/20 | 1 | 6 | 0 | 6,046 |
| IQ2_M | 3.77 GB | 3.3 | 12/20 | 0 | 8 | 4 | 4,593 |

\* File size over Q8_0's parameter count. The embeddings and LM head (a 248K-token vocabulary) are kept at higher
precision, so the average sits well above the nominal 3 or 2 bits of the layers.

Paired on the same problems against Q8_0 (exact McNemar):

| Build | Lost | Gained | p | Median tokens on problems both solved |
|---|---|---|---|---|
| Q4_K_M | 0 | 0 | 1 | identical correct sets |
| IQ3_XXS | 3 | 0 | 0.25 | 1,146 → 2,158 |
| IQ2_M | 4 | 0 | 0.125 | 1,134 → 2,596 |

- **Q4 is free.** It solved exactly the same problems as Q8.
- **Below 4 bits, the model doesn't get answers wrong; it fails to finish.** Every loss at IQ3_XXS and IQ2_M is a
  cap-limited runaway. Even on the problems it still solves, reasoning roughly doubles.
- **IQ2_M adds loops.** 4 traces have more than 20% repeated lines, the worst at 78%. That's the full collapse
  signature, the same as EXL3 2.0 on the 27B (12/20, 2 loopy, reasoning 2× longer).
- 20 problems can't separate IQ3_XXS from IQ2_M. Each is 3–4 problems down on Q8, and neither is significant alone.
  The trend (0 → 3 → 4 lost, lengths doubling, loops appearing) is the evidence.

## Ornith 1.5 35B-A3B

The same probe with 4 attempts per problem (80 per build), served on a rented A6000 by the same llama.cpp build
(`../pod/probe_ladder.sh`), 16 at a time. Q8_0 is Ornith's official build; the rest are bartowski's imatrix builds.
Paired against Q8_0 per problem (each problem scores 0–4), with an exact sign-flip test (`compare.py`):

| Build | File | Avg bits/weight | Correct | Wrong | Hit the cap | Loopy | Lost | Gained | p | Median tokens, both solved |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 37.8 GB | 8.5 | 66/80 | 4 | 10 | 0 | | | | 1,089 |
| Q4_K_M | 21.7 GB | 4.9 | 68/80 | 3 | 9 | 0 | 1 | 3 | 0.75 | 1,374 |
| IQ3_XXS | 15.3 GB | 3.4 | 65/80 | 5 | 10 | 0 | 2 | 1 | 1 | 1,297 |
| IQ2_M | 12.5 GB | 2.8 | 59/80 | 7 | 14 | 0 | 7 | 0 | 0.06 | 1,452 |

- **Q4 and IQ3_XXS are both free.** Where the 9B at IQ3_XXS was already 3 problems down, the 35B at fewer bits per
  weight is within one attempt of Q8.
- **IQ2_M costs something, not everything.** 7 attempts lost, none gained, on the edge of significance (p = 0.06).
  Losses split between runaways (+4) and wrong answers (+3), reasoning is a third longer, and there are no loops.
  The 9B at IQ2_M lost a quarter of its score, doubled its reasoning and looped.
- Scaled to each model's own Q8_0: at IQ3_XXS the 35B keeps 98% of its score and the 9B 81%; at IQ2_M, 89% vs 75%.

## Predictions

**Riel (2026-09-28):** the 9B breaks earlier than the 27B, "a dense model, so by default there is less to compress
relative to an MoE."

The 27B is also dense, so dense vs MoE can't be what separates them. Size can: a smaller model has fewer redundant
weights to absorb each rounding error. The dense-vs-MoE question belongs to the 35B-A3B ladder. There, each token
only passes through ~3B active parameters, which cuts the other way.

**Riel (2026-09-28):** the 35B-A3B MoE "holds up better for sure" than the 9B. The test: at the same build
(IQ3_XXS, IQ2_M), the 35B loses a smaller share of its own Q8_0 score than the 9B does. The Q8_0 baseline (37.8 GB)
doesn't fit on the 3090, so it runs on a rented pod; the low-bit builds (10–15 GB) run locally. Riel's mechanism:
the extra stored weights act as spare capacity.

Caveat on that mechanism: within one token, idle experts can't cover for a damaged active one, since the router
doesn't reroute around rounding error. What can help is averaging. Each token mixes 8 experts, so their independent
rounding errors partly cancel (roughly by √8). The always-on path (attention, router, shared expert) gets no such
averaging. The discriminating experiment is our own mixed build: experts at 2 bits and everything else at 8. If
it holds up, the experts really are the cheap part (canada-quant's bet). If it still breaks, the damage is in the
per-token path.

**Verdict on "the 35B holds up better": right.** At every low-bit build the 35B keeps a larger share of its
Q8_0 score (98% vs 81% at IQ3_XXS, 89% vs 75% at IQ2_M), at fewer bits per weight, without the 9B's loops or doubled
reasoning. The caveat is sample size on the 9B side: one attempt per problem, so its ladder is noisier by ±2–3
problems. A 4-attempt rerun of the 9B would firm this up, but the gap at IQ3_XXS (0 vs 3 problems lost) is well
outside that noise. Why it holds up (spare capacity or error averaging) is what the ablation below tests.

**Verdict on "the 9B breaks earlier" (after IQ2_M): leaning right, not proven.** At about 3 bits the 27B was
intact (EXL3 3.0: 19/20, reasoning barely longer). The 9B at IQ3_XXS already runs out of budget 3 more times, with
reasoning doubled. At 2 bits both look alike (12/20, loops).

Evidence so far leans Riel's way. IQ3_XXS already loses 3 problems at an average of 3.7 bits, while EXL3 3.0 on the
27B scored 19/20. It's confounded twice, though: EXL3's rotation plus trellis is a stronger quantizer than llama.cpp's
IQ formats at the same bits, and the two models start from different baselines (16/20 vs 18/20).

## Ablation: which part of the 35B breaks at 2 bits?

bartowski's IQ2_M puts 88% of its bytes in the routed experts (IQ2_S gate/up, IQ2_S/IQ3_S down). The other 12%, the
always-on path (attention, DeltaNet projections, shared expert, embeddings, LM head), is mixed at 2–6 bits. Two
builds split that damage. Both are requantized from Ornith's official Q8_0 with bartowski's imatrix
(`ablation/*.types`, `llama-quantize --allow-requantize --tensor-type-file`):

- **A, experts-2bit:** experts exactly as in IQ2_M, everything else Q8_0.
- **B, rest-2bit:** experts Q8_0, everything else exactly as in IQ2_M.

Readout against Q8_0 and IQ2_M on the reasoning probe (20 problems × 4 samples):
- If A matches IQ2_M, the experts carry the damage (evidence against spare capacity).
- If B does, the always-on path does (canada-quant's bet).

**Riel's prediction (2026-09-29): B collapses.**

### Result

| Build | File | Avg bits/weight | Correct | Wrong | Hit the cap | Lost | Gained | p | Median tokens, both solved |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 37.8 GB | 8.5 | 66/80 | 4 | 10 | | | | 1,089 |
| A, experts-2bit | 13.8 GB | 3.1 | 66/80 | 5 | 9 | 0 | 0 | 1 | 1,534 |
| B, rest-2bit | 36.6 GB | 8.2 | 65/80 | 4 | 11 | 2 | 1 | 1 | 1,098 |
| IQ2_M (both) | 12.5 GB | 2.8 | 59/80 | 7 | 14 | 7 | 0 | 0.06 | 1,452 |

Per problem, A's scores are identical to Q8_0's on all 20 (each 0–4 of 4).

- **Neither half alone breaks the model; only both together do.** A and B are each within noise of Q8_0. IQ2_M's
  7 lost attempts fall on problems both A and B still solve 4 of 4 times. The damage compounds: 2-bit experts
  feeding a 2-bit always-on path that also has to absorb their errors.
- **The two halves hurt differently.** 2-bit experts leave accuracy alone but lengthen reasoning by 40% on the
  problems both solve (1,089 → 1,534 tokens). The 2-bit always-on path leaves length alone (1,098). Rounding the
  experts adds noise the model spends tokens working around. The always-on path at 2 bits did no measurable harm on
  its own.
- **The practical build is A.** At 13.8 GB, 1.2 GB more than IQ2_M, it matches Q8_0 problem for problem. Keeping
  the 12% always-on path at 8 bits buys back everything IQ2_M lost. That is canada-quant's recipe (quantize the
  experts hard, protect the rest), and it fits a 3090 with room for long context.

**Verdict on "B collapses": wrong.** B scored 65/80 against Q8_0's 66. The always-on path at IQ2_M's precision is
harmless when the experts beneath it are intact.

**What this says about "spare capacity":** it supports the claim that the experts are the cheap part to squeeze. The
2-bit experts cost no accuracy by themselves, only some extra reasoning. But it doesn't separate spare capacity from
error averaging across the 8 active experts; both predict cheap experts. The interaction is the new finding. The
model tolerates one damaged half and not both, so the lean builds come from protecting whichever half is smaller.
