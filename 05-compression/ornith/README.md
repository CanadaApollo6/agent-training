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
| IQ2_M | 3.77 GB | 3.3 | | | | | |

\* File size over Q8_0's parameter count. The embeddings and LM head (a 248K-token vocabulary) are kept at higher
precision, so the average sits well above the nominal 3 or 2 bits of the layers.

Paired on the same problems: Q8_0 vs Q4_K_M is 0–0 (identical correct sets); Q8_0 vs IQ3_XXS is 3–0 (exact McNemar
p = 0.25, not yet significant). All three IQ3_XXS losses are cap-limited runaways, not wrong answers: the model
reasons ~50% longer and runs out of budget. That's the same signature as the 27B at 2 bits, only milder.

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

Evidence so far leans Riel's way. IQ3_XXS already loses 3 problems at an average of 3.7 bits, while EXL3 3.0 on the
27B scored 19/20. It's confounded twice, though: EXL3's rotation plus trellis is a stronger quantizer than llama.cpp's
IQ formats at the same bits, and the two models start from different baselines (16/20 vs 18/20).
