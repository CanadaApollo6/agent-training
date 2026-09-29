# Speed hill-climb on the 3090

How fast can the chosen local builds answer? This climbs the builds picked in
[05-compression/ornith](../../05-compression/ornith/).
- **APEX-I-Compact** (16.5 GB): no accuracy loss, reasoning near Q8's length.
- **Build A, experts at 2-bit** (13.8 GB): no accuracy loss, reasoning 40% longer.

## Rules

- **Score:** time to answer for one request, at fixed prompt lengths. That's decode rate × reasoning length, plus
  prefill. Decode tokens per second is reported alongside. Batched throughput is a separate, secondary track.
- **Gates:** a speedup counts only if it passes.
  - Changes that should keep the same output (speculative decoding, kernel swaps, CUDA graphs) must match plain
    decoding token for token.
  - Changes to the numbers (quantization recipes) must pass the paired reasoning probe, 4 attempts per problem.
- **Ceiling:** batch-1 decoding reads the weights it uses once per token, so the limit is bytes read per token over
  829 GB/s. Speculative decoding can beat it by checking several tokens per read.
- **Where:** candidates grind on rented cards, with winners confirmed on the 3090 in short runs.
  - A rented 3090 or 4090 is best, when Prime has one in stock.
  - Otherwise an A10 24 GB (same GA102 chip, sm_86, ~600 GB/s).

## Starting point

Build A reads about 2.5 GB per token, a ceiling of about 330 tok/s.

| Part | In the file | Read per token |
|---|---|---|
| Routed experts (IQ2_S/IQ3_S) | 11.0 GB | 0.34 GB (8 of 256) |
| Always-on path (Q8_0), including the 0.54 GB LM head | 2.2 GB | 2.17 GB |

For this MoE, the experts set the size but the always-on path sets the speed.

Baseline, PrismML's llama.cpp build (b10743), flash attention on, `llama-bench` with 3 repeats, on the 3090:

| Build | Prefill 512 | Decode 128 | Decode after 8K of context | Share of ceiling |
|---|---|---|---|---|
| A, experts-2bit | 3,504 tok/s | 148.2 tok/s | 143.8 tok/s | 45% |

At 45% of the ceiling, more than half the time per token goes somewhere other than reading weights. On an MoE at
batch 1, that's usually per-layer overhead (40 layers of small kernels, expert routing and gathering, and launches)
rather than bandwidth. That makes CUDA graphs and kernel fusion likely early wins.

## TensorFold on the 3090

TensorFold refuses GPUs below compute capability 9.0 (Hopper), and the 3090 is 8.6. The port:
- Build its kernels for sm_86, leaving out the FP8 ones.
- Send the 4-bit matmul down its existing non-cluster path.
- Pass its exactness tests.

Its CUDA engine reads MLX 4-bit weights only, so it runs a different build from build A.

**Riel's prediction (2026-09-29):** ported TensorFold decodes about 15% faster than llama.cpp on build A.

Claude's prediction, for contrast: 1.2–2×, most likely about 1.5×. Two things push it up:
- The MLX 4-bit build's always-on path is 4-bit rather than 8-bit, so it reads fewer bytes per token than build A.
- MTP drafts add on top.

Two things push it down:
- TensorFold's kernels were tuned on GB10, which has a 24 MB L2 against the 3090's 6 MB.
- For an MoE, each drafted row brings its own 8 experts, so drafts cost more to verify.

Measured as batch-1 decode tok/s on the same prompts, each engine in its best exact setting.

### The port (2026-09-29, rented A10)

It took a small patch, [`tensorfold/sm86.patch`](tensorfold/sm86.patch), against TensorFold 0.3.7:
- The capability floor drops to 8.0.
- Split-K's sum across K slices uses thread-block clusters on 9.0+. Below that, the slices go through the existing
  slice buffer and are added in the same order, so the bits don't change.
- Two instructions don't exist on Ampere.
  - The bf16 subtract in 4-bit unpacking becomes a single-rounding FMA, which gives the same bits.
  - FP8 MMA can't be used, so prompts take the bf16 kernels instead of FP8 ones.
- An env var overrides the memory reserve (below).

TensorFold's own tests pass on sm_86: 203 for the Qwen dense and MoE engines, plus 103 for the kernels. Left out are
FP8 prefill (by design) and EXL3, which isn't used here.

The MLX 4-bit Ornith checkpoint dropped the MTP layer. [`tensorfold/make_mtp.py`](tensorfold/make_mtp.py) rebuilds
it from the bf16 weights the way MLX quantized the rest. MLX also stores RMSNorm weights as 1 + w; check that
against the main model before trusting a side file.

### Result on the A10

Decode tok/s, 256 tokens, median of 3, on the same A10:

| Engine, build | Size | Chat prompt | Code prompt |
|---|---|---|---|
| llama.cpp, build A (`llama-bench` tg128) | 12.8 GiB | 101.7 | 101.7 |
| TensorFold, MLX 4-bit + MTP, greedy | 18.9 GiB | 190.9 | 276.0 |
| TensorFold, MLX 4-bit + MTP, sampled (T = 1) | 18.9 GiB | 188.7 | 231.4 |
| TensorFold, drafts off (its serial reference path) | 18.9 GiB | 39.0 | 38.8 |

- **Exact:** greedy output with drafts matched drafts-off token for token on 4 prompts × 1,024 tokens.
- **Speed:** about 1.9× llama.cpp on chat text and 2.7× on code, all from MTP drafting. The drafts-off path is
  TensorFold's slow reference, not a fair engine-only number.
- **Predictions:** Riel's +15% was too low, and Claude's 1.2–2× (most likely 1.5×) was also low.
- **Quality gate passed:** the MLX 4-bit build (plain round-to-nearest, 6 GiB bigger than build A) scored 68/80 on
  the reasoning probe, served by TensorFold. That's against Q8's 66/80, p = 0.75.
- **Time to answer:** its correct answers ran 18% longer than Q8's, against build A's 41%. With the A10 decode rates,
  it answers about 2.2× faster than llama.cpp with build A.

On a rented RTX 4090 (24 GB like the 3090, sm_89), the same tests pass, FP8 prefill included (306 tests). Decode is
312–376 tok/s on chat and 391–494 on code; prefill is 8–13K tok/s.

### The catch: context

On the A10's 22 GiB, TensorFold ran out of memory on a 30K-token prompt:
- The weights take 18.9 GiB. TensorFold's working memory (KV growth copies, the MTP head's own cache, state
  checkpoints, prefill buffers) takes about 2.4 GiB more.
- Its startup estimate is conservative for a discrete card: 4 GiB held back, a full-vocabulary logits bound, and 4
  KV copies. By its own estimate it wouldn't start at all, so `TENSORFOLD_CUDA_RESERVE_GIB` overrides the reserve.
- The prompt cache defaults to an eighth of system RAM. That's fine on GB10's unified memory but not on a 24 GB card.
  Run with `--prompt-cache-gib 0`.

On the 4090's 24 GB, a 26K-token prompt worked (recall correct, peak 22.8 GiB); 45K and up ran out of memory. The
3090 has the same 24 GB, so about 26K is its limit too, short of agent-length context.

Two ways out:
- Give TensorFold low-bit experts. Build A's trick applied to TensorFold would save about 8 GiB, but its grouped
  expert kernels read 4-bit only.
- Trim its working memory for 24 GB cards.

## Low-bit experts in TensorFold (2026-09-29)

Riel's call: give TensorFold low-bit experts, trim its buffers later, and end up with a custom Ornith 35B recipe.

### Kernels

The grouped expert kernels now also read 2- and 3-bit experts (groups of 64), for decode and prompts alike. Patch:
[`tensorfold/sm86-lowbit.patch`](tensorfold/sm86-lowbit.patch), which includes the sm_86 port.
- **Bit planes.** A 2-bit expert is stored as one plane of 2-bit fields. A 3-bit expert adds a second plane holding
  each level's third bit.
- **Same arithmetic.** The kernel rebuilds the exact 4-bit word the old kernel read from the planes. Everything after
  that (unpacking to bf16, the matmul, the epilogue) is unchanged. So a 2-bit expert gives the same bits as the same
  levels stored at 4 bits.
- **Shared expert.** It stays at 4 bits in its own tensor. Each tile picks its width by expert id.
- **Loader.** It reads each projection's width from the tensor shapes, so a checkpoint can mix widths.

Tests on a rented 4090:
- 19 new tests pass. They check that narrow experts give exactly the same bits as the 4-bit kernels (decode, prompts,
  fp32 and bf16 outputs, five width mixes), and that a row's output doesn't depend on the other rows in the batch.
- The existing 36 expert and MoE tests still pass.

### Recipe quantizer

[`tensorfold/quantize_experts.py`](tensorfold/quantize_experts.py) re-quantizes only the routed experts, from the bf16
weights. It keeps MLX's format (w = s·q + b, per group of 64). Everything else is copied from the MLX 4-bit checkpoint:
the always-on path, the routers, the shared expert and the MTP head.

The scale and offset per group come from a search rather than plain min/max rounding:
- Each input column gets an error weight from bartowski's importance matrix (the one build A used). A column that sees
  big activations costs more when it's wrong.
- For each group, the search tries a sweep of ranges. For each range it rounds, refits s and b by weighted least
  squares, then rounds them to bf16. The lowest weighted error wins.

Mean weighted relative error per projection, across the 40 layers:

| Projection | Width | Plain rounding | Search |
|---|---|---|---|
| gate | 2-bit | 0.171 | 0.103 |
| up | 2-bit | 0.176 | 0.107 |
| down | 3-bit | 0.037 | 0.027 |

At 4 bits the plain path reproduces MLX's own rounding: 97% of words are bit-identical, and the rest differ only in
rounding order.

Recipes:
- **R2:** gate/up at 2-bit, down at 3-bit. That's close to build A's layout (IQ2_S / IQ3_S). The checkpoint is 13 GB,
  against 19 GB for MLX 4-bit.
- **R1:** everything at 3-bit, the safer option.

**Riel's prediction (2026-09-29, before the probe result):** "R2 passes the probe but it takes 3-4x longer."

**Claude's predictions, written before measuring:**
- **Decode:** routed experts are about a third of the bytes read per token, and R2 cuts them by about 37%. Rebuilding
  words from the planes adds a little arithmetic. MTP drafts may be accepted slightly less often. Net: about 5% faster.
- **Context:** about 6.5 GiB freed, at about 70 MB per 1K tokens, gives roughly 100K+ tokens on 24 GB.
- **Probe:** R2 is riskier than build A, because 2-bit affine is coarser than IQ2_S's codebook. The score should hold
  within noise, but reasoning will run 30–50% longer, as it did for build A. R1 should pass cleanly.

### R2 results (rented 4090)

**Speed:** unchanged.

| Build | Chat, sampled | Code, sampled | Chat, greedy | Code, greedy |
|---|---|---|---|---|
| MLX 4-bit | 312 | 391 | 376 | 494 |
| R2 | 319 | 406 | 360 | 468 |

Different weights write different text, which changes how many MTP drafts get accepted. That moves the numbers either
way by a few percent. The bytes saved on experts don't show up, because the always-on path sets the speed. Claude's
+5% was within noise of right, but for the wrong reason.

**Probe:** it passes the gate, but it's the weakest build that has passed so far.

| Build | Correct | Hit the cap | Lost / gained vs Q8 | p | Median tokens, correct answers |
|---|---|---|---|---|---|
| Q8 | 66/80 | 10 | – | – | 1,089 |
| MLX 4-bit (TensorFold) | 68/80 | 6 | 1 / 3 | 0.75 | 1,290 (+18%) |
| **R2 (TensorFold)** | 62/80 | 12 | 5 / 1 | 0.31 | 1,283 (+18%) |
| Build A (llama.cpp) | 66/80 | 9 | 0 / 0 | 1.00 | 1,534 (+41%) |

- **Riel** (passes, reasoning 3–4× longer): right that it passes. Wrong on length: correct answers ran 18% longer, the
  same as MLX 4-bit.
- **Claude** (score holds, 30–50% longer): wrong on length too.
- The 4 lost problems are not significant at 80 samples, but they lean the wrong way. R2 reasons as briefly as MLX 4-bit
  and more briefly than build A, but gets fewer right. Build A's 2-bit codebook quants (IQ2_S) keep more accuracy than
  2-bit affine at the same size.

**Context:** better, but TensorFold's working memory turned out to be the real limit. With 6.5 GiB freed, a 59K-token
prompt fit (peak 20.3 GiB), but 78K ran out of memory. Memory grew about 100 MB per 1K tokens, where the KV cache
itself needs only 20 MB (10 attention layers × 2 KV heads × 256 × 2 bytes × K and V). An audit of the engine found the
same context held several times over:
- The prompt pass's KV cache grows by doubling, and each growth copies it.
- The decode graphs keep a second KV cache of their own, and the prompt is copied into it.
- The prompt cache keeps the states of earlier prompts, each with its own KV buffers.
- On top of that, the MTP head absorbed prompts in 4,096-row pieces with about 450 MB of scratch. That's where the
  out-of-memory error hit.

### Holding the context once

A second patch (in the same file): the engine allocates the whole window's attention rows once, at startup.
- Prompts are written straight into them. Decoding reads them in place.
- Kept prompt states point into the same rows. So the cache keeps only prefixes of the latest prompt, which is exactly
  what a growing agent conversation reuses. Switching between unrelated conversations re-reads the prompt, at about
  10K tok/s.
- The MTP head takes prompts in pieces of at most 1,024 rows.
- The startup estimate counts one copy of the KV cache instead of four. It no longer needs the negative-reserve
  override.

Gates:
- **Unit tests:** 4 new tests, plus the existing 12. Each request in a mixed sequence (resume, unrelated prompt, resume
  again, shorter prompt) matches serial decoding. Every kept state's KV rows match a fresh prompt pass bit for bit.
  Dropping the prefix-only rule makes that test fail; token equality alone didn't catch it.
- **Real model:** 4 prompts × 1,024 greedy tokens with drafts matched the serial path token for token.

R2 with a 131K window, on the 4090:

| Prompt | Prefill | Recall | Peak memory |
|---|---|---|---|
| 39K tokens | 11.5K tok/s (after warmup) | correct | 20.20 GiB |
| 79K | 8.1K tok/s | correct | 20.26 GiB |
| 106K | 6.8K tok/s | correct | 20.27 GiB |
| 126K | 6.0K tok/s | correct | 20.27 GiB |

Memory is flat now: the whole window is paid for at startup. Decode speed didn't change (same bench numbers). The first
prompt after startup is slow (68 s for 39K) while kernels compile, so warm the server up. A 3090 has the same 24 GB,
so a 131K context should fit there too; that still needs confirming on the 3090 itself.

### R1 (all experts at 3-bit)

R1 is 15 GB and fits a 131K window with a 2 GiB reserve. Decode is the same as R2 (chat 324 / code 404 sampled, 367 / 446
greedy). Probe:

| Build | Correct | Lost / gained vs Q8 | p | Median tokens, correct answers |
|---|---|---|---|---|
| R1 | 63/80 | 5 / 2 | 0.63 | 1,268 (+16%) |
| R2 | 62/80 | 5 / 1 | 0.31 | 1,283 (+18%) |

Adding a bit to gate/up bought one problem, which is inside the noise. At 80 samples the probe resolves differences of
about ±4 problems. So it can't rank R1, R2 and MLX 4-bit (62–68). All of them pass. Telling them apart needs a finer
measure, such as KL divergence from the full-precision model's next-token distribution on a fixed text.

### Why low-bit experts don't speed up decode

[`tensorfold/bench_experts.py`](tensorfold/bench_experts.py) times one MoE layer (route, gate/up, down) at Ornith's sizes,
in µs:

| Widths (gate-up / down) | 1 row | 2 rows | 4 rows | 6 rows |
|---|---|---|---|---|
| 4 / 4 | 38.0 | 38.7 | 74.8 | 98.7 |
| 3 / 3 | 35.4 | 42.6 | 56.8 | 80.7 |
| 2 / 3 | 38.7 | 44.5 | 62.0 | 72.4 |
| 2 / 2 | 37.6 | 42.9 | 58.7 | 68.6 |

- **At 1–2 rows:** width doesn't matter. The layer's 38 µs goes to routing, launches and latency, not to reading bytes:
  9 experts × 3 projections of 2048 × 512 at 4 bits is about 15 MB, which is 15 µs at the 4090's bandwidth.
- **At 4–6 rows,** the size of an MTP verification step, narrow experts save 20–30%.
- **Over 40 layers,** the 38 µs floor adds up to about 1.5 ms of every decode step.

That floor is the next target, and fused kernels are the way to attack it. Route, gate/up, SwiGLU and down can become
one launch per layer at batch 1.
