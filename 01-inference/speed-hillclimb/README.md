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

### A faster expert kernel (local 3090)

On the 3090 the same layer costs 60 µs at 1 row (4/4 widths). Reading its 15.9 MB at 936 GB/s would take 17 µs.

**Prediction (Claude; Riel gave none):** the layer at 1 row drops from 60 µs to about 28 µs, and at 6 rows from 147 µs
to about 100 µs. The plan was to spread each unit's K groups over several warps.

**Step 1: split K across warps.**
- **Diagnosis (nsys):** gate/up took 40 µs and down 11 µs. The kernel gives each (expert, 32-column tile) to one warp,
  and that warp walks all 32 groups of K in order. At 1 row that's 144 warps for 82 SMs.
- **Change:** S warps share a tile.
  - Warp w computes the MMA products of groups w, w + S, … and puts them in shared memory.
  - The warp that owns each 8-column slice replays the original per-group accumulation in group order.
  - The arithmetic and its order are unchanged, so the output is bit-identical.
- **Result:** gate/up dropped 40 → 21.5 µs with 4 warps. Down didn't move (11.3 → 12.0). At 4–6 rows the split made
  things worse.

**Step 2: the real limit was the tensor cores.**
- **Why down didn't move:** an MMA multiplies 16 token rows by 8 weight columns, and at 1 row 15 of the 16 rows are
  padding. With fp32 accumulate the 3090's tensor cores do about 71 TFLOPS, so the padded work alone costs about
  17 µs for gate/up and 8.5 µs for down. That matches what was left.
- **Fix:** swap the operands. Weights take the 16-row side (two 8-column tiles' worth) and tokens the 8-column side,
  which halves the MMAs for any item of ≤8 pairs. Items of 9–16 pairs take two passes.
  - The weight registers already had the right layout; only the scale, bias and output indexing changed.
- **Does it keep the bits?** It does only if the tensor core gives the same result for C = AB as for (BᵀAᵀ)ᵀ.
  [`tensorfold/mma_operand_symmetry.py`](tensorfold/mma_operand_symmetry.py) checks this. On the 3090 there were 0
  mismatches in 30.7M outputs, and nearly every output involved rounding.
- **Gates:**
  - `test_experts_split.py` checks the new kernel against the old one bit for bit: every width, 1–16 rows, 1/2/4/8
    warps, and groups that don't divide evenly.
  - A mutant (a wrong scale half) fails all 6 tests.
  - The 65 expert and MoE-engine tests pass, including drafts equal to serial decoding and the arena.

**Result:** µs per MoE layer, median of 3 runs
([`results/bench-3090-swap.txt`](tensorfold/results/bench-3090-swap.txt)):

| Widths | 1 row | 2 rows | 4 rows | 6 rows |
|---|---|---|---|---|
| 4 / 4 | 60.5 → **33.0** | 68.7 → 51.8 | 99.9 → 88.5 | 159 → 126 |
| 3 / 3 | 61.3 → **31.9** | 71.1 → 49.8 | 102 → 77.8 | 156 → 120 |
| 2 / 3 (R2) | 60.2 → **30.3** | 70.3 → 46.3 | 100 → 68.1 | 146 → 105 |
| 2 / 2 | 59.6 → **29.2** | 69.0 → 44.3 | 97.3 → 63.8 | 145 → 96.2 |

- **The prediction** was 28 µs at 1 row and 100 µs at 6 rows. The measurements are 33 µs and 126 µs at 4/4, and 30 µs
  and 105 µs at R2's widths. So the 1-row number came in near the prediction, but for a different reason than
  expected: operand waste, not latency.
- **Narrow experts now pay off at 4–6 rows.** At 6 rows R2 costs 105 µs vs 126 µs at 4/4, a 17% gap (it was 8%
  before), because the kernel is closer to reading bytes.
- **Estimated effect on a decode step:** 40 layers × 27–30 µs is about 1.1–1.2 ms saved per step at 1 row, and 0.5–1.3
  ms at 4 rows (MTP verification). That estimate still needs an end-to-end run.
- **Where the rest goes at 1 row** (µs): route 3, gate/up 17.5 at a 11.3 read floor, down 8.9 at 5.7, plus about
  1.3 of gap per launch. The next step is fusion into one launch: route inside it, and down fetching its weights while
  gate/up finishes.

### Fusing the MoE block (local 3090)

The decode path runs a MoE block as six launches: router GEMV, top-k, plan, gate/up, down, and combine (the weighted sum
of the 9 slots). [`tensorfold/bench_moe.py`](tensorfold/bench_moe.py) times the whole block in a CUDA graph.

**Plan:** one kernel does plan, gate/up, down and combine.
- Each block works out the grouping of pairs by expert itself. That's at most 72 pairs, too few to deserve a launch.
- Gate/up units run first. A down unit waits on a counter for its expert's gate/up tiles.
- The last down unit to finish a (row, 32-column) tile sums that row's slots in slot order.
- The arithmetic is unchanged, so the output keeps the same bits.
- Router and top-k stay as they are for now.

**Prediction (Claude; Riel gave none):** the block at 1 row drops from 42 to about 30 µs, 4/4 widths.

**Result: the one-kernel block lost, and two smaller fusions won.** The block at 1 row went from 43 to 40.5 µs (4/4),
not 30.
- **One kernel for everything was slower.** Holding the gate/up, down and combine code paths at once took all 255
  registers per thread and spilled 200-270 bytes, so fewer warps fit per SM. It beat the separate launches only at
  4/4 widths and 1 row. Dropped.
- **Top-k and plan in one launch (`select`).** One warp per row finds the top 8 with the GPU's warp-wide max and min
  instructions (two per pick where a shuffle tree takes ten). One scan over the experts then lays out the items. It
  copies Triton's softmax arithmetic, so picks and weights keep their bits. The first version was slower than the two
  launches it replaced (11.5 vs 6.9 µs at 1 row). The per-row pick loop and a per-pair plan loop cost 7k and 12k
  cycles. The rewrite takes 4.3 µs at 1 row and 4.9 at 6 rows.
- **Combine inside down's launch.** The last unit to finish a (row, 32-column) tile sums its slots. It saves about
  0.5 µs at 1 row but costs 1.5-6 µs at 2-6 rows, where the last unit's serial sum gets long. So it runs only for
  1-row steps.

Whole block (µs):

| Widths | Step | 1 row | 2 rows | 4 rows | 6 rows |
|---|---|---|---|---|---|
| 4 / 4 | old expert kernel | 72.3 | 79.5 | 108.8 | 169.4 |
| 4 / 4 | swapped | 43.3 | 62.9 | 97.2 | 136.2 |
| 4 / 4 | + select | 40.5 | 59.7 | 94.4 | 136.1 |
| 4 / 4 | + combine (all rows) | 41.6 | 59.9 | 98.0 | 141.8 |
| 2 / 3 | old expert kernel | 71.5 | 80.8 | 110.9 | 159.1 |
| 2 / 3 | swapped | 42.0 | 57.2 | 78.7 | 112.9 |
| 2 / 3 | + select | 38.8 | 54.5 | 76.4 | 111.6 |
| 2 / 3 | + combine (all rows) | 38.2 | 57.0 | 80.9 | 118.2 |

Repeated three times at 1 row, combine's gain ranged from 0 to 1.7 µs (4/4) and 0.2 to 0.9 µs (2/3). Every step gives
the old path's output bit for bit (tests: `test_moe_select.py`, `test_moe_decode_path.py`, plus the 75 expert and MoE
engine tests, where drafted decoding equals serial).

**Why 30 µs was wrong:** the prediction assumed the launch gaps (about 1.3 µs each) plus the separate route were most
of the 12 µs to cut. They were only about half. The rest is work the fusion doesn't remove: the router GEMV (5.6 µs;
Triton gives a 1 MB read only 9 programs) and gate/up and down themselves (about 25 µs together against a 17 µs
weight-read floor).

**What's next:** a wider router kernel (about 3 µs to gain at 1 row), then end-to-end tokens per second on the 3090.

### End to end on the 3090

The MLX 4-bit build, served by TensorFold on Riel's 3090 with MTP drafts, 16K context
([`tensorfold/bench_e2e_3090.sh`](tensorfold/bench_e2e_3090.sh)). The old and new MoE decode paths are the same server
with the kernels switched ([`tensorfold/serve_kernels.py`](tensorfold/serve_kernels.py)).

**Getting it to fit:**
- The build needs 18.4 GiB of weights, and TensorFold's estimate adds 2.7 GiB of working memory even at zero context.
  Most of that is a conservative bound for full-vocabulary logits.
- With Slack, Claude Desktop and ChatGPT open, the desktop held 2.1 GB of VRAM, and the build didn't fit. With only
  T3 Code open it held 1.4 GB. At 16K the estimate was 21.56 GiB within a 21.62 GiB budget, and the server used
  21.5 GB in all.
- TensorFold also caps a discrete GPU at free host RAM less 4 GiB, which is 13 GiB on this 32 GB desktop.
  `serve_kernels.py` can size by GPU memory alone. The server runs under a 14 GB memory cap in case loading did need
  the RAM; it didn't.

**No prediction was logged before this run** (Claude's slip). From the block numbers, a verify step of 4 rows saves
about 14 µs per layer, 0.56 ms over 40 layers. That suggested roughly +8%.

Decode tok/s, 256 tokens, median of 3:

| Prompt | Temperature | Old kernels | New kernels | Change |
|---|---|---|---|---|
| code (completion) | 0 | 352.6 | 391.8 | +11% |
| chat, no thinking | 0 | 260.6 | 291.8 | +12% |
| code (completion) | 1.0 | 300.6 | 328.8 | +9% |
| chat, no thinking | 1.0 | 238.8 | 261.8 | +10% |

Greedy, 1,024 tokens each, end to end including prefill:

| Prompt | Old | New |
|---|---|---|
| infinitely many primes | 415.8 | 457.3 |
| ISO 8601 parser | 306.1 | 332.4 |
| history of the transistor | 491.3 | 534.9 |
| train word problem | 410.7 | 465.9 |

- **Exact:** old and new kernels give the same 4 × 1,024 greedy tokens. Drafted decoding equals the serial path
  (`--no-drafts`, about 100 tok/s, TensorFold's reference) on all 4.
- **Against the rented 4090's release build:** the 4090 did 391 and 312 tok/s at temperature 1. The 3090 now does 329
  and 262, 84% of it, on 86% of its memory bandwidth (936 vs 1,008 GB/s).
- The texts differ from the A10 run's after 56-864 characters. That run used TensorFold 0.3.7 with the first sm_86
  patch, not this 0.3.6.3 build, so the two aren't expected to agree bit for bit.
- The GPU peaked at 69 °C over three short runs.

### A faster router, and R2 on the 3090

**Router.** The router is a bf16 GEMV, 257 × 2048. At decode, Triton ran it as 9 programs of 32 experts, so 9 of the 82
SMs read its 1 MB.
- A sweep of tile shapes tried 84 settings. All 84 gave the same bits, because each logit's chain of K steps runs in
  the same order whatever the tile.
- The best settings: 16 experts per program, 2 warps and K steps of 512 at 1 row, 4 warps and 256 above it. The
  router went from 5.6 to 4.6 µs at 1 row and from 5.6 to 5.1 at 4.
- **Prediction (Claude; Riel gave none):** 2-3 µs per layer. It came to 0.5-0.7 µs in the block, A/B under the same
  conditions. A router that keeps its bits can't go much lower: each logit is 128 dependent tensor-core steps.
- The first try broke the engine tests. The kernel doesn't mask K, and the test model's rows (256) are shorter than
  a 512 step. The tile is now clamped to divide the row, and `test_moe_router.py` covers it (a mutant fails).
- **Side finding:** the whole block measured 3-4 µs faster at 1 row (and 11 at 4 rows) once Slack, Claude Desktop and
  ChatGPT were closed. The desktop's other GPU apps share the GPU. Compare runs only under the same conditions.

**R2 (gate/up 2-bit, down 3-bit, 13 GB) built on the 3090.** It took `quantize_experts.py` about 10 minutes from the
local bf16 checkpoint, peaking at 77 °C. Its per-tensor errors match the pod build within 5e-6.
- It serves the full 131K window in 21.2 GB, estimated 18.15 GiB of a 21.8 GiB budget. No need to close anything but
  the heaviest GPU apps.

Decode tok/s, 256 tokens, median of 3, with MTP drafts:

| Build | Prompt | Temperature | Old kernels | New kernels | Change |
|---|---|---|---|---|---|
| R2, 131K window | code | 0 | 366.4 | 425.5 | +16% |
| R2, 131K window | chat | 0 | 289.0 | 337.1 | +17% |
| R2, 131K window | code | 1.0 | 301.5 | 345.0 | +14% |
| R2, 131K window | chat | 1.0 | 254.3 | 293.8 | +16% |
| MLX 4-bit, 16K window | code | 0 | 352.6 | 391.8 | +11% |
| MLX 4-bit, 16K window | chat | 0 | 260.6 | 291.8 | +12% |

- R2 gains more than the 4-bit build: the low-bit expert kernels were further from their read floor under the old
  kernel.
- Old and new kernels give the same 4 × 1,024 greedy tokens on R2 as well. The texts differ from the 4090's R2 run
  within a few words, as expected: the 4090 prefills with its FP8 path, and the two quantizations differ by rounding
  noise.
- R2 on the 3090 now decodes faster than the 4-bit build did on the rented 4090 (345 vs 391 code at temperature 1,
  294 vs 312 chat, with 7% less bandwidth). It also holds 8× the context.

**The next lever is the verify step's expert kernels.** With drafts, each step verifies 4 rows. At 4 rows the 4/4
block reads about 33 experts' weights (55 MB) and takes 83 µs against a read floor of about 59. That's 24 µs per
layer, about 1 ms of each ~7.5 ms step. *(Overstated: that floor used the 936 GB/s spec. See the next section.)*

### Where a verify round goes, and the attention tail

**Profile.** `tensorfold/profile_step.py` runs the engine in-process and profiles one greedy 256-token generation with
the torch profiler, which sees the kernels inside CUDA graphs. R2 with drafts took 84 rounds, about 3 tokens a round
and ~8 ms of GPU time a round:

| Kind | Share of GPU time |
|---|---|
| Dense 4-bit linears (`qmm`, 211 launches a round) + their reduce | 35% + 4.4% |
| Routed and shared experts | 25.6% |
| Full-attention tail over the KV cache (`_tail`, 13 a round) | 9% |
| Draft tree | 4% |
| Router | 3.2% |
| Top-k + plan (`select`) | 2.3% |
| Norms | 3% |

**The floor was wrong.** The 3090 reads about 850-890 GB/s in practice (a plain sum over 1 GB reads at 887), not the
936 of the spec. Against that, the 4-bit expert kernels at 4 rows are already near their floor: a mutant that only
loads the weights and does no math takes 46.7 µs against the real kernel's 48.7. The lever named above is mostly gone
for the 4-bit build.

**R2's low-bit kernels do have room.** At 4 rows, gate/up (2-bit) takes 32.8 µs at 674 GB/s and down (3-bit) 23.5 at
648. Two things tried:
- **Deeper prefetch (3 stages instead of 2):** a noisy run showed 87 -> 77 µs. An interleaved in-process A/B showed no
  gain. The first number was clock drift.
- **Register bloat.** Low-bit gate/up kernels use 231 (2-bit) and 248 (3-bit) registers, because the shared expert
  (always 4-bit) runs inside the same launch through its own inlined path. Without that path they need 127 and 158.
  Moving it to a `__noinline__` function didn't help: the register budget is shared. A mutant with the shared path
  emptied (so ~3% less work) measured:

| Rows | gate/up, now | gate/up, no shared path | down, now | down, no shared path |
|---|---|---|---|---|
| 1 | 15.4 | 20.2 | 7.9 | 8.2 |
| 4 | 32.8 | 31.9 | 23.5 | 21.9 |
| 6 | 55.0 | 43.7 | 32.0 | 29.6 |

  So running the shared expert apart is worth ~2.5 µs a layer at the 4-row verify step (~0.1 ms a round), and much
  more at 6 rows. At 1 row it would lose.

**The attention tail skipped its dead tiles.** `_tail` walks one chunk of keys in 64-key tiles. It ran every tile of
the chunk even past the last valid key, where every key is masked and the tile adds nothing. It now stops at the last
valid tile. The masked tiles contributed exact zeros, so the bits don't change.
- Microbench: 54 -> 38 µs at 4 rows and 300 keys, 59 -> 35-38 µs at 1 row.
- 99 attention, expert and MoE tests pass.
- **End to end, R2, 131K window** (median of 3, 256 tokens):

| Prompt | Temperature | Before | After | Change |
|---|---|---|---|---|
| code | 0 | 425.5 | 443.5 | +4% |
| chat | 0 | 337.1 | 351.6 | +4% |
| code | 1.0 | 345.0 | 344.3 | noise |
| chat | 1.0 | 293.8 | 288.0 | noise |

- The 4 × 1,024 greedy texts are identical to the run before the change.
- **Prediction (Claude; Riel gave none):** 16 µs × 13 calls = ~0.2 ms of an 8 ms round, +2.5%. Greedy gained 4%.

**Levers left, by size per ~8 ms round:**
- The draft passes read the full 286 MB LM head three times a round. A draft that scores only a frequent-token subset
  of the vocabulary (as in FR-Spec) would save ~0.6-0.7 ms. Output stays exact because verification still uses the
  full head.
- The dense 4-bit linears run at ~71% of the practical floor. At 85% they'd save ~0.5 ms.
- R2's shared expert in its own concurrent launch: ~0.1 ms at 4 rows.

### Lever 1: a smaller draft vocabulary (no gain)

Riel asked for the three levers in order and gave no prediction. **Claude predicted +5-7% greedy.** The premise was
wrong: TensorFold's MTP head already scores a 79,591-token subset (`draft_vocab.txt`), not the full 248K vocabulary.
- **Cost of the head at 1 row:** 123 µs for the 79.6K subset. At smaller sizes it's 99 µs (64K), 74 (48K), 63 (32K),
  38 (24K), 31 (16K), and 337 for the full vocabulary. A round runs ~2.4 draft steps.
- **Ranking:** `tensorfold/draftvocab/coverage.py` ranks ids by frequency in Ornith's own terminal-bench traces (35B
  and 9B, reasoning, replies and tool arguments). It ranks on half the tasks and holds out the other half. Coverage of
  the held-out tokens is 92.6% at 16K, 95.9% at 32K, 97.8% at 48K, and 99.4% for the default list. A draft can only
  be right when its token is in the list.
- **A/B:** `draftvocab/ab.py` is in-process and interleaved. It uses R2, 6 prompts (4 benchmark and 2 held-out agent
  tasks), 512 tokens, 2 reps, and swaps heads with their own captured graphs:

| Head | Greedy tok/s | Tokens a round | ms a round | Temperature 1 tok/s |
|---|---|---|---|---|
| default 79.6K | 430.1 | 3.36 | 7.81 | 346.7 |
| 48K | 435.2 (+1.2%) | 3.29 | 7.55 | 342.6 (-1.2%) |
| 32K | 426.7 (-0.8%) | 3.22 | 7.54 | 348.4 (+0.5%) |
| 24K | 423.8 (-1.5%) | 3.19 | 7.54 | 348.9 (+0.6%) |
| 16K | 419.8 (-2.4%) | 3.13 | 7.45 | 336.6 (-2.9%) |

- A smaller head saves the expected 0.26-0.36 ms a round, but the tokens kept per round fall by the same share. The
  net is within ±1-2%, which is noise. Greedy outputs are identical for every head, as they must be.
- **Kept the default.** The head is ~4% of a round, so no cut can win much while it costs acceptance.
