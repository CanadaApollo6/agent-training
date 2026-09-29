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
- **Not yet settled:**
  - The builds differ. The MLX 4-bit build is plain round-to-nearest, 6 GiB bigger than build A.
  - Its reasoning probe is running.
  - Time to answer should favor it further if it keeps near-Q8 reasoning length. Build A reasons 40% longer.

### The catch: context

On the A10's 22 GiB, TensorFold ran out of memory on a 30K-token prompt:
- The weights take 18.9 GiB. TensorFold's working memory (KV growth copies, the MTP head's own cache, state
  checkpoints, prefill buffers) takes about 2.4 GiB more.
- Its startup estimate is conservative for a discrete card: 4 GiB held back, a full-vocabulary logits bound, and 4
  KV copies. By its own estimate it wouldn't start at all, so `TENSORFOLD_CUDA_RESERVE_GIB` overrides the reserve.
- The prompt cache defaults to an eighth of system RAM. That's fine on GB10's unified memory but not on a 24 GB card.
  Run with `--prompt-cache-gib 0`.

16K works. The 3090 has about 1.5 GiB more than the A10, which isn't enough for agent-length context.

Two ways out:
- Give TensorFold low-bit experts. Build A's trick applied to TensorFold would save about 8 GiB, but its grouped
  expert kernels read 4-bit only.
- Trim its working memory for 24 GB cards.
