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
