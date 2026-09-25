# Decode gap: how far below the hardware ceiling does ordinary PyTorch decode run?

Qwen3.5-0.8B, batch 1, greedy, 256 new tokens after a 23-token prompt, on the RTX 3090.

## The ceiling

Decode reads every weight once per token, so `tok/s ≤ bandwidth ÷ bytes read per token`:

| bytes read per token | MB |
|---|---|
| text weights (embedding table counted once: it is tied to the LM head, which reads all of it) | 1,505 |
| Gated DeltaNet state: 18 layers × 16 heads × 128×128 fp32, read and written | 37.7 |
| KV cache: 6 full-attention layers, ~150 positions | 1.9 |
| **total** | **1,544** |

At the measured 829 GB/s that is **537 tok/s, or 1.86 ms per token**. The embedding table alone is 248,320 × 1,024
= 254M parameters, a third of the weights; the vision tower and MTP head are never read in text decode.

## Rungs

| rung | tok/s | ms/token | % of ceiling |
|---|---|---|---|
| HF transformers, DeltaNet as a PyTorch loop (`bench_hf.py --variant torch`) | 78.0 | 12.8 | 14.5% |
| HF transformers + flash-linear-attention kernels (`--variant fla`) | 76.1 | 13.1 | 14.2% |
| CUDA graphs, PyTorch DeltaNet (`bench_graphs.py --variant torch`) | 180.9 | 5.53 | 33.7% |
| CUDA graphs + fla kernels (`bench_graphs.py --variant fla`) | 212.4 | 4.71 | 39.6% |
| vLLM 0.30, `enforce_eager` (fused kernels, no CUDA graphs) | 93.3 | 10.72 | 17.4% |
| vLLM 0.30, default (fused kernels + CUDA graphs + torch.compile) | 366.1 | 2.73 | 68.2% |

## What the profiler says (`profile_hf.py`)

Per decode step (the profiler itself adds wall time, so read the ratios, not the absolute wall clock):

| | torch | fla |
|---|---|---|
| kernel launches | 2,021 | 1,535 |
| GPU busy | 5.9 ms | 5.1 ms |
| GPU busy as a share of wall clock | 31% | 31% |

Two separate losses, in order of size:

1. **The GPU is idle about two thirds of the time.** Python and PyTorch's dispatcher spend ~10 µs of CPU per kernel
   launch; the kernels finish faster than the CPU can issue them. This is launch overhead, and it is why swapping in
   fla's fused DeltaNet kernels changed nothing: it removed ~490 launches out of ~2,000, and the remaining ~1,500 still
   leave the GPU waiting.
2. **Even the busy time is 2.7× the ideal 1.86 ms.** Most launches are tiny elementwise kernels (copies, adds,
   `rsqrt`, `pow`, `mean`: RMSNorm alone is several unfused kernels, ~79 `mean` reductions per step). Each has a fixed
   cost of a few microseconds and moves almost no data, so the busy time is not spent streaming weights.

CUDA graphs attack loss 1 (record the launches once, replay them with one call). Only fusion attacks loss 2:
`torch.compile`, an inference engine's hand-written kernels, or, at the limit, one megakernel.

## CUDA graphs: the prediction and the result

Model: time per token ≈ max(CPU time to issue the launches, GPU time to run them). Eager decode is CPU-bound (the GPU
is busy 5.1 ms of each ~13 ms token). A graph replays all ~1,500 launches with one call, so the CPU term vanishes and
the prediction is the profiled GPU busy time: **5.1 ms → ~196 tok/s** (fla), 5.9 ms → ~170 tok/s (torch).

Measured: **4.71 ms (212 tok/s) and 5.53 ms (181 tok/s)**, slightly better than predicted because the profiler
inflates kernel times a little. The graph changed nothing on the GPU; it only removed the gaps between kernels.

Two lessons:

- **Reason in milliseconds per token, not tokens per second.** Times add, rates don't: 78 → 537 tok/s is 12.8 ms →
  1.86 ms, and each fix removes milliseconds from that sum.
- **An optimization is invisible until its part is on the critical path.** fla's fused kernels gained nothing in eager
  mode because the GPU was waiting on the CPU anyway. With launch overhead gone, the same kernels are worth 0.8 ms
  per token (5.53 → 4.71 ms).

The graph and eager runs agree on the first 238 (torch) and 111 (fla) tokens, then diverge. The static cache runs
full attention over a fixed-size buffer with a mask, a different bf16 summation order from the dynamic cache; a
near-tie between two tokens eventually flips the argmax. A logic bug would diverge at the first token.

What remains is loss 2: 4.71 ms of GPU time against 1.86 ms of necessary DRAM traffic, spread over ~1,500 small
kernels. Only fusion removes it.

## Where the remaining GPU time goes (`profile_hf.py --variant fla`, per decode step)

| category | kernels | GPU µs | µs/kernel | share |
|---|---|---|---|---|
| matmul/matvec (reads the weights) | 193 | 2,557 | 13.2 | 50% |
| elementwise (add, mul, rsqrt, ...) | 825 | 1,368 | 1.7 | 27% |
| copies and casts | 414 | 878 | 2.1 | 17% |
| reductions (mean, sum, argmax) | 84 | 254 | 3.0 | 5% |
| Triton / other (fla DeltaNet, conv) | 19 | 72 | 3.8 | 1% |

(Profiled total 5.13 ms; the unprofiled graph replay takes 4.71 ms, so scale by ~0.92.)

Two buckets with different physics:

- **Weight streaming, 2.56 ms.** The matvecs read 1,505 MB, which at 829 GB/s would take 1.82 ms, so they run at
  ~590 GB/s, 71% of achievable bandwidth. Fusion cannot remove these bytes; only better matvec kernels close
  the last 0.7 ms.
- **Small-kernel overhead, 2.57 ms.** 1,342 kernels averaging 1.9 µs that move almost no data. Fusion removes most
  of this, but a fused model still launches some kernels per layer, each with a fixed cost of a few µs.

## What makes a launch "tiny" or "big" (`kernel_size.py`)

A kernel is fixed code; each *launch* runs it on some amount of data. One copy kernel, launched back to back from a
CUDA graph (no CPU overhead), cycling through 96 MB of buffers so nothing is served from the 6 MB L2 cache:

| bytes read | µs/launch | GB/s | fixed-cost share | example launch in decode |
|---|---|---|---|---|
| 2 KB | 1.5 | 3 | 100% | RMSNorm or residual add on one hidden vector |
| 33 KB | 1.6 | 42 | 96% | DeltaNet `in_proj_a` / `in_proj_b` weight (16 × 1,024) |
| 1 MB | 3.8 | 551 | 39% | k or v projection in a full-attention layer (512 × 1,024) |
| 4 MB | 11.3 | 740 | 13% | DeltaNet `out_proj` or `in_proj_z` (1,024 × 2,048) |
| 8 MB | 21.5 | 779 | 7% | one MLP weight matrix (3,584 × 1,024 = 7.3 MB) |
| 12.6 MB | 31.6 | 796 | 5% | DeltaNet `in_proj_qkv`, already merged in the checkpoint (6,144 × 1,024) |
| 67 MB | 163 | 825 | 1% | |
| 533 MB | 1,302 | 818 | 0% | LM head (whole embedding table) |

`t ≈ 1.5 µs + bytes ÷ ~820 GB/s`, with a ramp in between: below ~8 MB a launch doesn't keep enough memory requests in
flight to reach full bandwidth. Most per-layer weight matrices here are 4–12.6 MB, partly inside that ramp, and the
tiny `in_proj_a/b` and k/v matvecs are mostly fixed cost; together with cuBLAS's matvec kernel itself, that is why
the matvecs average ~590 GB/s rather than ~820. The first run of this script, without buffer rotation,
reported 1,835 GB/s at 2 MB: the data was coming from L2, not DRAM.

## Where the 1,505 MB of weights are

| component | MB | share |
|---|---|---|
| embedding table (tied: read in full by the LM head) | 508.6 | 33.8% |
| MLPs, 18 DeltaNet layers | 396.4 | 26.3% |
| DeltaNet mixing projections, 18 layers | 379.6 | 25.2% |
| MLPs, 6 full-attention layers | 132.1 | 8.8% |
| attention projections, 6 full-attention layers | 88.1 | 5.9% |
| norms | 0.1 | 0.0% |

Every layer has an MLP of three 3,584 × 1,024 matrices (22 MB); a DeltaNet layer adds ~21 MB of projections, a
full-attention layer ~14.7 MB.

## vLLM: the predictions and the result

Predictions, starting from the CUDA-graph rung (4.71 ms) and the category breakdown:

- Riel: weight reads ~2.0 ms (bigger merged launches, ~90% efficient) + ~1.03 ms of remaining small kernels
  (275 elementwise, 212 copies, 14 reductions at the measured µs/kernel) = **~3.0 ms, ~330 tok/s**. A first draft
  scaled the weight reads by launch count to 780 µs and totalled 1.81 ms, below the 1.86 ms floor: impossible,
  because the 1,505 MB must be read however the launches are grouped.
- Claude: ~2.0 ms + ~0.6 ms (more aggressive fusion, almost no leftover copies) = **~2.6 ms, ~385 tok/s**.

Measured: **2.73 ms, 366 tok/s, 68% of the ceiling.** Without CUDA graphs the same fused engine runs at 10.72 ms,
slower than our HF + graphs rung: fusion cut launches, but the ones left are still issued one by one from Python.
Both fixes are needed, and launch overhead was the larger of the two.

The remaining 0.87 ms is the next question: profile vLLM's decode step the way we profiled HF's.
