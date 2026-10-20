# sm_86 speed on upstream TensorFold main: profile and ranked plan — 2026-10-20

Base: upstream `origin/main` d1018b31 (1.1.1 + one test fix), built `-Dsm=86 -Doptimize=ReleaseFast`, binaries kept at
`~/.cache/sm86-speed/base/`. Fork: the M3TiS #2 release binary (r1s-moe 0d167c4f). RTX 3090, one stream, 8K window.
Raw data: `prof/` (nsys reports and sqlite, `summary-*.txt`, serve logs with `TF_TIMING=1`, bench JSON).
Scripts: `scripts/serve_bench.sh`, `scripts/kern_top.py`, `scripts/rounds.py`, `scripts/per_launch.py`.

Status: GPU work paused at the coordinator's request (desktop apps crashing on GPU memory) after the three nsys runs
and two plain-step timings below. The ncu pass on the 27B's matmuls was killed before it wrote anything.

## What was measured

nsys runs: `bench_openai --tokens 128 --reps 2 --temperatures 0` (code + chat), `--cuda-graph-trace=node`, so the
tok/s here are under tracing (upstream R1s 165/130 vs 193/136 untraced).

| | R1s upstream | R1s fork | 27B upstream (DFlash2) |
| --- | --- | --- | --- |
| greedy tok/s code / chat (traced) | 165 / 130 | 471 / 305 | 155 / 110 |
| plain step (`tf-qwen35-cuda step`, 64 ids, untraced) | 8.3 ms | — | 25.9 ms |
| GPU kernel time, whole bench | 4648 ms | 1810 ms | 6126 ms |
| GPU busy fraction (bench window) | 0.86 | 0.81 | 0.93 |
| verify forward | one 735-node graph, 12.7 ms kernels | ten 51-53-node graphs, ~4.6 ms | one 1113-node graph, 27.4 ms |
| eager kernels a round | 110 (2.7 ms, mostly MTP) | ~80 (~1.2 ms) | 194 (DFlash hold 3.3 ms) |

Upstream already replays each verify shape as one captured CUDA graph (`cuda_engine.zig` `graphs`), and the GPU is
busy 86-93% of a round. So the 2.5x gap is not launches or host gaps: the same tokens cost 2.6x the kernel time.

Per R1s verify forward (4-row window), upstream vs fork kernel time:

| kernel group | upstream | fork | upstream kernel |
| --- | --- | --- | --- |
| routed experts (gate/up + down, 40 layers) | 6.19 ms | ~1.9 ms | `tf_experts3::expert_kernel` 3-bit, ~101 + 53 us a layer |
| dense projections (190 a forward) | 2.49 ms | ~1.3 ms | `tf_lane_gemv_split`, 13 us each |
| router logits (40) | 1.89 ms | ~0.28 ms | `tf_moe_router`, 47 us each for 4 rows |
| LM head | 0.36 ms | | `group_kernel` |
| everything else | ~1.8 ms | | GDN scans, attention, norms, topk, plan |

R1s eager per round (MTP drafter, 3 levels): head over the full 248K vocab 3 x 342 us, one-block argmax 3 x 75 us,
MTP attention 3 x 53 us, a host sync per level (`putIds` and `fetch`). `TF_TIMING`: absorb 1.0 ms + hold 1.5 ms +
verify ~10.5 ms a round.

27B: the verify forward's 27.4 ms is 11.4 ms `group_kernel` (sk = 1 shapes such as gate/up at 17408 columns, 272
CTAs: 3.3 waves) and 11.2 ms `tf_lane_gemv_split` (352 launches). A plain step is 25.9 ms against a ~15 ms floor for
reading the 4-bit weights once at 936 GB/s: the decode matmuls run at roughly 55-65% of bandwidth.

## Ranked pieces (expected gain on this card, evidence above)

1. **Routed-expert decode kernels (3- and 4-bit, `experts3.cu`/`experts.cu`)** — R1s and every affine qwen3_5_moe.
   6.2 of 12.7 ms a verify; the fork's kernels read the same bytes in ~1.9 ms. Upstream gets ~300 GB/s: one warp a
   32-column unit walks all K groups with two stages in flight. Exact route: the bits are fixed per group (`p` from
   zero by 4 mma steps, then `acc = fma(xs, b, fma(p, s, acc))` in group order), so groups can be computed by several
   warps or with deeper stages and the fma chain run in order afterwards. Expected −3 to −3.5 ms a verify
   (~25% of an R1s round, roughly 175 -> 230 tok/s). Does not help the 27B.
2. **Pre-sm_90 dense decode matmul (`lane_gemv_split` / `group_kernel` on sm_86)** — both models. 27B: verify
   27 ms and plain step 26 ms at ~60% of bandwidth; a kernel at ~85% would cut ~25% (plain ~39 -> ~50 tok/s, drafted
   similar). R1s: 2.5 -> ~1.3 ms a verify (~9%). Evidence of headroom: the fork's `qmm_seg` (TensorFold-prB
   `qmm_lane.cu`: weights as the mma's A operand, shared-input weights in one segmented launch) serves R1s's dense
   projections in about half the time, and gave +52% served on the 27B at 32K over the split-K base. Exactness: the
   bits are fixed by K-slice partials summed in slice order (`splitK` by shape); the swapped-operand form claims the
   same bits and must be shown byte-equal against `lane_gemv_split`/`group_kernel` on every shape before use.
3. **Decode router logits (`tf_moe_router`)** — MoE only. 1.9 ms a verify for 4 rows; a 16x16 tile wastes 15/16
   of its threads at one row and loads 2-byte words between barriers. One fma chain a (row, expert) in index order
   can stay exactly as is with one thread an expert and staged rows. Expected −1.5 ms a verify (~12%). Small, safe.
4. **MTP drafter on the device** — R1s only. ~2.5 ms a round eager: split argmax for draws (75 -> ~7 us x3),
   no host sync per level (device-side chain), optionally a draft-vocabulary head (342 -> ~110 us x3; needs a
   vocabulary file, a harder sell upstream). Expected −1 to −1.4 ms a round (~8-10%).
5. **Graph pieces / launch work** — upstream already graphs the verify forward; the fork's 4-layer pieces cut a
   ~110 us launch block. Expected ≤3%. Not worth a PR now.
6. **Weight pool (128 MiB blocks)** — upstream already places every weight in one arena (`Engine.arena`), so the
   2 MiB rounding waste the pool fixed does not exist here. Dropped.

## Plan

- Piece A (both models): item 2, the sm_86 dense decode matmul, branch `qwen35-sm86-qmm`.
- Piece B (largest R1s gain): item 1, the expert decode kernels, branch `qwen35-sm86-experts`.
- Item 3 (router) is the obvious next small PR; not started per the two-piece limit.

GPU-memory note for resumed runs: R1s serves in ~15 GiB and the 27B with DFlash2 in ~20 GiB at 8K; with nsys or ncu
attached the desktop apps (~1 GiB) were starved. Keep the 27B runs to `--context 4096`, avoid nsys on the 27B, and
check `nvidia-smi` headroom before each run.
