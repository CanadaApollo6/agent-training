## What this changes

`qwen3_5`'s decode projections that share one input now run as one launch on GPUs that take the split-K path (every CUDA GPU but GB10). Branch `qwen35-sm86-qmm`, one commit on `main` d1018b31.

- `lane_gemv.cu`: new `tf_lane_gemv_seg`. It is `tf_lane_gemv_split` over up to four projections of one input. Block b belongs to projection i when `first[i] <= b < first[i+1]`, and within it does exactly what split's block `b - first[i]` does: the same tile, the same K slice, the same staged loads and mma steps, and the same last-CTA sum of slice partials in slice order. Each projection keeps its own `splitK(n, k)`, so its bits are the ones `decode()` gives it alone. A projection whose `splitK` is 1 (the 27B's attention q, 12288 columns) stores its one partial straight out, which equals what the last CTA's in-order sum of a single partial stores. `tf_lane_gemv_split` itself compiles as before (the one-slice store is behind a template flag it does not set).
- `qlinear.zig`, `affine4_kernels.zig`: `Seg`/`Segs` (the launch's parameter block), `Split.initFor(d, parts)` (partials and tickets for four projections), and `Affine4Kernels.decodeMany`. On GB10 (`split == null`) `decodeMany` calls `decode()` per projection, so GB10's path does not change.
- `qwen3_5` engine: `projMany` / `projManySums` take a list of affine-4 projections of one input. They segment when every projection is affine-4 with the same K and the call is decode-sized (prompt rows over 16 keep the prompt GEMMs). Otherwise they fall back to `project()` per projection. Call sites, each a set of projections of one input:
  - Gated-delta in: qkv, z, a, b.
  - Attention in: q, k, v.
  - MTP attention in: q, k, v.
  - Draft-frame attention in: q, k, v.

  The MLP's gate and up are left alone: at 2 x 17408 columns they are already bandwidth-bound and one launch over both measured slower (below).
- One path change beyond sm_86: on every GPU that takes the split form, a projection whose `splitK` is 1 used to run `decode()`'s `cluster()` (qmm_group's cluster tile). Inside a segmented launch it now runs the lane kernel's one-slice store instead. The bits are the same (checked above and by the contract). I timed this only on the A10; on sm_80, sm_89, sm_90 and sm_120 its speed is untimed.

## Receipt

- Environment: NVIDIA A10 (24 GB, sm_86, 72 SMs), Linux, driver 570.148.08, CUDA 13.0 `nvcc`, Zig 0.17.0, built `-Dsm=86 -Doptimize=ReleaseFast`. Every number here is A10 (sm_86); base is `main` d1018b31 built the same way on the same machine.
- Checkpoints:
  - `TensorFold/Qwen3.8-27B-MLX-4bit` rev 22d8d538 with `z-lab/Qwen3.8-27B-DFlash2`.
  - `CanadaApollo6/Ornith-1.5-35B-A3B-M3TiS-3090` rev 66da2f4e (Qwen3.5-MoE 35B-A3B), drafted by its own MTP layer.
- Exactness:
  - `tf-cuda-test affine4-decode` (new; bits only, no timing). It runs eleven sets of projections at the 27B's and the 35B-A3B's shapes plus a short-K set, at 1, 2, 3, 4, 8 and 16 rows. It checks that:
    - `decode()` at each row count equals its 16-row call;
    - one segment equals `decode()`;
    - every set's segmented launch equals `decode()` per projection, byte for byte.

    PASS.
  - The kernel contract launches `tf_lane_gemv_seg` with two segments of the shape and checks the second, so its partials and tickets sit past the first's.
  - `tf-qwen35-cuda` over 64 ids, both models: `step`, `window`, `tree`, a second stream beside (`TF_Q35_BESIDE=64`), and a resumed state (`TF_Q35_PREFILL=32 TF_Q35_RESUME=1`).
    - Logits are byte-equal to base's in every mode.
    - On this branch, `window` and `tree` equal `step`, beside equals `step`, and resumed equals fresh.
  - Served output, base and this branch alternately on one A10, each a fresh server:
    - The 35B-A3B (MTP drafts, `--context 8192`) at `--parallel 2`: three rounds each. A round is `bench_openai` T=0 code and chat, then `bench_concurrent --tokens 128 --reps 2 --alone --serial --levels 1,2` at T=0 and seeded T=1, then `bench_cache`.
    - The 27B (DFlash2, `--context 4096`): two rounds each, of `bench_openai --reps 3` at T=0 and seeded T=1 on `--parallel 1`, then the same sequence at `--parallel 2`.
    - Every request's token hash, in order, is the same on this branch as on base in every round (28 requests a round on the 35B-A3B; 9 + 28 on the 27B).
    - Every `bench_concurrent` run reports alone 24 equal / 0 unequal and serial 12 / 0, on both binaries and both models. `bench_cache` qualifies on every run.
  - An earlier A10 machine gave unequal alone/serial cells on base as well as on this branch: base 16 / 8 alone on the 35B-A3B and 17 / 7 on the 27B. It also failed `bench_cache` on the 27B for both binaries, and its base T=1 hashes differed from this machine's base. None of that reproduces on this machine, for either binary. A device-side check of the split tickets after every launch (a debug build, not in this branch) found every ticket back at 0. The engine records on one stream, so two graphs never run at once.
- Decode speed: `bench_openai --tokens 128 --reps 3`, median tok/s, one stream, `--parallel 1`, code / chat.

  | model, serve | base T=0 | this T=0 | base T=1 | this T=1 |
  | --- | --- | --- | --- | --- |
  | 35B-A3B, MTP drafts, `--context 8192` | 152.5 / 120.3 | 164.8 / 126.3 | 138.4 / 113.9 | 141.3 / 118.0 |
  | 27B, DFlash2, `--context 4096` | 109.8 / 82.5 | 111.9 / 84.0 | 99.8 / 68.0 | 106.6 / 68.4 |

  The alternating runs above on the second machine, mean of the rounds' medians, code / chat:

  | model, serve | base T=0 | this T=0 | base T=1 | this T=1 |
  | --- | --- | --- | --- | --- |
  | 35B-A3B, `--parallel 2` | 152.0 / 114.7 | 156.4 / 116.9 | | |
  | 27B, `--parallel 1` | 102.8 / 77.2 | 105.7 / 79.5 | 96.8 / 62.8 | 99.3 / 64.3 |
  | 27B, `--parallel 2` | 98.9 / 75.2 | 102.2 / 77.5 | | |

  - `tf-qwen35-cuda step` (plain, 64 ids):
    - 35B-A3B: 8.9 -> 8.4 ms a step (112.5 -> 119.5 tok/s).
    - 27B: 32.9 -> 32.2 ms (30.4 -> 31.1 tok/s).
  - Per set, one captured graph of calls, 1 row, `decode()` per projection vs segmented, in µs (a local timing loop, not part of this branch):

    | set | `decode()` | segmented |
    | --- | --- | --- |
    | 27B gated-delta in | 104.7 | 92.1 |
    | 27B attention in | 98.6 | 76.7 |
    | 35B-A3B gated-delta in | 47.0 | 32.0 |
    | 35B-A3B attention in | 33.4 | 23.5 |
    | 27B MLP in (not used) | 182.7 | 189.9 |
- Prompt speed: prompt rows over 16 take the same calls as before. Served `ttft_s` medians:

  | prompt | 35B-A3B base | 35B-A3B this | 27B base | 27B this |
  | --- | --- | --- | --- | --- |
  | code | 0.0199 | 0.0191 | 0.0423 | 0.0414 |
  | chat | 0.0458 | 0.0462 | 0.1646 | 0.1663 |

  Every pair is within 1%, which is run-to-run noise.
- Kernel contract: `tf_lane_gemv_seg` is registered as "lane_gemv segmented split-K", replacing `decode()`. Its served shapes are the 27B's and 35B-A3B's sets above. `zig build kernel-contract` on the A10:
    - `PASS` on nonfinite, repeat, shapes ("14 shapes bit-equal to the kernel it replaces (worst 0.000 of the bound)") and sums.
    - Live limits: 56 registers, 128 + 17664 shared bytes, record current.
    - `SUMMARY 76 kernels: 76 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 932 limit problems`. The limit problems are all records of kernels this branch does not touch: nvcc 13.0.88 here compiles them differently from the registry (base d1018b31 reports the same 932, with 75 kernels).
  - `kernel-contract-limits` on `-Dsm=80,86,89,120,121` fatbins:
    - The `tf_lane_gemv_seg` record is `62 / 56 / 56 / 54 / 54` registers (sm_80/86/89/120/121), 128 static bytes, bound 128.
    - `tf_lane_gemv_split`'s record is unchanged and current: 62 / 56 / 56 / 44 / 44.
- Tests:
  - `zig build test native -Dsm=86`: 2960 of 2974 tests pass, 14 skipped. On the same machine base d1018b31 and `qwen35-sm86-experts` each fail one test, `cuda_cost_identity`'s "a changed payload at the same path invalidates it". It passed in this branch's run, so it is flaky on this machine rather than fixed here.
  - `tf-cuda-test affine4-decode`: passes.
  - `tools/zig/lean_check.py`: 444 problems on this branch and on base, none added. `zig fmt --check`: clean.
- Not run:
  - nvcc 13.2's record set: I have no 13.2 here.
  - Metal: this touches only the CUDA engine.
  - GB10, sm_89, sm_120 and an RTX 3090. GB10's path is unchanged by construction (`split == null`).

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals plain (`window`/`tree` == `step`), a resumed state equals a fresh one, and concurrent equals solo, at the engine level and served (`bench_concurrent` alone and serial all equal, three rounds against base).
- [x] No precision traded for speed: per-projection partition, slice partials and slice-order sums unchanged, checked byte for byte.
- [x] Prompt processing is no slower: same calls past 16 rows; TTFT within noise.
- [x] The description says which platforms I ran (CUDA sm_86 on an A10) and which I could not.
- [ ] New tests fail before the change: not applicable. `affine4-decode` calls `decodeMany`, which this change adds, so it does not build on base. It passes after the change.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
