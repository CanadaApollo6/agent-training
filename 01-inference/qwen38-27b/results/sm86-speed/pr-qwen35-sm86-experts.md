## What this changes

Decode rows of the 3-bit routed experts (`experts3`) get a second form of the SwiGLU pair kernel for calls with few column blocks. Today a call with one or two token rows leaves most of the GPU idle. Branch `qwen35-sm86-experts`, one commit on `main` d1018b31.

- `experts3.cu`: new `split_kernel<2, 2, 2, 4>`, the pair form of `expert_kernel` (M = 2, SwiGLU epilogue).
  - One CTA takes one unit (an expert's 32-column block), where `expert_kernel` gives each of its four warps a unit of its own.
  - Its four warps share the unit's K groups. Warp w loads and multiplies groups w, w + 4, ..., with two stages in flight, just as `expert_kernel` stages them.
  - For each group a warp computes the same terms `compute_stage` computes: `p` from zero by the same four mma k16 steps, the group's scale and bias words, and the rows' input sums. It parks them in shared memory.
  - After each round of four groups, warp j runs tile j's chain `acc = fma(xs, b, fma(p, s, acc))` over those groups in group order. That is the same chain, in the same order, from the same terms, so the sums are the same bits.
  - The epilogue is `expert_kernel`'s SwiGLU store for one tile.
  - `expert_kernel`, its down form and the prompt kernels do not change.
- `experts3.zig`: `run()` takes the split form for the pair when the call has fewer units than 7/8 of four times the warps the pair kernel keeps resident. Otherwise, and always for down, it launches `expert_kernel` as before. The bound is fixed per GPU at `resolve()`. Both forms give the same bits, so the choice affects only speed.
- Registry: "experts3 decode split" (128 threads, 25600 static shared bytes).
- Where it runs: the choice is not gated by architecture, so every CUDA GPU takes the split form under its own bound (`experts3.zig` `run()`). The bound scales with SM count. On the A10 (72 SMs) it covers 1 to 3 token rows. By the same count it covers about 3 rows on a 3090 (82 SMs) and 2 on GB10 (48 SMs). It covers about 6 rows on a 4090 or an H100 and about 8 on a 5090, which takes in a drafted 4-row verify window there. The bits are the same either way. I timed it only on the A10: on GB10, H100, 4090, 5090 and 3090 the speed of this form is untimed.

## Receipt

- Environment: NVIDIA A10 (24 GB, sm_86, 72 SMs), Linux, driver 570.148.08, CUDA 13.0 `nvcc`, Zig 0.17.0, built `-Dsm=86 -Doptimize=ReleaseFast`. Numbers are A10 (sm_86) unless marked RTX 3090. Base is `main` d1018b31, built the same way on the same machine.
- Checkpoint: `CanadaApollo6/Ornith-1.5-35B-A3B-M3TiS-3090` rev 66da2f4e (Qwen3.5-MoE 35B-A3B with 3-bit routed experts and a 4-bit shared expert), drafted by its own MTP layer.
- Exactness:
  - `tf-cuda-test experts3` runs `sameBits` with the split form forced off and forced on. The shapes are the existing one (128 by 256) and two new ones. One is wide (512 by 2048, 16 experts, 4 rows, 9 picks a row, the shared expert in the last slot). The other is short (128 by 320): its 5 K groups leave the split form's last round with one group for four warps. The pairs' SwiGLU rows and the down sums are byte-equal to affine-4 on the same codes in both forms. PASS.
  - `tf-qwen35-cuda` over 64 ids: `step`, `window`, `tree`, a second stream beside (`TF_Q35_BESIDE=64`) and a resumed state (`TF_Q35_PREFILL=32 TF_Q35_RESUME=1`).
    - Logits are byte-equal to base's in every mode.
    - On this branch, `window` and `tree` equal `step`, beside equals `step`, and resumed equals fresh.
  - `bench_concurrent --tokens 128 --reps 1 --alone --serial --levels 1,2` at `--parallel 2` with MTP drafts, code and chat, T=0 and seeded T=1: 24 of 24 alone and 12 of 12 serial equal, 0 failed. This is a single run.
    - An earlier run on another A10 machine served base's binary by mistake and gave 6 unequal chat cells. That was base's own nondeterminism on that machine. It also gave base unequal cells in the separate `qwen35-sm86-qmm` runs, and none of it reproduces on this machine.
  - `bench_cache` (2 streams, 2 repeats): qualified.
- Kernel time: `tf-cuda-test experts3 200` times the wide shape over 64 experts. Each call gets fresh picks and the calls run as one captured graph. Times are µs a call.

  | rows | up, `expert_kernel` | up, split | taken | down (unchanged) |
  | --- | --- | --- | --- | --- |
  | 1 | 46.2 | 22.5 | split | 11.6 |
  | 2 | 50.5 | 42.2 | split | 23.6 |
  | 3 | 67.5 | 60.1 | split | 32.6 |
  | 4 | 70.6 | 75.5 | `expert_kernel` | 40.3 |

  These are the `TIME` lines of one run on the final commit. An earlier run on another A10 gave 45.9 / 21.2, 49.2 / 40.1, 64.6 / 57.5 and 68.2 / 75.6.

- RTX 3090 (sm_86, 24 GB, 82 SMs, driver 615.78.08, CUDA 13.4 `nvcc`, the same build flags), base and this branch alternately, each a fresh server with `TENSORFOLD_MEMORY_RESERVE_GIB=4 --prompt-cache-gib 2`, `--context 8192 --parallel 2`:
  - A round is `bench_openai --reps 3` at T=0 and seeded T=1, then `bench_concurrent --tokens 128 --reps 2 --alone --serial --levels 1,2` at T=0 and seeded T=1, then `bench_cache`. Plain: two rounds each. MTP drafts: one round of this branch against three of base.
  - Every request's token hash (34 a round) is the same on this branch as on base, plain and drafted, and plain equals drafted.
  - Every `bench_concurrent` run reports alone 24 / 0 and serial 12 / 0, and `bench_cache` qualifies on every run.
  - Decode, median tok/s, one stream, code / chat (mean of the rounds):

    | serve | base T=0 | this T=0 | base T=1 | this T=1 |
    | --- | --- | --- | --- | --- |
    | plain | 121.8 / 118.9 | 137.0 / 134.5 | 113.6 / 114.1 | 127.3 / 128.3 |
    | MTP drafts | 191.8 / 146.0 | 188.1 / 146.9 | 168.7 / 137.3 | 172.2 / 137.3 |

    With drafts the change is within base's own spread (185.9 to 194.8 code at T=0 over its three rounds): the verify windows are 4 rows, which keep `expert_kernel` as on the A10.
- Decode speed: `bench_openai --tokens 128 --reps 3`, median tok/s, one stream, `--parallel 1 --context 8192`, code / chat. Base and this branch were run back to back on the same machine.

  | serve | base T=0 | this T=0 | base T=1 | this T=1 |
  | --- | --- | --- | --- | --- |
  | plain | 111.7 / 109.8 | 125.1 / 123.1 | 104.7 / 105.4 | 116.5 / 117.6 |
  | MTP drafts | 152.7 / 120.6 | 160.4 / 121.0 | 137.3 / 111.7 | 137.8 / 111.9 |

  - `tf-qwen35-cuda step` (plain, 64 ids): 8.9 -> 7.9 ms a step (112.5 -> 126.8 tok/s).
  - Plain decode gains about 12%: each step calls the pair kernel with one row. A drafted verify window is 4 rows, which stays on `expert_kernel`, so drafted decode gains only where rounds call the experts with fewer rows.
  - The 27B has no routed experts and is not affected.
- Prompt speed: prompt rows take `experts3_prefill`, which this does not touch.
- Kernel contract:
  - `kernel-contract-limits` on `-Dsm=80,86,89,120,121` fatbins:
    - The "experts3 decode split" record is `254 / 255 / 255 / 255 / 255` registers (sm_80/86/89/120/121), 25600 static shared bytes, bound 128.
    - `cuobjdump --dump-resource-usage experts3.fatbin`, `split_kernel<2, 2, 2, 4>`:

      | arch | REG | STACK | LOCAL | SHARED |
      | --- | --- | --- | --- | --- |
      | sm_80 | 254 | 0 | 0 | 25600 |
      | sm_86 | 255 | 0 | 0 | 25600 |
      | sm_89 | 255 | 0 | 0 | 25600 |
      | sm_120 | 255 | 0 | 0 | 26624 |
      | sm_121 | 255 | 0 | 0 | 26624 |

      No stack and no local memory on any of the five, so nothing spills at 254/255 registers. On sm_120/121 cuobjdump reports 26624 shared bytes, while the contract's limits check reads 25600 static bytes there; I have not traced the 1024-byte difference. For comparison, the unchanged pair `expert_kernel<2, 2, 2, 4>` shows 255 registers and a 48-byte stack on sm_80 in the same dump.
    - The pair and down records are unchanged and current.
    - The problem count is 877 on this branch and on base. All of them are records of kernels this branch does not touch, compiled differently by nvcc 13.0.88 here.
  - `zig build kernel-contract` on the A10:
    - Live limits: "experts3 decode split", 255 registers, 25600 + 0 shared bytes, record current.
    - `SUMMARY 75 kernels: 75 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 932 limit problems`. Base d1018b31 gives the same line.
    - The experts kernels are limit-checked only. They have no gate 1-4 entry, on base or here.
- Tests:
  - `tf-cuda-test experts3`: passes.
  - `zig build test native -Dsm=86`: 2958 of 2973 tests pass, with 14 skipped and 1 failure. The failure is `cuda_cost_identity`'s "a changed payload at the same path invalidates it". Base d1018b31 fails the same test in `zig build test native -Dsm=86` on this machine (2958 of 2973), so the failure does not come from this branch. It passed once in a `qwen35-sm86-qmm` run on the same machine, so it is flaky here.
  - `tools/zig/lean_check.py`: 444 problems on this branch and on base, none added. `zig fmt --check`: clean.
- Not run:
  - The 4-bit routed experts (`experts.cu`). The same split applies there, but this PR keeps to one kernel.
  - nvcc 13.2's record set: I have no 13.2 here.
  - GB10, sm_89 and sm_120.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals plain (`window`/`tree` == `step`), a resumed state equals a fresh one, and concurrent equals solo, at the engine level and in one served `bench_concurrent` run.
- [x] No precision traded for speed: same terms and the same fma chain in group order, checked byte for byte against affine-4 in both forms.
- [x] Prompt processing is no slower: the prompt kernels are untouched.
- [x] The description says which platforms I ran (CUDA sm_86 on an A10 and an RTX 3090) and which I could not.
- [ ] New tests fail before the change: not applicable. The test forces each form through `split_below`, which this change adds, so it does not build on base. It passes after the change.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
