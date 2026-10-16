## What this changes

3-bit routed experts for `qwen3_5_moe` on the CUDA engine, the width increment agreed on #803. It stacks on #856 (on #836): the last two commits are this pull request's.

- `cuda: 3-bit affine experts beside affine-4`: new `Affine3Experts` in `Grouped`, beside `Affine4Experts`. MLX 3-bit codes (groups of 64) are stored as bit planes: a 2-bit plane (a uint4 a lane), a 1-bit plane (a uint2 a lane), and the same scales and biases. That is 224 words a block against 288. The kernels rebuild the nibble words affine-4 reads, so the mma operands, sums and roundings are affine-4's on the same codes. The pack widens the checkpoint's 3-bit words to 4-bit, runs the existing affine-4 pack, then splits the planes. One expert index (the shared expert) keeps 4-bit tables. A unit or CTA is one expert's column block, so the branch is uniform. The prompt stages fit 48 KiB.
- `qwen3_5: 3-bit routed experts on the CUDA engine`: the geometry reads the quantization overrides.
  - Accepted: routers at 8 bits, and every layer's `switch_mlp` at 3 or 4 bits, one width throughout.
  - Refused (`UnsupportedQwenQuantization`): any other override that isn't the default 4/64, and any partial or mixed expert width.
  - A 3-bit checkpoint's experts pack 32 at a time through scratch in the existing stage. The shared expert and the MTP layer's experts stay 4-bit. 4-bit checkpoints load exactly as before.

## Proposed gate

On #803 I proposed per-projection fp64 references, an engine-vs-reference KL bar no worse than 4-bit's, and the RTX 3090 contracts. Here is what this branch measures for each.

1. **Each projection against fp64** (`tf-cuda-test experts3`): gate, up and down at 3 bits, decode and prompt checked separately.
   - Each is no further from its fp64 product than affine-4 on the same codes, and equal to it to the last digit.
   - Decode worst column: 1.1e-6 (gate), 1.2e-6 (up), 6.7e-7 (down).
   - Prompt worst column: 4.1e-2, 3.7e-2, 3.1e-2. That is the prompt form's bf16 output rounding, and affine-4 shows the same figure.
   - Decode and prompt rows over 3-bit experts plus a 4-bit shared expert are byte-equal to the all-affine-4 tables on the same codes.
2. **The engine against a reference of the same checkpoint.** The reference is the checkpoint with its 3-bit expert words rewritten as MLX 4-bit words (same codes, scales and biases), loaded through #856's affine-4 path.
   - `tf-qwen35-cuda` over 300 ids: `step`, `window` and `prompt` logits from the 3-bit path are byte-equal to the reference's.
   - So the engine's KL against any fp32 reference of this checkpoint is the affine-4 engine's on the same weights, which meets the bar by construction.
   - I can still run a teacher-forced fp32 comparison at 1K/8K/32K if you want it on top.
3. **Contracts on an RTX 3090**: below.

## Receipt

- Environment: RTX 3090 (24 GB, sm_86), Linux, driver 615.78, CUDA 13.4 `nvcc`, Zig 0.17.0, this branch at `967dd763`.
- Checkpoint: `CanadaApollo6/Ornith-1.5-35B-A3B-M3TiS-3090` at `82f19799` (Qwen3.5-MoE 35B-A3B, 3-bit routed experts, 4-bit elsewhere, an MTP layer). Two local adaptations, neither touching weights:
  - its `rope_parameters` spells `type` for `rope_type`, so the config was copied with the key renamed;
  - its `mtp-4bit.safetensors` names tensors `mtp.*`, so they were re-keyed to `language_model.mtp.*` (same bytes).
  - I'll fix both in the checkpoint rather than have the engine accept them.
- Exactness:
  - `bench_concurrent.py --tokens 128 --reps 2 --alone --serial` with `--drafter <the checkpoint>`, code and chat, T=1 seeded and T=0:
    - `--levels 1,2` at `--parallel 2 --context 8192`: 36 of 36 equal;
    - `--levels 1,4` at `--parallel 4 --context 4096`: 60 of 60 equal;
    - 0 unequal, 0 failed.
  - `tf-qwen35-cuda` over 300 ids: `window` and `tree` equal `step` byte for byte.
  - `bench_cache.py --require-cold` (2,571 tokens, drafter on): qualified, with repeated and concurrent first outputs equal.
- Decode speed: `bench_openai.py --tokens 128 --reps 3`, median tok/s, one stream, code / chat.

  | serve | T=1.0 | greedy |
  | --- | --- | --- |
  | `--parallel 2 --context 8192`, plain | 107 / 108 | 111 / 110 |
  | `--parallel 2 --context 8192`, MTP drafts | 164 / 132 | 176 / 134 |
  | `--parallel 4 --context 4096`, MTP drafts | 165 / 129 | 176 / 131 |

  - Aggregate from `bench_concurrent`: up to 306 steady at 2 streams and 524 at 4.
  - `tf-qwen35-cuda step`: 8.9 ms a step.
  - These match #856's 4-bit numbers on the same card. One-stream decode here is bound by launch and latency, not expert bytes.
  - What the width buys is memory: the model takes 14.87 GiB on the device against 18.62 GiB for the 4-bit checkpoint, leaving 8 streams at a 32K window.
- Prompt speed: cold `ttft_s` 1.21 s on 2,571 tokens, against 1.24 s for the 4-bit checkpoint on #856, same card.
- Tests:
  - `tf-cuda-test experts3`, `moe` and `glue` (1,242 of 1,242): all pass.
  - `zig build test`: 48 of 48 steps succeed. A new geometry test covers the 3-bit overrides and refuses a partial or 2-bit set.
  - `tools/zig/lean_check.py`: 527 problems on this branch and on #856, none added. `zig fmt --check`: clean.
- Not run:
  - Metal, since this touches only the CUDA engine.
  - Windows and sm_89, since I have neither. The kernels have no architecture-specific paths beyond affine-4's.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [x] No precision traded for speed. The 3-bit path's sums are affine-4's on the same codes, bit for bit.
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
