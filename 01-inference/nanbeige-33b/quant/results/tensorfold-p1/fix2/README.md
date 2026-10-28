# Llama fix round 2

Only findings 2 and 5 changed. The input commit is `c4af57935cc5fb8f24252f1ae9ce298c3b9d335b` on `nanbeige`.
The final local commit is `fd409155c09fadf5bea00232fe59deec621ead11`. Source, reference, binary and receipt hashes are in `manifest.json`.
No push, issue or pull request was created. Checkpoint files remain read-only.
No process outside this task was signaled. The production forward, loader and native adapter are unchanged.

The target is Nanbeige4.3-33B, EXL3 mul1 3.5 bpw, one RTX 3090 sm_86.
The engine uses FP32 residuals, FP16 RMS outputs and BF16 KV. The trusted references use exllamav3 1.6.5.
This round changes test layout, validation and mutation controls. It does not change inference arithmetic.

## Finding 2

`zig/tests/cuda/llama_tests.zig:19` adds a test-only cache layout to `Rig`.
It loads two real checkpoint layers, then expands the layer descriptors to 52 entries that borrow those weights.
The weight buffers still have one owner. `Model.init` allocates the production 52-layer cache and offset table.
An additional memory admission check accounts for the expanded cache before allocation.

`zig/tests/cuda/llama_boundary.zig:13` uses the full layout and executes weight layers 0 and 1 in cache slots 50 and 51.
It reads back the two late offset pairs and checks both against the real K and V allocation bases.
The normal forward stores, QK kernel, attention offset lookup and head execute on all 16,384 tokens.
Both final layer slots therefore reach the final K and V cache rows. Total KV is 3,489,660,928 bytes, or 3.25 GiB.
The oracle is the unchanged two-layer reference from `fix1/boundary`, whose files are hashed in the manifest.
This covers full-model cache addressing, not 52 layers of model arithmetic.
No smaller-context fallback was necessary.

`boundary.log` exits 0 and reports:

```text
SUMMARY llama boundary heads=56/8 weight_layers=2 cache_layers=52 slots=50,51 kv_bytes=3489660928 context=16384 positions=511,512,2047,2048,16383 refused=16384 fresh_kl=1.1330218638119827e-4 fresh_repeat=bit_equal device_peak=4944083460
```

All five logit comparisons have matching top tokens and KL below 0.000084.
The run takes 58.60 seconds, peaks at 4,984 MiB process VRAM and leaves at least 2,932 MiB free.
Its guard estimate is 5,200 MiB plus the 2,048 MiB reserve.

## Finding 5

`zig/src/families/llama/validation.zig:3,36` compares each position's hidden vector with its own reference scale.
`zig/tests/cuda/llama_tests.zig:157` checks all positions in each layer before accepting the layer.
The bounds are:

- Relative L2 at each position is at most `0.03`.
- Maximum absolute error at each position is at most `0.0001 + 0.03 * max(abs(reference_position))`.
- Every input value must be finite. Empty or misaligned comparisons fail.

RMSE remains a diagnostic. It no longer supplies a whole-sequence acceptance criterion.
A synthetic host regression shows that a large residual at another position can hide corruption under whole-sequence normalization.
The per-position validator rejects that case.

The final clean `llama-layers` run passes all 12 layers at all 217 positions across sequences of length 1, 6, 15 and 195.
That is 2,604 layer-position comparisons, including every position of the longest sequence.
Both worst metrics occur at sequence 2, layer 2, position 3:

| Metric | Observed worst | Bound | Bound above observed worst |
| --- | --- | --- | --- |
| Per-position relative L2 | 0.0218219981 | 0.03 | 37.48% |
| Per-position max error / reference peak | 0.0229484354 | 0.03 plus the absolute floor | 30.73% |

The worst maximum-error bound utilization is 0.7649327.
`summary.json` computes these margins from the saved final-engine residuals, rather than from an archived clean run.
`layers12.log` exits 0 and reports:

```text
SUMMARY llama layers=12 positions=217 mean_kl=5.8298328377966356e-5 max_kl=5.630279789195827e-4 top1=214/217 kl_above_0.01=0 device_peak=4054395204 pinned=2392850496
```

The unchanged logit check remains separate from the layer assertions. No logit KL exceeds 0.01.

### Mutation controls

`zig/tests/cuda/llama_layer_controls.zig:9` implements controls solely in the test runner.
Each control runs the real 12-layer engine on all four sequences and compares layer 11 at every position.
`rope_theta` uploads inverse frequencies for theta `10000` only while layer 11 executes, then restores the original frequencies.
`norm_plus_one` copies layer 11's input RMS weight vector into a test-owned device buffer and adds one to every weight.
`swapped_kv` exchanges only layer 11's K and V projection descriptors. Their dimensions match.
`zeros` zeroes the reference vectors in test memory. It does not change the checkpoint.
The `clean` control runs the identical test loop without any mutation and passes.

Every mutation returns exit 1 with `FAIL llama-layer-control: LlamaLayerDrift` after recording all 217 metrics.
These are assertion failures, not setup errors. `control-*.jsonl` records each position and its scale.
The maximum-error fraction column is diagnostic for the zero reference; its actual bound is the absolute floor.

| Control | Worst relative L2 | Worst max error / position peak | Bound broken | Failed positions |
| --- | --- | --- | --- | --- |
| Zero reference | 3.28597224e19 | 3.24532871e34 | L2 > 0.03 and max 32453.2871 > 0.0001 | 217 |
| Layer-11 theta 1e4 | 0.0808181193 at sequence 3, position 149 | 0.0258311906 at position 172 | L2 > 0.03 | 103 |
| Layer-11 input norm 1+w | 0.617570055 at sequence 3, position 38 | 0.264186396 | L2 > 0.03 and position-scaled max | 208 |
| Layer-11 swapped K/V | 1.088161875 at sequence 3, position 190 | 0.328935582 | L2 > 0.03 and position-scaled max | 209 |

The old whole-sequence formula remains a test-only diagnostic in the control runner.
The final RoPE control prints `legacy_failures=0`, reproducing the original miss on all four sequences.
The new assertion prints `failed_positions=103` and fails. Initial `calibrate-*` files are diagnostic history.
All final evidence is in `boundary.log`, `layers12.log`, `control-*.log`, their VRAM receipts and `summary.json`.

## Reproduce

From the TensorFold worktree, use Zig 0.17.0 and the sm_86 images already built in round 1:

```sh
TF_RESULTS=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p1
TF_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
TF_ZIG=/home/riels/.local/share/zig-x86_64-linux-0.17.0/zig
"$TF_ZIG" build tf-llama-test test-llama-host -Dfatbins="$PWD/zig-out/fatbin" -Doptimize=ReleaseFast --prefix "$TF_RESULTS/fix2/install" --summary all
/home/riels/exl3-local/bin/python tools/zig/llama_fix2_verify.py "$TF_MODEL" "$TF_RESULTS" "$TF_RESULTS/fix2" --binary "$TF_RESULTS/fix2/install/bin/tf-llama-test"
```

The verification script runs every device command through `llama_gpu_run.py`.
It expects the four mutation assertion failures and rejects setup failures or unexpected passes.
It checks the measured reserve and peak estimate in each receipt.
The layer estimate is 4,600 MiB. The measured layer process peak is 4,170 MiB.
It also attempts full-model admission and runs logits, greedy and the five speed pairs if admitted.
The baseline speed binary remains `fix1/tf-llama-before`, with only `forward.zig` restored to `fe0bd8fb` as documented in fix1.

## Other verification and remaining limits

`validation-host.log` freshly passes 2/2 host tests, including the new residual-scale regression.
`host-focused.log` passes the Llama host build and its cached test result.
`host-all.log` runs `zig build test test-kernel-contract test-golden fuzz-cortheon -j4 -Dfatbins=...`.
It reports 457/460 build steps succeeded and 844/848 tests passed, with 3 skipped and 1 failed; other tests are cached.
The sole failing test is `exl3_load.test.a scan that fails an allocation frees every group prefix it held` with `NondeterministicMemoryUsage`.
That source is byte-identical to `31ce9eb3`. Fix1's `exl3-isolated-base.log` already reproduces this failure on that baseline.
A fresh baseline execution is NOT RUN in this round. No unrelated fix is included.
`test-kernel-contract`, `test-golden` and `fuzz-cortheon` succeed in that same invocation.
The plain build passes all 24/24 steps in `build-plain.log`.
`lean-new.txt` reports `426 problems, 426 on the base, 0 new` from a fresh archive of `31ce9eb3` and the same checker.
All changed source files are at most 600 lines, and `git diff --check` passes.
A failed compiler-name-shadowing iteration and an unsupported host seed argument were corrected before the final checks.
No additional CUDA architecture, backend or native checkpoint is qualified by this round.

Full 52-layer logits, 128-token greedy replay and the five alternating before/after speed pairs are NOT RUN.
They remain BLOCKED by shared GPU memory. The final admission attempt sees 7,950 MiB free and refuses 17,000 MiB plus the 2,048 MiB reserve.
The largest free-memory observation across calibration and final runs is 8,553 MiB. No observation reaches the requested 19 GB threshold during this task.
No speed or full-model correctness claim is made.

## Files over 1 MB

The following generated files remain local under this directory and can be gitignored.
The list uses 1,000,000 bytes as its threshold. No new weight reconstruction dump was made.

| File | Bytes |
| --- | --- |
| `host-cache/o/c381262eb7fe45adb1d3d9bb0f68bbdf/test` | 17,234,765 |
| `install/bin/tensorfold` | 37,255,912 |
| `install/bin/tf-cuda-test` | 90,015,240 |
| `install/bin/tf-llama-test` | 14,109,216 |
| `install/bin/tf-qwen35-cuda` | 20,986,672 |
| `install/bin/tf-qwen35-learn` | 20,127,080 |
| `install/bin/tf-tp-test` | 7,499,480 |
| `install/bin/tf-tp-watch-check` | 4,004,048 |
| `layers12/engine-12-s1-logits.f32` | 4,005,888 |
| `layers12/engine-12-s2-logits.f32` | 10,014,720 |
| `layers12/engine-12-s3-layer0.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer1.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer10.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer11.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer2.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer3.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer4.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer5.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer6.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer7.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer8.f32` | 5,591,040 |
| `layers12/engine-12-s3-layer9.f32` | 5,591,040 |
| `layers12/engine-12-s3-logits.f32` | 130,191,360 |
