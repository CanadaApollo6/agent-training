# Llama fix round 1

Local branch `nanbeige`, starting at `fe0bd8fb934b15e3365132bf677c07627c711490`, baseline `31ce9eb3`.
Final commit is `c4af57935cc5fb8f24252f1ae9ce298c3b9d335b`, also recorded in `manifest.json`. No push, issue, PR or weight write. No process outside this task was signaled.

The requested fixes are implemented. Bounded device checks pass on RTX 3090 sm_86.
Full 52-layer replay and the requested speed comparison remain BLOCKED by shared GPU memory.

## Changes and evidence

| Finding | Changed source | Verification |
| --- | --- | --- |
| 1. Budget admission | `native.zig:21,46`, `memory.zig:7,25,31`, `weights.zig:74` | Native options receive the existing room callback. KV, row buffers, attention scratch and peak EXL3 reserve growth must fit before `Model.init`. Host test refuses a 15 GiB cap, a 6 GiB reserve, zero/failed probes and one byte below the requirement. `host-restored.log` passes 6/6. Disabling admission fails with `expected error.LlamaMemoryBudget, found void` in `budget-control-red.log`. |
| 2. Default context | `zig/tests/cuda/llama_boundary.zig:8,55`, `tools/zig/llama_boundary_reference.py` | Two real checkpoint layers, 56/8 heads, full 16384-slot production cache. All five requested positions pass. Position 16384 refuses. A fresh BOS request has KL 1.13302e-4 and two fresh requests have bit-equal logits. `boundary-final.log`, `boundary-summary.json`. |
| 3. Kernel qualification | `cuda_kernels.zig:115`, `dense_pointwise_records.zig`, `contract_dense_pointwise.zig`, `docs/kernel-contract.md` | Real image and all three mangled symbols registered. Three kernels pass gates 1-4, with gate 4 N/A for pointwise formulas. Eight controls are red as planted, two limit controls are caught, zero limit problems. Compiled and live registers are 20/26/12, shared bytes 0, bound 256. `pointwise.log`. Build and loader restrict this image to sm_86. Other SMs are NOT QUALIFIED. |
| 4. Independent head accuracy | `tools/zig/llama_head_fp64.py:23,45,65`, `llama_tests.zig:261` | Identical final-norm FP16 inputs from the archived 52-layer residuals. Native EXL3, exllamav3 one-row and original multi-row heads compared against CPU FP64 dots of reconstructed FP16 weights. Native has the smallest maximum error in all three cases. `head/comparison.json`, `head-engine-final.log`, `head-fp64.log`. |
| 5. Layer assertions | `validation.zig:24`, `llama_tests.zig:144` | RMSE <= 1, max error <= 0.0001 + 0.01 * reference peak, relative L2 <= 0.01. Final 12-layer/217-position run passes all assertions. A physical zero reference returns `FAIL llama-layers: LlamaLayerDrift`, exit 1. `layers12-final.log`, `layer-zero-control.log`. |
| 6. Prefix work and fence | `forward.zig:101,109,141`, `tools/zig/llama_fix_bench.py` | Attention launches groups for position + 1, while scratch remains sized for the full context. A completed download clears metadata-pending state, so greedy skips a second wait. Prefill retains its metadata lifetime fence. Default-context and 12-layer replay pass. Timing is NOT RUN because the guard refuses the full model. |

Source locations without a directory are under `zig/src/families/llama`, except the named build and contract files.
The complete source and binary hashes are in `manifest.json`.

## Device outputs

```text
SUMMARY llama boundary heads=56/8 layers=2 context=16384 positions=511,512,2047,2048,16383 refused=16384 fresh_kl=1.1330218638119827e-4 fresh_repeat=bit_equal device_peak=1588639460
SUMMARY llama layers=12 positions=217 mean_kl=5.8298328377966356e-5 max_kl=5.630279789195827e-4 top1=214/217 kl_above_0.01=0 device_peak=4054395204 pinned=2392850496
SUMMARY dense_pointwise kernels=3 pass=3 fail=0 numerical_controls=8 red=8 missed=0 limits=0 sm=86
```

| Position | KL against exllamav3 | Top token matches |
| --- | --- | --- |
| 511 | 6.18360e-5 | yes |
| 512 | 7.44019e-5 | yes |
| 2047 | 7.99388e-5 | yes |
| 2048 | 7.50188e-5 | yes |
| 16383 | 8.28623e-5 | yes |

The two-layer oracle uses exllamav3 FP16 cache and chunks of at most 128 tokens.
The engine uses its own serial loop and production BF16 cache. Both use the same saved token IDs.
The final boundary run took 55.56 seconds and peaked at 1784 MiB process VRAM.
The 12-layer check took 9.48 seconds and peaked at 4170 MiB. All successful final runs retained the reserve.

## FP64 head results

Maximum absolute errors cover all 166912 vocabulary columns.
Multi-row shapes are the original 15, 195 and 212 rows, respectively.
The FP64 reference uses CPU float64 sums, 1024 vocabulary columns per chunk, with no full dense GPU head.

| Case | Native max error | exllamav3 one-row max error | exllamav3 multi-row max error | FP64 winner token | FP64 top-2 margin |
| --- | --- | --- | --- | --- | --- |
| seq2-pos7 | 0.05450899 | 2.95465966 | 1.65333294 | 926 | 1.45734375 |
| seq3-pos110 | 0.05380445 | 3.65816840 | 1.01001600 | 152361 | 1.65328393 |
| greedy-token17-prefix | 0.05438064 | 3.01257804 | 0.83221656 | 13 | 0.82935039 |

| Case | Native winner / margin | exllamav3 one-row winner / margin | exllamav3 multi-row winner / margin |
| --- | --- | --- | --- |
| seq2-pos7 | 926 / 1.47154999 | 13 / 1.58337021 | 926 / 0.69144440 |
| seq3-pos110 | 152361 / 1.64955139 | 152361 / 1.62301636 | 152361 / 1.69961929 |
| greedy-token17-prefix | 13 / 0.80783463 | 2269 / 0.29316711 | 13 / 1.35191727 |

`head/inputs.json` hashes the archived residuals and common norm input. This isolates head arithmetic.
It does not claim a fresh full-model body replay. The selected FP64 winner matches native in all three cases.

An initial attempt used exllamav3's unbounded full-head reconstruction and OOMed.
Its workspace estimate was too small and free VRAM briefly fell to 644 MiB, below the required reserve.
The failed attempt is retained in `head-prepare-unbounded-oom.log` and its VRAM receipt.
The corrected script reconstructs 1024 columns at a time. Its successful run peaked at 1318 MiB process VRAM.
The first contract diagnostic also exposed a temporary VTable pointer; static hook storage fixes that test-runner bug.
`pointwise-temporary-hooks-fail.log` retains the failure. The final contract run exits 0.

## Builds and host checks

`build-focused.log` builds `tf-llama-test` from nvcc 13.4 sm_86 images, 114/114 steps.
`build-prebuilt.log` shows the same focused runner using those already-built images.
`build-plain.log` passes the plain build using prebuilt images, 23/23 steps.
`host-focused.log` passes 10/10 tests across Llama and contract image routing.
`host-final.log` passes 1810/1819 tests, skips 8, and fails one inherited test.
The failing test is `exl3_load.test.a scan that fails an allocation frees every group prefix it held`.
It returns `NondeterministicMemoryUsage`. Both `exl3-isolated-base.log` and `exl3-isolated-head.log`
reproduce it with an isolated test, including on `31ce9eb3`. No unrelated fix is included.
`test-golden` reports 55/55 equal. `fuzz-cortheon` passes. Full `zig build test` passes 2087/2105 tests, skips 17, and fails the same inherited test.
Its complete output is in `test-complete.log`.
`lean-new.txt` reports 426 problems versus 426 on `31ce9eb3`, zero new. `git diff --check` passes.
Every changed or added source file is at most 600 lines. sm_80, sm_89, sm_120, sm_121 and non-CUDA device checks are NOT RUN.
The focused pointwise receipt qualifies this image, not the entire repository's CUDA kernel suite.

The final model binary is `tf-llama-after`, SHA256 `512157f25f60be190ed40f9ffe7c76b37b462546b6ba7a2a29d1a3e3a1dc47e6`.
The paired baseline `tf-llama-before` uses the final sources with only `forward.zig` restored from `fe0bd8fb`.
Its SHA256 is `a7f3fa76bbe166c88a66d06a021d215dc0be2724a2455e2d67a84205eab64b2b`.
`build-perf-before.log` records its build. Final source and installed focused binary are restored.

## Blocked full-model checks and speed

The guard returned `BLOCKED need 17000 MiB plus 2048 MiB desktop reserve; free 8740 MiB`
for final 52-layer logits, final 128-token greedy replay, and the first paired benchmark run.
Vesuvius PID 3473960 still holds 13390 MiB. No other process was signaled.
`full52-logits-blocked.log`, `full52-greedy-blocked.log` and `bench-blocked.log` retain exact commands and refusals.

Before/after tok/s is NOT RUN. The author's earlier 28.12 tok/s, three repetitions, is historical evidence only.
It is not the requested five-run alternating baseline and cannot establish this fix's speedup.
The new benchmark runs five A/B pairs, one warmup and one 127-forward measured continuation per process,
context 4096, 32 prompt tokens and 128 output tokens. It requires identical token arrays across both sides.
The existing profile identifies EXL3 linears and launch gaps as the limiter; no new performance claim is made.

Run `bash fix1/rerun-blocked.sh` from this results directory after the guard can admit the model.
That file sets absolute paths and runs these exact final-binary checks without changing any process or memory limit.
