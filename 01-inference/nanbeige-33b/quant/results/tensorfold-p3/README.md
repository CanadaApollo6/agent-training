# TensorFold Nanbeige P3

Implementation is local on `nanbeige`, based on `5bd3231f`. The commit is recorded in `source-manifest.json`.
Full-model qualification is incomplete. There is no measured full-model speedup or quantization acceptance result.

## Implementation

`families/dflash2` owns the shared config, graph, operator protocol, execution schedule, session and selector.
Qwen27's old paths re-export those declarations. Its parser still applies `requireStock` at the Qwen boundary.
Llama parses the trained checkpoint dimensions and binds the target's hidden width, vocabulary and layer count.
The CUDA adapter reuses existing affine-4, dense rows, norms, QK, activation and Qwen35 draft kernels.
No CUDA kernel source or reduction contract changes.

Llama captures the five configured post-block residuals in bf16. It batches up to eight target projection rows,
keeps serial per-row attention arithmetic, and commits only the accepted prefix into the rolling draft context.
The existing lane core owns acceptance and keyed target draws. Sampled proposals are deterministic chains.
Matching a target draw implements the point-mass proposal accept/reject case and preserves the target distribution.
This uses the same target-pick acceptance machinery as native Qwen27. It does not implement stochastic draft proposals.
The drafter is chain-only, with at most seven proposals. Trees and multiple streams refuse explicitly.

Enable with `serve MODEL --drafter DRAFT`. `TF_LLAMA_DRAFT_FORMAT=affine4` is the default draft storage.
Set it to `bf16` for the reference mode. `--drafter-bits 4` also selects affine-4.
Target KV remains independent. Set `TF_LLAMA_KV_BITS=4` for KV4, or unset it for bf16 KV.
`--no-drafts` skips the drafter. This is an experimental path until the full gates below pass.

## Results

| Check | Result | Receipt |
|---|---|---|
| Native build | sm_86 native and bounded runner built | `build-wire-final.log`, `build-native-final.log` |
| Fresh CPU suite | 1,452/1,452 invocations pass | `final-host.log` |
| Qwen regressions | 59 existing host invocations plus 57 DFlash2 host invocations pass | `final-host.log` |
| Llama host | 61/61 pass, including portable graph tests | `final-host.log` |
| Two real target layers | 96 logit rows byte-equal at widths 2, 3, 8 in both KV modes, per draft-format run | `final-bf16.log`, `final-affine4.log` |
| Rejected suffix | Both KV modes overwrite a rejected suffix and reproduce serial logits | Same |
| Production lane integration | 32 greedy and 32 sampled tokens equal plain, per format; two cases use penalties | Same |
| Draft context seam | Diagnostic window 8 wraps and starts after a skipped prompt prefix | Same |
| Independent CPU draft reference | 50,176 elements; bf16 relative L2 0.009393, max absolute 0.5; affine-4 0.015429, max absolute 1.0 | `reference-bf16.log`, `reference-affine4.log` |
| Lean | 426 inherited, zero new; files at most 600 lines | `lean.log` |
| Full greedy, sampled, acceptance, speed | NOT RUN; admission refused before launch | `full-admission.log` |
| Five-layer server wire | Three JSON and three SSE bodies equal under P2 envelope normalization; actual proposals required | `wire-partial/summary.json`, `wire-partial-kv4/summary.json` |
| Full server wire bytes | NOT RUN | Command below |
| sm_89 | NOT RUN | Explicit qualification option below |

The partial lane test uses real two-layer target weights with a diagnostic tap mapping `[0,1,0,1,0]` and a
shortened draft window. It exercises the production backend and lane acceptance, including prompt gaps,
ring wrap, repeated requests and presence/frequency penalties. It does not establish Nanbeige acceptance.
The independent reference uses five captured real hidden vectors from layers 0 through 4 as diagnostic inputs,
not the trained tap set `[4,15,26,38,49]`. The same inputs enter both implementations. Its bound is relative L2 0.03.
Native affine-4 versus native bf16 final hidden rows differ by relative L2 0.143949 on that one diagnostic input.
All seven selected proposals happened to agree on that input. Neither observation measures acceptance impact.
The CPU reference is a PyTorch transcription of the supplied DFlash2 dataflow, independent of the Zig callbacks.

The local full-model admission required 15,449 MiB plus 2,048 MiB free reserve, with only 13,395 MiB free.
No full target launched. The partial runs used `llama_gpu_run.py --need-mib 6500`, with a 180-second timeout.
Final process peaks were 5,476 MiB for bf16 drafting and 2,856 MiB for affine-4. Minimum free memory stayed above 7 GiB.
No Vesuvius process was signaled or waited for. No push or agent-training commit was made.

## Drafter footprint

The checkpoint contains 81 tensors and 4,047,202,816 tensor bytes. It contains neither an embedding nor an LM head.
The CUDA drafter borrows the target's pinned host embedding and device EXL3 head. Both selector codebooks remain
mapped on the host, totaling 170,917,888 bytes. The mapped checkpoint is host virtual memory, not device allocation.
MLP matrices dominate at 2,367,160,320 native bytes. Attention projections/norms use 734,005,760 bytes,
fusion 513,802,240, convolutions 257,474,560, selector projection 3,670,016, and other norms 172,032.
`inventory.json` records the checkpoint header inventory and config hash.

| Allocation | bf16 draft | Affine-4 draft |
|---|---:|---:|
| Resident draft weights | 3,876,634,112 | 1,091,091,968 |
| Draft KV ring | 41,943,040 | 41,943,040 |
| Target tap capture | 573,440 | 573,440 |
| Affine kernel split workspace | 6,294,528 | 6,294,528 |
| Reusable forward pool | 24,187,648 | 16,966,720 |
| Measured attachment total | 3,949,632,768 | 1,156,869,696 |
| Extra target batch buffers | 7,483,424 | 7,483,424 |
| Largest temporary quantization staging | 0 | 144,506,880 |

These are tracked `DeviceBuffer` bytes. Both measured attachment totals equal the planner exactly.
The pool planner runs the shared execution schedule with allocation callbacks. The live pool is reserved before
serving. Capacity also accounts for the loader's ordered staging peak and the production target allocator budget.
Affine-4 uses the existing group-64 MLX layout and `core.affine4.prepare`, including bf16 scales and biases.
Preparation happens at load without modifying the checkpoint. There is no second resident bf16 copy of quantized linears.

## Capacity

These are allocator ceilings, not observed full-model physical fits. Target weights occupy 14,356,819,968 device bytes.
All rows use the P4a `llama.memory.runtimeBytesMode` budget, including sampler, attention partials, BOS storage and
transient EXL3 workspace. Draft rows add the measured/planned fixed attachment and target batch buffers above.
All rows leave 2 GiB free. The second pair also leaves a 512 MiB driver allowance.

| Draft format | bf16 KV | KV4 | bf16 KV + driver allowance | KV4 + driver allowance |
|---|---:|---:|---:|---:|
| None | 43,219 | 153,351 | 40,700 | 144,424 |
| bf16 | 24,647 | 87,556 | 22,127 | 78,630 |
| Affine-4 | 37,754 | 133,992 | 35,235 | 125,065 |

Thus affine-4 plus KV4 projects enough capacity for 115,488 prompt tokens plus 256 generated tokens.
That fit remains unverified on the full target, and P4a's full-model KV4 quality gate also remains open.
For 48 GiB with the same reserve and allowance, bf16-KV ceilings are 161,650 without a draft, 143,077 with bf16
drafting, and 156,185 with affine-4. KV4 ceilings are 524,288, 507,102 and 524,288, capped by model context.
See `capacity-24g.log`, `capacity-24g-driver.log` and `capacity-48g-driver.log`.

## Reproduce the bounded checks

From the engine root, with the supplied Zig toolchain:

```bash
~/.local/share/zig-x86_64-linux-0.17.0/zig build native tf-llama-test -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j4 --summary all
export TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
export TF_LLAMA_DRAFT=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/drafter/data/draft-cool-step570
export P3_RECEIPTS=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p3
CUDA_VISIBLE_DEVICES='' /home/riels/exl3-local/bin/python tools/zig/llama_draft_reference.py --draft "$TF_LLAMA_DRAFT" --model "$TF_LLAMA_MODEL" --captures /tmp/llama-prof/out --work /tmp/nanbeige-p3
nvidia-smi
python3 tools/zig/llama_gpu_run.py --need-mib 6500 --timeout 180 --receipt "$P3_RECEIPTS/repeat-vram.json" -- zig-out/bin/tf-llama-test llama-draft-partial "$TF_LLAMA_MODEL" "$TF_LLAMA_DRAFT" /tmp/nanbeige-p3/taps.bf16 /tmp/nanbeige-p3/native-affine4.bf16 affine4
CUDA_VISIBLE_DEVICES='' /home/riels/exl3-local/bin/python tools/zig/llama_draft_reference.py --draft "$TF_LLAMA_DRAFT" --model "$TF_LLAMA_MODEL" --captures /tmp/llama-prof/out --work /tmp/nanbeige-p3 --compare /tmp/nanbeige-p3/native-affine4.bf16 --format affine4
```

The captured inputs are pre-existing temporary diagnostics. They are not committed. Change `affine4` to `bf16`
in both commands to check reference storage. Large binary outputs remain under `/tmp/nanbeige-p3`.

## NOT RUN full-model commands

On the rented card, use `TF_ENGINE`, `TF_LLAMA_MODEL` and `TF_LLAMA_DRAFT` to override paths.
The scripts read the actual card total and compute admission from allocator bytes. They have no 24 GiB limit.
The checkpoint and review-7 prompt fixture must be present. Override `P3_REVIEW` for the fixture location.
Each invocation checks three prompts at 256 tokens, alternates five greedy pairs and checks 32 sampled seeds.
The sampled check requires identical keyed token sequences, which is stronger than a marginal histogram comparison.
The result records token arrays, proposed/accepted counts, mean tokens per round, prefill time, decode time,
paired median speed ratio and range. Drafter quantization acceptance is measured by comparing the two formats' results.

```bash
bash "$P3_RECEIPTS/rerun-full.sh" short 16 bf16
bash "$P3_RECEIPTS/rerun-full.sh" short 16 affine4
bash "$P3_RECEIPTS/rerun-full.sh" short 4 bf16
bash "$P3_RECEIPTS/rerun-full.sh" short 4 affine4
P3_SEEDS=0 bash "$P3_RECEIPTS/rerun-full.sh" long 16 bf16
P3_SEEDS=0 bash "$P3_RECEIPTS/rerun-full.sh" long 16 affine4
P3_SEEDS=0 bash "$P3_RECEIPTS/rerun-full.sh" long 4 bf16
P3_SEEDS=0 bash "$P3_RECEIPTS/rerun-full.sh" long 4 affine4
python3 tools/zig/llama_draft_wire.py --model "$TF_LLAMA_MODEL" --draft "$TF_LLAMA_DRAFT" --format affine4 --kv 16 --out "$P3_RECEIPTS/wire-kv16"
python3 tools/zig/llama_draft_wire.py --model "$TF_LLAMA_MODEL" --draft "$TF_LLAMA_DRAFT" --format affine4 --kv 4 --out "$P3_RECEIPTS/wire-kv4"
```

Long defaults to 20,000-token prefixes. Set `P3_LENGTH=68000` for the pilot p75 region.
Both prefixes require three eligible real prompts. `P3_TIMEOUT` changes the bounded guard timeout. Long suites default to four hours because cold prefill is serial
and five pairs over three long prompts repeat that work. Short suites default to one hour.
Wire checks apply the existing P2 normalization from `llama_bytes_test.zig` to UUID `id` and wall-clock `created`.
Every remaining JSON and SSE byte compares exactly. The initial raw-envelope mismatch is retained in
`wire-metadata-control.log`. The diagnostic native server uses five real layers, a matching diagnostic
draft config with tap IDs `[0,1,2,3,4]`, and 16 reply tokens. Its trace must show offered proposals. Both target KV modes pass.
This catches a loader-to-host width mismatch that direct backend checks alone would miss.
For sm_89, rebuild with `-Dsm=89`, set `TF_LLAMA_ALLOW_UNQUALIFIED_SM=89`, run the partial checks first,
then run the commands above. This explicit option does not turn sm_86 evidence into sm_89 qualification.

The first unavailable `test-qwen27` target and the temporary checkpoint facade compatibility failure remain in
`portable.log` and `qwen-fresh.log`. Restored/fresh checks pass. Expected failure-fixture diagnostics in the host
logs are not failed tests. ATC was unavailable locally; no issue, PR or remote coordination message was posted.
