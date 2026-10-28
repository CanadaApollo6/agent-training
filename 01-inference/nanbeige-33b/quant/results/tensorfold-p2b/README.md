# Nanbeige TensorFold P2b

Sampled serial decoding and both native API endpoints are implemented on local branch `nanbeige`.
Input is `e8d79bcc1b1b510a74cdd58784031bd7a2e83d73`. Lean base is `31ce9eb3`.
Production commit is `b3774166b884925c34d268bad55d278e06151b16`.
Verification commit is `3aeb8e97513bfb0c2a66f2e32e5defad664bdd3d`.
No push, issue, PR, external message, checkpoint edit, or signal to another job was made.
Independent review is NOT RUN in this implementation sub-agent.

## Supported request settings

| Field | Behavior |
| --- | --- |
| `temperature` | Defaults to 1.0 for the Nanbeige profile. Zero uses the unchanged FP32 argmax when penalties are neutral. Negative values are refused before SSE. |
| `top_p` | Defaults to 0.95. Accepted range is [0,1]. The shared sampler treats 0 and 1 as no nucleus cut. |
| `top_k` | Defaults to 0, off. A positive value filters candidates. The existing parser maps negative values to off. |
| `min_p` | Defaults to 0, off. Accepted range is [0,1]. |
| `seed` | Request seeds override defaults. The shared sampler keys draws by seed, absolute position and token ID. Signed integer seeds preserve their u64 bit pattern. Omitted seeds use fresh request entropy unless `--prompt-seed` is configured. |
| `presence_penalty`, `frequency_penalty` | Supported in [-2,2], for sampled and greedy requests. Only generated reply tokens are counted. The pending token is counted once. The shared CUDA operator subtracts presence plus frequency times count. |
| `repetition_penalty` | Omitted, null, or numeric 1 is accepted. A non-neutral value receives a clear JSON HTTP 400 before SSE. |
| `max_tokens`, `max_completion_tokens` | Enforced across reasoning and answer. `max_completion_tokens` wins. Context limits remain enforced. |

Request values override configured defaults. Null uses the default, as the pod's `param` helper does.
The Nanbeige profile requires model_type `llama` and `<|im_start|>` token ID 166100.
Generation-config values and CLI flags override that profile's fallback defaults.
The supplied generation_config.json contains no sampling settings or `do_sample`, so this checkpoint starts at T 1, top_p 0.95, top_k 0, min_p 0.
The startup log now displays the defaults that the server actually resolves.
Existing stop strings, chat-template kwargs, automatic tools and SSE usage remain supported.
Required or named tool calls and token grammars retain their existing explicit refusals.

The adapter uses `cuda.sample.Draws`, `lanes.gpu_rule`, the shared BF16 conversion and the shared penalty operator.
There is no new sampling algorithm, CUDA source, scheduler, weight arithmetic or activation-precision change.
As in Kolibri, positive-temperature logits are converted from FP32 to BF16 for the shared sampler.
Identical seeded streams across TensorFold and exllamav3 are not claimed.

## Changed source

- [sampling.zig:6](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/sampling.zig:6) defines the typed draw. Its sampler owns pinned staging, device buffers and reply counts.
- [kernels.zig:40](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/kernels.zig:40) loads the shared sampler. [forward.zig:176](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/forward.zig:176) exposes token selection without changing logits arithmetic.
- [backend.zig:69](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/backend.zig:69) applies request sampling and penalties to first, queued and verified rows at their absolute positions.
- [native.zig:12](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/native.zig:12) removes the greedy-only capability. [memory.zig:22](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/memory.zig:22) accounts for sampler allocation.
- [nanbeige.zig:6](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/nanbeige.zig:6), [server.zig:199](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/server.zig:199) and [serve.zig:242](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/serve.zig:242) set and display profile defaults. Numeric validation runs on merged, normalized settings before SSE.
- [llama_sampling.zig:27](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_sampling.zig:27) checks distributions, seeds, FP32 greedy ties, penalties and serial/lane/verify equality.
- [llama_sampling_check.py:20](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_sampling_check.py:20) computes independent statistical targets, reconstructs a pi request and checks native HTTP.
- [llama_output_test.zig:228](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/llama_output_test.zig:228) captures actual engine requests from JSON and SSE routes, including defaults, overrides, penalties, seeds and token limits.

## Check 1. Temperature-zero regression

PASS against the saved native fixtures.
`greedy-slice-regression.json` compares the saved P2 two-layer oracle with the final serial runner and both HTTP endpoints.
All four arrays match at all 128 positions. `greedy-serve/summary.json` also verifies queueing, streamed IDs and cancellation/reset.

The additional 12-layer run matches the pre-change engine at all 128 positions.
The old and new engines each match their lane backend at all 128 positions.
Both differ from the saved exllamav3 12-layer fixture starting at token 3.
`greedy12-regression.json` proves this difference is inherited, rather than silently treating that reference as green.
The baseline is a fresh e8d79bcc archive, with only the test-only layer-count entry added.
`baseline-greedy-test-only.diff` records every change to that archive. Both binaries use the identical final CUDA images.
Those two runs exit 1 for the inherited trusted-reference assertion. Their before/after token arrays are byte-identical.
Full-model dense-head fixture equality is BLOCKED by shared GPU memory.

## Check 2. Statistical correctness

PASS for all seven cases, using 204,800 draws through the actual Llama sampler adapter on CUDA sm_86.
The six small cases each take 32,768 draws. The wide case takes 8,192 draws.
The target uses an independent FP64 softmax, top-k normalization, an inclusive top-p cut, min-p filtering and final normalization.
All fixture logits and penalty-adjusted logits are exactly representable in BF16.
The pilot case tests T 1, top_p 0.95, top_k 0 and min_p 0.
The wide case has 2,048 equal logits and retains 1,946 tokens, which exercises nucleus selection past 1,024 candidates.
For its chi-square test, adjacent groups of 256 tokens form eight predeclared bins.

The fixed acceptance threshold is Pearson chi-square below 40, with at most seven degrees of freedom.
The df=7 tail probability at 40 is 1.25879038737e-6. All expected bin counts exceed 5.
No excluded token may be drawn. `statistics.json` retains targets, counts and statistics.
The deliberately wrong uniform target in `statistics-control.log` fails this same assertion, exit 1.

| Case | Draws | Retained tokens | Degrees of freedom | Chi-square, threshold 40 |
| --- | ---: | ---: | ---: | ---: |
| pilot | 32768 | 6 | 5 | 4.459651 |
| temperature | 32768 | 8 | 7 | 11.572233 |
| top_k | 32768 | 3 | 2 | 3.496678 |
| min_p | 32768 | 4 | 3 | 2.184441 |
| combined | 32768 | 3 | 2 | 3.336479 |
| penalties | 32768 | 7 | 6 | 4.198079 |
| wide_nucleus | 8192 | 1946 | 7 | 5.270368 |

## Check 3. Seeds

PASS. `sampling.log` reports 256/256 identical draws for the same seed and position.
Changing the seed changes 228/256 draws. All 128 FP32 near-tie greedy draws select the original FP32 winner.
Two direct greedy penalty checks also pass, including an empty history that must apply no penalty.
`sampled-decode.log` verifies 64 tokens across serial, lane and explicit verify/keep paths, with penalties off and on.
Native chat and completions each repeat 32 seeded tokens exactly through JSON and SSE.
Each endpoint also changes its sequence with a different seed. Concurrent seeded requests equal their solo sequence.

## Check 4. Defaults and pi-shaped SSE

PASS on the real native server and the original two-layer checkpoint slice.
`sampled-serve/server.log` prints `sampling: temperature 1.0, top_p 0.95, top_k 0, min_p 0.0`.
Requests with no sampling fields succeed on both endpoints.
On both routes, omitted filters with seed 1234 equal explicit pilot defaults and the 32-token serial sample.
Presence/frequency requests equal the independent serial route's 32 tokens on each endpoint.
Explicit T/top_p/top_k/min_p/seed overrides succeed with exactly eight tokens.
Host tests verify the resolved values directly and verify max_completion_tokens precedence.

`pi-fields.json` examines 19 traces and all 485 recorded calls.
471 calls send max_tokens 16384, seven send 8192, and seven send other positive token caps.
Every recorded per-call sampling object has only `max_tokens`.
Temperature, filters, seed and all three penalty fields are absent, including neutral penalties.
`pi-trace-semantics.json` records the installed dialect's whitelist and parse_sampling function.
That whitelist includes penalties, filters, seed, tool_choice and other generation controls, so their absence is observable.
No raw HTTP body is retained in these traces.
`pi-request.json` reconstructs the first call's model, messages, tools and exact max_tokens 16384, then adds stream=true.
The native server returns HTTP 200 and three SSE events before the client deliberately disconnects.
The following seeded request matches its pre-cancellation sequence. This does not claim a completed 16,384-token pi turn.

| Pi request field | Handling |
| --- | --- |
| `model` | Resolves the configured nanbeige-exl3-a1 alias. |
| `messages` | Uses the existing Nanbeige HF template and tokenizer. |
| `tools` | Uses the existing template and typed qwen tool-output parser. No new strict grammar guarantee is claimed. |
| `max_tokens` | Reserves the requested 16,384 completion tokens within the configured context. |
| `stream` | Emits SSE and handles disconnect/cancellation. |
| No sampling controls | Uses T 1, top_p 0.95, top_k 0 and min_p 0. |

This pi check uses context 32768 so the 16,384-token completion allowance fits beside the prompt.
The family's default context remains 16384. A production pi launch needs an explicitly large enough `--context`.
The engine still refuses a prompt plus requested completion that exceeds that configured context.

## GPU receipts and blocked full-model checks

Each device invocation runs under `tools/zig/llama_gpu_run.py` with an estimate above measured peak and 2048 MiB reserve.
Only the task's child servers receive SIGTERM. No process belonging to Vesuvius or another job is touched.
All retained final device runs last at most 61 seconds.

| Run | Seconds | Peak process MiB | Minimum free MiB | Exit |
| --- | ---: | ---: | ---: | ---: |
| sampling | 60.11 | 262 | 7491 | 0 |
| sampled-decode | 7.20 | 1690 | 6137 | 0 |
| greedy12 | 22.93 | 4350 | 3559 | 1 |
| baseline-greedy | 12.52 | 4350 | 3562 | 1 |
| sampled-serve/server | 25.20 | 1912 | 5738 | 0 |
| greedy-serve/server | 6.92 | 1666 | 6207 | 0 |
| greedy-serve/oracle | 3.52 | 1690 | 6182 | 0 |

The largest initial free-memory observation in the retained final device receipts is 8163 MiB.
No observation reaches the requested 19.5 GB threshold.
The clean final P1 admission sees 7783 MiB and refuses 17000 plus 2048 MiB.
The clean final P2 admission sees 7783 MiB and refuses 15500 plus 2048 MiB.
Neither full-model binary starts. Missing VRAM receipts for blocked attempts do not imply a run.
Full 52-layer logits, dense-head 128-token equality, five alternating speed pairs, full-model bash tool use and full-model server throughput remain BLOCKED and NOT RUN.
No slice throughput substitutes for any of those gates.

## Other verification

- The requested native build with nvcc and sm_86 passes 125/125 steps and 7/7 host tests in `build-engine.log`.
- The final source build passes 20/20 steps and 27/27 focused tests in `build-final.log`. The added runner entry builds in `build-runner.log`.
- Plain build passes 24/24 steps in `build-plain.log`.
- The broad host run reports 457/460 steps, 1780/1785 tests passed, four skipped and one failed in `host-all.log`.
- The sole failing test is the inherited exl3_load allocation test at line 319, NondeterministicMemoryUsage. `exl3-base.log` freshly reproduces it on 31ce9eb3. The source is byte-identical to that base. `exl3-base-setup.log` is an earlier missing-module setup error and supplies no test evidence.
- `test-kernel-contract`, `test-golden` and `fuzz-cortheon` pass. Goldens report 322/335 and 55/55 equal, zero unexpected differences and 13 inherited known differences.
- `lean-new.txt` reports 426 problems, 426 on a fresh 31ce9eb3 archive, zero new. All changed files have at most 600 lines. `git diff --check` and Python compilation pass.
- Full model, other CUDA chips, Metal, HIP, Windows, multi-GPU, drafts and strict tool grammars are NOT RUN or unsupported as noted above.

Model the Domain kept draw inputs typed and sampler buffers owned by the serial model.
Prove It Works required actual native endpoints and a fresh pre-change greedy comparison.
Test Behavior, Not Implementation kept statistical targets independent of the shared sampler.
Sequence Work into Verifiable Units kept runtime support and repeatable verification in two ordered commits.
The throughput checkpoint serialized shared GPU runs and kept the coupled adapter under one writer.

## Reproduce

Run from the assigned worktree. Commands below use the original model and the P2 slice read-only.
Expanded device commands and exact settings are also retained in each VRAM receipt and `*/commands.json`.

```sh
TF_ROOT=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
TF_QUANT=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant
TF_OUT="$TF_QUANT/results/tensorfold-p2b"
TF_MODEL="$TF_QUANT/models/3.5-nb"
TF_SLICE="$TF_QUANT/results/tensorfold-p2/slice"
TF_P1="$TF_QUANT/results/tensorfold-p1"
TF_ZIG=/home/riels/.local/share/zig-x86_64-linux-0.17.0/zig
cd "$TF_ROOT"
"$TF_ZIG" build native tf-llama-test -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j4
TF_LLAMA_MODEL="$TF_MODEL" "$TF_ZIG" build test-llama-host test-llama-server-host -Dfatbins="$TF_ROOT/zig-out/fatbin" -j4
python3 tools/zig/llama_gpu_run.py --need-mib 1000 --receipt "$TF_OUT/sampling-vram.json" --timeout 180 -- zig-out/bin/tf-llama-test llama-sampling "$TF_OUT/distributions.jsonl"
python3 tools/zig/llama_sampling_check.py stats "$TF_OUT/distributions.jsonl" "$TF_OUT/statistics.json"
python3 tools/zig/llama_gpu_run.py --need-mib 3000 --receipt "$TF_OUT/sampled-decode-vram.json" --timeout 120 -- zig-out/bin/tf-llama-test llama-sampled "$TF_SLICE" "$TF_P1" "$TF_OUT/sampled-serial.jsonl"
python3 tools/zig/llama_sampling_check.py serve zig-out/native/bin/tensorfold-native "$TF_SLICE" "$TF_OUT/sampled-serve" "$TF_OUT/sampled-serial.jsonl" "$TF_OUT/pi-request.json"
python3 tools/zig/llama_serve_check.py serve zig-out/native/bin/tensorfold-native "$TF_SLICE" "$TF_OUT/greedy-serve" --need-mib 3000 --context 1024 --oracle zig-out/bin/tf-llama-test --refs "$TF_P1"
python3 tools/zig/llama_gpu_run.py --need-mib 4600 --receipt "$TF_OUT/greedy12-vram.json" --timeout 180 -- zig-out/bin/tf-llama-test llama-greedy-slice "$TF_MODEL" "$TF_P1" 12 128 "$TF_OUT/greedy12.json"
```

The last command intentionally retains the inherited exllamav3-reference assertion, exit 1.
For the pre-change comparison, archive e8d79bcc, apply `baseline-greedy-test-only.diff` with `patch -p0`, and build tf-llama-test against this worktree's fatbins.
Then run that binary through the identical 4600 MiB guard and compare its 128 tokens to `greedy12.json`.
`greedy12-regression.json` retains the exact executed command and binary hash. Temporary baseline binaries were removed.

When at least 19.5 GB is free, rerun these blocked commands. Every underlying GPU run is still guarded.
The first helper's full_checks performs the P1 logits, greedy and five alternating speed-pair gates when admitted.

```sh
python3 tools/zig/llama_fix2_verify.py "$TF_MODEL" "$TF_P1" "$TF_OUT/full-p1" --binary "$TF_ROOT/zig-out/bin/tf-llama-test"
python3 tools/zig/llama_serve_check.py serve zig-out/native/bin/tensorfold-native "$TF_MODEL" "$TF_OUT/full-serve" --need-mib 15500 --context 4096 --timeout 600 --full --smoke "$TF_QUANT/smoke_serve.py" --oracle zig-out/bin/tf-llama-test --refs "$TF_P1"
```

The first command also reruns the existing boundary and mutation controls before those full-model gates.
The actual blocked P1 attempt invoked that helper's full_checks alone and saved `full-p1.json` and `full52-blocked.log`.
The actual blocked P2 commands are in `full-serve/oracle-command.json` and `full-serve/commands.json`.

## File identities and size limits

`source-manifest.json` hashes all changed source, both final binaries, every compiled CUDA image and the used reference identities.
Binaries stay in `/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig-out`, outside this receipt directory.
`large-files.json` lists every receipt file above 1,000,000 bytes. It is empty.
The P2 slice is reused in place. No additional weight file is copied here.

| Changed file | Lines |
| --- | ---: |
| [tools/zig/llama_sampling_check.py](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_sampling_check.py) | 228 |
| [tools/zig/llama_serve_check.py](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_serve_check.py) | 281 |
| [zig/src/families/llama/backend.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/backend.zig) | 140 |
| [zig/src/families/llama/cuda.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/cuda.zig) | 24 |
| [zig/src/families/llama/forward.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/forward.zig) | 182 |
| [zig/src/families/llama/kernels.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/kernels.zig) | 111 |
| [zig/src/families/llama/memory.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/memory.zig) | 56 |
| [zig/src/families/llama/native.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/native.zig) | 74 |
| [zig/src/families/llama/sampling.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/sampling.zig) | 82 |
| [zig/src/server/llama_output_test.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/llama_output_test.zig) | 271 |
| [zig/src/server/nanbeige.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/nanbeige.zig) | 109 |
| [zig/src/server/serve.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/serve.zig) | 429 |
| [zig/src/server/server.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/server.zig) | 530 |
| [zig/tests/cuda/llama_main.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_main.zig) | 41 |
| [zig/tests/cuda/llama_sampling.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_sampling.zig) | 144 |
| [zig/tests/cuda/llama_tests.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_tests.zig) | 298 |
