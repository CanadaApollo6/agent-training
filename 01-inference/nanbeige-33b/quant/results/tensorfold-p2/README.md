# Nanbeige TensorFold P2

The greedy OpenAI-compatible serving path is implemented and verified on the CPU and a two-layer model slice.
Full 52-layer tool use, server throughput, and 128-token equality remain **BLOCKED** by shared GPU memory.
A TB2 harness that sends positive temperature must use temperature 0 for this implementation.
Positive-temperature sampling is explicitly rejected before SSE opens.

The source is branch `nanbeige`, final commit `e8d79bcc1b1b510a74cdd58784031bd7a2e83d73`.
P1 is `fd409155c09fadf5bea00232fe59deec621ead11`. The lean baseline is `31ce9eb3`.
These are local commits. No push, issue, pull request, or external message was created.
The original checkpoint stayed read-only. Only task-owned processes were signaled.
No CUDA operator, weight arithmetic, precision mode, or second scheduler was added.

## Acceptance results

| Check | Result | Evidence |
| --- | --- | --- |
| 1. Prompt tokens | PASS. 15 conversations, 2202 IDs, zero differences from HF. | [parity.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/parity.log) |
| 2. Output parsing | PASS. 15 raw texts through the actual pod parser, TF parser, and both JSON and SSE handlers. | [parity.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/parity.log) |
| 3. Full-model bash turn and decode speed | BLOCKED. No full-model tool or tok/s claim. | [full-serve/summary.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/full-serve/summary.json) |
| 3. Slice HTTP plumbing | PASS. Solo/concurrent 16-token replies equal, streamed IDs equal, cancellation leaves the next request equal. | [slice-serve/summary.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/slice-serve/summary.json) |
| 4. Full-model 128-token equality | BLOCKED. | [full-serve/oracle.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/full-serve/oracle.log) |
| 4. Slice equality | PASS. All 128 tokens match serial generation through both chat and completions, using the exact 195-token P1 chat prompt. | [slice-serve/greedy-chat.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/slice-serve/greedy-chat.json), [slice-serve/greedy-server.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/slice-serve/greedy-server.json) |
| 5. Request settings | PASS for the supported greedy path and explicit refusals below. | [parity.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/parity.log), [slice-serve/rejections.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/slice-serve/rejections.json) |

The prompt corpus includes system/user, the model's default system, prior assistant reasoning and its alias,
reasoning embedded in content, JSON-string tool arguments, malformed JSON arguments, consecutive tool results,
tools present and absent, mixed string/text parts, and the exact P1 prompt.
It exercises `preserve_thinking=false`, `enable_thinking=false`, and `tool_call_format=json`.
The existing Jinja engine renders the checkpoint template correctly for every case. Its implementation is unchanged.
Chat encoding adds no tokenizer post-processor prefix. Completion text adds a missing BOS and preserves an existing one.
Message-boundary `<|im_start|>` tokens remain part of the model's template.
The primary EOS is 166101. The pod's alternate stop 166102 is also honored.

The reference script extracts `prepare`, `typed`, `split_output`, and their regex constants from `serve_exl3.py` via AST.
It executes those actual definitions without importing Torch or starting the pod server.
It compares reasoning/content exactly and tool names plus decoded JSON arguments, excluding generated tool IDs.
The raw corpus covers string/number/integer/boolean/object/nullable parameters, Python object spelling,
multiple calls, missing thinking/call/parameter closers, unknown functions, malformed envelopes, trailing prose,
and Chinese/emoji output.
The focused build passes 19/19 tests, including the five P2 tests and imported module tests.
The stream boundary test checks all 35 valid split positions across its ASCII and Chinese/emoji examples.

[prompt-before.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/prompt-before.log) reproduces the original reasoning-alias token mismatch.
[output-mutation.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/output-mutation.log) records the intended assertion failures when generic thinking replaced the required-close policy.
That control predates the final UTF-8 fix. [parity-restored.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/parity-restored.log) records its restored green run.
[unicode-before.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/unicode-before.log) freshly reproduces a failing UTF-8 validity assertion from the new stream path.
The final [parity.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/parity.log) is green after holding back seven complete characters.

## GPU scope and admission

The actual native binary serves two original checkpoint layers with the original embeddings, head, and tokenizer.
The slice is a new safetensors file under this directory. The source weights were never edited.
The server uses context 1024, one lane, greedy decoding, no drafts, no prompt cache, and no keep-warm requests.
The serial oracle uses context 4096 and the same `generate` function as `llama-greedy`.
This verifies request/stream/state behavior and token equality on the slice. Its generated text is not model-quality evidence.

The final server run lasts 6.60 seconds, peaks at 1,664 MiB,
and leaves at least 6,147 MiB free.
The serial oracle lasts 3.12 seconds, peaks at 1,688 MiB,
and leaves at least 6,128 MiB free.
Both use an honest 3,000 MiB estimate and preserve the 2,048 MiB desktop reserve.
The concurrency receipt and server log show that the second request queues and receives its own solo token sequence.
The slice also verifies three pre-stream HTTP 400 refusals and reset after a disconnected stream.

The full-model attempt uses context 4096 to reduce KV memory from the default 16384-token setting.
Its guard requires 15,500 MiB plus the 2,048 MiB reserve, or 17,548 MiB free.
The final admission says `BLOCKED need 15500 MiB plus 2048 MiB desktop reserve; free 7859 MiB`.
The command exits 2 for BLOCKED. Neither the full oracle nor the full server starts.
Full-model `smoke_serve.py`, its bash call, and throughput are **NOT RUN**.
The prepared full run measures five 128-token single-stream replies and retains server decode rates separately
from the smoke client's latency including prefill. No slice speed is substituted for full-model throughput.

## Sampling and request settings

| Field | Behavior |
| --- | --- |
| `temperature` | 0 or omitted uses greedy. Positive values return HTTP 400 before SSE. Negative values are refused. |
| `top_p`, `top_k`, `min_p` | Accepted under temperature 0; filters do not change argmax. Positive-temperature combinations are refused. `top_p` and `min_p` must be in [0,1]. |
| `presence_penalty`, `frequency_penalty` | Zero or omitted accepted. Nonzero returns HTTP 400 before SSE. |
| `repetition_penalty` | Numeric 1 or omitted accepted. Other values return HTTP 400 before SSE. |
| `max_tokens`, `max_completion_tokens` | Enforced across reasoning and answer. `max_completion_tokens` wins when both are supplied. Requests exceeding the context get an explicit context error. |
| `stop` | A nonempty string or list of nonempty strings is honored by the existing stop handler. |
| `chat_template_kwargs` | Passed into the existing template renderer. Non-object values are refused. |
| `tool_choice` | Auto/none supported. Required/named calls are explicitly refused because this adapter has no token grammar. |
| Concurrent requests | One serial lane owns model state. HTTP requests queue through the existing lane host. |

## Reproduce

These commands run from the assigned worktree. All GPU commands inside the checker go through `llama_gpu_run.py`.
The checker records exact expanded commands and VRAM receipts. It never signals a process it did not launch.

```sh
TF_ROOT=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
TF_QUANT=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant
TF_OUT="$TF_QUANT/results/tensorfold-p2"
TF_ZIG=/home/riels/.local/share/zig-x86_64-linux-0.17.0/zig
cd "$TF_ROOT"
export ZIG_GLOBAL_CACHE_DIR="$TF_OUT/global-cache"
"$TF_ZIG" build native tf-llama-test -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast -j4 --prefix "$TF_OUT/install" --cache-dir "$TF_OUT/host-cache" --summary all
/home/riels/exl3-local/bin/python tools/zig/llama_serve_reference.py "$TF_QUANT/models/3.5-nb" "$TF_QUANT/serve_exl3.py" --zig "$TF_ZIG" --results "$TF_OUT"
python3 tools/zig/llama_serve_check.py slice "$TF_QUANT/models/3.5-nb" "$TF_OUT/slice" --layers 2
python3 tools/zig/llama_serve_check.py serve "$TF_OUT/install/native/bin/tensorfold-native" "$TF_OUT/slice" "$TF_OUT/slice-serve" --need-mib 3000 --context 1024 --oracle "$TF_OUT/install/bin/tf-llama-test" --refs "$TF_QUANT/results/tensorfold-p1"
```

When the GPU has enough free memory, run this exact full-model command.
It first runs the serial 128-token oracle, then starts the native server, checks queue/cancellation behavior,
runs the supplied smoke test with concurrency 1, measures five single-stream replies,
and compares all 128 P1 continuation IDs through both endpoints.

```sh
python3 tools/zig/llama_serve_check.py serve "$TF_OUT/install/native/bin/tensorfold-native" "$TF_QUANT/models/3.5-nb" "$TF_OUT/full-serve" --need-mib 15500 --context 4096 --timeout 600 --full --smoke "$TF_QUANT/smoke_serve.py" --oracle "$TF_OUT/install/bin/tf-llama-test" --refs "$TF_QUANT/results/tensorfold-p1"
```

The shared host invocation is below. Its inherited allocation-test failure returns exit 1.

```sh
TF_LLAMA_MODEL="$TF_QUANT/models/3.5-nb" "$TF_ZIG" build test test-kernel-contract test-golden fuzz-cortheon -j4 -Dfatbins="$TF_ROOT/zig-out/fatbin" --prefix "$TF_OUT/install" --cache-dir "$TF_OUT/host-cache" --summary all
python3 tools/zig/lean_new.py "$TF_OUT/lean-base.txt" "$TF_OUT/lean-head.txt"
git diff --check
```

## Shared-code review and verification limits

Review `hf_text.zig` closely for arbitrary kwargs, BOS deduplication, and the profile's alternate EOS.
Review `model_text.zig`, `errors.zig`, `prompt.zig`, and `chat.zig` for request-local kwargs propagation, including probe renders.
Review `tool_parse.zig`, `tool_params.zig`, and `tool_qwen.zig` for the explicit XML mode and schema conversions.
The generic parser retains its old strict behavior. Nanbeige uses the scored pod's tolerant qwen3 behavior.
Review `server.zig`, `serve.zig`, `nanbeige.zig`, and `openai.zig` for profile selection, required-close reasoning,
UTF-8 streaming, final complete tool-call events, and pre-stream refusals.
The profile requires config model_type `llama` and `<|im_start|>` ID 166100.
`openai_client.zig` is an unchanged socket-output adapter extracted to keep `openai.zig` below 600 lines.
Review `families/llama/native.zig` and `native/cuda.zig` for rows=1 and greedy capability reporting.

The final native/oracle build passes 121/121 steps. The plain build passes 24/24 steps.
The final shared run reports `457/460 steps succeeded (1 failed); 1573/1577 tests passed (3 skipped, 1 failed)`. Other results are cached.
Its sole failure is `exl3_load.test.a scan that fails an allocation frees every group prefix it held`,
with `NondeterministicMemoryUsage` at `zig/src/core/exl3_load.zig:319`.
That file is byte-identical to `31ce9eb3`, and [exl3-base.log](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/exl3-base.log) freshly reproduces the same assertion on that base.
`test-kernel-contract`, `test-golden`, and `fuzz-cortheon` succeed in the same final invocation.
The golden corpora report 322/335 and 55/55 equal, with zero unexpected differences and 13 inherited known differences.
The lean result is `426 problems, 426 on the base, 0 new`. [lean-new.txt](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/lean-new.txt) is the comparison receipt.
`git diff --check` passes. All 28 changed files are at most 600 lines; the largest is 597.
No full-model, Metal, HIP, Windows, multi-GPU, or sampled-decoding qualification is claimed.
Independent review is NOT RUN in this implementation sub-agent. The parent can review the named shared-code changes.

Model the Domain kept typed request options and the existing single lane owner.
Prove It Works drove the source-extracted HF/pod comparisons and actual native HTTP checks.
Sequence Work into Verifiable Units kept the protocol, admission, P1 route equality, and UTF-8 correction in ordered commits.

## Changed files and commits

Line counts are for the final commit. Full source hashes are in [source-manifest.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/source-manifest.json).

| File | Lines |
| --- | ---: |
| [tools/zig/llama_serve_check.py](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_serve_check.py) | 281 |
| [tools/zig/llama_serve_reference.py](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_serve_reference.py) | 146 |
| [zig/build/server.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/build/server.zig) | 126 |
| [zig/src/families/llama/native.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/families/llama/native.zig) | 76 |
| [zig/src/native/cuda.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/native/cuda.zig) | 555 |
| [zig/src/server/chat.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/chat.zig) | 597 |
| [zig/src/server/errors.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/errors.zig) | 56 |
| [zig/src/server/fixtures/llama/outputs.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/fixtures/llama/outputs.json) | 17 |
| [zig/src/server/fixtures/llama/prompts.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/fixtures/llama/prompts.json) | 17 |
| [zig/src/server/fixtures/llama/sources.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/fixtures/llama/sources.json) | 10 |
| [zig/src/server/hf_text.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/hf_text.zig) | 546 |
| [zig/src/server/llama_output_test.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/llama_output_test.zig) | 226 |
| [zig/src/server/llama_prompt_test.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/llama_prompt_test.zig) | 76 |
| [zig/src/server/model_text.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/model_text.zig) | 198 |
| [zig/src/server/nanbeige.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/nanbeige.zig) | 97 |
| [zig/src/server/openai.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/openai.zig) | 579 |
| [zig/src/server/openai_client.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/openai_client.zig) | 38 |
| [zig/src/server/prompt.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/prompt.zig) | 183 |
| [zig/src/server/reply_text.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/reply_text.zig) | 523 |
| [zig/src/server/root.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/root.zig) | 198 |
| [zig/src/server/serve.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/serve.zig) | 429 |
| [zig/src/server/server.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/server.zig) | 529 |
| [zig/src/server/tool_params.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_params.zig) | 428 |
| [zig/src/server/tool_parse.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_parse.zig) | 551 |
| [zig/src/server/tool_qwen.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_qwen.zig) | 40 |
| [zig/tests/cuda/llama_main.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_main.zig) | 38 |
| [zig/tests/cuda/llama_serve_oracle.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_serve_oracle.zig) | 21 |
| [zig/tests/cuda/llama_tests.zig](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/tests/cuda/llama_tests.zig) | 294 |

- `b1f865ea78878df4c03667b2b3fceb5dba6e1be3 Serve Nanbeige chat prompts and qwen3 output through the native API`.
- `2ab15b6364f17fcb1004644505961af0549e3351 Complete Nanbeige admission and guarded serving checks`.
- `30cccbc78e9fb9a2cff49c643716e339dbdfd025 Compare the P1 chat prompt through both serving routes`.
- `e8d79bcc1b1b510a74cdd58784031bd7a2e83d73 Keep streamed Nanbeige reasoning on UTF-8 boundaries`.

## Files over 1 MB

The threshold is 1,000,000 bytes. Every retained file above it is listed here.
The tokenizer symlinks point to the original read-only assets. Build caches and the baseline source archive were removed
after verification. Installed binaries, the slice, token dumps, logs, source hashes, and receipts remain here.
[manifest.json](/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2/manifest.json) records their identity. The full-model blocked attempts have no process-VRAM receipt because no process starts.

| File | Bytes | Ownership |
| --- | ---: | --- |
| `install/bin/tensorfold` | 37,256,216 | Generated here |
| `install/bin/tf-cuda-test` | 90,015,544 | Generated here |
| `install/bin/tf-llama-test` | 16,456,192 | Generated here |
| `install/bin/tf-qwen35-cuda` | 20,986,976 | Generated here |
| `install/bin/tf-qwen35-learn` | 20,127,384 | Generated here |
| `install/bin/tf-tp-test` | 7,499,784 | Generated here |
| `install/bin/tf-tp-watch-check` | 4,004,248 | Generated here |
| `install/native/bin/tensorfold-native` | 81,759,264 | Generated here |
| `slice/model.safetensors` | 3,808,140,252 | Generated here |
| `slice/tokenizer.json` | 18,450,979 | Read-only source symlink |
| `slice/tokenizer.model` | 2,782,298 | Read-only source symlink |
