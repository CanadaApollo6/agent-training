# TensorFold P2 fix 1

All six reported defects are fixed at 94c2c22be9cd429e8162e2284b963ee525ea76a0.
Local commits only. No push, issue or pull request was created.
Full-model qualification remains BLOCKED by shared GPU memory.

The input was 3aeb8e97513bfb0c2a66f2e32e5defad664bdd3d on nanbeige.
The broad and lean baseline is 31ce9eb3.
Weights, tokenizer files, pod server and previous receipts were read-only.
The original two-layer slice is reused from tensorfold-p2.

## Six corrections and regression evidence

The test-only commit 9633b5a3 preserves all six failing reproductions before production edits.
[regressions-before.log](regressions-before.log) reports 18/29 passed and 11 failed.
[regressions-final.log](regressions-final.log) reports all 29 tests passed.
Early fixture setup and compiler errors are excluded from failing-before evidence.

### 1. Spaced tool names and schema case

[tool_parse.zig:323](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_parse.zig:323) scans qwen3_coder names through the closing greater-than sign, rejects embedded newlines, and strips names with Python whitespace semantics.
Its function and parameter tags are case-sensitive.
[tool_params.zig:32](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_params.zig:32) collects exact schema keys.
[tool_params.zig:78](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_params.zig:78) looks them up without case folding.
[tool_qwen.zig:11](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/tool_qwen.zig:11) selects that schema policy.
Names with different case retain their own schemas.

The regressions are named Llama serve: spaced function names match the pod, Llama serve: spaced parameter names match the pod, and Llama serve: case-sensitive schemas match the pod.
Before, the function test returned the XML as content instead of a call.
The parameter and case tests failed with TestUnexpectedResult at the arguments equality assertion.
Their exact output is in [regressions-before.log](regressions-before.log).
After, [parser-after.log](parser-after.log) and [parser-case-after.log](parser-case-after.log) contain:

~~~text
OUTPUT spaced-function equal
OUTPUT spaced-parameter equal
OUTPUT case-function equal
OUTPUT case-parameter equal
OUTPUT case-collision equal
OUTPUT case-collision-upper equal
OUTPUT case-tags equal
~~~

[parser-all-after.log](parser-all-after.log) passes all 26 outputs from the actual pod parser.
The generic strict XML mode keeps its old name scan, case folding and tag handling.
Only Nanbeige calls the dedicated qwen3_coder mode.
Shared parser and HTTP visibility tests pass 56/56 in [parser-shared.log](parser-shared.log).
Commit b9bb7591 owns this fix.

### 2. Literal special-token text

[hf_text.zig:290](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/hf_text.zig:290) uses the ordinary special-token encoder for Nanbeige instead of request-span masking.
Other profiles retain masked encoding.
The regression is Llama serve: literal special-token text matches HF.

~~~text
Before: prompt special-user: HF=39 TF=45
Before: prompt special-tool: HF=375 TF=381
After:  PROMPT special-user tokens=39 equal
After:  PROMPT special-tool tokens=375 equal
~~~

See [regressions-before.log](regressions-before.log), [special-after.log](special-after.log) and [regressions-final.log](regressions-final.log).
Commit b56332ea owns this fix.

### 3. Message roles

[messages.zig:85](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/messages.zig:85) validates messages while preserving their roles and order.
[openai.zig:108](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/openai.zig:108) selects that policy during request preparation.
[prompt.zig:33](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/prompt.zig:33) selects it during the second normalization.
Generic families still merge leading instructions and use their existing late-system behavior.
The regression is Llama serve: developer roles match HF through both normalizations.

~~~text
Before: prompt developer-leading: HF=42 TF=21
Before: prompt developer-mixed: HF=42 TF=39
After:  PROMPT developer-leading tokens=42 equal
After:  PROMPT developer-mixed tokens=42 equal
~~~

See [roles-after.log](roles-after.log) and [regressions-final.log](regressions-final.log).
Commit fca75a7a owns this fix.

### 4. Template control precedence

[hf_text.zig:375](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/hf_text.zig:375) first collects raw kwargs.
For generic profiles it then applies the resolved enable_thinking, reasoning_effort and GLM clear_thinking values.
It removes raw values for deliberately unset controls and protects the resolved thinking_mode global.
For Nanbeige it passes raw kwargs without injecting those resolved controls, matching serve_exl3.py.

The regressions are Llama serve: resolved template controls retain upstream precedence outside Nanbeige and Llama serve: raw template kwargs match the pod.

~~~text
Before expected: high|True|False|kept
Before found:    low|False|True|kept
After: Llama serve: resolved template controls retain upstream precedence outside Nanbeige...OK
After: Llama serve: raw template kwargs match the pod...OK
~~~

See [controls-after.log](controls-after.log) and [regressions-final.log](regressions-final.log).
[controls-boundary-before.log](controls-boundary-before.log) preserves the additional thinking_mode failure.
[controls-boundary-after.log](controls-boundary-after.log) passes explicit, absent and unset-control cases.
Nanbeige separately checks that raw kwargs win and that absent kwargs stay absent.
Commits 66442f8e and 9feb8600 own this fix.

### 5. Context refusal before streaming

[openai.zig:494](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/openai.zig:494) sends Nanbeige context errors through the ordinary refusal path while the stream is unopened.
Direct preparation, compaction and unopened generation use the same policy.
Other profiles retain streamed context errors.

The regression is Llama serve: context overflow is HTTP 400 before SSE with and without compaction.
It exercises chat and completions, verifies no SSE events, and verifies no engine submission.

~~~text
Before: expected 400, found 200
After: Llama serve: context overflow is HTTP 400 before SSE with and without compaction...OK
~~~

See [context-after.log](context-after.log).
The real CUDA slice server also returns HTTP 400 with application/json for the oversized stream request.
See the fourth entry in [greedy-serve/rejections.json](greedy-serve/rejections.json).
[llama_serve_check.py:180](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_serve_check.py:180) retains that native regression.
Commit 8794a8af owns the production fix.
Commit 94c2c22b extends the native check.

### 6. Unicode stop strings

[reply_text.zig:389](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/reply_text.zig:389) bounds the decode window by the longest stop's UTF-8 byte length plus eight context tokens.
Both the engine stop hook and server generation consume that bound.
Ten lightbulbs occupy 40 bytes and 41 HF tokens.
The former 18-token window could never contain the complete stop.

The regressions are Llama serve: Unicode stops end generation across more tokens than characters and Unicode stop strings end JSON and SSE replies with byte-sized tokens.

~~~text
Before HF route expected: stop
Before HF route found:    length
After: Llama serve: Unicode stops end generation across more tokens than characters...OK
Before upstream generic test: expected .stop, found .length
After upstream generic test: All 15 tests passed.
~~~

See [unicode-after.log](unicode-after.log), [unicode-base-before.log](unicode-base-before.log) and [unicode-base-after.log](unicode-base-after.log).
The generic test runs without Nanbeige, weights or HF assets.
A fresh upstream archive fails with only the regression added and passes with the one-line bound correction.
[unicode-standalone.patch](unicode-standalone.patch) records the portable change.
**c523f51d is the standalone Unicode stop commit.**
It contains the shared fix and independent generic JSON/SSE regression.

## HF and pod fixtures

[llama_serve_reference.py](/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/tools/zig/llama_serve_reference.py) uses the installed HF AutoTokenizer and AST-extracted functions from the unchanged serve_exl3.py.
It does not reimplement the expected renderer, tokenizer or output parser.
[fixtures.log](fixtures.log) reports:

~~~text
HF prompts=23 tokens=3118 pod outputs=26
~~~

All 3,118 prompt IDs match.
Cases include literal special tokens in user and tool content, a leading developer, mixed developer history, kwargs with and without top-level reasoning_effort, and spaced tool-call history.
Parser cases include spaced function and parameter names, Unicode whitespace, empty trimmed names, embedded newlines, case variants and case-colliding schema names.
All 26 raw outputs also pass actual JSON and SSE handlers with the real tokenizer.
[parity-counts.json](parity-counts.json) records the counts.
The source fixture manifest records pod/tokenizer hashes and HF package versions.

## Existing host and slice checks

The final native build succeeds at 128/128 steps and 29/29 executed tests in [build-final.log](build-final.log).
The seven Llama loader tests pass in [regressions-after.log](regressions-after.log) and are cached in the final build.
The final focused standalone execution passes 29/29 in [regressions-final.log](regressions-final.log).
The plain build passes 24/24 steps in [build-plain.log](build-plain.log).

[host-final.log](host-final.log) reports 457/460 steps, 704/708 tests passed, three skipped and one failed.
Only the known exl3_load allocation test fails with NondeterministicMemoryUsage at line 319.
The first broad head invocation reports 5,498/5,525 passed, 26 skipped and that same single failure in [host-all.log](host-all.log).
A fresh 31ce9eb3 archive reports 458/458 steps and 5,520/5,546 passed, 26 skipped and no failures in [host-base.log](host-base.log).
These are per-invocation execution counts, not distinct test inventories.
The final head run reuses cached work.

A targeted fresh base run also passes in [exl3-base.log](exl3-base.log).
This run does not claim a fresh base failure.
[inherited-exl3.json](inherited-exl3.json) verifies that exl3_load, safetensors and save_guard match the fresh base byte-for-byte.
The prior tensorfold-p2b/exl3-base.log retains an upstream failure of that same test.
No change in this task touches those files.

Other families' server tests pass in the aggregate.
Generic context preflight tests, Qwen-family chat tests, Dory, GLM clear_thinking and DeepSeek prompt tests remain in that server run.
The focused shared parser/visibility run passes 56/56.
Kernel contracts, server goldens and Cortheon fuzzing pass.
Goldens report 322/335 and 55/55 equal, zero unexpected differences, and 13 known differences.

[greedy-serve/summary.json](greedy-serve/summary.json) passes on the original two-layer slice.
Chat and completions each equal the independent serial oracle at all 128 tokens.
Queued requests, streamed token IDs and cancellation/reset reuse agree.
Four pre-stream HTTP refusals pass, including context overflow.

[sampling.log](sampling.log) and [statistics.json](statistics.json) pass all seven cases and 204,800 draws.
The deliberately wrong target fails in [statistics-control.log](statistics-control.log).
[sampled-decode.log](sampled-decode.log) passes serial/lane/verify equality with penalties off and on.
[sampled-serve/summary.json](sampled-serve/summary.json) passes both HTTP endpoints, defaults, seeds, filters, penalties, queues, SSE and cancellation/reset.
It reuses the trace-derived pi request read-only, gets HTTP 200, and disconnects after three SSE events.
This does not claim a completed 16,384-token pi turn.

[greedy12-regression.json](greedy12-regression.json) compares fresh native runners from 3aeb8e97 and the final implementation.
All 128 tokens are equal.
Both retain the inherited exllamav3-reference failure, exit 1, rather than count it as a pass.
No full-model equality or performance claim follows from a slice.

## GPU admission and blocked work

[GPU telemetry](gpu-summary.json) records every actual device run.
Every command used llama_gpu_run.py with an estimate above its observed peak and a 2,048 MiB reserve.
Only owned child servers were signaled.
The largest observed pre-admission free memory was below 8 GiB.
No observation reached the requested 19.5 GB threshold.

[full-p1-blocked.log](full-p1-blocked.log) refuses 17,000 MiB plus the reserve.
[full-serve/summary.json](full-serve/summary.json) refuses 15,500 MiB plus the reserve.
Neither full-model binary starts.
The 52-layer logits, full-model dense-head 128-token equality, five alternating speed pairs, full-model bash tool use and server throughput remain BLOCKED and NOT RUN.

Run these exact commands when memory admits them.
The P1 helper also runs the older boundary/control checks before its full-model gates.

~~~sh
TF_ROOT=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
TF_QUANT=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant
TF_OUT="$TF_QUANT/results/tensorfold-p2-fix1"
TF_MODEL="$TF_QUANT/models/3.5-nb"
TF_P1="$TF_QUANT/results/tensorfold-p1"
cd "$TF_ROOT"
python3 tools/zig/llama_fix2_verify.py "$TF_MODEL" "$TF_P1" "$TF_OUT/full-p1" --binary "$TF_OUT/install/bin/tf-llama-test"
python3 tools/zig/llama_serve_check.py serve "$TF_OUT/install/native/bin/tensorfold-native" "$TF_MODEL" "$TF_OUT/full-serve" --need-mib 15500 --context 4096 --timeout 600 --full --smoke "$TF_QUANT/smoke_serve.py" --oracle "$TF_OUT/install/bin/tf-llama-test" --refs "$TF_P1"
~~~

The full P1 helper itself was NOT RUN in this task; guarded full-model admission was attempted directly.
Other chips, Metal, HIP, Windows, multi-GPU, drafts and strict tool grammars are NOT RUN.

## Reproduction and identities

[commands.jsonl](commands.jsonl) records expanded commands, working directories, input model and exit status.
[filter.py](filter.py) preserves the exact Zig module invocation for individual regressions.
[run.py](run.py) preserves the common environment, log routing, cache and install paths.
The fresh baseline invocation uses install/base-source with the same absolute fatbin source and dedicated host-cache/base.
The archived base and input checkouts match all 8,768 and 8,808 tracked Git blobs respectively, as recorded in their identity manifests.
[source-manifest.json](source-manifest.json) binds changed source, binaries, CUDA images and read-only references.
The executed binaries are preserved under install/verified with their original invocation paths and matching hashes.
An additional identical-source rebuild changed the native executable hash.
Its untested output is retained under install/rebuilt.
The tested executable was restored to install/native/bin/tensorfold-native.
The manifest distinguishes rebuild image inputs from images found byte-for-byte inside the tested executables.
[commits.json](commits.json) gives full commit SHAs.

[lean-new.txt](lean-new.txt) reports 426 problems, 426 on fresh 31ce9eb3, zero new.
Every changed file has at most 585 lines.
git diff --check, Zig formatting and Python compilation pass.
Native binaries generated for this task are under install/ or host-cache/.
[large-files.json](large-files.json) lists other receipt files above 1,000,000 bytes and is empty.

## Local commits

| Commit | Purpose |
| --- | --- |
| 9633b5a3 | Failing tests and HF/pod fixtures before production fixes. |
| c523f51d | Standalone Unicode stop fix and generic JSON/SSE regression. |
| b9bb7591 | Qwen XML names and exact schema keys. |
| b56332ea | HF-compatible literal special-token encoding. |
| fca75a7a | Preserve Nanbeige roles through both passes. |
| 66442f8e | Resolved template controls win outside Nanbeige. |
| 8794a8af | Context HTTP refusal before SSE. |
| 9feb8600 | Preserve unset controls and thinking_mode precedence. |
| 94c2c22b | Native context refusal check and regression formatting. |

Model the Domain kept role policy explicit and retained the existing profile boundary.
Prove It Works required actual HF IDs, pod parser outputs and native HTTP checks.
Test Behavior, Not Implementation checks IDs, calls, statuses and finish reasons.
Sequence Work into Verifiable Units put failing tests first and kept the Unicode fix separate.
Fix Root Causes changed scanning, encoding, normalization, precedence and admission at their owning boundaries.

Attention.
This implementation sub-agent performed its own source and evidence review.
No further agents or independent review were run.
The known allocation failure and VRAM-blocked full-model gates remain visible above.

