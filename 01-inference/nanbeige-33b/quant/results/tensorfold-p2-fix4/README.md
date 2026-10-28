# P2 fix round 4

Candidate 7fe2dce77a5913310a6edef719e278225b08b02d on local `nanbeige`, based on `8fd873cb`. No push or GPU execution.

## Before and after

| Finding | Before | After evidence |
| --- | --- | --- |
| LFM rewriting | Prose and string arguments became DSML | Both stay literal; `full-response.log`, permanent HTTP cases `lfm-prose` and `lfm-argument` |
| Numeric typing | Unicode integers and nonfinite floats became strings; `1e2` stayed float | Python integer and float conversion, integral-float rule, NaN and Infinity; 226 typed byte cases cover every Unicode 15 Nd block |
| Object and array literals | `[1,]\n ` became an array; `TRUE` became true | Exact JSON then literal subset, raw fallback on invalid syntax; trailing-newline indentation, tuples, escapes, signs, duplicate keys, big integers, typed fixtures |
| Parallel calls | `parallel_tool_calls:false` returned one of two calls | Both calls returned; reviewer `limit-calls` and permanent HTTP `parallel-false` |
| Envelopes | Empty string or omitted fields, extra metadata and SSE usage | Pod field order, nulls, empty arrays, initial delta, optional usage, UUID4 IDs, requested model; 61 exact JSON bodies and 61 exact SSE bodies |
| Argument serialization | Compact separators, `call_` IDs, wrong number type | Python JSON separators, Unicode, float repr, exact ints and dictionary order; 24,005 independent reviewer cases have zero semantic and zero serialization mismatches |

The reviewer's parser differential went from 2,323 semantic plus 15,590 serialization-only mismatches to 0 plus 0. The full-response aggregate went from 5 mismatches among 23 scenarios to 0. The permanent tests compare wire bytes after normalizing IDs and creation times, with matching generated text chunks; the reviewer's aggregate check intentionally uses different chunk boundaries.

## Implementation

Five new pod modules own output parsing, numeric conversion, Python literals, requests, and response envelopes. `nanbeige.zig` retains prepare and reasoning streaming. The old `tool_qwen.zig`, shared `decodeQwen`, exact-schema helpers, Qwen XML mode, and obsolete pod streaming adapter were deleted. Shared strict parsers retain their behavior. `source-hashes.json` binds these receipts to the complete changed tree.

Model the Domain led to the isolated pod boundary. Build the Lever produced the permanent AST-labelled grammar and response fixtures. Prove It Works led to production HTTP-writer byte comparisons.

Request-side checks also fixed null sampler defaults, zero or omitted completion limits, model echo, ignored tool choice, stop strings, penalties, logprobs and structured-output controls, media/context error envelopes, and assistant JSON arguments before rendering. The generator labels 27 prepared-message cases and 28 HF prompt cases, including scalar, null, array, invalid and duplicate-key history arguments.

## Verification

All commands set `CUDA_VISIBLE_DEVICES=''`. Zig is `~/.local/share/zig-x86_64-linux-0.17.0/zig`. Python oracle is `~/exl3-local/bin/python`, version 3.12.14 with Unicode 15.0.0.

| Gate | Result | Receipt |
| --- | --- | --- |
| Required five targets from `zig/` | 991/1002, same 11 inherited relative-fixture failures as review 4 | `tests-zig-cwd.log` |
| Required five targets from engine root | 17/17 build steps, 963/963 freshly executed tests | `tests-root.log` |
| Llama server host | 36/36 | `tests-root.log` |
| Server CPU | 887/887 | `tests-root.log` |
| Tool parse host | 39/39 fresh in first invocation, unchanged sources; final invocation cached | `tests-zig-cwd.log` |
| Recovery and HTTP visibility | 23/23 and 17/17 | `tests-root.log` |
| Pod fixtures | 3,000 tag soup, 1,005 grammar, 226 typed, 27 prepare, 28 prompts, 61 JSON and 61 SSE | `fixtures.log`, `tests-root.log` |
| Deterministic regeneration | All eight fixture files identical; 1,667,761 bytes total | `fixtures-repro.log` |
| Independent parser differential | 24,005 cases, 0 semantic, 0 serialization mismatches | `differential-after.json`, `differential.log` |
| Independent full responses | 23 scenarios, 0 aggregate mismatches | `full-response.log` |
| Lean against fix1 baseline | 426 base, 426 head, 0 new | `lean-head.txt`, `lean-new.log` |
| Native sm_86 compile | 116/116 steps, exit 0 | `native-build.log` |
| GPU generation and TB2 | NOT RUN, user prohibited GPU use | No generation-equivalence or score claim |

Reproduce the permanent generator from the engine root:

```sh
TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
POD=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py
CUDA_VISIBLE_DEVICES='' ~/exl3-local/bin/python tools/zig/llama_serve_reference.py "$TF_LLAMA_MODEL" "$POD"
CUDA_VISIBLE_DEVICES='' TF_LLAMA_MODEL="$TF_LLAMA_MODEL" ~/.local/share/zig-x86_64-linux-0.17.0/zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all
CUDA_VISIBLE_DEVICES='' ~/.local/share/zig-x86_64-linux-0.17.0/zig build native -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j4 --summary all
python3 tools/zig/lean_check.py > /tmp/head.txt
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt /tmp/head.txt
```

The reviewer scripts ran from `/tmp/nanbeige-fix4-review/`, copied from `/tmp/nanbeige-review-8fd873cb/`, with `source` pointing to the engine. The parser harness imports `nanbeige_output.zig` and omits the deleted maximum-call argument. The response harness is unchanged. Adapted sources are copied here as `reviewer-*`; the original sources and review evidence remain in `review4/`.

## Deliberate divergences

`divergences.log` verifies raw-string fallback for sets, bytes, complex numbers, Ellipsis, and tuple dictionary keys. The pod throws uncaught serialization `TypeError` for those. Named Unicode escapes also fall back to the raw string because this small native subset has no Unicode-name database; ordinary, raw, hex, code-point and octal string escapes are supported.

Malformed request types, invalid sampler ranges, nonpositive completion counts, and native admission/runtime failures retain native validation rather than reproducing every HF/FastAPI/EXL3 exception. Context admission uses the engine's actual window, which can be smaller than the pod's 262144. See the committed `docs/recipes/nanbeige-pod.md` for the compatibility boundary. No remaining mismatch was found in the reviewer's two corpora.

ATC is not installed in this environment. Work stayed within the user's explicitly assigned local branch and commit scope.
