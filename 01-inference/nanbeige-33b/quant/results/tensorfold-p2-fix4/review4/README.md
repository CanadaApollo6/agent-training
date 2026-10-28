# Independent read-only review of 8fd873cb against 6d56a1c4

Verdict: REJECT for P2 acceptance. The three claimed fixes pass. Remaining parity defects predate this commit.

No tracked files were edited. No commit, push, native model load, or GPU operation was performed.
All generated review sources and data are in this directory. `source` links to the reviewed repository.

## Methods and results

- `differential.py` is an independent grammar generator, seeded with 0x8fd873cb. It generates valid model-like calls and mutates their headers, closers, whitespace, truncation, duplicate calls, and schema values. Schemas include integer, number, boolean, object, array, string, enum, and nullable integer.
- The Python oracle compiles the actual pod's `typed`, `split_output`, and regex constants from its AST. No copied implementation or committed fuzz generator supplies labels.
- `parser.zig` imports the actual engine's Nanbeige split and qwen parser, with real JSON serialization. Both sides run on all 24,005 cases. Results were repeated under the requested `~/exl3-local/bin/python` (3.12.14), with identical counts to the initial system-Python run.
- 18,917 cases have tool calls. There are 2,323 semantic mismatches and 15,590 additional serialized-arguments-only mismatches. Random call IDs are excluded from these counts and examined separately.
- `classify.json` groups differing argument values. There are no split, content, function-name, or parameter-boundary differences in this corpus. The semantic differences are Unicode numeric conversion, nonfinite numbers, and Python literal acceptance. Counts in `classify.json` count arguments, and can exceed mismatching cases.
- `minimal.json` and `minimal-pod-python.log` contain reduced examples.
- `base_check.py` executes a scratch overlay of the two changed parser files at the parent commit. It confirms four failing directed cases at the parent pass at the head. The fifth (terminal newline after all spans are unclosed) passes both because the old split pre-stripped the answer. Head: 5/5 directed checks pass.
- `routes.zig` uses the actual `openai.run` path, real model tokenizer, and CPU-only scripted engine, capturing whole JSON and streaming events. `full_response.py` executes the pod's actual `chat` function via AST, with fake generation/HF admission and a streaming-response collector. Usage counts are supplied by the fake job; this does not validate generation or prompt admission. There are 23 scenarios, each JSON and SSE. Five have differing aggregate stream values: Unicode integer, Python literal, parallel-call limit, LFM prose, and LFM argument text. Finish reasons match 23/23. Envelope fields, argument JSON strings, and call-ID formats differ separately.
- Streaming reasoning preserves raw `<think>` and surrounding newlines on both servers; nonstream reasoning applies the pod's split normalization. This difference between streaming and nonstreaming is intentional reference behavior.

## Findings and minimal inputs

All tool cases offer a function `run`. Parameter types are indicated below.

1. `zig/src/server/chat.zig:492` rewrites LFM-looking text on Nanbeige. Raw `</think><|tool_call_start|>[run()]<|tool_call_end|>` is unchanged content on the pod, but becomes `<｜DSML｜tool_calls>\n<｜DSML｜invoke name="run"></｜DSML｜invoke>\n</｜DSML｜tool_calls>` on the engine. The same replacement corrupts a string parameter containing that literal. Skip LFM rewriting for `pod_chat`.
2. `zig/src/server/tool_params.zig:165` converts `</think><tool_call><function=run><parameter=i>١` as string `"١"` instead of integer `1`. Number parameters also reject Unicode digits; nonfinite `nan`/`inf` are strings instead of Python's NaN/Infinity. Match Python numeric conversion on the Qwen path, including Unicode digits/whitespace, nonfinite serialization, and the number-to-integer rule when no dot occurs.
3. `zig/src/server/tool_params.zig:176` applies a broader literal decoder than `ast.literal_eval`. For array parameter `a`, raw `</think><tool_call><function=run><parameter=a>[1,]\n ` gives the pod string `"[1,]\n "` but engine array `[1]`. Object value `TRUE` is also incorrectly accepted as boolean true. Use Python-equivalent literal acceptance without unconditional stripping/case folding on this path.
4. `zig/src/server/openai.zig:312` applies `parallel_tool_calls:false` to Nanbeige, while the reference ignores it. With raw `</think><tool_call><function=run></tool_call><tool_call><function=run></tool_call>`, pod returns two calls and engine one. Match the reference's unlimited-call policy for Nanbeige, or change the scoring reference and contract together.
5. `zig/src/server/openai.zig:396` has different empty-field behavior. Empty output yields pod content/reasoning/reasoning_content null and tool_calls []; engine content "" and other fields absent. Engine choices omit logprobs:null. SSE starts with role only instead of role+content:"", and includes usage without include_usage. Scope pod-equivalent envelope serialization to Nanbeige.
6. `zig/src/server/tool_parse.zig:206` emits compact arguments and `call_` + 24 hex digits, versus the pod's default JSON separators and `chatcmpl-tool-` + 16 hex digits. Integer parameter 1 gives `{"i":1}` versus `{"i": 1}`. Number 1e2 gives 100.0 versus 100. Use a Nanbeige-specific call serializer and Python-equivalent number typing.

## Commands and gates

From the repository's `zig/` directory, with TF_LLAMA_MODEL set to
`/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb`
and CUDA_VISIBLE_DEVICES empty:

```
~/.local/share/zig-x86_64-linux-0.17.0/zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all
```

`tests.log`: Nanbeige 32/32 (including 3,000 committed fuzz cases), parser 39/39, recovery 23/23, visibility 17/17, JSON 5/5. Server CPU initially 872/883 because 11 tests use repository-root-relative fixture paths. The root-directory build was cached; the exact newly compiled server test executable was therefore run directly, fresh, from the repository root:

```
TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb CUDA_VISIBLE_DEVICES='' .zig-cache/o/e4f3a5976b8cbecba4e185871eeca217/test
```

`server-cpu-fresh.log`: all 883 tests pass.

```
python3 tools/zig/lean_check.py > /tmp/nanbeige-review-8fd873cb/lean-head.txt
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt /tmp/nanbeige-review-8fd873cb/lean-head.txt
```

426 base, 426 head, zero new problems. All seven changed files are at most 600 lines; changed sources have no single-line comments exceeding 120 columns. `git diff --check 6d56a1c4 8fd873cb` passes. Final working tree is clean.

Independent checks:

```
~/exl3-local/bin/python /tmp/nanbeige-review-8fd873cb/differential.py
~/exl3-local/bin/python /tmp/nanbeige-review-8fd873cb/minimize.py
~/exl3-local/bin/python /tmp/nanbeige-review-8fd873cb/base_check.py
python3 /tmp/nanbeige-review-8fd873cb/build_routes.py
~/exl3-local/bin/python /tmp/nanbeige-review-8fd873cb/full_response.py
```

Parser build command:

```
~/.local/share/zig-x86_64-linux-0.17.0/zig test --dep json -Mroot=/tmp/nanbeige-review-8fd873cb/parser.zig -Mjson=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama/zig/src/server/json.zig -lc --test-filter independent --test-no-exec -femit-bin=/tmp/nanbeige-review-8fd873cb/parser
```

Full-model GPU checks: NOT RUN by instruction.
