# Read-only review of 9a5732e0

Verdict: ACCEPT under the supplied TB2 relevance bar. No tracked files were edited. No commit, GPU work, or TB2 evaluation ran.

The checkout stayed at 9a5732e0d6a4c8a2ab8bb4b4a5ae55d3f2e8306b. The baseline is a git archive of 891015b9 under base/source. source-hashes.json records the reviewed files and reference pod. The review inspected every nanbeige*.zig file, the changed shared modules, and the openai/chat call sites.

## Differential replay

sweep.py is a new deterministic generator with seed 0x9a5732e0. It reconstructs the same 55 first/middle/last pilot histories as review5, using the original trace and its actual read/bash/edit/write schemas. Each history receives 182 new continuations. They include multiple calls, JSON and Python argument representations, long heredocs, JSON strings inside arguments, CRLF, tabs, ANSI escape characters, named Unicode escapes, invalid escapes, non-ASCII text, incomplete call envelopes, and length-limited replies. The largest prompt is 94,967 tokens; the largest generated continuation is 44,959 tokens. Expected replies use the actual Python 3.12 FastAPI/Starlette route, with only generation and tensor/sampler objects stubbed. The tokenizer and chat template are real. Prompt rendering is cached only on the Python oracle side.

routes.zig adapts the prior production-wire driver. It feeds the same tokenizer IDs to a CPU scripted engine, calls openai.run through ClientOut and Conn, and captures production HTTP writes. It checks preparing returns to zero and output is valid UTF-8. Prompt token sequences are compared exactly for each of the 55 histories. For all 10,010 requests, comparison checks HTTP status, content type, HTTP-body completion, and exact application bytes in both JSON and SSE. Only random completion/tool IDs and timestamps are normalized. Usage and finish_reason remain significant. Result: 20,020/20,020 matches and 55/55 prompt matches.

The first run exposed a test setup error: eight long scripts exceeded a 32,768-token engine request limit, while the stub emitted their full text. The corrected corpus requests 65,000 tokens, retaining explicit length-limit cases. The entire corpus was regenerated and rerun. harness-limit-correction.json retains the obsolete mismatch count; comparison-summary.json is the final result. This was a test setup correction, not an engine finding.

## TCP failures, cancellation, and allocation ownership

llama_serve_failures.py regenerated all 148 fixture rows and seven injected-fault rows with the reference interpreter. Both generated files are byte-identical to the committed fixtures. The fresh shared tests exercise all 148 JSON and SSE expectations and seven fault boundaries.

tcp.zig starts Server.serve on an ephemeral loopback port with the real tokenizer, routes, HTTP reader/writer, connection lifecycle, and request allocation ownership. Only inference is replaced with an asynchronous CPU engine that admits one active request and supports cancellation. tcp_check.py also starts the actual pod app with uvicorn/h11. Every failure fixture runs in JSON and SSE against both servers, immediately followed by a successful request on that same server. Twelve mid-reasoning disconnects per server are followed by successful JSON and SSE requests.

Result: 1,232 exact TCP response comparisons, including 640 recovery responses, with zero mismatches. All 24 disconnect probes succeed. Both generators receive all 12 disconnect cancellations. The engine also cancels one generation whose model-name surrogate prevents its first role frame from encoding. All 617 quiescent engine allocation snapshots are exactly 19,692 live bytes. The counting DebugAllocator reports no leak at teardown; preparing and active connections/jobs drain to zero. The TCP test exits zero and its 13 linked tests pass. This proves CPU HTTP/server ownership, not GPU scheduler behavior.

## Success-path cost

bench.py alternates baseline/head five times on 108 pilot continuations using identical ReleaseSafe builds and tokenizer/history input. The first timing line is prefixed by the Zig test runner and excluded from timing aggregation, leaving 107 measured cases per run. All 2,160 output comparisons pass, including usage, finish_reason, and SSE frame bytes. Framing/content-type changes in the baseline are intentionally decoded before application-body comparison.

Machine load was 7.09/8.41/11.73 on 20 logical CPUs, with unrelated workloads present. They were not disturbed. The total measured JSON+SSE time per run had overlapping ranges: baseline 1,061.7–1,567.8 ms, head 1,064.9–1,875.6 ms. Median totals were 1,200.9 and 1,158.7 ms. This supports no latency-regression finding; it is not a speedup claim or a GPU inference measurement. profile.zig separately instruments the interval before engine submission to identify request preparation versus output processing.

The request-arena capacity increase is deterministic. Median JSON increase is zero; maximum increase is 1,024,188 bytes, for pilot-54-00109. New templateFailure reparses historical string arguments at nanbeige_request.zig:70 before prepare reparses them at nanbeige.zig:46. Reusing normalized arguments is a MINOR allocation follow-up. Arena capacities include allocator growth, not just decoded argument bytes. All request memory is reclaimed.

Resource guards in nanbeige_failure.zig remain tied to the supplied interpreter and call stack. Their portability is a documented MINOR follow-up under the user's bar.

## Commands

All model-aware commands used CUDA_VISIBLE_DEVICES=''. Python used ~/exl3-local/bin/python (3.12.14), PYTHONDONTWRITEBYTECODE=1 can be set for reruns, and PYTHONPATH=/tmp/nanbeige-fix6/deps:/tmp/nanbeige-review-891015b9/deps. The latter are the prior CPU dependency directories. No network dependency installation was needed.

From the engine root:

```bash
TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb CUDA_VISIBLE_DEVICES='' ~/.local/share/zig-x86_64-linux-0.17.0/zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all
```

The same command with `--cache-dir /tmp/nanbeige-review-9a5732e0/zig-cache` freshly passes 1,015/1,015 tests, 17/17 build steps. Breakdown: 40 llama, 891 server, 5 bundle, 39 parser, 23 recovery, 17 HTTP visibility.

```bash
python3 tools/zig/lean_check.py > /tmp/nanbeige-review-9a5732e0/lean-head.txt
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt /tmp/nanbeige-review-9a5732e0/lean-head.txt
```

Lean has 426 inherited findings and zero new. Changed source files are at most 600 lines and single-line comments at most 120 columns. Zig format and git diff whitespace checks pass.

Scratch drivers, in execution order:

```bash
python3 /tmp/nanbeige-review-9a5732e0/build_routes.py
~/exl3-local/bin/python /tmp/nanbeige-review-9a5732e0/sweep.py
TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb REVIEW_INPUT=/tmp/nanbeige-review-9a5732e0/input.jsonl /tmp/nanbeige-review-9a5732e0/routes > /tmp/nanbeige-review-9a5732e0/routes.log 2>&1
~/exl3-local/bin/python /tmp/nanbeige-review-9a5732e0/compare.py
python3 /tmp/nanbeige-review-9a5732e0/build_tcp.py
~/exl3-local/bin/python /tmp/nanbeige-review-9a5732e0/tcp_check.py
python3 /tmp/nanbeige-review-9a5732e0/bench.py
python3 /tmp/nanbeige-review-9a5732e0/summarize_bench.py
```

The failure-fixture regeneration command was:

```bash
~/exl3-local/bin/python tools/zig/llama_serve_failures.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py --output /tmp/nanbeige-review-9a5732e0/failures.json
```

The separate instrumented profile attributes 94.9% of the measured whole-request CPU time to work before engine submission, primarily request normalization, template rendering, and tokenization. Its timings are excluded from the baseline/head comparison. The Prove It Works principle drove the real TCP failure/recovery checks and live allocation accounting.
