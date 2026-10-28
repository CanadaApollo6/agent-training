# Review 891015b9

Read-only review of `7fe2dce7..891015b9` in `/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama`. No tracked edits or commits. Every execution was CPU only with `CUDA_VISIBLE_DEVICES=''`. Tokenization used local model files. The fake Engine submitted scripted token IDs to the production `openai.run`, `openai_client.ClientOut`, and HTTP connection writer. No model inference ran.

## Environment and commands

```sh
export CUDA_VISIBLE_DEVICES=''
export TOKENIZERS_PARALLELISM=false
export TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
export PYTHONPATH=/tmp/nanbeige-review-891015b9/deps
cd /home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
bash /tmp/nanbeige-review-891015b9/rerun.sh
```

The review 5 rerun was copied with only its scratch directory changed. Its script runs the requested command from the engine root with a fresh scratch cache:

```sh
~/.local/share/zig-x86_64-linux-0.17.0/zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all --cache-dir /tmp/nanbeige-review-891015b9/cache
python3 tools/zig/lean_check.py > /tmp/nanbeige-review-891015b9/lean-head.txt
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt /tmp/nanbeige-review-891015b9/lean-head.txt
```

The suite passed 1,009/1,009, all 17 steps. `rerun.sh` subsequently exits 1 because lean_check reports 426 inherited problems and the script has set -e. lean_new was run separately and passed with zero new problems. The misleading `failed command` lines printed alongside Zig warning output are followed by the final successful summary in tests-fresh.log.

Review 5 produced 22,074 cases, with 230 reference exceptions. All 21,844 serializable cases matched both JSON and SSE, with no prompt mismatch or missing result. Its targeted follow-up matched 64/64 cases.

Independent corpus:

```sh
~/exl3-local/bin/python /tmp/nanbeige-review-891015b9/extended.py
~/exl3-local/bin/python /tmp/nanbeige-review-891015b9/run_extended.py
~/exl3-local/bin/python tools/zig/llama_serve_unicode.py --check --ucd-dir /tmp/nanbeige-review-891015b9/ucd --zig ~/.local/share/zig-x86_64-linux-0.17.0/zig
```

The reference is Python 3.12.14, Unicode 15.0.0. The corpus enumerates all 1,114,112 code points, obtaining 143,041 names. Every canonical name and all 473 official aliases are checked as original, lower, and alternating mixed case. All 461 named sequences receive the same three spellings. Positive literals are batched into arrays of at most 128 strings. Rejected literals are separate parameters, 12 per request; each rejection is independently observable. There are 431,925 name literals and 2,330 additional edge literals, comprising 22,155 requests. Each request is driven through JSON and SSE separately with the real tokenizer and scripted CPU engine. All 21,936 serializable requests match both formats; 219 reference exceptions are recorded separately rather than counted as matches. No missing result or prompt mismatch.

Boundary literals include whitespace and hyphen variants, empty and unclosed name escapes, 50,000-character names, CJK interval endpoints and numeral formats, Hangul syllables, 508 names from Unicode 17 absent in Unicode 15, truncated and signed/underscored hex, octal, lone and paired surrogate escapes, raw/bytes/f-string prefixes, concatenation, triple quotes, NUL, dict keys, tuples, sets, complex values, and Ellipsis.

The eight exhaustive worker processes all exited 0. The original log join failed when decoding/re-encoding deliberately ill-formed surrogate bytes. The logs were joined as binary and then compared without rerunning the engine; run_extended.py now preserves binary bytes when joining. `independent.compare` reads WTF-8 with surrogatepass and serializes diagnostic files with ensure_ascii. Actual wire bytes are retained. Only random IDs and creation times are normalized. Reference errors are 205 TypeErrors and 14 UnicodeEncodeErrors.

The committed generator's additional reproduction check passes 295,962 Python and 295,961 Zig comparisons, with the gzip and metadata byte-identical. The one difference in counts is its excluded newline-containing line-oriented Zig input. Our HTTP corpus includes that newline case.

## Error response verification

```sh
~/exl3-local/bin/python /tmp/nanbeige-review-891015b9/exceptions.py
~/exl3-local/bin/python /tmp/nanbeige-review-891015b9/minimal.py
REVIEW_INPUT=/tmp/nanbeige-review-891015b9/minimal-input.jsonl /tmp/nanbeige-review-891015b9/routes > /tmp/nanbeige-review-891015b9/minimal-engine.log 2>&1
```

exceptions.py AST-extracts the actual pod chat/typed/split_output code and registers the chat route with FastAPI. It invokes ASGI directly, retaining response-start, body, completion, and exception events. EXL3 generation is replaced by the same scripted CPU job. There is no network listener and no GPU import or initialization from the pod script. For object-typed payloads `b'a'` and `'\ud800'`, the reference returns JSON HTTP 500 with `Internal Server Error`. SSE sends only the role frame, then aborts without completion. The native production route returns HTTP 200 and finishes SSE for both. The bytes value becomes raw text; the surrogate produces literal ED A0 80 bytes, invalid UTF-8, in JSON and SSE. `exceptions-asgi.json` and `minimal-engine.json` retain outputs. The same literal conversion and permissive WTF-8 write behavior exists at the parent commit. The bytes/set/complex/Ellipsis divergence is documented in the parent recipe. These are retained strict-parity blockers, not regressions introduced by 891015b9.

## Allocation and timing

```sh
python3 /tmp/nanbeige-review-891015b9/build_measure.py
/tmp/nanbeige-review-891015b9/measure > /tmp/nanbeige-review-891015b9/measure-final.log 2>&1
python3 /tmp/nanbeige-review-891015b9/build_routes_measure.py
~/exl3-local/bin/python /tmp/nanbeige-review-891015b9/bench_cases.py
REVIEW_INPUT=/tmp/nanbeige-review-891015b9/bench/input.jsonl /tmp/nanbeige-review-891015b9/routes_measure > /tmp/nanbeige-review-891015b9/bench/routes.log 2>&1
```

ReleaseSafe CPU-only measurements. This is handler/parser overhead with scripted generation, not inference latency or a GPU benchmark. The machine has 20 CPUs and unrelated work was running. The large exhaustive run finished before final timing. Five interleaved pairs per request width compare identical semantic plain and named values. All 30 HTTP scenarios match the reference's JSON and SSE. The timing includes openai.run and wire output but excludes tokenizer initialization, input-file reading, result logging, and model inference. `bench/timings.json` records all attempts. No CPU profiler was available, so no speedup claim is made; source and allocation measurements attribute the new per-literal work to gzip inflation.

For one named parameter, median JSON handler time is 3.602 ms, range 3.553–3.780 ms; SSE median is 3.271 ms. JSON arena capacity is 5,060,272 bytes. With 32 separate named parameters, JSON median is 98.408 ms, range 98.164–103.481 ms, arena capacity 57,903,662 bytes. These are arena capacity readings, not whole-process RSS.

The actual inflated table is 1,154,877 bytes. One Names instance keeps the same pointer and allocation capacity across 1,000 lookups. The parser constructs a new Names for each literal parameter and therefore inflates again per parameter and request. The owning HTTP connection arena resets before its next request with a 1 MiB retention cap and is deinitialized at connection teardown. Eight threads, 100 parses each, passed and the testing allocator reported no leaks. There is no shared mutable table state to race.

Changed Zig/Python sources meet 600-line and 120-column single-line-comment limits; maximum source length is 597 lines. git diff --check passes. GPU inference, full-model checks, actual EXL3 runtime, and TB2 are NOT RUN.
