# P2 fix round 6

Local `nanbeige`, based on `891015b9`. The engine commit is recorded in `commit.txt`. No push, agent-training commit, or GPU execution.

## Before and after

| Failure | Before | After |
| --- | --- | --- |
| Lone surrogate | HTTP 200 with ED A0 80; SSE completed with invalid UTF-8 | Plain HTTP 500; SSE preserves earlier frames and aborts before the bad frame |
| Bytes, sets, complex, ellipsis, unsupported keys | Raw parameter returned and stream completed | Pod serialization failure, plain 500 or aborted SSE |
| Recursion and parser stack limits | Raw fallback or native limit at depth 900 | Separate decode, AST, and serialization failures; 901-deep valid JSON stays typed |
| Template history exceptions | Engine accepted invalid XML arguments | Plain 500 before SSE opens; JSON-format history remains accepted |
| Output allocation failure | Silent return or successful raw fallback | Fixed 500 body, aborted stream, or producer error frame at the pod's boundary |

`minimal-before.json` is a fresh run of the review's baseline executable. The two minimal repros each mismatch JSON and SSE. `tests-root.log` records their permanent corrected wire assertions. The baseline invalid bytes remain in the hashed `/tmp` raw log.

## Enumerated failure paths

All unsent output failures return HTTP 500, `text/plain; charset=utf-8`, content length 21, and `Internal Server Error`. SSE opens with HTTP 200 and `text/event-stream; charset=utf-8`.

| Pod path and exception | Verified behavior after the fix |
| --- | --- |
| `literal_eval` accepts bytes, sets, complex numbers, `...`, or unsupported dictionary keys; `json.dumps(args)` raises `TypeError` | 500; SSE role and reasoning survive, then abort, with no content, call, error frame, or DONE |
| Literal dict or set construction raises `TypeError` for an unhashable item | Same; evaluation order is retained |
| Integer-to-complex conversion raises `OverflowError` | Same; this escapes `typed` rather than becoming raw text |
| A base-prefixed integer exceeds 4300 decimal digits at argument serialization; `ValueError` | Same; decimal conversion failures inside `typed` remain raw strings |
| `json.loads`, argument serialization, or AST construction raises `RecursionError` | Same; whole and stream cutoffs are separately labelled, including adjacent depth boundaries |
| Long unary, power, conditional, or lambda expressions overflow Python's parser stack with `MemoryError` | Same, without exhausting host memory |
| Argument JSON, literal evaluation, role chunks, or whole-response rendering raises injected `MemoryError` | 500 or the precise incomplete stream prefix; six injection sites are permanent fixtures |
| Surrogates survive typing and `ensure_ascii=False`, then response encoding raises `UnicodeEncodeError` | Whole 500; SSE keeps content and earlier call frames, aborts before the offending frame, and never writes invalid UTF-8 |
| Schema lookup raises `AttributeError`, `KeyError`, or unhashable-name `TypeError` | Whole 500; SSE role and reasoning then abort; reasoning-only output bypasses schema lookup |
| A reasoning-producer allocation or generator `RuntimeError` reaches `produce`'s handler | Error frame, completed HTTP body, no DONE; the runtime-failure fixture is the seventh injected case |
| A caught `ValueError` or `SyntaxError` | Raw text, normal 200 and normal SSE completion; `frozenset()` and the name `Ellipsis` are rejected literals |

Duplicate dictionary keys and parameter names settle before serialization. An overwritten unserializable value can therefore succeed. Literal evaluation errors still escape before a later parameter can overwrite them. Nonfinite argument values remain accepted.

`PodFailure` explicitly distinguishes an unsent reply, an aborted stream, and a producer error. Chunked HTTP framing preserves incomplete-response detection. The real uvicorn probe captures EOF with a missing zero chunk for aborted streams, and a zero chunk after producer errors. The TCP failure probes omit `Connection: close` in the request, so server closure is observed directly. Server and date headers are transport-specific.

## Verification

All commands ran from the engine root with `CUDA_VISIBLE_DEVICES=''`. The oracle uses Python 3.12.14 and the prior review's CPU dependencies. Fixture metadata pins the pod source and dependency versions. Production HTTP writer tests use the real tokenizer and a scripted CPU engine.

| Gate | Result |
| --- | --- |
| Permanent failure inputs | 148 exact JSON and 148 exact SSE cases; 89 whole failures and 82 stream aborts |
| Fault injection | Six MemoryError sites and one generator RuntimeError; exact prefixes and HTTP completion |
| Actual uvicorn TCP | 20 responses, exact application bytes, content type, completion, and EOF |
| Review 6 extended | 22,155 pairs, including 219 formerly skipped reference exceptions; zero mismatches |
| Review 5 main and follow-up | 22,074 plus 64 pairs, including 230 formerly skipped reference exceptions; zero mismatches |
| Required CPU targets | 1,015 total passes; final execution 971 passes plus 44 unchanged cached tests |
| Native sm_86 | 116/116 build steps; compile only |
| Lean | 426 inherited findings, zero new |
| Reproducibility and format | All three new fixture files byte-identical; Zig format and git whitespace checks pass |

The adapted review scripts preserve each original corpus and tokenizer. They run the actual FastAPI app to label former exceptions, compare status and content type, decode HTTP chunk framing, and check exact bytes and completion. They no longer count reference exceptions as skipped comparisons. Final source replays rebuilt both harnesses. Oversized corpora, raw wire logs, and executables stay in `/tmp`; `artifact-hashes.json` identifies them.

Review 5's `rerun.sh` still exits 1 at `lean_check.py` because its `set -e` stops on inherited findings. Its comparisons and CPU suite pass. The required separate `lean_new.py` invocation returns zero new findings.

## Limits

Resource guards match the supplied interpreter and observed app call stack. CPython changes C recursion and parser-stack thresholds across builds and call stacks. The small native literal reader does not reproduce every resource counter for arbitrary mixed Python grammar. Genuine process-wide RAM exhaustion, OS OOM kills, entropy or clock failures, and reset packet details cannot have identical triggers across runtimes. Output allocation faults are tested with a bounded allocator and app fault injection. Memory must remain available for the fixed error header.

The Unicode table still inflates per literal. Its request ownership and thread safety remain unchanged. Other malformed requests, native admission, and native runtime validation retain the recipe's stated boundary. GPU inference, actual EXL3 generation, TB2, and device qualification are NOT RUN.

Model the Domain kept failure stage and deferred serialization explicit. Build the Lever produced the permanent app oracle, fixture generator, and TCP capture tool. ATC is unavailable; the user's explicit local worktree assignment supplies scope.
