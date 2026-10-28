REJECT for "P2 server parity fit for a TB2 pilot pending only full-model GPU checks" under the stated byte-equivalence requirement.

The Unicode changes pass. The remaining blockers are pre-existing response differences, not regressions introduced by this commit.

- **Major, retained:** `zig/src/server/nanbeige_literal.zig:283`, `zig/src/server/json_write.zig:59`. Minimal object-typed parameter `"'\\ud800'"`. The reference returns JSON HTTP 500 / `Internal Server Error`, or aborts SSE after its role frame. The engine returns HTTP 200 containing invalid UTF-8 bytes `ED A0 80` and completes SSE. Smallest fix: detect surrogate encoding failure and reproduce the reference's JSON/SSE failure path. Plausible, though uncommon, during coding-agent Unicode-fixture work.
- **Major, retained and documented:** `zig/src/server/nanbeige_literal.zig:88`, `zig/src/server/nanbeige_output.zig:37`. Minimal object-typed parameter `b'a'`. The reference raises `TypeError`, producing the same 500/aborted-stream behavior. The engine completes successfully with raw string `"b'a'"`. Smallest fix: distinguish valid but unserializable Python literals from syntax/value failures and propagate the former. Plausible in binary-fixture tool arguments; uncommon.

The table has no shared mutable initialization state. Eight threads completed 800 parses without allocator-reported leaks. The inflated table is 1,154,877 bytes. It is reused within one literal, then inflated again for another parameter/request. Request-arena reset/teardown releases it. Five CPU handler measurements for one named parameter gave 3.60 ms median, 3.55–3.78 ms range, and 4.83 MiB arena capacity. This excludes model inference.

Commands and results:

- Adapted `bash rerun.sh`: 22,074 review-5 cases plus 64 follow-ups; zero serialized JSON/SSE mismatches. The 230 original reference exceptions remain separately counted.
- `extended.py` and `run_extended.py`: scanned all `0x110000` code points; all 143,041 names and 473 aliases in three case variants, all 461 named sequences, and 2,330 edge literals. All 21,936 serializable response pairs match. Another 219 cases raise reference exceptions, including 14 surrogate failures.
- `llama_serve_unicode.py --check --ucd-dir … --zig …`: table reproducible; 295,961 Zig comparisons pass.
- From engine root, `zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all`: **1,009/1,009 pass** with the specified model and Zig.
- `lean_check.py` plus `lean_new.py`: **0 new**, 426 inherited. File/comment limits and diff whitespace pass.

Exact commands, normalized outputs, exception receipts, and timings are in `/tmp/nanbeige-review-891015b9/METHOD.md` and adjacent artifacts. Tracked files remain unchanged. GPU inference and TB2 are NOT RUN.
