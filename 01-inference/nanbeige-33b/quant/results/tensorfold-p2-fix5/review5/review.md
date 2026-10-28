REJECT for "P2 server parity fit for a TB2 pilot pending only full-model GPU checks".

CPU parity defects remain. Repros use `</think><tool_call><function=edit><parameter=edits>VALUE</parameter></function></tool_call>` with the recorded pilot's array-valued `edits` schema. Complete HTTP bytes are in [minimal-repros.json](/tmp/nanbeige-review-7fe2dce7/minimal-repros.json).

- **Major:** `nanbeige_literal.zig:258`. VALUE `[{'oldText':'x','newText':'\N{SNOWMAN}'}]` becomes an array containing `newText="☃"` on the pod. The engine returns the entire literal as a string. This documented divergence violates the specified contract and can break real TB2 code edits. Smallest fix: implement Python named Unicode escapes using a pinned name table.
- **Major:** `nanbeige_literal.zig:252`. VALUE `[{'oldText':'x','newText':'\u+041'}]` stays a raw string on the pod but becomes an array containing `newText="A"` on the engine. `\u0_41` and `\x+1` also decode incorrectly. Smallest fix: require exactly the specified number of hexadecimal digits before conversion. Plausible when a model emits malformed escapes while editing source or parser fixtures; the engine can apply different text.
- **Minor:** `nanbeige_literal.zig:215`. VALUE consisting of single quotes around one literal NUL retains its quotes on the pod; the engine returns just NUL. Smallest fix: reject embedded NUL before literal parsing. Unlikely in ordinary shell work, but an undocumented mismatch.

The other documented literal exceptions mostly involve invalid tool-value types. Plain string-valued shell commands preserve escapes correctly. Reduced context is relevant to long TB2 runs; sampled recorded histories reached 94,967 prompt tokens.

The new module boundary is substantially cleaner. `openai.zig:114-115` still contains an unreachable Nanbeige/chat arm after line 97's early return. No other-family regression was found. Shared generic literal parsing remains needed.

Commands and results, with full reproduction in [rerun.sh](/tmp/nanbeige-review-7fe2dce7/rerun.sh) and [method](/tmp/nanbeige-review-7fe2dce7/METHOD.md):

- `~/exl3-local/bin/python independent.py generate`, production HTTP-writer harness, then `independent.py compare`: **22,074 cases; 53 JSON and 53 SSE mismatches** among 21,844 serializable reference results. Separately, 230 reference exceptions, including 224 malformed histories. Distribution: 47 named-escape, four malformed-escape, two NUL mismatches.
- Prompt IDs: **0 mismatches across 826 renderable synthetic histories and 55 recorded pilot histories**. The 224 malformed histories fail in HF and receive engine HTTP 400. The 55 recorded histories also match JSON/SSE bytes.
- Parent/head shared-parser comparison: **3,000/3,000 byte-identical** after ID normalization.
- Requested five Zig targets, engine root, `-j4 --summary all`, fresh scratch cache: **1,007/1,007 tests**, 17/17 steps. Breakdown: 36 llama, 887+5 server, 39 parser, 23 recovery, 17 HTTP visibility.
- `lean_check.py` plus `lean_new.py` against fix1: **0 new**, 426 inherited. Changed source files ≤600 lines. Two overlong shared comments are inherited; no new overlong comments. `git diff --check` passes.

Build the Lever produced the independent generator; Prove It Works required production serializer capture. Tracked files remain untouched. GPU generation and TB2 scoring: **NOT RUN**.
