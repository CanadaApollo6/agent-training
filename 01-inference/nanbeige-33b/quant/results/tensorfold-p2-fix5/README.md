# P2 fix round 5

Local `nanbeige`, based on `7fe2dce77a5913310a6edef719e278225b08b02d`. Candidate commit is `891015b9952fc3cbadb6859149fe810ac5dcf826`. No push, GPU execution, or agent-training commit.

## Before and after

| Finding | Before on reviewed baseline | After evidence |
| --- | --- | --- |
| Named escapes, Major | 47 JSON and 47 SSE mismatches; SNOWMAN stayed raw | 0; generated names, aliases, ranges, and Hangul; `unicode.log`, `minimal-after.json`, `independent-after.json` |
| Hex escapes, Major | 4 JSON and 4 SSE mismatches; signs and underscores decoded | 0; exact ASCII hex digits required; 70 malformed hex fixtures; `minimal-after.json`, `tests-root.log` |
| Source NUL, Minor | 2 JSON and 2 SSE mismatches; source NUL accepted | 0; reject source NUL anywhere only after JSON decoding fails; decoded NUL controls pass; `minimal-after.json`, `tests-root.log` |
| Dead chat arm | Unreachable Nanbeige branch in shared chat planning | Deleted; legacy completions branch retained; `source-hashes.json`, required host tests |

`minimal-before.json` preserves the six review repros and their original byte outputs. `repros-before.log` freshly fails the added parser and HTTP tests on the old implementation, 35/37 tests pass. The corrected implementation passes all 128 new parser cases and the complete 189 JSON plus 189 SSE fixtures. All six review repros appear permanently in `tools/zig/llama_serve_escapes.py` and are labelled by the pod's AST-extracted code.

## Unicode database

The committed gzip file is 261,166 bytes. It contains 34,430 explicit-name and alias records. The inflated format is a little-endian count, four-byte sorted-name offsets, and records with a three-byte code point plus a NUL-terminated ASCII name. Generated ranges cover CJK unified, Khitan, and Nushu. Hangul uses syllable composition. CJK compatibility names stay in the explicit table. Inflation is lazy and owned by the literal's request allocator, with no shared mutable cache.

Python 3.12.14 has 143,041 named code points and 473 aliases. Every name and alias was tested in uppercase and lowercase against `ast.literal_eval`, then against the actual Zig lookup. Python performed 295,962 comparisons; Zig performed 295,961. The single newline-containing negative name is excluded from the line-oriented Zig corpus and checked by Python. Both report zero mismatches. This includes 461 named sequences in both cases, algorithmic range boundaries, invalid spellings, and whitespace variants. Permanent parser/HTTP cases sample the same families.

Python 3.12 requires case-sensitive CJK unified and Hangul spellings. Ordinary names, aliases, CJK compatibility, Khitan, and Nushu accept lowercase. Tangut names and named sequences are rejected. These results come from the required interpreter, rather than assuming that all families share casing rules.

The generator pins Unicode 15.0.0 and hashes of official NameAliases.txt and NamedSequences.txt, verifies every alias with `unicodedata.lookup`, and reproduces the committed gzip and metadata. The adjacent Unicode license covers the data.

Model the Domain kept Unicode lookup and its request-owned state in one module. Build the Lever produced the pinned generator, exhaustive cross-language check, and permanent pod-labelled cases.

## Verification

All executions used `CUDA_VISIBLE_DEVICES=''`. The reference interpreter was `~/exl3-local/bin/python`, Python 3.12.14, Unicode 15.0.0. Zig was 0.17.0. All build and test commands ran from the engine root.

| Gate | Result | Receipt |
| --- | --- | --- |
| New reviewer main corpus | 22,074 cases; 21,844 serializable; 0 JSON, SSE, prompt, or missing-result mismatches | `independent-after.json`, `reviewer-rerun.log` |
| New reviewer follow-up | 64 cases, including 55 recorded histories and 9 directed cases; 0 JSON or SSE mismatches | `targeted-after.json` |
| Earlier parser differential | 24,005 cases; 0 semantic and 0 serialization-only mismatches | `differential.log`, `differential-after.json` |
| Earlier full responses | 23 scenarios; 0 aggregate mismatches | `full-response.log`, `full-response.json` |
| Required five host targets, fresh scratch cache | 17/17 steps, 1,009/1,009 tests | `tests-root.log` |
| Host test breakdown | 37 llama, 888 server, 5 bundle, 39 parser, 23 recovery, 17 visibility | `tests-root.log` |
| Native sm_86 build | 116/116 steps, exit 0 | `native-build.log` |
| Lean | 426 base, 426 head, 0 new | `lean-head.txt`, `lean-new.log` |
| Unicode completeness and reproducibility | 0 mismatches; committed gzip and metadata identical | `unicode.log` |
| Fixture regeneration | All 11 fixture, data, and license files byte-identical | `fixtures-repro.log` |
| Format and whitespace | Zig format check and git diff check pass | `command-status.json` |

The main corpus retains 230 pre-existing reference exceptions, separately counted. These include invalid history argument types, the documented unsupported literal types, and an unpaired-surrogate encoding failure. No equality claim is made for those exception bodies.

The original `rerun.sh` was executed unchanged. Its comparisons and CPU suite passed. With `set -e`, it then exits 1 at `lean_check.py`, which reports the 426 inherited findings. Its remaining `lean_new.py` command was executed separately and returned 0 new. Its CPU rerun freshly executed 965 tests and cached the unchanged 39 parser and 5 bundle tests. The separate scratch-cache run freshly executed all 1,009.

`review-artifacts.json` hashes the full inputs, expected outputs, raw wire logs, and harness binaries retained under `/tmp`. Oversized corpora, raw wire logs, and binaries are not copied here. Harness sources and build logs are included. The earlier parser adapter changes only its deleted-module import to `nanbeige_output.zig` and drops the obsolete maximum-call argument. `source-hashes.json` binds every changed engine file and the pod to these results.

## Reproduce

```sh
export CUDA_VISIBLE_DEVICES=''
export TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
~/exl3-local/bin/python tools/zig/llama_serve_unicode.py --check --zig ~/.local/share/zig-x86_64-linux-0.17.0/zig
~/exl3-local/bin/python tools/zig/llama_serve_reference.py "$TF_LLAMA_MODEL" /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/serve_exl3.py
~/.local/share/zig-x86_64-linux-0.17.0/zig build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all
~/.local/share/zig-x86_64-linux-0.17.0/zig build native -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j4 --summary all
bash /tmp/nanbeige-review-7fe2dce7/rerun.sh
CUDA_VISIBLE_DEVICES='' ~/exl3-local/bin/python /tmp/nanbeige-fix5-review-8fd873cb/differential.py
CUDA_VISIBLE_DEVICES='' ~/exl3-local/bin/python /tmp/nanbeige-fix5-review-8fd873cb/full_response.py
python3 tools/zig/lean_check.py > /tmp/h.txt
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt /tmp/h.txt
```

GPU inference, TB2, model-output equivalence, and device qualification are NOT RUN. The ATC CLI is unavailable; the user's explicit local-branch assignment supplied ownership and scope. No open implementation findings remain within this task.
