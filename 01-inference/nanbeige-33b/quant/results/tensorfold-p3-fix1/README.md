# Nanbeige P3 fix 1

This local implementation fixes the b84fc6c8 review. All verification in this receipt is CPU-only.
GPU execution, full-model quality, acceptance and speed are NOT RUN. No protected job was touched.
The local engine commit is `97d4ac9caf8fcbdadd5ced51cd8fc2c54bc4c17d` on `nanbeige`. `source-manifest.json` records the final local commit and source hashes.
No commit was made in agent-training, and nothing was pushed. The installed outputs were restored to sm_86 after both builds passed. The existing `review/` directory is unchanged.

## Before and after

| Finding | Before | After and permanent regression |
| --- | --- | --- |
| Full acceptance loses draft context | Review reproduction reaches target 5, draft 2, then `BadCommittedTaps` | Production adapter with CPU target/draft fakes reaches 5/5, fully accepts again, rejects, then finishes 32 tokens. `backend_test.zig` is not a copied backend. |
| Grouped SSE differs from serial | `permanent-red.log` has the new frozen-response test fail on bytes, alongside the 5/2 position failure | Nanbeige commits each delivered token separately. All 189 success and 148 failure HTTP byte oracles run at block sizes 1 through 8 and every starting offset, 36 arrangements each. Fixtures are unchanged. |
| Draft allocations bypass cap | Target admission counted serial runtime only. Batch and draft attachment checked physical VRAM | `memory.admitRows` includes batch buffers. Attachment admission uses the callback and max of resident bytes and ordered loader peak, before device allocation. Tests admit a target, refuse an attachment-only overage, test loader-peak overage and unreadable room, and check batch-only overages in both KV modes. |
| Duplicate short inputs | Review found three prefixes with one unique input | Preparation asserts three unique inputs in every group. The corpus has only two unique 128-token prefixes, so short inputs use BOS plus distinct 127-token history suffixes. |
| Incomplete paid-run checks | Single sampler, disabled EOS, sparse positions, incomplete timing | Portable orchestration, four sampling settings across 32 seeds, penalties, configured EOS, forced full acceptance then rejection, 512 late positions and at least five interleaved pairs. |

`drafter-before.log` reruns the original review reproduction. `permanent-red.log` preserves the new tests failing before the production fixes.
The first invariant requires the previous drafter end to equal `DraftRequest.start`, and the accepted rows to end at the committed target position.
The second requires drafter end to equal target end, with the next draft position one beyond it.
`keep()` only rolls back the target. `draft()` ingests the kept captures exactly once even when depth is zero or every proposal matched.
The adapter is shared by both target KV modes. No GPU arithmetic or kernel source changed.

The drafter reserves its complete operation pool at load. An unexpected larger operation now returns `DraftPoolExceeded` instead of allocating beyond the admitted peak.
The room callback is used only during load and is not retained beyond its owner's stack lifetime.

## Qwen27

Qwen27 does not have this bug. `native/qwen27_drafts.zig` calls `settle()` before a solo proposal and `absorbEarly()` for shared rows.
`native/qwen27_lanes.zig` explicitly settles an all-kept verify, absorbs its taps, and checks `committed_end == offsets[0]`.
No Qwen implementation was changed. `qwen27-unchanged.json` verifies 76 Qwen source files byte-identical to b84fc6c8.
`p2-fixtures-unchanged.json` verifies all 14 Nanbeige fixture files byte-identical to that commit.

## Stopping and byte parity

`llama_blocks_test.zig` checks EOS, length and supported stop strings at output offsets 1 through 8, under all 36 arrangements.
It checks finish reason and completion usage explicitly, then compares the entire final JSON and SSE frame sequence with serial delivery, including DONE.
The scripted engine deliberately delivers beyond the requested limit to exercise server clipping inside a block.
Nanbeige chat retains the pod's existing behavior of ignoring top-level stop strings.
Stop-string tests therefore use `/v1/completions`. Only completion identity and native timing/speculation extensions are removed there.
Chat golden comparisons still normalize only the P2-approved id and created fields.
Existing allocation-fault oracles do not consume delivered blocks and remain unchanged.

## Run the pending full-model gates

Build both binaries for the rented card with the supplied Zig 0.17 toolchain and CUDA compiler, for example:

```bash
"$ZIG" build native tf-llama-test -Dnvcc="$NVCC" -Dsm=86 -j4 --summary all
```

Use `-Dsm=89` and `--sm 89` together on an sm_89 card. This is an explicit qualification run, not inherited sm_86 qualification.
The reference Python needs PyTorch, Transformers and exllamav3. The smoke argument is the P2 Python smoke probe, with `--url`, `--concurrency` and `--tokens` arguments.
The pod argument is the reference `serve_exl3.py` source. The baseline runner must support `llama-bench-once` and `BENCH_TOKENS` receipts.
Histories is a JSON array of at least three real BOS-prefixed histories of at least 100,000 tokens, distinct at every tested prefix length and short suffix.
No machine-specific path is embedded in the runner.

```bash
python3 tools/zig/llama_full_gate.py --sm 86 \
  --model "$MODEL" --draft "$DRAFT" --histories "$HISTORIES" \
  --pod "$POD_SOURCE" --smoke "$SMOKE" --baseline-runner "$BASELINE" \
  --runner "$RUNNER" --binary "$SERVER" --reference-python "$REFERENCE_PYTHON" \
  --work "$WORK" --out "$OUT"
```

Add `--plan-only` to produce the full command matrix without launching a model or querying a GPU.
Normal execution checks the card matches the requested SM and has at least 45,000 MiB.
All GPU children run through `llama_gpu_run.py` with a peak estimate and at least 2,048 MiB reserve.
P3 and P4 requirements come from CPU checkpoint capacity planners with a 512 MiB driver allowance.
P1 uses conservative 20,000 MiB reference and 17,000 MiB native estimates, and P2 reserves 18,000 MiB.
Only the task-owned server child is stopped. The scripts never enumerate and terminate unrelated jobs.

The 22 jobs cover:

- P1 independent 52-layer logits, 128-token greedy equality and five interleaved baseline/candidate speed pairs.
- P2 server smoke and exact chat JSON/SSE bytes from the pod coroutine supplied the P1 independent greedy tokens.
- P4a three histories each at 20K, 60K and 100K. Score the final 512 positions against bf16 KV, report mean, p99 and worst KL and top-token flips, compare 128 greedy tokens, then collect five interleaved timing pairs per history. Later timing repetitions skip diagnostic logit dumps. Quality still requires mean KL at most 0.01, worst at most 0.1 and greedy equality.
- P3 both bf16 and affine-4 drafters with both KV modes. Each cell checks three distinct short histories, 32 sampled seeds distributed across four settings including penalties, configured EOS, and five interleaved greedy pairs per history. A diagnostic replaces proposals with known matching tokens, then a wrong token, to require full acceptance followed by rejection. Every final token must still equal serial output.
- P3 JSON/SSE checks in all four cells include max_tokens 1 through 8, enabled EOS, ignored chat stop controls and supported completion stops.

P3 fails if mean acceptance length, `1 + accepted / target_rounds`, is not greater than one.
It reports paired speed ratios and spread separately for each history as well as aggregate results.
`$OUT/summary.json` includes all job statuses, nested results, source head, input hash and observed card.
A failed job leaves remaining jobs explicitly NOT RUN. Large logit dumps stay in `$WORK`, outside receipts.
P4 stops after the first pair if quality fails, before spending time on later timing repetitions.
P4 timing uses 127 actual decode forwards. The first pair also performs the quality readbacks, which are reported separately from decode timing.

## CPU verification

| Gate | Result |
| --- | --- |
| Fresh aggregate | 1,198/1,198 pass, 49/49 build steps |
| Requested server/parser steps | 1,017/1,017 pass |
| Llama host | 65/65 pass |
| Shared DFlash2 host | 57/57 pass |
| Qwen27 host | 59/59 pass |
| Focused stopping suite | 15/15 pass, includes imported prerequisites |
| Python admission/orchestration contracts | 5/5 pass |
| Final sm_86 native and model runner | 122/122 compile steps pass |
| Final sm_89 native and model runner | 122/122 compile steps pass |
| Lean | 426 inherited, zero new |
| Full-model GPU gates | NOT RUN |

The six extra aggregate invocations are four new Llama tests and the new stopping test in both server targets.
Success and failure delivery replays cover 12,132 JSON and 12,132 SSE comparisons, plus the explicit stopping checks.
The cached sweep also passed, but the fresh cache is the source of the 1,198 count.

`commands.txt` records the exact host, native-build and lint commands.
`script-tests.log` checks the complete 22-job plan without launching CUDA, distinctness refusal, five interleaved KV pairs, P3 refusal with zero acceptance despite proposals, and reserve-breach termination of only the guard-owned child.
`block-stops.log` is the focused stopping suite. The fresh aggregate host and both final native build results are recorded in their logs and summary.
Lean reports 426 inherited findings and zero new. Changed files are at most 600 lines, and comment and whitespace checks pass.
Build success does not qualify sm_86, sm_89, the full model, KV4 quality, or draft performance.

Model the Domain made committed position explicit. Fix Root Causes moved ingestion to the unconditional draft boundary.
ATC is unavailable on this machine. The user's explicit local worktree assignment supplies the work scope.
