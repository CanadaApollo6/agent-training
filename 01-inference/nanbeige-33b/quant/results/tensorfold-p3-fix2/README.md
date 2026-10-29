# Nanbeige paid-gate repair

Local engine commit `a3c96094081d20cb879807f66a39120948bb4d2f` against `97d4ac9c`. Final local commit and build identities are in `source-manifest.json`.
The existing `review/` directory is unchanged. No GPU execution, uploads, HF/Prime API calls or pushes were performed.
GPU correctness, speed and the actual L40S runtime remain **NOT RUN**. Compilation does not qualify either SM.

| Area | Before | After and evidence |
| --- | --- | --- |
| KV timing logits | `rows=0` used uninitialized logits | Final prefill row always produces logits. `kv-regression-red.log` fails the target assertion when the fix is removed; `kv-regression-green.log` passes. |
| P4 prompt work | 5,400,000 serial token forwards | 600,000 tokens through eight-row prefill. One 100K prefill per history and format. Score 100K, 60K, 20K in descending order, then five alternating resident-cache decode pairs at each length. Three CPU prefill tests include shorter-prefix equality after longer-context decode. |
| Job matrix | 22 jobs, nine separate P4 loads | 16 jobs, three P4 sweeps. All nine history/length KL cells remain, each with 512 scored positions and 128 greedy tokens. |
| Resume | Replaced completed statuses | SHA256 content binding for engine commit, binaries, fatbins, scripts, model, drafter, histories, reference scripts and exact job commands. Artifact hashes are checked before trusting PASS. Atomic fsync/rename checkpoints persist RUNNING. An exclusive output lock prevents simultaneous executions. |
| Recovery and selection | No restart/selection proof | `--only` / `--skip` support comma-separated names/globs. P1 reference dependencies are included. A crashed RUNNING job reruns, PASS jobs do not. `dry/dry-run.json` records an actual process exit 73, 14 resumed jobs, zero completed reruns, changed-weight refusal and zero GPU calls. |
| Runtime admission | No projection or parent deadline | First P4 measures both prefill formats and decode rates. Every next job prints the remaining projection with a 25% rate reserve. Refuse before starting a job if its estimate or the calibrated remaining plan exceeds the budget. Parent deadlines terminate only the owned process group. SIGTERM checkpoints incomplete work. |
| Sampling | Eight keyed comparisons per variant, claimed stronger than histograms | Independent seeds and 8,192 draws per side at fixed verification rows 1, 4, 7, three prompts and four settings. Real production verification with the attached drafter supplies the batched logits. Serial forwards supply reference logits. `sampling-design.json` records the statistical contract. |
| Wire stopping | Ordinary prompts might never stop | Test-only wrapper fixes draw/proposal IDs while real target/drafter CUDA forwards execute. EOS at accepted offsets 2 through 8 on both routes, and decoded stop `X` on completions. All 42 JSON/SSE cases require stop finish reason, exact completion-token usage and equal normalized HTTP writer bytes. Native TCP cases still run separately. CPU tests drive the same forced wrapper and full HTTP writer harness. Chat stop controls retain the existing ignored-stop contract. |
| Draft speed | Reported ratios without enforcing improvement | Each short greedy prompt needs median plain/drafted decode ratio above 1.0, plus positive acceptance. Five alternating pairs per prompt retain spread and a paired median 95% bootstrap interval. Negative CPU controls refuse zero acceptance, ratio 1.0 and ratio 0.9. |
| KL analysis | Python scalar loops over about 85 million logits per cell | FP64 NumPy row calculations, same KL and greedy bars, bounded to one vocabulary row. |

## Statistical contract

The bins are declared before observing data, `token_id % 8`. This is a binned next-token distribution test, not proof of
identity for every vocabulary item. Four settings cover temperature, top-k, top-p and presence/frequency penalties.
Thirty-six cells use three fixed conditional positions in each of three histories, without random-history confounding.
A union Hoeffding bound covers all 288 frequency differences at family alpha 0.01 per drafter/KV job. Each simultaneous upper bound must
be below a 0.10 bin-probability margin. At 8,192 draws per side the bound guarantees at least 84.55% probability of
accepting equal distributions, and at least 99% rejection power at a 0.10 difference in any declared bin.
The CPU biased-frequency control fails this same gate. Greedy sequence identity and forced accept/reject checks remain
separate requirements. No keyed sampled-sequence equality claim remains.

## Runtime and memory projection

`runtime-projection.json` contains every job estimate. These are planning assumptions, **not measured GPU rates**.
The 860 GB/s card assumption uses 240 prefill tokens/s and 35 decode tokens/s at a 60K baseline. The 100K prefill
estimate applies an additional 100K/60K attention factor. It includes serial quality-row work and decode at all three
lengths. The first real P4 measurement replaces those assumptions and can refuse the remainder if the card is slower.
Setup time is charged against the container's 150-minute deadline, with two minutes reserved for final sync.

| Job labels | Count | Planning seconds each |
| --- | ---: | ---: |
| `p4-100000-{0,1,2}` including 100K/60K/20K | 3 | 1684.708 |
| `p3-{bf16,affine4}-kv{16,4}` | 4 | 548.571 |
| `p3-{bf16,affine4}-kv{16,4}-wire` | 4 | 150 |
| `p1-reference` | 1 | 180 |
| `p1-logits` | 1 | 60 |
| `p1-greedy` | 1 | 60 |
| `p1-speed` | 1 | 133.143 |
| `p2` | 1 | 150 |
| Total gate, excluding setup/sync | 16 | 8431.551 seconds, 140.526 minutes |

The CPU allocator estimate for two shared-weight caches at context 100128 is 39,889 MiB. The guard adds 512 MiB driver
allowance and preserves 2,048 MiB free. Admission therefore needs 42,449 MiB free. See `capacity-cpu.log`. This is an
allocator projection, not a measured peak. Every actual device launch still goes through `llama_gpu_run.py`.
A full gate that cannot fit the measured budget stops with BUDGET, never a partial PASS. A selected subset ends PARTIAL.

## Operator staging and launch

Stage a private input bucket outside this task. Its root must contain `engine.bundle`, `baseline.bundle`, `model/`,
`draft/`, `histories.json`, `serve_exl3.py`, `smoke_serve.py`, `zig-0.17.tar.xz`, `llama_hf_job.sh`, and `SHA256SUMS`.
The engine bundle must contain the final reviewed commit. The baseline bundle must contain `97d4ac9c` and its
`llama-bench-once` harness. The Zig archive must be the exact compatible 0.17 toolchain, with one top-level directory.
`SHA256SUMS` must bind all staged files including model/drafter shards and the entrypoint. Both buckets are private.
Create the output bucket before launch. No bucket or job was created here.

The entrypoint installs Python 3.12, torch 2.10.0 cu128, exllamav3 1.6.5 and the reference dependencies, installs staged
Zig 0.17, clones source bundles from the read-only mount, checks out pinned commits, and builds for the requested SM.
Every ten minutes it snapshots receipts, syncs work first, then publishes the receipt snapshot. Large logits belong
under the output bucket's `work/`, outside the small `receipts/`. This ordering keeps saved PASS artifacts recoverable.
Final sync also runs on failure. `RESUME=1` restores both directories. Preserve the same mount paths and input bytes.

Exact launch template after setting the operator-owned bucket variables and final reviewed commit:

```bash
export IN_BUCKET=hf://buckets/OPERATOR/nanbeige-in/RUN
export OUT_BUCKET=hf://buckets/OPERATOR/nanbeige-out/RUN
export ENGINE_COMMIT=a3c96094081d20cb879807f66a39120948bb4d2f
export BASELINE_COMMIT=97d4ac9caf8fcbdadd5ced51cd8fc2c54bc4c17d
printf 'HF_TOKEN=%s\n' "$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")" |
  uvx --from huggingface_hub hf jobs run --detach --secrets-file - \
    --flavor l40sx1 --timeout 150m --name nanbeige-full-gate \
    -e ENGINE_COMMIT="$ENGINE_COMMIT" -e BASELINE_COMMIT="$BASELINE_COMMIT" \
    -e OUT_BUCKET="$OUT_BUCKET" -e SM=89 -e RESUME=0 \
    -v "$IN_BUCKET:/inputs:ro" \
    nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /inputs/llama_hf_job.sh
```

For a restart, use the identical command with `-e RESUME=1`. For an A6000 set `SM=86` and use the same entrypoint in the
operator's container launcher. No Prime launch/API call was made. Optional selection arguments follow the entrypoint,
for example `--only 'p3-*-wire'`. Do not change input bytes to reuse a completed checkpoint.

Local CPU orchestration proof, no model or license needed:

```bash
python3 tools/zig/llama_full_gate.py --dry-run --out /tmp/nanbeige-cpu-proof
python3 tools/zig/test_llama_gates.py
```

## Verification receipts

`commands.txt` records the exact commands. `host.log` is the fresh requested baseline aggregate with the first two new
prefill tests, 1,200/1,200. Follow-up host logs cover the forced wrapper, the descending-prefix test and full writer
harness on the final source. The final covered count is 1,205. The final aggregate executes 1,026 and reuses 179 verified cached tests.
The requested server/parser count is 1,020, Llama is 69, DFlash2 is 57 and Qwen27 is 59. Both SM builds pass 126/126.
Final counts and native build status are recorded in `source-manifest.json`.
`python-tests.log` covers the permanent CPU orchestration suite. `lean-new.txt` must report zero new findings.
Build attempts with errors are retained as attempt evidence; only `build-sm86-complete.log` and
`build-sm89-complete.log` are final native-build evidence. GPU tests, statistical device draws, speed intervals,
cloud installation and bucket recovery are NOT RUN.

Model the Domain made checkpoint states and job identities explicit. Prove It Works required a real crash/restart
and CPU execution of the actual wire harness, rather than trusting generated command lists.
