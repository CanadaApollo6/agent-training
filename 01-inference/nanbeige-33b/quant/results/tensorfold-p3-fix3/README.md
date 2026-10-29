# Nanbeige gate repair 3

Local commits `d1cf9fb4` and `a5454ed3931caa5dee27ef45c468f44f8ea76b82` against `a3c96094081d20cb879807f66a39120948bb4d2f`. Final commit and content hashes are in `source-manifest.json`.
The supplied `review/` directory is preserved. No GPU execution, HF/Prime API calls, uploads, launches, pushes or agent-training commits were made.
Full-model correctness, coupled CUDA draws, actual cloud setup and card throughput remain **NOT RUN**.

| Finding | Before | After and evidence |
| --- | --- | --- |
| Sampled lane | Verification had no sampling, discarded outputs, and compared the same sampler twice. | CPU executable instantiates the production generic Llama adapter and lane Engine, including first/queue/verify/keep, history, pending token, sampling propagation, acceptance, rejection and commit. Cached model execution and a CPU port of the CUDA sampling rule replace device execution. An independent Python categorical oracle checks emitted tokens. `power.json`, `mutation-exits.json`, and `replay-build.log`. |
| Modulo statistic | Eight bins missed within-bin transfers. | Deleted the old statistic. Every token has a singleton test. Six predeclared event constructions select tokens where the oracle exceeds each buggy law. Real job cells cover 3 prompts, rows 0/3/6, 4 settings and all 4 drafter/KV combinations. |
| GPU draw cost | Hundreds of thousands of synchronous GPU draws per configuration. | Export target/drafter logits once. Replay 32,768 CPU draws per conditional cell. Compare 16 coupled seeds exactly at each GPU cell across serial target, batched target and CPU rule, including the production verification output. Large raw caches stay in temporary storage, outside the receipt bucket. |
| Shell failures | Conditional `sync_out` hid work failures. | Every copy and upload records its own status. Work failure and any sync failure produce nonzero exit and durable FAIL. Output logs start on the mounted bucket before setup. `DRY=1` skips GPU work. Offline `LOCAL_DRY=1` exercises work/sync failures without networking. `python-tests.log` includes the exact sync body with a failing work upload and successful receipt upload. |
| Dependencies | Unpinned installs and incompatible future hub upgrades. | Hash-locked inference and HF CLI environments are separate. Python/uv bootstrap, all offline .deb files, image digest and lock files must appear in SHA256SUMS. Inference retains hub 0.36.0 for transformers 4.57.1. The CLI uses hub 1.10.1. Full cloud installation remains NOT RUN. |
| Deadline | Fingerprinting preceded the clock. | Absolute container-start deadline covers setup, fingerprinting, gate and sync. Python alarm and process-group limits stop owned work before the sync reserve. The outer container command has a hard 210-minute bound. Preflight and job receipts list unfinished jobs. |
| Order and calibration | KV first, unmeasured sampling/wire constants treated as calibrated. | P1 reference/logits/greedy; 60K/100K KL; P3 greedy/acceptance/coupled sampling; P2 wire; 20K KL/timings; P1/P3 speed. Every P4 length retains five alternating pairs. Long-context pairs also supply calibration while doing KL. Each job class calibrates from its first real wall-time unit, with a 25% reserve. |
| P1 speed | Two Zig binaries, no exl3 threshold. | Five alternating runs per side and all P1 prompts. Same first 32 prompt IDs, 128 output tokens, 127 timed plain decode forwards, one warmup, 4096 cache capacity, no drafter, 16-bit KV, INT8 GEMV off. Report medians, spread, ratios and a separate speed verdict at ratio >= 1. Speed never overrides correctness. CPU tests reject false blocking and missing timing cells. |
| Cheap false passes | Duplicate sampling cells, changed environments, and short/unstable timed work. | Require the exact 36-cell set, both accept/reject coverage, complete counts, exact emitted forced history and bonus token, stable timed token lists, positive finite speeds, nested failure statuses, source/lock/shell hashes, and installed reference-package versions in resume identity. |

## Real-lane mutation power

These are actual CPU production-lane executions, not multinomial draws from a predicted bug distribution.
The CPU model boundary injects six named sampler-law defects, then the production adapter and Engine emit their tokens.
Residual replacement/reversal are hypothetical corrections at this boundary. Production target-token matching has no residual sampler.
The first proposal is the drafter-logit argmax, as in the production greedy proposal path.
Each trial uses 32,768 draws and a disjoint deterministic seed block. All six explicit mutant gate commands exit 1.

| Mutation | TV | Rejected trials | Simultaneous 95% power lower bound |
| --- | ---: | ---: | ---: |
| residual_replaced | 0.09944 | 128/128 | 0.9633 |
| residual_reversed | 0.20378 | 128/128 | 0.9633 |
| draft_support | 0.06243 | 128/128 | 0.9633 |
| temperature_twice | 0.04907 | 128/128 | 0.9633 |
| penalties_draft_only | 0.03568 | 128/128 | 0.9633 |
| within_bin_transfer | 0.05521 | 128/128 | 0.9633 |

The confidence bounds use Bonferroni over six power estimates, `(0.05/6)^(1/128)` after 128/128 rejections.
The oracle never imports the production sampler or RNG. It computes softmax after bf16 narrowing, reply-only penalties,
temperature, top-k, nucleus and min-p. Statistical family alpha is 0.01 across all four configurations and all token/event tests.
The union bound conservatively allocates `4 * 36 * (166912 + 7)` two-sided tests. Radius is 0.01844370.
Against a declared event whose probability changes by 0.03, the analytic rejection-power lower bound is 0.99984189.
The baseline synthetic corpus has 12 cells. The real job requires all 144 cached model cells.

**Limit.** Power is established for these predeclared bug directions with TV at least 0.03, not every possible 166,912-token alternative.
A finite set of directional events and 32,768 draws cannot justify a universal high-dimensional TV guarantee.
Within-bin transfers now fail even when their old modulo-eight frequencies are identical.

## Runtime projection

`runtime-projection.json` totals 197.619 minutes, including 25 minutes of setup and 3 minutes of final sync.
The 26 gate jobs total 169.619 minutes. The hard limit is 210 minutes.
This is a planning projection, **not measured L40S/A6000 throughput**. KV assumptions are 300 prefill tokens/s and 45 decode tokens/s,
with attention scaling at 100K. Other job-class estimates are explicit in `llama_gate_budget.py`.
CPU replay, wire, reference, speed and setup costs on the rental remain unmeasured. Each class replaces its assumption after its first real unit.
Admission checks the next job against the remaining absolute deadline and preserves the final-sync reserve.
A slower card or setup produces a nonzero budget abort with partial receipts, never a full PASS.
Resume requires identical input bytes, artifacts, engine commit and command matrix.

## Remaining ways a wrong port could pass

- The independent P1 model reference covers short histories. A shared long-context error in bf16 and 4-bit KV could pass P4 relative KL.
- The CPU replay replaces model execution and the CUDA kernel. Sixteen coupled GPU seeds per cell cover only those cached shapes/settings.
- The replay tests fixed conditionals and commits. It does not establish every sampled trajectory, seed, tie, penalty history or scheduler interleaving.
- The declared event tests have no universal power guarantee for an adversarial probability transfer unrelated to their directions.
- Model-fidelity thresholds and the existing cross-engine tie policy admit bounded numerical differences.
- P1 speed failure is deliberately nonblocking and appears as `p1_speed=FAIL` beside correctness PASS.

The job is not hardware-qualified by these CPU receipts. No real speed or setup-success claim is made.

Exact launch, resume, CPU setup rehearsal and local reproduction commands are in [launch.md](launch.md).

## Verification

`host.log` reports 53/53 build steps and 159/159 freshly executed tests. Other requested host lanes use valid cached results.
The earlier source-matching aggregate covered 1,205 tests; that is not a claim of 1,205 fresh executions here.
`python-tests.log` reports 18 tests. Both SM native builds and the model/wire harness compile. Compilation earns no device credit.
`lean-new.txt` records 426 inherited findings and zero new findings. `commands.txt` records commands and explicit NOT RUN work.
`atc board --json` was attempted. ATC is not installed. No issue or public presence was created for this local-only assignment.

Model the Domain made job classes and checkpoint states explicit. Prove It Works required real lane mutations and real crash/resume runs.
