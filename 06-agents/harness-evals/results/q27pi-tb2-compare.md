19 tasks (dropped: qemu-startup)

| Build | Attempts (solved of 19) | Mean | Median output tokens a task |
|---|---|---|---|
| Qwen3.8-27B base, reasoning_effort medium, T 1.0 | 13, 11, 12 | 12.0 (63%) | 6813 |
| Qwen3.8-27B base, thinking off, T 0.8 | 8, 5, 8 | 7.0 (37%) | 5459 |
| Qwen3.8-27B + harness-hub pi LoRA, thinking off, T 0.8 | 14, 13, 12 | 13.0 (68%) | 6802 |
| Ornith 35B R1s (3090, TensorFold) | 11, 10, 9 | 10.0 (53%) | 12587 |
| Nanbeige4.3-33B BF16 (H200 per attempt, vLLM) | 11, 10, 13 | 11.3 (60%) | 20641 |

| Task | q27pi:med | q27pi:off | q27pi:ad | ornith35b:r1s-c4s | nanbeige:bf16 |
|---|---|---|---|---|---|
| build-pov-ray | 2/3 | 0/3 | 2/3 | 0/3 | 2/3 |
| compile-compcert | 2/3 | 1/3 | 2/3 | 1/3 | 3/3 |
| custom-memory-heap-crash | 3/3 | 0/3 | 3/3 | 0/3 | 2/3 |
| financial-document-processor | 3/3 | 1/3 | 2/3 | 2/3 | 0/3 |
| git-leak-recovery | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| git-multibranch | 2/3 | 1/3 | 3/3 | 1/3 | 3/3 |
| hf-model-inference | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| kv-store-grpc | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| make-mips-interpreter | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| modernize-scientific-stack | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| mteb-retrieve | 0/3 | 0/3 | 1/3 | 0/3 | 1/3 |
| nginx-request-logging | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| portfolio-optimization | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 |
| protein-assembly | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 |
| pypi-server | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| pytorch-model-cli | 3/3 | 3/3 | 3/3 | 1/3 | 1/3 |
| rstan-to-pystan | 1/3 | 0/3 | 0/3 | 2/3 | 3/3 |
| schemelike-metacircular-eval | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 |
| sqlite-db-truncate | 0/3 | 1/3 | 2/3 | 2/3 | 0/3 |

Paired against q27pi:med (per-task solve rates, permutation test):
- q27pi:off: -26.3 points a task on average; better on 1 tasks, worse on 8; p = 0.021
- q27pi:ad: +5.3 points a task on average; better on 5 tasks, worse on 3; p = 0.566
- ornith35b:r1s-c4s: -10.5 points a task on average; better on 3 tasks, worse on 6; p = 0.351
- nanbeige:bf16: -3.5 points a task on average; better on 5 tasks, worse on 4; p = 0.852

| Build | Solved-run tokens vs q27pi:med | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| q27pi:med | - | - | - | 363 |
| q27pi:off | 0.55x | 0.44-0.68 | 0 of 9 tasks | 161 |
| q27pi:ad | 0.70x | 0.52-0.93 | 2 of 13 tasks | 283 |
| ornith35b:r1s-c4s | 1.35x | 1.08-1.74 | 8 of 12 tasks | 406 |
| nanbeige:bf16 | 3.39x | 2.55-4.59 | 13 of 13 tasks | 1074 |

| Build | Gave a wrong answer | Hit the turn cap | Hit the time cap | Other |
|---|---|---|---|---|
| q27pi:med | 9 | 9 | 3 | 0 |
| q27pi:off | 11 | 22 | 3 | 0 |
| q27pi:ad | 7 | 7 | 4 | 0 |
| ornith35b:r1s-c4s | 10 | 14 | 3 | 0 |
| nanbeige:bf16 | 18 | 2 | 3 | 0 |
