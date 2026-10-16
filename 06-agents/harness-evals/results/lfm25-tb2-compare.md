19 tasks (dropped: qemu-startup)

| Build | Attempts (solved of 19) | Mean | Median output tokens a task |
|---|---|---|---|
| Liquid's Q8_0 (pod, llama.cpp) | 7, 6, 4 | 5.7 (30%) | 12561 |
| M3TiS #3: Liquid's MLX 4-bit + 4-bit table (sm_86 pod, TensorFold Zig) | 3, 7, 5 | 5.0 (26%) | 16195 |

| Task | q8 | m3tis |
|---|---|---|
| build-pov-ray | 0/3 | 0/3 |
| compile-compcert | 0/3 | 0/3 |
| custom-memory-heap-crash | 0/3 | 0/3 |
| financial-document-processor | 0/3 | 0/3 |
| git-leak-recovery | 3/3 | 2/3 |
| git-multibranch | 0/3 | 0/3 |
| hf-model-inference | 1/3 | 1/3 |
| kv-store-grpc | 3/3 | 3/3 |
| make-mips-interpreter | 0/3 | 0/3 |
| modernize-scientific-stack | 3/3 | 3/3 |
| mteb-retrieve | 0/3 | 0/3 |
| nginx-request-logging | 2/3 | 1/3 |
| portfolio-optimization | 3/3 | 1/3 |
| protein-assembly | 0/3 | 0/3 |
| pypi-server | 1/3 | 3/3 |
| pytorch-model-cli | 1/3 | 1/3 |
| rstan-to-pystan | 0/3 | 0/3 |
| schemelike-metacircular-eval | 0/3 | 0/3 |
| sqlite-db-truncate | 0/3 | 0/3 |

Paired against q8 (per-task solve rates, permutation test):
- m3tis: -3.5 points a task on average; better on 1 tasks, worse on 3; p = 0.750

| Build | Solved-run tokens vs q8 | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| q8 | - | - | - | 328 |
| m3tis | 1.40x | 0.98-1.86 | 7 of 8 tasks | 349 |

| Build | Gave a wrong answer | Hit the turn cap | Hit the time cap | Other |
|---|---|---|---|---|
| q8 | 19 | 21 | 0 | 0 |
| m3tis | 18 | 24 | 0 | 0 |
