19 tasks (dropped: qemu-startup)

| Build | Attempts (solved of 19) | Mean | Median output tokens a task |
|---|---|---|---|
| Ornith 35B Q8_0 (pod, llama.cpp) | 9, 10, 12 | 10.3 (54%) | 9182 |
| Ornith 35B R1s (3090, TensorFold) | 11, 10, 9 | 10.0 (53%) | 12587 |
| Flint BF16 (H200, Recursive's SGLang kit) | 9, 11, 9 | 9.7 (51%) | 23766 |

| Task | ornith35b:q8 | ornith35b:r1s-c4s | flint:bf16 |
|---|---|---|---|
| build-pov-ray | 0/3 | 0/3 | 2/3 |
| compile-compcert | 2/3 | 1/3 | 1/3 |
| custom-memory-heap-crash | 3/3 | 0/3 | 1/3 |
| financial-document-processor | 2/3 | 2/3 | 2/3 |
| git-leak-recovery | 3/3 | 3/3 | 3/3 |
| git-multibranch | 2/3 | 1/3 | 3/3 |
| hf-model-inference | 3/3 | 3/3 | 3/3 |
| kv-store-grpc | 2/3 | 3/3 | 0/3 |
| make-mips-interpreter | 0/3 | 0/3 | 0/3 |
| modernize-scientific-stack | 3/3 | 3/3 | 3/3 |
| mteb-retrieve | 0/3 | 0/3 | 3/3 |
| nginx-request-logging | 2/3 | 3/3 | 0/3 |
| portfolio-optimization | 2/3 | 3/3 | 2/3 |
| protein-assembly | 0/3 | 0/3 | 0/3 |
| pypi-server | 3/3 | 3/3 | 3/3 |
| pytorch-model-cli | 1/3 | 1/3 | 2/3 |
| rstan-to-pystan | 1/3 | 2/3 | 0/3 |
| schemelike-metacircular-eval | 0/3 | 0/3 | 0/3 |
| sqlite-db-truncate | 2/3 | 2/3 | 1/3 |

Paired against ornith35b:r1s-c4s (per-task solve rates, permutation test):
- ornith35b:q8: +1.8 points a task on average; better on 3 tasks, worse on 4; p = 1.000
- flint:bf16: -1.8 points a task on average; better on 5 tasks, worse on 5; p = 1.000

| Build | Solved-run tokens vs ornith35b:r1s-c4s | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| ornith35b:q8 | 0.78x | 0.54-1.08 | 6 of 13 tasks | 374 |
| ornith35b:r1s-c4s | - | - | - | 406 |
| flint:bf16 | 1.81x | 1.36-2.45 | 9 of 10 tasks | 694 |

| Build | Gave a wrong answer | Hit the turn cap | Hit the time cap | Other |
|---|---|---|---|---|
| ornith35b:q8 | 14 | 12 | 0 | 0 |
| ornith35b:r1s-c4s | 10 | 14 | 3 | 0 |
| flint:bf16 | 15 | 10 | 3 | 0 |
