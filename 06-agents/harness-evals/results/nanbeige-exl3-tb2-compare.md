19 tasks (dropped: qemu-startup)

| Build | Attempts (solved of 19) | Mean | Median output tokens a task |
|---|---|---|---|
| Ornith 35B Q8_0 (pod, llama.cpp) | 9, 10, 12 | 10.3 (54%) | 9182 |
| Ornith 35B R1s (3090, TensorFold) | 11, 10, 9 | 10.0 (53%) | 12587 |
| Flint BF16 (H200, Recursive's SGLang kit) | 9, 11, 9 | 9.7 (51%) | 23766 |
| Nanbeige4.3-33B BF16 (H200 per attempt, vLLM) | 11, 9, 13 | 11.0 (58%) | 20641 |
| Nanbeige4.3-33B EXL3 3.5 bpw own-text calibrated (H200, serve_exl3.py, 120-min cap) | 10, 12, 12 | 11.3 (60%) | 25904 |

| Task | ornith35b:q8 | ornith35b:r1s-c4s | flint:bf16 | nanbeige:bf16 | nanbeige:exl3 |
|---|---|---|---|---|---|
| build-pov-ray | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 |
| compile-compcert | 2/3 | 1/3 | 1/3 | 3/3 | 2/3 |
| custom-memory-heap-crash | 3/3 | 0/3 | 1/3 | 2/3 | 3/3 |
| financial-document-processor | 2/3 | 2/3 | 2/3 | 0/3 | 0/3 |
| git-leak-recovery | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| git-multibranch | 2/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| hf-model-inference | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| kv-store-grpc | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| make-mips-interpreter | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| modernize-scientific-stack | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| mteb-retrieve | 0/3 | 0/3 | 3/3 | 0/3 | 0/3 |
| nginx-request-logging | 2/3 | 3/3 | 0/3 | 1/3 | 2/3 |
| portfolio-optimization | 2/3 | 3/3 | 2/3 | 2/3 | 2/3 |
| protein-assembly | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 |
| pypi-server | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| pytorch-model-cli | 1/3 | 1/3 | 2/3 | 1/3 | 0/3 |
| rstan-to-pystan | 1/3 | 2/3 | 0/3 | 3/3 | 3/3 |
| schemelike-metacircular-eval | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| sqlite-db-truncate | 2/3 | 2/3 | 1/3 | 0/3 | 2/3 |

Paired against nanbeige:bf16 (per-task solve rates, permutation test):
- ornith35b:q8: -3.5 points a task on average; better on 4 tasks, worse on 6; p = 0.836
- ornith35b:r1s-c4s: -5.3 points a task on average; better on 4 tasks, worse on 6; p = 0.722
- flint:bf16: -7.0 points a task on average; better on 4 tasks, worse on 6; p = 0.645
- nanbeige:exl3: +1.8 points a task on average; better on 3 tasks, worse on 3; p = 1.000

| Build | Solved-run tokens vs nanbeige:bf16 | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| ornith35b:q8 | 0.34x | 0.24-0.46 | 0 of 12 tasks | 374 |
| ornith35b:r1s-c4s | 0.37x | 0.27-0.52 | 1 of 11 tasks | 406 |
| flint:bf16 | 0.59x | 0.41-0.81 | 1 of 10 tasks | 694 |
| nanbeige:bf16 | - | - | - | 1074 |
| nanbeige:exl3 | 1.21x | 1.04-1.43 | 10 of 12 tasks | 952 |

| Build | Gave a wrong answer | Hit the turn cap | Hit the time cap | Other |
|---|---|---|---|---|
| ornith35b:q8 | 14 | 12 | 0 | 0 |
| ornith35b:r1s-c4s | 10 | 14 | 3 | 0 |
| flint:bf16 | 15 | 10 | 3 | 0 |
| nanbeige:bf16 | 19 | 2 | 3 | 0 |
| nanbeige:exl3 | 19 | 3 | 1 | 0 |
