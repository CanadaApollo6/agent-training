An Ampere receipt for this branch, as promised on #803: RTX 3090 (24 GB, sm_86, 82 SMs, 420 W default limit), Linux, driver 615.78, CUDA 13.4 `nvcc`, Zig 0.17.0, this branch at `3d6f96ee`, `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` at `f84b054c`. It builds with `-Dsm=86` and serves with no changes.

**Exactness**
- `tf-cuda-test` info, smoke, moe, glue (1,242 of 1,242), sample and quant-ref: all pass. moe experts within 9.8e-7 of fp64.
- `tf-qwen35-cuda` over 300 ids: `window` and `tree` equal `step` byte for byte.
- `bench_concurrent.py --tokens 128 --reps 2 --alone --serial` with `--drafter <the checkpoint>`: `--levels 1,2` at `--parallel 2 --context 8192` gives 36 of 36 equal; `--levels 1,4` at `--parallel 4 --context 4096` gives 60 of 60. 0 unequal, 0 failed.

**Speed** (`bench_openai.py --tokens 128 --reps 3`, median tok/s, one stream; code / chat)

| serve | T=1.0 | greedy |
| --- | --- | --- |
| `--parallel 2 --context 8192`, plain | 104 / 108 | 112 / 111 |
| `--parallel 2 --context 8192`, MTP drafts | 171 / 130 | 180 / 167 |
| `--parallel 4 --context 4096`, MTP drafts | 168 / 138 | 192 / 162 |

- Aggregate from `bench_concurrent`: 295 to 318 steady at 2 streams, up to 493 at 4.
- `tf-qwen35-cuda step`: 9.0 ms a step (111 tok/s).
- `bench_cache.py --require-cold`, 2,571 tokens with the drafter: cold `ttft_s` 1.24 s, qualified, with repeated and concurrent replies equal.

Plain decode beats the 4090 here (bandwidth-bound). Drafted decode trails it by about 25%, and the cold prompt takes 1.24 s against your 0.72 s, so the prompt and verify paths are where sm_86 falls behind. I'll look at those for the 3-bit width's PR and keep any sm_86 change as its own pull request.
