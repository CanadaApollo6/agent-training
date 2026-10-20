# M3TiS #2 (R1s) on upstream TensorFold 1.1.1, RTX 3090 — 2026-10-20

Upstream v1.1.1 (b10231b9), built `zig build native -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast`.
Checkpoint: the published bundle (HF 66da2f4e, `~/.cache/m3tis2/bundle`), served as-is with `--drafter <bundle>`.

- Loads unchanged: 15.15 GiB on the device, 16 s.
- Exactness (`bench_concurrent.py --levels 1,2 --tokens 128 --reps 2 --alone --serial`, --parallel 2, 8K):
  36 of 36 equal (24 alone, 12 serial), 0 failed. Steady 2-stream aggregate 253 code / 216 chat at T=1.
- Speed at --parallel 2 (`bench_openai.py --tokens 128 --reps 3`): code/chat 139/119 T=1, 179/125 greedy — level
  with the #972 receipt (164/132, 176/134).
- Same bench at --parallel 1, 8K window, same files:

  | tok/s | release fork binary (r1s-moe 0d167c4f) | upstream 1.1.1 |
  | --- | --- | --- |
  | code T=1 | 416 | 173 |
  | chat T=1 | 361 | 132 |
  | code greedy | 478 | 193 |
  | chat greedy | 339 | 136 |

Verdict: keep the release on the fork binary. The ~2.5x gap (CUDA graphs, draft-vocab head, device-side draft chain,
sm_86 matmuls) is the upstream speed work to port onto 1.1.x.
