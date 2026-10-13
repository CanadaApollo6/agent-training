Stacked on #608: this targets its branch, so the diff is only the four commits below (13 files, +132 −55). With them the qwen27 CUDA engine serves on an **RTX 3090 (sm_86, 24 GB)**, still token-for-token 0.6.6's.

## What changes

- **`f3307446` Split-K matmuls before sm_90.** qmm's split-K path meets its K slices in a thread-block cluster, which sm_86 doesn't have. Before sm_90, split-K matmuls take `lane_gemv` (byte-equal to the cluster path) 16 rows a launch.
- **`2eac7a73`, `bc3b4859` Round staging on 16 bytes.** A shared round packs its host tables into one upload. Back to back, `_gdn_pre`'s `WIN` and `_tail`'s `STREAM` land off 16 bytes at some window sizes, and no captured variant matches (`no captured Triton variant of _tail for this launch`). Each part now starts on 16 bytes. An aligned pointer may also fall back to a variant captured without the alignment hint (never the reverse).
- **`903b23fc` The scratch sized by `--parallel`.** The server sized it for 16 streams of 16 rows at any `--parallel`, and the attention partials in it grow with the window. On a 3090, `--context 32768 --parallel 1` failed with *fits no stream: one at a 32768-token window takes 2.22 GiB, and 1.14 GiB is left after the model's 18.10 GiB*. The scratch now holds `--parallel` windows (capped at 16): the model plus scratch is 16.82 GiB and the stream fits. At `--parallel 16` the layout is unchanged.
  - Two opt-in env settings for tight cards:
    - `TF_QWEN27_PROMPT_ROWS` sets a prompt chunk's rows (default 4096). At 1024 it frees ~1 GiB, and a 28K-token prompt reads ~10% slower.
    - `TF_QWEN27_HOST_EMBED=1` keeps the token table (0.67 GiB) in mapped host memory. Only prompt rows read it.

## Checks (RTX 3090, CUDA 13.4, driver 615.71)

The glue set was captured from 0.6.6 on the same card with the scripts in `zig/tests/cuda/qwen27`. `capture_multi.py` ran with `--streams 3`, because 0.6.6's memory gate refuses the default 6 on 24 GB. The set has 41 kernels including `MULTI=1`.

| Check | Result |
|---|---|
| `tf-qwen27-cuda run` (serial, a 4722-token prompt among them) | 4/4 MATCH |
| `tf-qwen27-cuda draft` | 4/4 MATCH, rounds within 1 of Python's |
| `tf-qwen27-cuda multi --context 8192` | 3/3 streams MATCH, same rounds |
| `draft` with `TF_QWEN27_HOST_EMBED=1 TF_QWEN27_PROMPT_ROWS=1024` | 4/4 MATCH |
| Served at `--parallel 1` and `--parallel 3`: 3 greedy chats alone, with `"draft": false`, and concurrent | all equal |
| `zig build test` | pass |

Served at `--context 32768 --parallel 1` (`tools/bench_openai.py`, 256 tokens, medians of 3): code 140 / chat 96 tok/s greedy, 168 / 95 sampled.

Not checked: the RTX PRO 6000 or GB10 (I have neither). The layout there is unchanged at `--parallel 16`.

## Next, if useful

I also have a pre-sm_90 decode matmul and an opt-in 4-bit attention cache on top of this. Together they make 235 code / 160 chat tok/s and a 262K window on the 3090. The 4-bit cache changes arithmetic, so I'll open an issue for its gate first, per CONTRIBUTING. @jschmied's generic glue kernels would also remove the capture step for sm_86.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
