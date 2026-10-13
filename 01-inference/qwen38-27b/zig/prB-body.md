Stacked on #3 (RTX 3090 support), which is stacked on ashhart/TensorFold#608. GitHub can't target a fork's branch, so this diff also shows #3's five commits. **This PR's change is the last commit, `083532cf`** (5 files, +637 −18). Once #3 lands, the diff shrinks to that commit.

On an RTX 3090 it makes served decode **~52% faster** with identical tokens. Every row's logits are byte-equal to 0.6.6's.

## What changes

- **`qmm_lane.cu`: a pre-sm_90 decode matmul.** With #3, sm_86 runs split-K matmuls as `lane_gemv`, 16 rows a launch. `qmm_lane` puts the weights in the mma's A operand, so 1 to 16 rows fill a tile. Each tile's K slices are added in slice order, either in one block (2 or 4 slices side by side) or by the tile's last block. That is the order 0.6.6's `qmm.cu` uses on the same card. Wider rounds take row tiles of 16, 32 or 64, at most 128 rows a launch.
- **One launch for weights sharing an input.** `Ops.denseMany` runs GDN's qkv/z/b/a, attention's q/k/v and the MLP's gate/up as one segmented launch for each K split they share. Each weight keeps its own split, so every output equals its lone launch.
- Its fragment helpers come from the existing `qmm_frag.cuh`. sm_90 and newer, and `TF_QWEN27_GEMV_ROWS`, keep their current paths.

## Checks (RTX 3090, CUDA 13.4, driver 615.71, the same 0.6.6 glue set as #3)

| Check | Result |
|---|---|
| `tf-qwen27-cuda windows` against a fresh `capture_windows.py` capture from 0.6.6 on the same card | cases 0-18 (chains of 1-16 rows, shared rounds of 2-16 streams, trees): **every row's logits sha256 equal**. Case 19 (16 streams) runs out of memory in the check tool on 24 GB, with and without this PR. `--wide` doesn't fit 0.6.6 on 24 GB. |
| `run` (serial) | 4/4 MATCH |
| `draft` | 4/4 MATCH |
| `multi --context 8192` | 3/3 streams MATCH, 30 rounds as Python |
| `draft` with `TF_QWEN27_HOST_EMBED=1 TF_QWEN27_PROMPT_ROWS=1024` | 4/4 MATCH |
| `zig build test` | pass |
| `tools/zig/lean_check.py` | 542, the same as ashhart/TensorFold#608's branch |

## Served decode, before (#3) and after, alternated

`tensorfold-native serve --context 32768 --parallel 1` with the DFlash2 drafter. `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`, run as A, B, A, B on one machine:

| tok/s, greedy | #3 | this PR |
|---|---|---|
| Code (`fibonacci-raw`) | 131.9, 131.3 | **200.4, 203.7** |
| Chat (`gpu-chat-no-think`) | 90.0, 90.1 | **133.8, 138.3** |

The token hashes of every reply are the same in all four runs.

Not checked: sm_80/sm_89 cards (I have neither). On sm_90 and newer the code path is unchanged.

## Not in this PR

A rule that runs shared-input weights unsplit once they fill the card is worth another ~2-3% on the 3090. It changes K splits, and so the bits, so it needs a gate per CONTRIBUTING. I'll raise it with the 4-bit cache issue.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
