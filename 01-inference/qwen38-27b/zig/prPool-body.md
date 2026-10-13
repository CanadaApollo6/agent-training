Against ashhart/TensorFold#608's branch, one commit (`549653b1`, 2 files, +36 −6). It doesn't depend on #3 or #4.

## The problem

The qwen27 CUDA loaders give every weight tensor its own `cuMemAlloc`. The driver rounds each allocation up to whole 2 MiB pages. The 27B has ~900 tensors over 2 MiB, and many sit just past a page: a 2.8 MiB scale table takes 4 MiB. Summed over the checkpoint's tensor sizes, the rounding costs **~1.1 GiB**. The server's own count (`model … GiB`) doesn't see it, because it adds up the requested lengths. So the memory is gone before the stream gate runs.

## The change

`weights.Pool`: both loaders (the 27B's and DFlash2's) take tensors first fit out of 128 MiB blocks, 256-byte aligned as `cuMemAlloc` gives them. A simulation over the checkpoint's tensors leaves ~70 MiB of block tails unused. `Weights.buffers` owns the blocks, so `deinit` and `bytes()` work as before. `bytes()` now counts whole blocks, which is the memory actually held. No kernel, layout or arithmetic changes: only the addresses move.

## Checks (RTX 3090, CUDA 13.4, driver 615.71)

sm_86 needs #3 to run the 27B, so the checks ran on #3 with this commit applied:

| Check | Result |
|---|---|
| `tf-qwen27-cuda windows` (fresh 0.6.6 `capture_windows.py` capture, same card) | cases 0-18: every row's logits sha256 equal |
| `run` / `draft` / `multi --context 8192` | 4/4, 4/4, 3/3 MATCH |
| Served greedy replies (`bench_openai.py`, code and chat, 2 reps) | token hashes equal with and without this commit |
| `tools/zig/lean_check.py` | 542, the same as ashhart/TensorFold#608's branch |

Memory, `serve --context 32768 --parallel 1` with the drafter (the startup line):

| | #3 | #3 + this |
|---|---|---|
| Model as counted | 16.82 GiB | 16.92 GiB (whole blocks now) |
| **Left for streams after the 1 GiB reserve** | 3.0 GiB | **3.9 GiB** |

Speed is unchanged (code 127.4 → 130.5, chat 90.6 → 89.6 tok/s, single runs). On my 3090 fork, this is what lets a 262K window load with the default reserve.

Not checked: GB10 and the RTX PRO 6000. The change is host-side allocation only, but their unified or large memory may show a smaller effect.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
