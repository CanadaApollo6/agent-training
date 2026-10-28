# TensorFold llama P1: Nanbeige4.3-33B EXL3 3.5 bpw on the RTX 3090

Engine commit: `fe0bd8fb` on branch `nanbeige` (worktree `01-inference/tools/TensorFold-llama`, base TF 1.1.8 `31ce9eb3`).
Reference: exllamav3 1.6.5 (`~/exl3-local`), `tools/zig/llama_reference.py`.

| Check | Reference | Result |
|---|---|---|
| 12 layers, 217 positions | exllamav3 (`ref-12-*`) | mean KL 5.8e-5, max 5.6e-4, top-1 214/217, 0 above 0.01 |
| 52 layers, 217 positions | exllamav3 body + fp32 dense head (`dense/`) | mean KL 8.7e-5, max 2.1e-3, top-1 214/217, 0 above 0.01 |
| 52 layers, 217 positions | exllamav3 with its EXL3 head (`ref-52-*`) | mean KL 1.1e-3, max 0.055 (seq 2 pos 7), 2 above 0.01, top-1 212/217 |
| 128 greedy tokens | dense-head reference (`dense/ref-52-greedy.json`) | 128/128 match; lane backend equals serial |
| 128 greedy tokens | exllamav3 greedy (`ref-52-greedy.json`) | first divergence at token 17 |
| 561-token prompt, 2 layers | exllamav3 (`boundary/`) | mean KL 6.1e-5, max 1.8e-4 (crosses the 512-key span) |
| Serial decode, context 4096, 32-token prompt | none | median 28.12 tok/s, 3 runs, range 28.03 to 28.20; device peak 15.3 GB |

Why the exllamav3 head disagrees: its EXL3 head kernel gives different logits at 1 to 2 rows than at 212 rows.
At the token-17 prefix, `\n` scores 47.40 at 1 row, 49.00 at 212 rows and 48.51 with the dense fp32 head.
The engine's body plus the dense head stays below 0.01 KL everywhere.

Speed limiter (nsys on the 12-layer run, `/tmp/llama-prof`): EXL3 linears take about 508 us a layer for about 258 MB.
That is about 510 GB/s, 54% of the card's 936 GB/s. The 6-bit head runs at about 846 GB/s.
Kernel time is about 29.2 ms of the 35.6 ms a token takes. The other 18% is launch gaps and the per-token host sync.

VRAM receipts: `*-vram.json` (`tools/zig/llama_gpu_run.py`, nvidia-smi sampled every 200 ms).
