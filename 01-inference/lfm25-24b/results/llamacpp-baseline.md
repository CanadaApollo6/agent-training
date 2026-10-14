# llama.cpp baseline on the RTX 3090 (2026-10-14)

llama.cpp a0e6dc7, CUDA 13.4, sm_86, `llama-bench -ngl 99 -fa 1 -p 2048 -n 128 -d 0,8192,32768 -r 3`.
Desktop in use (1.6 GB VRAM held, load average ~8). No drafting.

| build | size | decode @0 | decode @8K | decode @32K | prefill 2K @0 | prefill 2K @32K |
|---|---|---|---|---|---|---|
| Q4_K_M | 13.52 GiB | 256.0 | 242.8 | 206.4 | 5610 | 4359 |
| QAD-Q4_0 (Liquid's quantization-aware build) | 12.64 GiB | 270.1 | 255.6 | 218.4 | 6183 | 4678 |

Bytes read per token (`gguf_bytes.py`, experts counted at 4 of 64):

| part | Q4_K_M GB/token | QAD-Q4_0 GB/token |
|---|---|---|
| routed experts | 0.865 | 0.807 |
| short conv (30 layers) | 0.284 | 0.284 |
| embed = LM head (tied, Q6_K) | 0.215 | 0.215 |
| dense MLP (2 layers) | 0.094 | 0.082 |
| attention (10 layers) | 0.061 | 0.059 |
| norms, routers | 0.020 | 0.020 |
| **total** | **1.539** | **1.467** |
| ceiling at 830 GB/s | 539 tok/s | 566 tok/s |

llama.cpp reaches 47-48% of the ceiling at empty context.
