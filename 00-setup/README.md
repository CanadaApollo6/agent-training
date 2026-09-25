# 00 · Setup

## Python environment

The root project pins Python 3.12 (PyTorch, vLLM and NeMo lag new Python releases; the system default is 3.14)
and pulls torch from the CUDA 13.2 index, the newest one with torch wheels.

```bash
uv sync
uv run 00-setup/roofline.py
```

Later modules that pin their own torch (vLLM, NeMo RL, prime-rl, Kev) get their own `uv` projects so they don't
fight over versions.

## CUDA toolkit

PyTorch wheels bundle the CUDA runtime, so `nvcc` is only needed to compile our own kernels (Module 2), and for
packages that build CUDA extensions (flash-attn, causal-conv1d). It needs root:

```bash
sudo pacman -S cuda        # installs to /opt/cuda: nvcc, Nsight Compute (ncu), Nsight Systems (nsys), compute-sanitizer
```

Open a new shell afterwards so `/etc/profile.d/cuda.sh` puts `/opt/cuda/bin` on `PATH`, then `nvcc --version`.

## After a system update

If `nvidia-smi` reports `Driver/library version mismatch`, the update installed a new NVIDIA userspace while the
old kernel module is still loaded. Reboot.

## Baseline (2026-09-25)

EVGA RTX 3090, driver 615.71.09, torch 2.14.0+cu132:

| | achieved | spec | |
|---|---|---|---|
| DRAM bandwidth (D2D copy) | 829 GB/s | 936 GB/s | 89% |
| bf16 matmul 8192³ | 76.7 TFLOPS | 71.0 | 108% (factory overclock) |
| tf32 matmul 8192³ | 39.6 TFLOPS | 35.6 | 111% |

Ridge point ≈ 93 FLOPs/byte. Batch-1 decode is ~1 FLOP/byte in bf16, so it is entirely bandwidth-bound; the
decode ceiling for a model is roughly `829 GB/s ÷ weight bytes` (Bonsai 2 27B: ~140 tok/s, Qwen3.8 27B at
4.25 bpw: ~58 tok/s).
