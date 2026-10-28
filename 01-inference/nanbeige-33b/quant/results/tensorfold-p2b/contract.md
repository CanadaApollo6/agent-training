# P2b contract

Input is local branch nanbeige at e8d79bcc. Lean base is 31ce9eb3. No publication is authorized.
Nanbeige4.3-33B EXL3 mul1 3.5 bpw, CUDA sm_86 on a shared RTX 3090, one serial lane.
FP32 residuals, FP16 normalization outputs and BF16 KV remain unchanged.

The missing capability is sampled serial decoding and chat/completion serving with T 1, top_p 0.95, top_k 0, min_p 0.
Request values override model and CLI defaults. Temperature zero retains the original FP32 argmax.
Llama uses cuda.sample.Draws, its keyed rule and BF16 logit conversion, as Kolibri does.
The alternative of using the host sampler would add full-logit readbacks each token, so it is rejected.
The sampler owns pinned staging, device buffers and token counts. The lane retains request ownership and absolute positions.
Presence and frequency use the shared CUDA penalty operator. Non-neutral repetition penalty is refused before SSE.

Verify fixed-logit distributions independently with chi-square and a predeclared threshold, same-seed equality,
different-seed differences, slice greedy token equality, actual HTTP defaults and trace-derived pi SSE.
All device execution goes through llama_gpu_run.py. The card's other processes are never signaled.
Full P1/P2 checks run only when 19.5 GB is free and the admission guard allows them.
Binaries stay in the worktree's zig-out. Receipts stay here. ATC is absent from PATH; the user confirms sole local ownership.

Throughput checkpoint. Contracts, existing fixtures and pi fields precede editing. One writer owns this coupled adapter.
GPU checks serialize. Sampler correctness precedes server integration. No performance improvement is claimed.
