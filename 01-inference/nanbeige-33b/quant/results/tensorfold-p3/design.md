# DFlash2 module boundary

The caller loads a checkpoint config, binds target hidden/vocabulary/layer geometry, chooses draft storage, and attaches a drafter to the existing lane core.

## Candidates

A family adapter can borrow Qwen's portable graph while leaving all source under qwen27. That keeps paths stable but makes target-neutral code look Qwen-specific. It also leaves the stock parser restriction at an ambiguous boundary.

A shared dflash2 module can own config, tensor graph, operator protocol, execution, committed context and selector. Qwen's public paths re-export the same declarations. Qwen's parser retains requireStock at its boundary. Llama's CUDA adapter binds the generic config to its own target. This is the chosen shape. Model the Domain puts geometry and storage in explicit types.

## Types and ownership

- `Config` owns dimensions, tap order, window, convolution and selector settings. `parse` validates external JSON. `requireStock` remains a Qwen boundary check.
- `Graph` borrows checkpoint tensors and the target embedding. It never owns target weights.
- `Ops` owns backend-specific projection, norm, convolution, RoPE, attention and head execution. Shared `execution` owns the DFlash2 schedule.
- `Preparation` records weight storage independently of target KV precision.
- Llama owns its target KV, sampler and target hidden taps. The drafter owns only its draft weights, rolling context and forward scratch.

The target embedding and LM head must be borrowed. Codebooks can stay on the host because only selected candidate rows enter selector scores. Full-model output equality and speed remain separate gates. A serial verification implementation can prove mechanics but cannot establish a speedup.

## Throughput checkpoint

First prove portable contracts and real checkpoint accounting. Then prove a bounded CUDA drafter operation against an independent reference. Advertise no full-model correctness, quantization acceptance or speed until the target fits and those exact gates run.
