# RL smoke test: Ornith-1.5-9B, GRPO, prime_agent on Terminal-Bench 2 (2026-10-07)

Does the whole RL loop run end to end? The model serves rollouts, the agent works in Prime sandboxes, the tasks'
tests give rewards, and the trainer updates the weights and sends them back to the model server.

- **Trainer:** [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) at `81052082`. GRPO-style groups of 4
  attempts per task, 16 attempts per step, 3 steps, learning rate 1e-6.
- **Tasks and agent:** 63 Terminal-Bench 2 training tasks, with the 6 heaviest left out. The agent is prime_agent:
  up to 60 turns, 90K tokens per attempt, and 32K per reply.
- **Hardware:** a Hugging Face Job on h200x4 at $20/h. 2 GPUs run vLLM and 2 run the trainer.
- **Network:** the sandboxed agent reaches the model through a Prime tunnel. `tunnel_check.py` showed an HF Job can
  open one.

```bash
CONFIG=ornith9b-tb2-smoke-h200x4.toml FLAVOR=h200x4 ./hf_rl_job.sh submit
./watch.sh <job id> <run>          # job log, finished steps, error count, stage changes
```

## Result: it works

| Step | Time | Average reward | Attempts cut off by the length cap | Turns per attempt | Trainer throughput |
|---|---|---|---|---|---|
| 1 | 14m 25s | 0.50 | 50% | 36.0 | 1,032 tok/s (includes warm-up) |
| 2 | 9m 50s | 0.75 | 25% | 36.8 | 4,012 tok/s |
| 3 | 8m 44s | 0.44 | 44% | 30.3 | 4,616 tok/s |

- **The loop runs.** Each step collected its 16 attempts and trained on them. The new weights reached vLLM over
  NCCL, and the next step's attempts used them: the off-policy lag never went past 2 steps.
- **The model server and trainer agree.** The KL between vLLM's and the trainer's token probabilities was
  0.002–0.006, so the trainer is learning from the same probabilities the model actually sampled from.
- **Trainer memory:** the peak was 70.4 GiB per GPU, with 131K-token rows. That is half an H200, so there is
  headroom for longer attempts or bigger batches.
- **The rewards mean nothing yet.** Three steps at lr 1e-6 can't move a model; the swings are which tasks each step
  drew.
- **Prime sandboxes still drop out.** 20 tracebacks over the run: sandboxes terminated mid-attempt, plus "Bad Gateway"
  on exec. Those attempts were retried (up to 2 times). Step 3 still lost 5% of its attempts.
- **Many attempts hit the 90K-token cap** (25–50%). On a real run, that cap and the turn limit decide how much of
  each attempt the model learns from.
- **No sandboxes leaked:** none with the run's label were left after the job ended.

## Cost

| | Minutes | Cost |
|---|---|---|
| Three failed starts (missing `prime_kernels`, missing `vllm-router`, NCCL broadcast sized before `dp` was set) | ~11 | ~$4 |
| The run (setup ~1 min, model load ~17 min, 3 steps, save) | ~42 | ~$14 |
| **Total** | | **~$18** |

The a100x8 version queued for over an hour and was cancelled before it ran. Sandbox fees aren't included.

Once warm, a step of 16 attempts takes about 9–10 minutes, which is **~$3 per step** on h200x4. Rollouts are the
bottleneck: the trainer finishes its part in a fraction of that.

## Fixes it took (in `hf_rl_job.sh` and the h200x4 config)

1. `uv sync` needs the `kernels` extra (`prime_kernels`) and the `disagg` extra (`vllm-router`).
2. On Hopper, `attn = "auto"` picks FlashAttention 3, which the job doesn't have, so the config forces
   `flash_attention_2`.
3. `rl` sizes the NCCL weight broadcast before `[deployment]` sets vLLM's data parallelism. Set
   `[inference.vllm] data_parallel_size` explicitly, or vLLM's workers have no broadcast receiver.
4. HF Jobs reject a timeout written as `4h30m` (use `270m`) and single-letter env names.
