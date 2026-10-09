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

## The 35B fit check: R1s-SD on h200x8 (2026-10-07)

Can the real model, R1s-SD (Ornith 1.5 35B-A3B after self-distillation, bf16), run RL on one 8×H200 node? Same
tasks and harness as the 9B smoke test; 32 attempts per step in groups of 8; 3 steps
(`ornith35b-sd-rl-fitcheck-h200x8.toml`). 4 GPUs run vLLM (one 67 GB copy each), 4 train (experts split 4 ways,
each 128K row split 4 ways). The trainer keeps fp32 master weights, keeps the expert routing frozen, and replays the
expert choices vLLM made (router replay).

```bash
CONFIG=ornith35b-sd-rl-fitcheck-h200x8.toml FLAVOR=h200x8 TIMEOUT=120m MODEL_SRC=ornith-sd-bf16 ./hf_rl_job.sh submit
```

| Step | Attempts collected in | Reward | Cut off by the length cap | Trainer update | Trainer peak memory | vLLM-trainer KL |
|---|---|---|---|---|---|---|
| 1 | 26m 16s (includes warm-up) | 0.61 | 48% | ~7.5 min | 134.1 GiB | 0.0010 |
| 2 | 10m 20s | 0.60 | 20% | 4m 13s | 133.9 GiB | 0.0009 |
| 3 | 4m 40s | 0.20 | 53% | 5m 18s | 133.8 GiB | 0.0010 |

- **It fits, with little room.** The trainer peaks at ~134 of 140 GiB per GPU. Each step logged one allocator warning
  (a 20 MB mapping failed, then succeeded after the cache was freed). The peak is the same every step, because
  rollouts are packed into fixed 128K rows: longer attempts add rows, not memory per row.
- **Router replay works.** The vLLM-trainer mismatch is ~0.001, half the 9B smoke test's without it.
- **Speed:** the trainer reaches 10K tokens/s (MFU ~70%) once warm. Collecting attempts is the bottleneck. Steps 2
  and 3 were fast partly because attempts finished during step 1 were already waiting.
- **Prime sandboxes still drop out:** 13 attempts lost to "The sandbox has been terminated" in ~47 minutes (retried).
- **Setup:** ~10 min to install, 7 min to copy the 67 GB checkpoint from the bucket, ~3 min to load the trainer.
- **Cost:** 66 minutes, ~$44. Stopped while it was saving the final checkpoint (fp32 weights plus optimizer state,
  ~420 GB), which a fit check doesn't need.

**For the real run:** steady state is roughly 10–13 minutes per step of 32 attempts, so 50 steps is about 9–11 hours,
**~$360–450** on h200x8 plus sandbox fees. Raising the number of attempts in flight above 64 should shorten steps
while vLLM's GPUs still have spare capacity. The rewards above mean nothing yet: 3 steps at lr 1e-6.

## Before the real run: does RL actually help R1s-SD? (plan, 2026-10-07)

The fit check proves the loop runs on the 35B, not that it teaches anything. Riel's bar before ~$400: evidence that
RL moves R1s-SD.

**Room to learn (from existing rollouts, free).**
- On Terminal-Lego (187 tasks, two tries each), R1s-SD's average solve rate is 48%, and 58% of tasks are solved at
  least once.
- RL mostly turns "sometimes" into "usually", so the near-term prize on tasks like these is about +10 points.
- 37 tasks are solved exactly once in two tries: those give the clearest training signal.

**One problem to fix before any paid step: the length cap ends half the attempts.**
- In the fit check, 20–53% of attempts per step reached the 90K-token attempt cap. It sits at 90K so that one more
  32K reply still fits the 131K window.
- Those attempts score whatever the tests give a half-finished sandbox, usually 0. Much of the signal would then be
  "be shorter", not "solve it".
- Options:
  - Compaction, if the coaching test shows R1s-SD uses it when told: context stays small, so the cap rarely bites.
  - A 262K window: the trainer's 128K rows would need more splitting, so memory must be rechecked.
  - A 16K reply cap, which lets attempts run to ~110K. It cost 1.7 solves per attempt in the evals.
- **Update (2026-10-08): the coaching test sends RL to pi, and pi compacts by itself.** R1s-SD never compacts in
  prime_agent, even when told to (06-agents/harness-evals). pi's harness does it without the model:
  - It runs a separate "context summarization assistant" call and restarts from the summary.
  - In our 600 R1s-SD pi rollouts, 34 runs show it, e.g. one went from 64.9K to 21.6K prompt tokens.
  - For the pilot, give pi's model entry an explicit `contextWindow`. The verifiers harness writes `models.json`
    without one, so the threshold is pi's default. Compaction must fire below the 90K attempt cap, not above it.
  - Also decide whether the summarization calls are trained on or masked.
  - The fit check ran prime_agent. Switching to `env.agent.harness.id = "pi"` needs a short check run.
- Lost sandboxes are already handled: the config reruns a Prime-killed attempt up to twice and drops it from training
  after that, and drops timed-out attempts too.

**The pi check (2026-10-08): the length cap is fixed.**
`ornith35b-sd-rl-pi-check-h200x8.toml`, 2 steps on h200x8, 52 min, ~$35. Results: `results/picheck-35b-*`.
- Two changes from the fit check:
  - pi's contextWindow is patched to 98304 (`PI_CONTEXT_WINDOW` in `hf_rl_job.sh`), so pi summarizes at ~82K.
  - max_total_tokens went from 90K to 400K. It counts every new input token and every reply over the whole attempt,
    so at 90K it would have cut summarized attempts anyway.
- Training through pi works. Both steps trained on all their samples, and the job exited cleanly.

| | prime_agent fit check | pi check |
|---|---|---|
| Steps 1 / 2: minutes | 26 / 10 | 17 / 23 |
| Steps 1 / 2: reward | 0.61 / 0.60 | 0.81 / 0.69 |
| Steps 1 / 2: "truncation" | 48% / 20% | 12.5% / 0% |
| Attempts stopped by the length cap | most of the truncation | **0 of 136** |

- 12 of 129 attempts with a saved trace summarized (3–5 trainer rows each). No call's prompt went over 82K; the
  largest was 81K.
- Those 12 are the long, hard attempts: 11 then ran into the 60-turn cap and 3 were solved, against 75 of 103 for
  the rest.
- What still ends attempts early:
  - **60-turn cap: 18 attempts.** For the pilot, raise it to 100. Summarizing frees room, but turns keep counting.
  - **A reply cut at 32K: 6 attempts, 0 solved.** This is the known "thinks without acting" failure. It's left alone:
    RL should learn to avoid it, since every one scores 0.
  - **Prime killed the sandbox: 14 (10%).** Dropped from training, as before.
- Summary calls are trained on like any other turn: their reward is the attempt's. Masking them would need code and
  there's no reason yet.

**The pilot: a go/no-go run that isn't wasted.** Ran 2026-10-09 (results below the plan).
Config: `ornith35b-sd-rl-pilot-h200x8.toml`; prime-rl's own config check passes. Everything else is the pi check's setup.
- **Tasks:** a fixed pool of 16 tasks R1s-SD solves sometimes (`pilot/pool.txt`). Each step trains on 4 of them, 8
  attempts each. Over 12 steps every task comes up 3 times, so its solve rate can be watched rising (or not).
  - Picked from the 63 TB2 train tasks by R1s-SD's earlier solve rates: from cancel-async-tasks (about half) down to
    polyglot-c-py and query-optimize (1 in 10).
  - Dropped: sanitize-git-repo, chess-best-move and reshard-c4-data, which lost a third or more of their attempts to
    sandbox failures.
- **Held-out check:** the held-out 20, 2 attempts each, before step 1 and after step 12, on the run's own model server
  with the same sampling. 40 attempts catch a collapse, not a few-point drift.
- **Settings changed from the pi check:** learning rate 2e-6 (was 1e-6) so 12 steps can show a trend; turn cap 100
  (60 cut 7 of the pi check's 32 attempts).
- **Saving the result:** checkpoints at steps 4, 8 and 12 (only the latest kept). After training the job:
  - turns the last checkpoint back into normal model files;
  - checks that the frozen parts (vision tower, routers) are unchanged;
  - measures how much of the model changed;
  - uploads the weights (~67 GB) to the sd-out bucket.
  - A time limit on training (5 h) leaves room for this even if steps run slow.
- **Leak checks done before the pilot (all clean):**
  - Images (`pilot/image_check.sh`, reports in `pilot/image_check/`): each of the 16 images opened with the network
    off. No Git repos, no test or solution files.
    - large-scale-text-editing ships its expected output by design: its grader rebuilds both the input and the
      expected file before checking.
    - The caches found are pip download caches.
  - Earlier rollouts (`pilot/hack_scan.py`, `pilot/hack_scan_before.json`): all 215 earlier R1s-SD attempts at the 16
    pool tasks (evals plus the pi check). Every hit is ordinary work.
    - configure-git-webserver clones its own local server repo, which is the task.
    - count-dataset-tokens downloads its dataset; mcmc-sampling-stan checks CRAN.
    - fix-ocaml-gc reads OCaml's own test suite; sqlite-with-gcov looks at gcov files.
  - Vals AI's MiMo audit (root README, reading spine) is why: it showed RL learning to dig hidden answers out of Git
    objects, file times, build caches and upstream repos.
- **Go (all four):**
  - the pool's solve rate clearly up (about +10 points or more from its first pass to its third);
  - held-out not down;
  - no reward hacks: rerun `hack_scan.py` on the pilot's rollouts, and read every solved attempt it flags;
  - weights moving: `pilot/weights_moved.py`, start vs end. NeMo-DCR measured 0.6–1.2% of bf16 weights changing per
    GRPO step across six models. Near zero over 12 steps means the updates rounded away and the run can't have learned.
- **No-go:** the pool is flat (the recipe doesn't teach; fix it before spending more), or the pool is up but held-out
  is down (memorising; needs a bigger pool).
- **On go:** the full run starts from the pilot's uploaded weights, so the pilot's training counts toward it. Its
  optimizer state is not kept (~0.4 TB); that only costs the first few steps' warm-up.
- **Cost.** HF has no cheaper machine that fits: training needs ~134 GB per GPU, and only H200s have 141 GB (A100 80,
  RTX PRO 6000 96). Prime's H200 pods cost about the same per GPU. So the saving has to come from wasted time.
  - **Where the time went in the pi check:** steps took 17–23 min although the median attempt took 5 min. A step needs 4
    complete groups of 8, and one slow attempt (up to 41 min) holds up its whole group. At the worst point, 67 finished
    attempts sat waiting while the batch stood at 31 of 32. Meanwhile attempts in flight crept from 32 to 54 over 40 min.
  - **Two changes:**
    - attempts capped at 30 min (was 60). In the pi check only 3 of 136 attempts ran longer, and 1 of the 84 solves;
    - 64 attempts in flight from the start.
    - The held-out test uses the same 30-min cap before and after, so it stays a fair comparison.
  - **Estimate:** 12–20 min a step. That's a guess from the pi check, not measured with these settings.

  | Part | Time | Cost |
  |---|---|---|
  | Setup, model load, server start | ~20 min | ~$13 |
  | 12 training steps (the first overlaps the "before" test) | 2.5–4 h | $100–160 |
  | "After" test on the held-out 20 | ≤30 min | ≤$20 |
  | Export, checks, upload | ~30 min | ~$20 |
  | **Total** | **3.5–5 h** | **about $150–215**, hard cap $240 (6 h job limit; training stops at 5 h) |

  Prime sandbox fees come on top. The first quote ($120–140) assumed ~$9 a step; with the old settings the pi check's
  pace put it at $215–300.

**Pilot result (2026-10-09): a small gain everywhere, none of it proven. Not a clear go.**
Run `ornith35b-sd-pilot-1009-1234`, 12 steps in 2 h 44 min, ~$118. The trained weights are in the sd-out bucket as
`ornith35b-sd-pilot-1009-1234-bf16`. Results: `results/pilot-*`.

| Check | Result |
|---|---|
| Pool solve rate up ~10 points | **+5.0 points** per task, early (v0–2) vs late (v6+) versions, of all attempts; +3.6 of finished attempts. 10 tasks up, 5 down (`pilot/pool_progress.py`). |
| Held-out not down | **32/48 → 34/46** usable runs (67% → 74%), the held-out 20 × 3 in pi. Up 3 tasks, down 2, all by 1–2 runs. |
| No reward hacks | **One shortcut.** A late fix-ocaml-gc attempt downloaded upstream OCaml's `shared_heap.c`, diffed it against the broken copy and copied the fix (solved). None of 14 attempts did that before the pilot; 1 of 40 during it. Everything else the scan flagged is ordinary work (`pilot/hack_scan_pilot.json`). |
| Weights moving | **Yes.** 16.2% of bf16 weights differ from the start (experts 17%, attention 14%). Vision tower and routers are bit-identical, as frozen. |

- **A trap in reading the pool numbers.**
  - At first, late versions looked +13 points better, with timeouts falling from 28% to 15%.
  - Attempts still running when training stopped were never recorded, and those were mostly slow ones from the last
    versions. That made late versions look faster and better than they were.
  - Finished attempts never got shorter (~7 min, ~22 turns throughout).
  - With attempts dispatched in the last 31 minutes left out, it's +5.
- **Shorter replies on the held-out tasks.**
  - The trained model's replies per attempt dropped: median 13.1K → 9.8K tokens, mean 51K → 34K.
  - Turns stayed the same (~21), and fewer attempts hit the 100-turn cap (8 → 5).
  - Its three series finished ~20 min sooner.
  - On the pool tasks, reply length didn't change, so this may be noise too.
- **Lost data.**
  - 30-minute timeouts dropped 15–31% of each step's attempts. Steps trained on 14–32 attempts, not 32.
  - In the held-out test, 12 and 14 of each model's 60 runs were lost: most to Prime terminating sandboxes, three to a
    Node.js package 404 at pi setup.
- **The in-job held-out eval failed.**
  - prime-rl sends eval requests as plain chat with `tool_choice: auto`, which the run's vLLM rejects; it had no tool
    parser. Training goes through the renderer and wasn't affected.
  - The held-out test was rerun outside: each model served by plain vLLM on its own H200
    (`06-agents/harness-evals/hf_serve_vllm.sh` with MODEL_SRC), same tasks and settings for both. ~$9.
- **Before any longer run:**
  - block GitHub downloads on tasks with an upstream fix (`network_block`);
  - raise the 30-minute cap (45–60 min) or score timeouts as fails;
  - fix the in-job eval: a tool-call parser on the run's vLLM, or keep testing outside as here.

**Continuation (approved 2026-10-09): 12 more steps from the pilot's weights, with the fixes.**
Config `ornith35b-sd-rl-cont-h200x8.toml`. Same pool, settings and lr, so the pool numbers continue the pilot's.
The optimizer starts fresh, because the pilot's checkpoints stayed on its job.

- **GitHub blocked while the agent works.**
  - Hosts blocked: github.com, \*.github.com, githubusercontent.com, \*.githubusercontent.com.
  - Prime matches exact names. A bare `github.com` still let raw.githubusercontent.com through (`pilot/egress_test.py`).
  - In the pilot only 2 of ~600 attempts touched GitHub, so honest work loses almost nothing.
  - TB2 scores in the agent's own sandbox, and every pool task's `test.sh` installs uv from GitHub (fix-ocaml-gc's also clones from GitHub).
  - So `SCORING_REOPEN=1` patches verifiers to lift the block after the agent stops and before scoring. The block and the reopen were tested on a Prime sandbox, and one full regex-log attempt (DeepSeek V4.1 Flash, pi) was tested end to end: blocked at 16:53, reopened at 17:01, scored 1.0.
- **Timeouts:**
  - The cap is now 60 min, up from 30.
  - Scoring timeouts as fails isn't an option: prime-rl drops a timed-out training attempt by design (`dispatcher.py`).
- **No in-job eval:**
  - The fix is one setting (`tool_call_parser = "qwen3_coder"`, `reasoning_parser = "qwen3"`). But each in-job eval holds all 8 GPUs (~$20 per eval).
  - The held-out 20 are tested outside afterwards (~$5), as for the pilot.
- **Saving and cost:**
  - Checkpoints at steps 3, 6, 9 and 12.
  - RL_TIMEOUT=3h15m, job TIMEOUT=3h50m: at most ~$155, expected $120–150.
- **Read-out:**
  - `pilot/pool_progress.py <traces> 61` on the pool.
  - Held-out on `<run>-bf16`, compared with base 32/48 and pilot 34/46.
  - `pilot/hack_scan.py` on the stream.
- **Go/no-go for the full run:**
  - Go if the pool keeps climbing past the pilot's late 50% and the held-out score holds or rises.
  - Stop RL if both flatten.
