# R1s-SD solve rates on Terminal-Lego (prime_agent)

Which Terminal-Lego (TL) tasks Ornith R1s-SD solves sometimes: the band RL can learn from. A task solved in every
try gives no signal, and so does one never solved. Each task got 2 tries under prime_agent, with 60 turns and a
1-hour limit, in the local Docker runtime against R1s-SD on HF a10g-large servers. Run with `solve_rates.py pilot hard`.
Per-task rows are in `solve_rates_<set>.jsonl`.

| Set (2026-10-05) | Tasks | Solved | Always | Once | Never |
|---|---|---|---|---|---|
| Pilot, medium (16K reply cap) | 36 | | 26 | 4 | 6 |
| Pilot, hard (16K reply cap) | 12 | | 4 | 2 | 6 |
| Pilot total | 48 | 65/96 | 30 | 6 | 12 |
| Hard round (32K cap, `tl_round.sh`) | 140 | 117/280 (42%) | 43 | 31 | 66 |

- **The hard pool is used up.**
  - 152 of the 153 eligible hard tasks have been tried.
  - The one left out, task_12598, failed to build: `cabal update` failed.
- **How the 163 failed hard attempts ended:**
  - 70 hit the 1-hour limit.
  - 38 ran out of 60 turns.
  - 55 finished with a wrong answer.
- **Some solves came at a limit.** 10 solves came after the 1-hour limit and 11 at the turn limit: the tests grade whatever state the sandbox is left in.
- **Solved runs average 33 turns.** Hard-task rollouts are long, and this sets the cost of RL rollouts.
- **The local docker runtime doesn't enforce the hour limit when a sandbox command hangs.**
  - `tl_watchdog.sh` removed 6 such containers after an hour. Those attempts count as timeouts.
  - No attempt failed for infrastructure reasons.
- **RL candidates: 37 tasks solved once** (31 hard, 2 pilot hard, 4 pilot medium).
  - Two tries make a coarse label. A task solved half the time lands in "once" only half the time.
  - Tasks at 20-30% often show as never, and tasks at 70-80% as always.
  - The learnable band is therefore wider than these 37. Retrying the 109 always/never hard tasks would find more.
- **Medium is mostly too easy:** 26 of 36 were always solved.
  - There are 4,431 eligible medium tasks, though.
  - At the pilot's 4 in 36, about 490 of them would land in the solved-once band.
