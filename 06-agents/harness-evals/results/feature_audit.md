# prime_agent feature use by R1s-SD

1653 rollouts read, 1439 usable (not lost to infrastructure), 441 tasks. Solve rate overall: 42.8%.

## By experiment

| Experiment | Usable runs | Solved |
|---|---|---|
| 16K reply cap | 52 | 50% |
| 32K reply cap | 54 | 57% |
| HF pod runs | 39 | 8% |
| Terminal-Lego hard | 280 | 42% |
| Terminal-Lego pilot | 95 | 68% |
| Terminal-Lego smoke | 1 | 100% |
| big budget | 21 | 14% |
| fix smoke | 8 | 50% |
| fix test, off | 156 | 39% |
| fix test, on | 164 | 44% |
| lessons loop | 230 | 25% |
| round-2 data | 339 | 52% |

## Features (pattern found in at least one ipython cell)

| What | Meaning | Runs using it | Share | Solved with | Solved without | Same tasks: with / without (tasks) |
|---|---|---|---|---|---|---|
| `edit` | the edit skill for file changes | 163 | 11.3% | 53% | 42% | 53% / 43% (103) |
| `compact` | context compaction | 0 | 0.0% | nan% | 43% | – |
| `spawn` | start a sub-agent | 0 | 0.0% | nan% | 43% | – |
| `agent_msg` | talk to or watch sub-agents | 0 | 0.0% | nan% | 43% | – |
| `harness_layer` | memory/skills/refine layer | 14 | 1.0% | 79% | 42% | 75% / 45% (12) |
| `goal` | the goal skill | 1 | 0.1% | 100% | 43% | – |
| `websearch` | web search skill | 48 | 3.3% | 15% | 44% | 12% / 8% (23) |
| `attach_image` | look at an image | 15 | 1.0% | 33% | 43% | 33% / 31% (9) |
| `skill_docs` | read a skill's docs | 27 | 1.9% | 26% | 43% | 28% / 30% (20) |
| `bash_background` | start a shell job and keep its handle | 48 | 3.3% | 42% | 43% | 39% / 40% (40) |
| `bash_await` | run a shell command and wait | 45 | 3.1% | 47% | 43% | 48% / 50% (42) |
| `py_file_io` | read/write files in Python | 1359 | 94.4% | 44% | 20% | 32% / 14% (35) |

## Habits the system prompt warns against

| What | Meaning | Runs using it | Share | Solved with | Solved without | Same tasks: with / without (tasks) |
|---|---|---|---|---|---|---|
| `subprocess` | subprocess/os.system instead of bash() | 1191 | 82.8% | 46% | 29% | 32% / 25% (90) |
| `heredoc` | heredoc inside bash() | 4 | 0.3% | 50% | 43% | 50% / 45% (4) |
| `shell_loop` | shell loop inside bash() | 2 | 0.1% | 0% | 43% | 0% / 0% (2) |
| `sleep_poll` | polling with sleep | 294 | 20.4% | 40% | 43% | 34% / 35% (72) |
| `sed_inplace` | sed -i edits | 1 | 0.1% | 100% | 43% | 100% / 100% (1) |

## Calls to tools that don't exist

969 runs (67.3%) called a tool other than `ipython`: `bash` x1780, `websearch` x62, `edit` x16, `attach_image` x4, `goal` x1, `websearch.run` x1.
Each got back "Tool <name> not found": 1,864 wasted turns, 4.1% of all turns.
Solved with: 45%; without: 39%.

## How runs end

| Ending | Runs | Solved |
|---|---|---|
| agent_completed | 942 | 59% |
| max_turns | 300 | 13% |
| agent_timeout | 197 | 13% |

Runs whose last model reply was cut off by the length cap: 155 (10.8%), solved 0%.
Turns per run: median 29. Peak prompt size: median 44,373 tokens, 90th percentile 108,975, max 130,475; runs past 100K: 228.
Runs with more than one trace (sub-agents ran): 0.

## By peak prompt size (no run ever compacts its context)

| Peak prompt | Runs | Solved | Ended: done / turn cap / time limit |
|---|---|---|---|
| 0-32K | 551 | 53% | 484 / 16 / 51 |
| 32-64K | 367 | 47% | 264 / 50 / 53 |
| 64-100K | 293 | 35% | 129 / 109 / 55 |
| 100K+ | 228 | 20% | 65 / 125 / 38 |
