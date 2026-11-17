Paired against q27pi:off (per-task solve rates, permutation test):
- q27pi:med: +26.3 points a task on average; better on 8 tasks, worse on 1; p = 0.021
- q27pi:ad: +31.6 points a task on average; better on 11 tasks, worse on 1; p = 0.004
- ornith35b:r1s-c4s: +15.8 points a task on average; better on 6 tasks, worse on 1; p = 0.139
- nanbeige:bf16: +22.8 points a task on average; better on 9 tasks, worse on 4; p = 0.105

| Build | Solved-run tokens vs q27pi:off | 95% CI | Longer on | Output tokens per call (median run) |
|---|---|---|---|---|
| q27pi:med | 1.81x | 1.47-2.26 | 9 of 9 tasks | 363 |
| q27pi:off | - | - | - | 161 |
| q27pi:ad | 1.01x | 0.81-1.31 | 4 of 10 tasks | 283 |
| ornith35b:r1s-c4s | 2.55x | 1.89-3.37 | 10 of 10 tasks | 406 |
| nanbeige:bf16 | 6.47x | 4.57-9.54 | 8 of 8 tasks | 1074 |

| Build | Gave a wrong answer | Hit the turn cap | Hit the time cap | Other |
|---|---|---|---|---|
| q27pi:med | 9 | 9 | 3 | 0 |
| q27pi:off | 11 | 22 | 3 | 0 |
| q27pi:ad | 7 | 7 | 4 | 0 |
| ornith35b:r1s-c4s | 10 | 14 | 3 | 0 |
| nanbeige:bf16 | 18 | 2 | 3 | 0 |
