# P2 fix round 2: qwen3_coder header recovery (2026-10-28)

Engine commit 6d56a1c4 on the local `nanbeige` branch (parent 94c2c22b).

The second P2 review found one parity gap: TensorFold took the first `<function=` in a tool call even when its header was
invalid, so an empty or newline-broken header before a valid `bash` header gave zero calls where the pod gives one.
`tool_qwen.zig` now searches for the first header that matches the pod's `<function=([^>\n]+)>`.

| Check | Result | Log |
| :-- | :-- | :-- |
| Three new pod-generated fixtures (`recover-*`) before the fix | fail (raw text left as content, no call) | `recover-before.log` |
| Same after the fix, parser and JSON+SSE routes | equal to the pod; host server tests 30/30 | `recover-after.log` |
| HF prompt fixtures after regenerating | unchanged (23 conversations, 3,118 ids) | |
| lean_check vs the 31ce9eb3 base | 426 / 426, 0 new | `lean-head.txt` |
| Native build (sm_86) | exit 0 | `build-native.log` |

`<function=<function=bash>` is kept as a case on purpose: the pod reads it as a call named `<function=bash`, and
TensorFold now does the same.
