# Served, and staged as M3TiS #3 (2026-10-16)

## The server

`tensorfold-native serve` now runs LFM2.5 (fork branch `lfm2`, 9115658a). It speaks the OpenAI API and turns
Liquid's Python-style tool calls into OpenAI `tool_calls`, streamed too. Greedy requests draft the way
`drafting.md` describes. Sampled requests pick from the top k on the GPU's logits row: 300+ tok/s, where a
full-vocabulary pick managed only 71.

Agent loops needed one more piece. LFM2.5's chat template drops the thinking of earlier turns, so every new turn
changes text the server already read. The short-convolution state can't be rewound, so the server snapshots it
every 512 tokens (about 240 KB each) and resumes from the last snapshot before the change. Prompts are always cut
on the same 128-token grid from the start, so a resumed turn does the same arithmetic as a cold read and gives the
same reply. On a 14.7K-token history the next turn takes 0.24 s to read instead of 7.0 s.

## The bundle

Staged at `~/.cache/m3tis3/bundle`; uploaded with Riel's OK to `CanadaApollo6/LFM2.5-24B-A2B-M3TiS-3090`. Card, launcher,
NOTICE and SOURCE.md live in `m3tis/`. The server is built portable (glibc 2.28+, x86-64-v3, sm_86) and stripped
to 10 MB. The weights are Liquid's MLX 4-bit files hard-linked unchanged, plus `embed4.safetensors`.
`make_embed4.py --stamp` wrote a change notice into embed4's header, which the LFM Open License asks for
(section 4b). The tensor bytes are untouched.

Checks on the bundle:

- `server_check.py` against `./m3tis-serve.sh` passes all seven checks: tool calls, resumed turn == cold turn,
  streamed calls, greedy and sampled code, the 14.7K prompt, and a long history's next turn.
- Two requests at once both finish; the second waits for the first.
- `repetition_penalty`, `frequency_penalty` and `presence_penalty` are accepted and ignored (same reply).
- `chat_template_kwargs {"enable_thinking": false}` answers without thinking.

## Release numbers (`m3tis/bench.py` → `m3tis-bench.json`)

Full 131K window, one user, 512-token replies, a warm-up then the median of 3, started below 55 °C.

| tok/s | greedy | Liquid's T 0.1, top_k 50 | T 1.0 |
|---|---|---|---|
| code | 334 | 316 | 310 |
| chat | 324 | 316 | 317 |
| tool call | 334 | 318 | 318 |

- The 14.7K-token rewrite prompt is read in 7.0 s. Replies then run at 347 greedy (drafted) or 279 with Liquid's
  settings.
- The first token takes 34 ms.
- The card peaked at 18.1 GB, including about 1.6 GB of desktop.

## Drafts and greedy text

With drafts on vs off, greedy replies split at a near-tie: after about 950 characters on the code prompt and 360 on
the long one. The tool-call reply was identical. This is the rounding difference `drafting.md` already proved: with
both paths forced to round alike, they match token for token. The card says so.

## After upload

The repo holds all 20 files at their exact sizes (Liquid's shards were already on the Hub, so they weren't sent
again). A fresh `hf download` into an empty folder matches the bundle file for file by sha256. Run as the card says
(`chmod +x`, `./m3tis-serve.sh`), it passes all seven `server_check.py` checks. Speeds were 3–8% under the release
numbers because the card was warm from the session before.

## Not done

No agent-task evals on this build. The KL figure comes from 443 positions of one answer.
