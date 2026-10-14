# Prompt-lookup drafting (2026-10-14)

TensorFold fork branch `lfm2`, commit 6afe44f7, `tf-lfm2-cuda run ... --draft 15`.

LFM2.5 has no draft head, so the drafts come from the text itself: find the latest earlier place where the last
few tokens appeared, and guess that what followed then follows again. One pass checks the real next token plus the
guesses. Every guess the model agrees with is a free token.

## Why it pays less here than on Ornith

Checking more tokens in one pass is not free on this model. Each extra row brings its own 4 experts out of 64, and
those weights have to be read too. Measured cost of one pass by rows checked (short context):

| rows | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ms | 3.1 | 3.9 | 4.8 | 6.2 | 9.2 |

So a guess has to be right more than about a quarter of the time to earn its row. The model's own outputs show
how often plain guessing is right. On fresh code about 12% of guessed tokens were accepted. On the long
rewrite-this-code prompt it was about 30%.

## The policy

We replayed the model's greedy outputs against the cost table above to compare policies (`draft_policy.py`). The
winner:

- draft only when the last 2 or more tokens match earlier text, using the longest match;
- start at 2 guesses, double after a fully accepted draft, otherwise drop to the number accepted;
- with no draft, run the plain one-row step.

## Results

Greedy, 1024 tokens at most, same prompts as the serial tuning:

| prompt | plain tok/s | drafted tok/s | tokens per pass |
|---|---|---|---|
| code | 327 | 345 | 1.17 |
| tools | 322 | 354 | 1.24 |
| chat | 319 | 330 | 1.03 |
| 14.7K-token rewrite | 285 | **351** | 1.72 |

Drafting changes which kernels run, and their sums round a little differently, so the text can split from plain
decoding at a near-tie. To prove the drafting itself is exact we forced both paths to round alike. Then drafted
output matched plain greedy token for token on all four prompts.

Reproduce: `tf-lfm2-cuda run ~/models/lfm25-24b/mlx4 --prompts prompts.json --max-tokens 1024 --draft 15`;
window costs with `TF_LFM2_BENCH_WINDOWS=1`; the exactness check adds `TF_LFM2_ATTN_SPLITS=8 TF_LFM2_SPLITK=0` to
both runs and `--expect`.
