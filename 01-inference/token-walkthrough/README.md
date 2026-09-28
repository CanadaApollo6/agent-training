# What is a transformer, actually

One token walked through Qwen3.5-0.8B, the model profiled in [decode-gap](../decode-gap/). Each part opens one
more box, and every hand-written piece is checked against the model's own output. [CONCEPTS.md](CONCEPTS.md)
explains every part in plain English, with no code.

| Part | Script | Opens |
|---|---|---|
| 1 | `skeleton.py`, `lens.py` | The whole model as a loop: tokens → embedding row → 24 layers → LM head → next token |
| 2 | `mine.py`, `check.py`, `try_rmsnorm.py`, `anatomy.py` | Inside a layer: RMSNorm and the MLP |
| 3 | `attention.py` | Full attention and the KV cache |
| 4 | | Gated DeltaNet: memory that doesn't grow |
| 5 | | Decoding: sampling, and why decode is one token at a time |

## Words

- **Token**: a chunk of text, usually a word or part of a word, with a fixed id. The tokenizer is a lookup table built
  before training and is not part of the network. `' Patrick'` is id 19026; the leading space is part of the token.
- **Vocabulary**: every token the model knows. There are 248,320 of them.
- **Embedding table (`E`)**: a 248,320 × 1024 matrix with one row per token. A token's row is its starting vector.
- **Hidden size**: the length of that vector, 1024. Everything between the lookup and the LM head is 1024 numbers
  per token.
- **Residual stream**: the running 1024-number vector for each token. A layer never replaces it. It reads it
  and adds its own contribution (`h = h + layer_output`), like people adding notes to a shared page.
- **Layer**: one block that updates the stream: 18 Gated DeltaNet layers and 6 full-attention layers, in the
  pattern 3 DeltaNet, 1 full, repeated.
- **LM head**: turns the final vector into one score per vocabulary token. Here it is the embedding table again
  ("tied" weights): score(token) = dot product of the final vector with that token's row.
- **Logits**: those 248,320 raw scores. **Softmax** turns them into probabilities: exponentiate, then divide by
  the sum.
- **Greedy decoding**: always take the highest-probability token, append it, and run again.

## Part 1: the skeleton

```python
h = E[ids]                      # [1, T, 1024]: one row per token
for layer in model.layers:      # 24 times
    h = layer(h)                # each layer adds its contribution to h
h = norm(h)
logits = h @ E.T                # [1, T, 248320]; the last position predicts the next token
```

`skeleton.py` runs this loop by hand with the layers as black boxes. Its logits are **bit-for-bit identical** to
`model(ids).logits`, so this loop is the whole model.

The first version scored only the last position. It differed from the model by up to 0.06 in some logits. In
bf16, a different matmul shape sums in a different order and rounds differently. Scoring all positions, as the
model does, makes the results identical.

On `"The Kansas City Chiefs quarterback is Patrick"` it predicts `' Mah'` at 44.6%, then `'omes'`.

### Where the 1,505 MB goes, in the skeleton's terms

| Stage | Read per decoded token | Share |
|---|---|---|
| Embedding lookup (1 row) | 2 KB | 0.0% |
| 18 DeltaNet layers × 43.1 MB | 776.0 MB | 51.6% |
| 6 full-attention layers × 36.7 MB | 220.2 MB | 14.6% |
| Final norm | 2 KB | 0.0% |
| LM head (every row of `E`) | 508.6 MB | 33.8% |

The same 508.6 MB table is used twice in one step. The lookup reads 2 KB of it. The LM head reads all of it,
because the model must score every possible next token to find the best one. A third of all decode bytes exist
to rank 248,320 candidates. That is why small models with big vocabularies are dominated by the head.

### The logit lens (`lens.py`)

Applying the final norm and the LM head after *any* layer shows what the stream would predict if the model stopped
there. Row 0 means no layers: the embedding goes straight to the LM head.

Riel's predictions, written before running:

1. With zero layers the model predicts `'.'` or `' Star'`.
2. `' Mah'` becomes the top guess in the middle layers.

**Result 1: zero layers predicts `' Patrick'` at 100%.**

| Token | Logit | Rank of 248,320 | Cosine with the `' Patrick'` row |
|---|---|---|---|
| `' Patrick'` | 63.75 | 1 | 1.000 |
| `'Patrick'` (no space) | 54.0 | 2 | 0.78 |
| `' Mah'` | 6.75 | 65,465 | 0.10 |
| `' Star'` | 1.75 | 219,575 | 0.02 |
| `'.'` | -9.88 | 248,316 | -0.09 |

With no layers, the LM head computes the dot product of the `' Patrick'` row with every row, including itself. A
vector always points exactly along itself (cosine 1). In 1024 dimensions, unrelated rows are close to perpendicular
(median cosine 0.06). A logit gap of about 10 between first and second place is a factor of e^10 ≈ 22,000 in
probability, so softmax gives 100%. `' Star'` and `'.'` are good guesses for *what follows* Patrick. But "what
follows" is knowledge, and knowledge lives in the layers. The embedding table only says what each token *is*.
Without layers the model is an identity map: it predicts its own input.

**Result 2: `' Mah'` is the top guess only after layer 24.** Its rank shows the answer being built:

| After layer | 0–14 | 15 | 16 | 17 | 18 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| Rank of `' Mah'` | 65K–212K | 40,485 | 7,532 | 1,734 | 611 | 188 | 37 | 14 | **1** |

Through layer 14, `' Mah'` sits no better than a random token. It climbs steadily from layer 15 on and wins only at
the end. There's a caveat on reading this. The lens assumes every layer writes in the "language" the LM head
reads, and for a model this small that is only true near the end. The middle rows are garbage tokens at 1–4%
confidence (`'ablemente'`, `'不是吗'`). That is what it looks like to read a vector in a working format the head
can't decode. It doesn't prove the model knows nothing about Mahomes before layer 15. It shows that the knowledge
isn't in vocabulary form until then. A "tuned lens" trains a small translator for each layer to read the middle.
