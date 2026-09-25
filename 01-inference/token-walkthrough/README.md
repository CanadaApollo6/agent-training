# What is a transformer, actually

One token walked through Qwen3.5-0.8B, the model profiled in [decode-gap](../decode-gap/). Each part opens one
more box, and every hand-written piece is checked against the model's own output.

| Part | Script | Opens |
|---|---|---|
| 1 | `skeleton.py`, `lens.py` | The whole model as a loop: tokens → embedding row → 24 layers → LM head → next token |
| 2 | | Inside a layer: RMSNorm and the MLP |
| 3 | | Full attention and the KV cache |
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

Predictions, written before running:

- *(pending)*
