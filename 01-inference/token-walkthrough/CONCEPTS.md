# Part 1, conceptually

The reference explanation, with no code. The measurements behind it come from `skeleton.py` and `lens.py` (see the
README).

## The big picture

A language model is one function: **given the text so far, it gives a probability for every possible next
token.** Writing a whole answer is just calling that function over and over: pick a token, append it, call again.

Part 1 follows one call, from `The Kansas City Chiefs quarterback is Patrick` to the prediction `Mah` (the first
piece of "Mahomes"). There are six stages.

## Stage 1: Tokenizer (text → token IDs)

**What happens:** The text is cut into chunks called **tokens**, each from a fixed list of 248,320 (the
**vocabulary**). Each token has an ID number. The sentence becomes 7 tokens:

| Token | `The` | ` Kansas` | ` City` | ` Chiefs` | ` quarterback` | ` is` | ` Patrick` |
|---|---|---|---|---|---|---|---|
| ID | 760 | 19535 | 4170 | 43256 | 18889 | 369 | 19026 |

The leading space belongs to the token: `" Patrick"` and `"Patrick"` are different tokens.

**Why word pieces?** Letters would make every sentence very long, with little meaning per step. Whole words can't
handle new words, typos, names or code. Word pieces are the compromise: common words get one token, and rare ones
are built from pieces ("Mahomes" = "Mah" + "omes").

**Where and when:** On the CPU, before the GPU does anything. It's a lookup table, not a neural network. It was
built before training and never changes.

## Stage 2: Embedding (ID → vector)

**What happens:** The **embedding table** has one row per vocabulary token, and each row is 1,024 numbers. Token
19026 picks row 19026. That list of numbers is a **vector**, and 1,024 is the model's **hidden size**.

**Why:** Networks can only do arithmetic, and an ID is just a label. A vector can hold many properties at once.
The table is learned in training, so similar tokens get similar rows: `" Patrick"` and `"Patrick"` are 78%
aligned, while unrelated tokens are close to 0%.

**Where and when:** On the GPU. One row, 2 KB, is read per token. It's the model's starting description of each
token, with no context yet.

## Stage 3: The residual stream and 24 layers

**What happens:** Each token's vector flows through 24 **layers**. A layer never replaces the vector. It reads it
and **adds** its result back in. That running vector is the **residual stream**. It works like a shared document
that 24 editors annotate in turn, where nobody erases earlier notes.

Each layer does two jobs, each preceded by **RMSNorm** (a normalization step that keeps the numbers at a steady
size, like a volume knob):

1. **Mixing between tokens:** how "Patrick" reads "Chiefs" and "quarterback". It's done by **attention** in 6
   layers and **DeltaNet** in 18 (Parts 3 and 4).
2. **Per-token processing:** the **MLP**, where stored knowledge lives (Part 2).

**Why add instead of replace?** Early information survives to the end, each layer only contributes a refinement,
and training stays stable.

**Where and when:** On the GPU, every layer for every token. It's 996 MB of the 1,505 MB read per generated
token.

## Stage 4: LM head (vector → a score for every token)

**What happens:** After a final RMSNorm, the vector is compared against **every row of the embedding table**. Each
comparison gives a score, called a **logit**.

**Why the same table?** It's called **tied embeddings**. The table already describes what each token is, and
reusing it saves 508 MB of parameters.

**Where and when:** On the GPU. It reads the **entire 508 MB table for every generated token**: 34% of all bytes
read per token.

## Stage 5: Softmax and choosing

**What happens:** **Softmax** turns scores into probabilities that add up to 100%. It exaggerates gaps: a 10-point
lead becomes about 22,000× more likely. `Mah` got 44.6% and ` E` got 9.3%. Then the model picks: **greedy** takes
the top token, and **sampling** draws at random by probability, with **temperature** controlling how adventurous
the draw is.

**Why probabilities?** Training nudges the model to give higher probability to the word that really came next.

## Stage 6: Repeat, one token at a time

Append `Mah` and run again, which gives `omes`. This is **autoregressive** generation. There are two phases:

- **Prefill:** the whole prompt at once, in parallel. The GPU is busy computing, so it's fast.
- **Decode:** one token per step. Each step reads all ~1.5 GB of weights for a single token, so decode speed is
  limited by **memory bandwidth**: 829 GB/s ÷ 1.5 GB gives the 537 tokens/second ceiling.

## Where and when, per generated token

| Stage | Runs on | Reads per token | Share |
|---|---|---|---|
| Tokenizer | CPU | (a lookup) | — |
| Embedding lookup | GPU | 2 KB | 0% |
| 24 layers | GPU | 996 MB | 66% |
| LM head | GPU | 508.6 MB | 34% |
| Softmax + choice | GPU | tiny | — |

## What the experiments showed

1. **With all 24 layers removed,** the model predicts its own input: `Patrick` at 100%. The embedding table only
   knows what each token *is*. Knowledge of what comes *next* exists only in the layers.
2. **The answer forms late.** `Mah` ranks about 100,000th through layer 14, climbs from layer 15 onward, and
   becomes the top guess only after layer 24.
3. **There's no hidden machinery.** A hand-built version of these six stages matched the real model bit for bit.
