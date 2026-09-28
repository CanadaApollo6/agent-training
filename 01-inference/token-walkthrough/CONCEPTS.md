# The token walkthrough, conceptually

The reference explanation, with no code. Part 1's measurements come from `skeleton.py` and `lens.py`, and Part 2's
from `anatomy.py` and `check.py`.

# Part 1: the whole model as a loop

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

# Part 2: inside a layer (RMSNorm and the MLP)

## The big picture

Part 1 treated each of the 24 layers as a black box that adds something to the residual stream. Every layer
follows the same four-step recipe:

1. **RMSNorm**: resize a copy of the stream to a standard size.
2. **Mixer**: let this token gather information from the other tokens. This is attention in 6 layers and DeltaNet
   in 18 (Parts 3 and 4). Its result is **added** to the stream.
3. **RMSNorm** again, on a fresh copy.
4. **MLP**: process this token on its own, using knowledge stored in the weights. Its result is **added** to the
   stream.

Part 2 opens the two pieces that are the same in every layer: RMSNorm and the MLP. In short, **the mixer gathers
and the MLP thinks.**

## RMSNorm: a volume knob

**Terminology.** **RMS** stands for root mean square: square every number, average the squares, and take the
square root. It measures how big the numbers typically are, ignoring their signs. **Normalizing** means dividing by
the RMS so that the typical size becomes 1.

**Worked example:** a vector of 2 numbers instead of 1,024.

| Step | Value |
|---|---|
| Input | [3, 4] |
| Square each | [9, 16] |
| Average | 12.5 |
| Square root → RMS | 3.536 |
| Divide by it | [0.849, 1.131] |
| RMS of the result | 1 |

The direction is kept (the second number is still 4/3 of the first), but the overall size is reset.

**The learned gain.** After resizing, each of the 1,024 positions (called **channels**) is multiplied by its own
learned amount. It's like a mixing desk with 1,024 sliders: some channels get boosted and some get turned down.
Qwen stores each slider as an offset from 1, so an offset of 0 means "leave it alone". Training starts from plain
normalization and learns adjustments from there.

**Why it's needed.** Every layer adds to the stream, so the stream keeps growing. For "Patrick", the typical number
is 0.01 in the embedding and about 0.24 by layer 24, roughly 20× larger. Without the norm, layer 20 would see
inputs 20× louder than layer 2, and every layer would have to cope with any volume. With the norm, every mixer and
every MLP receives an input of about the same size (RMS 1.1–1.7 after the gains). Think of the automatic gain on a
microphone: whether you whisper or shout, the recording comes in at a usable level.

The norm only resizes the copy each part reads, never the stream itself. The stream keeps its full, growing
history.

**The final norm is different.** Its gains are large (median 4.5), so it turns the stream's 0.23 into 4.33. Bigger
numbers going into the LM head mean bigger gaps between logits, and so more confident probabilities.

**Where and when:** Twice in every layer, once more at the end, plus a few small ones inside the mixers. It runs
on every token, on the GPU. It's cheap: 2 KB of weights per norm, about 0% of the bytes. It's computed at higher
precision (fp32) because squaring and averaging 1,024 numbers in bf16, which keeps only about 3 significant
digits, would lose accuracy.

## The MLP: 3,584 detectors

**Terminology.** MLP stands for multi-layer perceptron, an old name for the plainest kind of neural network
block. It's also called the **feed-forward** block. Each of its 3,584 internal units is called a **neuron**.

**What happens**, for one token:

1. **Widen.** The 1,024-number vector is compared with 3,584 learned patterns, twice. The **gate** comparison asks
   each neuron "does this vector match your pattern?" The **up** comparison asks "by how much, and in which
   direction?" Each comparison is a dot product, the same "how aligned are these?" operation the LM head uses.
2. **Switch.** Each gate score goes through **SiLU**, a soft on/off switch:

   | Gate score | −6 | −4 | −2 | −1 | 0 | 1 | 2 | 4 | 6 |
   |---|---|---|---|---|---|---|---|---|---|
   | After SiLU | −0.01 | −0.07 | −0.24 | −0.27 | 0 | 0.73 | 1.76 | 3.93 | 5.99 |

   A positive score passes through almost unchanged, and a negative score is squashed to nearly 0. The result
   is multiplied by that neuron's up value. So a neuron is **on**, with a strength and direction, or **off**.
3. **Narrow.** Each neuron owns a 1,024-number "note" (the **down** weights). The output is the sum of every
   neuron's note, scaled by how strongly it fired. That sum is added to the stream.

**Analogy:** a filing cabinet with 3,584 drawers. The gate and up weights are the drawer labels, and the token's
vector is the query. Drawers whose labels match open, and the down weights are what's inside each drawer. The
answer is a blend of the contents of the open drawers. Researchers call this the **key-value memory** view of the
MLP.

**Why it's built this way:**
- **Wide (3,584 > 1,024)** gives room for many separate detectors.
- **Two comparisons multiplied together** let one signal control another: the gate decides whether, and up
  decides how much. This is **SwiGLU** ("Swish-gated linear unit"; Swish is another name for SiLU). It beats a
  plain on/off switch at the same size, which is why almost every modern LLM uses it.
- **No token mixing:** the MLP sees only its own token's vector. Anything it needs from other words must already
  have been written into the stream by a mixer. That's why the two alternate.

**Where and when:** Once per layer, on every token, independently, on the GPU. Each MLP is three 1,024 × 3,584
matrices, 22.0 MB. All 24 total **528.5 MB, 35.1% of the bytes read per generated token**, slightly more than the
LM head. The MLPs are the single biggest chunk of the model.

## What the measurements showed

**1. The fact is looked up in layer 15's MLP.** Remove one MLP at a time and see what happens to P(`Mah`), which
is 44.6% in the full model:

| MLP removed | P(`Mah`) | `Mah`'s rank | New top guess |
|---|---|---|---|
| none | 44.6% | 1 | `Mah` |
| layer 15 | **0.4%** | 39 | `Johnson` |
| layer 20 | 1.9% | 5 | `E` |
| layer 24 | 2.5% | 4 | `,` |
| layer 7 | 4.7% | 2 | `J` |
| layer 5 | **72.0%** (higher) | 1 | `Mah` |

Removing layer 15's MLP almost erases the answer. It's exactly the layer where the Part 1 lens saw `Mah` begin to
climb. The later MLPs (20–24) matter too: they turn it into a confident prediction. Removing some parts
*raises* the answer (layer 5), because parts also push against each other. The final prediction is a negotiation,
not a sum of agreeing votes.

**2. It's spread across many neurons, not a "Mahomes neuron".** In layer 15, for "Patrick":
- Only 24% of the 3,584 gates are positive, and only 78 neurons fire at more than 10% of the strongest one.
- Yet 495 neurons are needed to account for half of the output.
- Reading the strongest neurons' notes as vocabulary gives nonsense (`(P`, `seudo`, `ify`, ...).

As with the lens, layer 15 writes in the model's internal working format. Later layers read that format and turn
it into `Mah`.

**3. The MLPs vote for `Mah`; the mixers hedge.** The final gap between `Mah` and the runner-up ` E` is 1.56
logits. It can be split exactly into what each part wrote into the stream:

| Written by | Toward `Mah` over ` E` |
|---|---|
| All 24 MLPs | **+4.11** |
| All 24 mixers | **−3.11** |
| The embedding | +0.58 |
| Largest single parts | layer 24 mixer −3.22, layer 24 MLP +2.64, layer 22 MLP −1.65, layer 15 MLP +1.36 |

Layer 15 writes only +1.36 directly, yet removing it drops `Mah` from 44.6% to 0.4%. Most of its effect is
**indirect**: later layers build on what it wrote.

**4. The mixers are essential too.** Removing layer 1's *mixer* drops `Mah` to 0.0% (rank 81,380) and the model
predicts `.`. Without early context-gathering, "Patrick" never learns the sentence is about the Chiefs, so there's
nothing for the MLPs to look up. That's Parts 3 and 4.

## Checking the hand-built versions

The hand-written RMSNorm and MLP were fed real vectors from the model, and their outputs were compared with the
model's own. There were three possible verdicts:

| Version | Largest difference | Verdict |
|---|---|---|
| RMSNorm | 0 | **exact**: identical to the last bit |
| MLP | 0.0039 (values up to about 0.52) | **rounding**: the same math, rounded in a different order |
| MLP with the gate multiply left out | 0.518 of 0.523 | **wrong** |
| RMSNorm with the "+1" on the gain left out | 11.9 of 23.8 | **wrong** |

In bf16, doing the same additions in a different order rounds slightly differently. A difference of a few
thousandths is two correct answers disagreeing in the last digit. A difference as big as the values themselves
means the math is wrong.
