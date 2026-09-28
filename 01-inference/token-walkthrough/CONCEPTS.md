# The token walkthrough, conceptually

The reference explanation, with no code. Part 1's measurements come from `skeleton.py` and `lens.py`, Part 2's
from `anatomy.py` and `check.py`, and Part 3's from `attention.py`.

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

# Part 3: full attention and the KV cache

## The big picture

The MLP only sees its own token. On its own, "Patrick" could be Patrick Stewart, Saint Patrick, or Patrick Star.
To predict `Mah`, the vector at "Patrick" needs information from "Chiefs" and "quarterback". The **mixer** moves
information between tokens, and **attention** is the classic mixer. Qwen3.5 uses it in 6 of its 24 layers (4, 8,
12, 16, 20, 24). The other 18 layers use DeltaNet (Part 4).

In one sentence: **each token asks a question, every earlier token offers an answer, and the token takes a blend
of the answers, weighted by how well each one matches the question.**

## Query, key, value

Each token makes three vectors from its (normalized) stream vector:

- **Query**: "what am I looking for?"
- **Key**: "what do I contain?", like a label on the outside of a folder.
- **Value**: "what I hand over if I'm picked", the contents of the folder.

For the token doing the looking:

1. **Score**: compare its query with every earlier token's key, using the same dot product ("how aligned?") as
   before. The scores are divided by 16 (the square root of the 256 numbers per head) so they don't get too big.
2. **Share out**: softmax turns the scores into shares that add up to 100%. These are the **attention weights**.
3. **Blend**: take each token's value, multiply it by its share, and add them up. The result is added to the
   stream (after the gate and an output projection, below).

**Worked example.** There are three earlier tokens, with scores 2, 0 and 1. Softmax: e² = 7.39, e⁰ = 1, e¹ = 2.72,
and the total is 11.11. The shares are 66.5%, 9.0% and 24.5%. The result is 66.5% of the first token's value, plus
9.0% of the second's, plus 24.5% of the third's.

**It's like a search engine,** where the query is what you typed, the keys are page titles, and the values are the
pages. The difference is that you get a blend of every page, weighted by relevance, instead of a list of links.

## The pieces around the core

- **Causal mask.** A token can only look at itself and earlier tokens. The measured weight on later tokens is
  exactly 0. When generating, the future doesn't exist yet, so training works the same way.
- **Heads.** Attention runs as 8 independent **heads** in parallel. Each head has its own query, so each can look
  for something different, like 8 people reading the same sentence for different things. Each head works in 256
  numbers.
- **Position (RoPE).** On its own, attention is order-blind: "dog bites man" and "man bites dog" would look the
  same. **Rotary position embedding** rotates the query and key by an angle that depends on each token's
  position, like clock hands. Their match score then depends on how far apart the two tokens are. Qwen rotates
  only 64 of each head's 256 numbers (25%). The other 192 match on content alone.
- **Query/key norms.** A small RMSNorm on each query and key keeps the scores from blowing up.
- **Output gate** (Qwen-specific). Each head's result passes through a learned gate, a per-channel valve from 0
  (closed) to 1 (open). The token being processed computes it, so a head can say "I found nothing useful, let
  nothing through." It is mostly closed: on average only 3–25% gets through.
- **Shared keys and values (GQA,** grouped-query attention**).** There are 8 query heads but only 2 sets of keys
  and values. Each group of 4 heads shares one set. This makes the KV cache 4× smaller (below).

**Where and when:** 6 layers, every token, on the GPU. The attention weights are 14.7 MB per layer. With its MLP, a
full-attention layer is 36.7 MB. In prefill, all 7 prompt tokens compute their scores at once (a 7 × 7 grid,
masked to a triangle). In decode, only the new token's row is computed.

## The KV cache

**The observation.** A token only looks backward, so its key and value never change once they are computed. The
key for "Chiefs" is the same at step 8 as at step 500.

**What happens.** Store every token's keys and values: that store is the **KV cache**. Each decode step computes
the query, key and value for the one new token only, adds its key and value to the cache, and scores its query
against everything in the cache.

**Why:** without the cache, step 500 recomputes all 500 tokens from scratch. Measured on this model:

| Generate | Cache on | Cache off | Same text? |
|---|---|---|---|
| 64 tokens | 69.0 tok/s | 54.2 tok/s | yes |
| 512 tokens | 76.7 tok/s | 32.8 tok/s | yes |

The gap grows with length: the cache keeps each step's cost roughly flat, while recomputing gets slower every
step. (Both are slow here because this runs the plain, unoptimized reference code. The same model runs at
hundreds of tok/s in vLLM.) The "cache off" run also turns off DeltaNet's memory, which is Part 4's version of the
same idea.

**The cost: memory that grows with every token.**

| Context length | KV cache |
|---|---|
| 1 token | 12,288 bytes (2 for K and V × 2 KV heads × 256 × 2 bytes × 6 layers) |
| 8,192 tokens | 0.10 GB |
| 131,072 tokens | 1.61 GB |
| 262,144 tokens (the maximum) | **3.22 GB, more than the 1.5 GB of weights** |

- Each decode step reads the whole cache, as well as the weights. At 122,000 tokens of context, reading the cache
  costs as much as reading all the weights, so long-context decode gets slower.
- Without GQA, the cache would be 4× bigger (49 KB per token).
- If all 24 layers used full attention, it would be another 4× (again 49 KB per token, 12.9 GB at maximum
  context).
- Every user has their own cache. When serving many users on one GPU, the cache, not the weights, is what limits
  how many fit. That's why serving engines like vLLM manage it carefully (PagedAttention), and why Qwen made 18
  of 24 layers DeltaNet, whose memory doesn't grow (Part 4).

## What the measurements showed

**1. Layer 4: "Patrick" reads the job and the team.** Averaged over heads: `quarterback` 31%, itself 27%, `Chiefs`
21%. Per head, four heads (1, 2, 3, 6) put 40–49% on `quarterback` and 21–35% on `Chiefs`. Head 7 looks only at
itself (90%). `Kansas` and `City` never get more than 8% in any attention layer. The most likely reason is that
earlier layers have already folded "Kansas City" into the vector at "Chiefs", so reading "Chiefs" is enough.

**2. Layer 16: the attention sink.** `The` gets 51% on average, and heads 0, 4, 5 and 7 put 70–91% on it. `The`
carries nothing useful. But shares must add up to 100%, so a head with nothing to look up has to put its
attention *somewhere*. The first token is visible to every token, so models learn to use it as a parking spot.
This is called an **attention sink**, and it shows up in almost every LLM. Serving tricks for very long text
(StreamingLLM) always keep the first few tokens in the cache for this reason: remove the parking spot and the
model breaks. Qwen's output gate was partly designed to make sinks unnecessary (the head can close its gate
instead), but they still appear here.

**3. Averages hide specialists.** Layer 16 looks idle on average, yet removing it hurts more than removing any other
attention layer:

| Attention layer removed | P(`Mah`) |
|---|---|
| none | 45.3% |
| 4 | 31.8% |
| 8 | 42.7% |
| 12 | 45.8% |
| **16** | **8.2%** (tied with ` E`) |
| 20 | 39.3% |
| 24 | **64.1%** (higher) |

In layer 16, head 1 reads `Chiefs` (39%) and `quarterback` (30%), while its neighbours park on `The`. A few heads
do the real work.

**4. The last attention layer is a brake.** In layer 24, heads 2 and 6 look almost entirely at "Patrick" itself (95%
and 99%) and have the most open gates (0.23–0.24). Removing layer 24's attention *raises* `Mah` to 64%. This
matches Part 2, where layer 24's mixer wrote −3.22 against `Mah`. The final attention layer tempers the model's
confidence.

(These numbers come from the "eager" attention code, which writes out the attention weights. It rounds slightly
differently from the fast path, so the baseline is 45.3% here versus 44.6% in Parts 1–2.)

**Unexplained so far:** in layer 16, `The`'s value vector is about 3× larger than the others' (2.02 vs about
0.7). So the sink isn't simply "look at an empty token". Something else, perhaps the gate, must be cancelling what
it contributes.
