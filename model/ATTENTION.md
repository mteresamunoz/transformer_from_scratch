# `attention_head.py` + `multi_head_attention.py` — giving the model memory

A single self-attention head lets each position look at every earlier position (never the future — causal), instead of only the current one like the bigram did. Three vectors per token make this possible:

- **Query (Q)**: "what am I looking for?" — this position's question.
- **Key (K)**: "what do I offer?" — how each position advertises itself to others.
- **Value (V)**: "what do I actually communicate?" — the information passed along if attended to.

## Analogy: a library / search engine

Every book (token) has a **label on its spine** (`Key`) — how it advertises itself, what it can be found by — and **actual content inside** (`Value`) — what you get if you pick it. You show up with a **search query** (`Query`) — "I'm looking for something about X". You compare your `Query` against every book's `Key` (that's the `Q @ Kᵀ` dot product), and the books whose label matches best get more weight — so when you compile an answer, you take **more content (`Value`)** from the relevant books and almost none from the irrelevant ones.

Translated to tokens: each position acts as a "searcher" (its `Query`) over every earlier position (which offers its `Key` and `Value`), and builds its output as a weighted mix of the `Value`s of whichever positions are most relevant to its `Query`.

**The part that's easy to miss: Q, K, V all come from the *same* vector, through *different* paths.** They aren't three properties a token already had — all three are computed the same way, from the same input embedding `x`, just passed through three separate, independently-learned `nn.Linear` layers:

```python
Q = self.query(x)   # x → one linear transform → Q
K = self.key(x)     # the same x → a DIFFERENT linear transform → K
V = self.value(x)   # the same x → yet another DIFFERENT linear transform → V
```

Before training, these three projections are random noise — meaningless, like the bigram's table before training. Gradient descent is what shapes them, over many iterations, into projections that actually help minimize the loss (predict the next character well). Nobody hand-codes "this head looks for vowels" — if that kind of pattern emerges, it's discovered by training, the exact same way the bigram's table "discovered" real character-pair frequencies without anyone telling it what they were.

## Worked example: tracing shapes through one head

Forget the bigram's `(vocab_size, vocab_size)` table — here we track a `(batch, seq_length, n_embd)` tensor as it moves through the head. Small concrete numbers: sequence `"cab"` (`seq_length=3`), `n_embd=4`, a single head so `head_size=4` too (batch=1, to keep it simple).

| Step | Operation | Shape after | What it means |
|---|---|---|---|
| 0 | `idx` (token ids for `"cab"`) | `(1, 3)` | just integers, `[2, 0, 1]` |
| 1 | token embedding lookup | `(1, 3, 4)` | each of the 3 positions is now a 4-number vector |
| 2 | `Q = query(x)` — `nn.Linear(4, 4)` | `(1, 3, 4)` | one query vector per position |
| 3 | `K = key(x)` — `nn.Linear(4, 4)` | `(1, 3, 4)` | one key vector per position |
| 4 | `V = value(x)` — `nn.Linear(4, 4)` | `(1, 3, 4)` | one value vector per position |
| 5 | `scores = Q @ K.transpose(-2, -1)` | `(1, 3, 3)` | **the key shape change**: `(batch, seq, head_size) @ (batch, head_size, seq) → (batch, seq, seq)` |
| 6 | `scores = scores / sqrt(head_size)` | `(1, 3, 3)` | scaling, shape unchanged |
| 7 | mask future positions with `-inf` | `(1, 3, 3)` | still the same shape, just some entries overwritten |
| 8 | `weights = softmax(scores, dim=-1)` | `(1, 3, 3)` | each row now sums to 1 |
| 9 | `out = weights @ V` | `(1, 3, 4)` | **back to the original shape**: `(batch, seq, seq) @ (batch, seq, head_size) → (batch, seq, head_size)` |

The output of one attention head has the **same shape as its input** (`(batch, seq_length, head_size)`, and `head_size == n_embd` for a single head) — that's what lets it slot into the rest of the model (residual connections, stacking blocks) without anything downstream needing to know attention happened at all.

### Step 5 and 9 are the two shapes worth burning into memory

- **Step 5** (`Q @ K^T`) produces a **`(seq_length, seq_length)`** matrix *per sequence in the batch* — this is the "who attends to whom" affinity table. Entry `[i, j]` = how much position `i`'s query matches position `j`'s key.
- **Step 9** (`weights @ V`) **collapses that square matrix back down** to `(seq_length, head_size)` — for each position `i`, it's a weighted sum over all `V[j]`, using row `i` of the attention weights.

### The causal mask (step 7), concretely

For `seq_length=3`, the mask (before applying `-inf`) is a lower-triangular matrix — `True` where attending is allowed:

```text
       j=0    j=1    j=2
i=0:  True   False  False   ← position 0 only sees itself
i=1:  True   True   False   ← position 1 sees itself and position 0
i=2:  True   True   True    ← position 2 sees everything (0, 1, 2)
```

`torch.tril(torch.ones(seq_length, seq_length))` builds exactly this shape. Wherever it's `False`, `scores` gets overwritten with `-inf` *before* the softmax — after softmax, `-inf` becomes `0` probability, so no information ever flows backward from the future.

## `multi_head_attention.py`: several of these in parallel

A multi-head attention layer is literally `num_heads` independent `AttentionHead`s (each with its own Q/K/V projections, smaller `head_size = n_embd // num_heads`), run on the same input, then concatenated back together:

| Step | Shape |
|---|---|
| input `x` | `(batch, seq_length, n_embd)` |
| each of `num_heads` heads outputs | `(batch, seq_length, head_size)` where `head_size = n_embd // num_heads` |
| concatenate all heads along the last dim | `(batch, seq_length, n_embd)` — back to the original size |
| final `nn.Linear(n_embd, n_embd)` projection | `(batch, seq_length, n_embd)` — unchanged shape, mixes information across heads |

**What actually gets concatenated is outputs, not weights.** Each head has its own, fully independent weights (its own `query`/`key`/`value` layers) — those are never shared or combined between heads; each learns its own during training. What `torch.cat` combines is the **numbers each head produces** after running its own forward pass with its own current weights — nothing about any head's weights changes or merges here, this is a pure computation, same as everything inside a single head's `forward()`.

**Why the final projection is needed, not just the concatenation.** After `torch.cat`, the result is heads sitting *side by side*: the first `head_size` columns came from head 1, the next `head_size` from head 2, and so on — nothing has mixed between them yet. The final `nn.Linear(n_embd, n_embd)` is a learned layer whose job is exactly that mixing: it can learn to combine, say, "some of what head 1 found (maybe subject-verb relationships) with some of what head 2 found (maybe punctuation matching)" into one blended representation. Without it, each head's discovery would stay permanently isolated in its own slice of the output, with no way for the rest of the model to combine signals from different heads together.

Why split into multiple smaller heads instead of one big one? Each head can specialize in a different kind of relationship (e.g. one might learn "attend to the previous word", another "attend to matching punctuation") — splitting `n_embd` into `num_heads` independent subspaces gives the model room to learn several such patterns in parallel, at no extra parameter cost compared to one head of the full size (this is why `n_embd` has to be divisible by `num_heads`, as covered in `data/shakespeare_char/README.md`).
