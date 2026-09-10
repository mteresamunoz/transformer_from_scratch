# Model

The neural network itself, built up in stages — starting from a model with no attention at all, then adding self-attention piece by piece until it becomes a small decoder-only GPT.

## `bigram.py` — the baseline

A bigram model predicts the next token by looking **only at the current token** — no context, no history, no attention. It's the classical NLP "bigram": a table of transition likelihoods between pairs of adjacent tokens (`P(next character | current character)`).

It doesn't produce good text, but it's a useful starting point because it gives us, in its simplest possible form, every piece the rest of the project will reuse: how a model is defined in PyTorch, how the loss is computed, how text is generated. Once self-attention is added later, only "how logits are computed" changes — the loss calculation and generation loop stay the same.

## `nn.Embedding` vs `nn.Linear` — they are not interchangeable

This is worth being explicit about, because it's an easy mix-up: both are PyTorch layers with a weight matrix, but they do fundamentally different operations.

- **`nn.Linear(in_features, out_features)`**: expects **continuous vectors** as input, of size `in_features`, and computes `input @ W + b` — a real matrix multiplication.
- **`nn.Embedding(num_embeddings, embedding_dim)`**: expects **integer indices** as input, and performs a **lookup** — "give me row number `i` of my table." No multiplication, no addition between rows — just selection.

Our input (`idx`) is a tensor of integer token IDs, not continuous feature vectors, so `nn.Embedding` is the correct layer here — `nn.Linear` would either error on the shapes or silently do something meaningless.

### A tiny worked example

Forget 65 characters for a moment — imagine a 4-character vocabulary: `stoi = {'a':0, 'b':1, 'c':2, 'd':3}`.

```python
table = nn.Embedding(4, 4)   # vocab_size=4, embedding_dim=4 (bigram special case: they match)
```

**What's inside `table` right after creating it?** Nothing about characters, losses, or probabilities — just **random floating-point numbers**, initialized by PyTorch. Made-up example values:

```text
table.weight =
          logit for 'a'  'b'    'c'    'd'
row 0 ('a'):   0.12      -0.87   0.33   -0.05
row 1 ('b'):  -0.44       0.21  -0.19    0.66
row 2 ('c'):   0.05       0.09  -0.31    0.14
row 3 ('d'):   0.77      -0.22   0.40   -0.61
```

Just a `(4, 4)` grid of learnable numbers. No text is stored anywhere — characters only exist as a concept in `stoi`/`itos`; the table only ever sees integers.

**Calling `table(idx)`** doesn't multiply anything — it just returns the rows asked for. If the input context were `"ac"`, then `idx = [0, 2]`, and:

```text
logits[0] = row 0 = [ 0.12, -0.87, 0.33, -0.05]   # because idx[0] == 0
logits[1] = row 2 = [ 0.05,  0.09, -0.31, 0.14]   # because idx[1] == 2
```

`logits.shape = (2, 4)`: 2 because we asked for 2 indices, 4 because each row holds 4 numbers (one score per possible next character).

**No "sliding" happens inside the table.** Each index in `idx` triggers one independent lookup — looking up index 0 has no relationship to looking up index 2. The idea of "moving through the data" (`i`, `i+1`, `i+2`, ...) belongs to the **data loader** (cutting consecutive chunks out of the text), not to the embedding table, which is a static structure: give it a number, it gives back a row.

### Scaled up to the project's real shapes

`idx` in practice is 2D: `(batch_size, seq_length)`, e.g. `(64, 256)` — all integers, no floats. `table(idx)` looks up one row per integer, so the output gains an extra dimension: `logits.shape = (batch_size, seq_length, vocab_size)`, e.g. `(64, 256, 65)`. Every position of every sequence gets back its own full row of `vocab_size` logits.

### Why `(vocab_size, vocab_size)` here, and not a "real" embedding

A typical model separates two steps: `nn.Embedding(vocab_size, n_embd)` (token → a vector of `n_embd` numbers, a meaningful learned representation) followed by `nn.Linear(n_embd, vocab_size)` (that vector → logits). The bigram model **collapses both into one table**, `(vocab_size, vocab_size)`: row `i` *is directly* the logits vector, with no representation step in between. It works only because the bigram has nothing else to compute — once real attention is added later, `n_embd` will be its own, separate hyperparameter (see `data/shakespeare_char/README.md` and the earlier discussion on `d_model`/`n_embd` vs `seq_length`).

## `forward()`: with `targets` vs without

```python
def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    ...
```

- **During training**: `targets` is always passed (it's `yb` from the data loader) — we *know* the real next character, since it's sitting right there in the text. That lets us compute a loss: how far off were the model's logits from the truth.
- **During generation**: `targets=None` — we're producing text that doesn't exist yet, so there is no "correct answer" to compare against. `forward()` still returns `logits`, just no `loss`.

`forward()` itself never updates any weights — it's a pure calculation: given `idx` (and optionally `targets`), produce `logits` (and optionally a loss number). Nothing about the model changes when you call it.

## Loss and cross-entropy

`logits` are raw, unnormalized scores. Running them through `softmax` turns them into a proper probability distribution (values between 0 and 1, summing to 1). **Cross-entropy loss** measures how far that distribution is from the truth:

```text
loss = -log(P(correct next character))
```

Only the probability assigned to the *actual* next character matters. Two intuitive cases:
- Probability near **1** for the correct character → `-log(1) = 0` → loss near zero (great prediction).
- Probability near **0** for the correct character → `-log(~0) → very large` → huge loss (terrible prediction).

**Sanity-check number worth remembering**: right after creating the model (random, untrained weights), it hasn't learned anything, so it should assign roughly *uniform* probability across all `vocab_size` characters. Expected loss in that case:

```text
-log(1 / vocab_size) = -log(1/65) ≈ 4.17
```

If the very first loss you print is close to that number, the model and loss are wired correctly — a good check before training even starts. (In practice, with truly random init, it'll be *close* but not exact — e.g. `4.66` instead of `4.17` is still a healthy sign, same order of magnitude, no shape bugs. It'll come down once training starts.)

## Where gradient descent fits in (spoiler: not here)

`forward()` only answers "how good/bad are the model's current predictions?" — it never touches the weights. **Minimizing that loss is the whole goal of training**, but the mechanism that actually does the minimizing — computing gradients (`loss.backward()`) and nudging every weight toward reducing the loss (`optimizer.step()`) — lives in `train.py`, not here. `train.py`'s job is to call `forward()` repeatedly and, after each call, use gradient descent to push the weights so the loss trends downward over many iterations.

## `generate()`, step by step

Once the model can produce `logits` for a context, generating text is a loop. Using the same tiny 4-character vocabulary, starting from context `"a"` → `idx = [[0]]` (shape `(1, 1)`):

**Iteration 1:**
1. `logits, _ = self.forward(idx)` → no `targets` passed, so `loss = None`; `logits.shape = (1, 1, 4)`.
2. `logits = logits[:, -1, :]` → keep only the last position → shape `(1, 4)`, e.g. `[0.12, -0.87, 0.33, -0.05]`.
3. `probs = softmax(logits)` → e.g. `[0.30, 0.11, 0.37, 0.22]` (sums to 1).
4. `torch.multinomial(probs, 1)` → sample according to those probabilities — say index `2` (`'c'`) comes out.
5. `idx = torch.cat((idx, [[2]]), dim=1)` → `idx` is now `[[0, 2]]` → text so far: `"ac"`.

**Iteration 2**: repeat with `idx=[[0, 2]]`. The bigram has no memory, so only the last character (`'c'`) actually drives the next prediction — but the loop always appends to the *whole* growing context, which is exactly the pattern reused later when the model does have real memory of everything within `seq_length`.

**Why sample instead of just taking the highest logit?** Always picking the single most likely next character (*greedy decoding*) would make the model deterministic and repetitive for a given context ("the the the the..."). Sampling with `torch.multinomial` still favors likely characters but keeps variety across generations.

### With `batch_size` added

Starting from `idx` of shape `(batch_size, 1)` instead of `(1, 1)` — e.g. `batch_size=3`:

```text
idx = [[0],   # sequence 0, starts at 'a'
       [1],   # sequence 1, starts at 'b'
       [3]]   # sequence 2, starts at 'd'
```

Every step above happens **identically, for all 3 rows at once**, as tensor operations:
- `logits.shape` goes from `(1, t, 4)` to `(3, t, 4)`.
- `softmax` normalizes each row's own 4 numbers independently — rows don't interact.
- `torch.multinomial` samples **one character per row**, independently — row 0 might get `'c'`, row 1 might get `'a'`, row 2 might get `'b'`; nothing is shared between them.
- `torch.cat` appends the new character to each row separately.

So `batch_size` in generation (same as in training) doesn't make sequences more accurate or related to each other — it just runs `batch_size` **fully independent** sequences through the same tensor operations in one shot, instead of looping over them one at a time in Python.

## `max_new_tokens`

A parameter of the `generate()` call, chosen by us — not learned, not part of the model's architecture:

- It's the number of *new* characters appended, not the total output length: starting from a 5-character context with `max_new_tokens=100` gives a `5 + 100 = 105`-character result.
- It has no effect on training or on the model's weights — the same trained model can be called with `max_new_tokens=10` once and `max_new_tokens=1000` another time.
- For the bigram model it can be any size, since there's no context limit — the model only ever looks at the last character anyway. Once real attention with a `seq_length` is added, `idx` will need to be **cropped** to the last `seq_length` tokens before each `forward()` call inside the loop, since the model won't be able to use more context than that. That crop isn't needed yet, but it's coming.
