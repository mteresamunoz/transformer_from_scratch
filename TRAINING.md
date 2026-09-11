# Training walkthrough

Three distinct phases that are easy to blur together. The confusing part is usually: *where does the "input" come from, and is there a target or not?* — the answer is different in each phase, so let's pin that down explicitly, using the tiny 4-character (`a,b,c,d`) table example from `model/README.md`.

## Phase 1 — model initialization: random, untrained

```python
model = BigramLanguageModel(vocab_size)
```

This allocates the `(vocab_size, vocab_size)` table and fills it with **random** numbers (see the worked example in `model/README.md`). No data has been touched yet, no training has happened. There isn't really an "input" concept yet at this point — we've just created empty, meaningless parameters. If you called `generate()` right now, you'd get garbage — which is exactly what we saw in `model/test_bigram.py`.

## Phase 2 — training: the input comes from the dataset, automatically

```python
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
loss.backward()
optimizer.step()
```

Here, the "input" (`xb`/`yb`) is **not** something you type or hand-pick — it's sampled automatically from `data/shakespeare_char/train.npy` by `get_batch()`. You, the human, never see or choose these examples one by one; the training loop pulls thousands of random batches on its own, over and over.

Crucially, in this phase **we always know the target** (`yb`) — it's simply "whatever character actually comes next in the real text". Knowing the target is exactly what makes a loss computable, and a loss is exactly what makes a gradient (and therefore learning) possible.

This phase's whole job: nudge the table's numbers, over thousands of iterations, from random garbage toward numbers that reflect the real statistics of the training text — e.g. if `"ab"` appears far more often than `"ac"` in Shakespeare, the entry for `a→b` gets pushed up much more consistently than `a→c` (see the worked gradient example further down).

## Phase 3 — inference / generation: the input comes from you, no target, no more learning

```python
context = encode("ROMEO:")            # or a plain seed token, e.g. torch.zeros((1,1))
generated = model.generate(context, max_new_tokens=200)
```

Now the "input" (`idx`/context) is provided by **you** — a prompt you encode yourself, or a minimal seed token to start from scratch. There is **no target**: we're producing text that doesn't exist yet, so there's nothing to compare it against. No loss, no `backward()`, no `optimizer.step()`. The weights are **frozen** at whatever values training left them in — generating text never changes the model.

## Side by side

| | Phase 1: init | Phase 2: training | Phase 3: inference / generation |
|---|---|---|---|
| Where does the input come from? | n/a — no data yet | `get_batch()`, sampled automatically from `train.npy` | you — a prompt you `encode()`, or a seed token |
| Do we know the target? | n/a | yes, always (the real next character) | no — that's exactly what we're trying to produce |
| Is a loss computed? | no | yes, every iteration | no |
| Do the weights change? | n/a (just created) | yes, every iteration (`optimizer.step()`) | no — frozen |
| Model call used | just `BigramLanguageModel(vocab_size)` | `model(xb, yb)` | `model.generate(idx, max_new_tokens)` |

## The gradient step, worked through numerically (Phase 2, zoomed in)

Same 4-character vocab as before (`stoi = {'a':0,'b':1,'c':2,'d':3}`), and say the sampled batch happened to include the pair `x='a', y='b'` (the text contained `"ab"` somewhere):

```text
table.weight (before this step) =
row 0 ('a'):   0.12   -0.87   0.33   -0.05
```

1. **Forward**: `logits = table(0) = [0.12, -0.87, 0.33, -0.05]` (only row 0 is touched, since `idx=0`). `softmax(logits) ≈ [0.34, 0.13, 0.42, 0.12]`.
2. **Loss**: the true next character is `'b'` (index 1), and the model gave it only `0.13` probability → `loss = -log(0.13) ≈ 2.04` (a bad prediction).
3. **Backward**: gradient for row 0 = `probs - one_hot(target)` = `[0.34, 0.13, 0.42, 0.12] - [0, 1, 0, 0]` = `[0.34, -0.87, 0.42, 0.12]`. Only **row 0** gets a gradient — rows 1, 2, 3 (`'b'`, `'c'`, `'d'`) weren't looked up in this step, so they're untouched (a "sparse" update, specific to `nn.Embedding`).
4. **Optimizer step**: with e.g. `lr=0.01`, plain SGD would do `row 0 -= 0.01 * [0.34, -0.87, 0.42, 0.12]` → the `'b'` column (negative gradient) goes **up**, the others go down slightly. `AdamW` does something similar in spirit but adapts the step size per weight using historical gradient statistics, rather than a flat `lr` for everything.

One step barely moves anything — the point is that this happens thousands of times, with a different random batch each time, and the *frequent* pairs (like `a→b` if it's common in the text) get reinforced far more consistently than the rare ones, so the table gradually converges toward the real transition statistics of the dataset.

## Device placement (CPU / GPU)

The model and the data both need to live on the same device, or PyTorch errors out ("tensors on different devices"). Two separate things need moving, at two different times:

1. **Detect the device once**, near the top of the script: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
2. **Move the model once**, right after creating it: `model = model.to(device)`. Since `nn.Module` already tracks every registered parameter (the same mechanism behind `model.parameters()`), `.to(device)` moves the whole embedding table in one call.
3. **Move every batch, every iteration**: `get_batch()` always returns plain CPU tensors — it stays deliberately unaware of `device`, so it remains simple and reusable regardless of hardware. It's the caller's job (`train.py`) to move each new batch right after fetching it: `xb, yb = xb.to(device), yb.to(device)`. This can't be done "once" like the model, since a new tensor is created on every call.

## `estimate_loss()`: a stable train vs val reading

A single training batch's loss (what we've printed so far) is noisy — see the earlier discussion on batch sampling variance. `estimate_loss()` fixes that by averaging the loss over many batches, for **both** splits, purely for measurement — never for training:

```python
@torch.no_grad()
def estimate_loss(eval_iters=200):
    out = {}
    model.eval()  # see note below
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # see note below
    return out
```

A few details worth calling out:

- **`get_batch('val')` reads from `val.npy`** — data the model never trains on. Crucially, this function never calls `.backward()` or `optimizer.step()` on either split: we're only *measuring*, not learning, even for the `'train'` split's numbers here (this is a *cleaner* train-loss reading than the one printed inside the training loop, precisely because it's averaged over `eval_iters` batches instead of just one).
- **`@torch.no_grad()`** tells PyTorch not to build the computation graph it would need for a `.backward()` call. Since nothing here ever calls `.backward()`, this saves memory and compute for no downside.
- **`model.eval()` / `model.train()`** toggle layer behaviors that differ between training and evaluation — most commonly `dropout` (turned off at eval time) and `batchnorm` (uses running statistics instead of the current batch's). The bigram model has neither, so right now this line changes nothing in practice — but it's the correct universal pattern, and it'll matter for free once dropout is added in `model/gpt.py`, without needing to remember to add it later.
- **Averaging over `eval_iters=200` batches** (same idea as the sampling-noise discussion earlier): individual batches can look "easy" or "hard" by chance; averaging many washes that out, giving a number that's actually comparable between one evaluation and the next.

Called periodically inside the training loop (not every iteration — that would be wasteful, since it runs 200 extra forward passes per split):

```python
if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
```

Watch both numbers move together: if `train` keeps dropping while `val` starts climbing, that's overfitting — the signal to stop training (or add regularization) discussed earlier.

## `eval_interval` vs `eval_iters` — two unrelated axes, easy to mix up by name

- **`eval_interval`**: the *outer* cadence — every how many **training loop iterations** we pause to evaluate at all. `eval_interval=100` over `max_iters=1000` means we stop and measure at `iter = 0, 100, 200, ..., 900` — 10 evaluations total; every other iteration just trains, nothing gets printed.
- **`eval_iters`**: the *inner* sample size — when an evaluation **does** happen, how many batches `estimate_loss()` averages over to produce one stable number (see the sampling-noise discussion above). This has nothing to do with how often we evaluate — it's about how trustworthy each individual evaluation's number is.

```text
iter=0   → evaluate! → inside estimate_loss(): average 200 fresh batches → print
iter=1..99 → just train, nothing printed
iter=100 → evaluate! → average 200 fresh batches → print
...
iter=900 → evaluate! → last one to land on a multiple of eval_interval
iter=901..999 → just train, nothing printed
```

**`eval_interval` does *not* need to evenly divide `max_iters`.** `iter % eval_interval == 0` simply fires on whichever multiples of `eval_interval` happen to fall inside `range(max_iters)`. If `max_iters=950` and `eval_interval=100`, evaluations land on `0, 100, ..., 900`, and the last 50 iterations (901-949) train without ever being measured again — not an error, just means the last printed number is a bit stale relative to the actual final model.

## Always evaluate on the very last iteration too

That staleness isn't just cosmetic — it can break the "save best checkpoint" logic. If the true lowest `val_loss` of the whole run happens to land on the very last iteration (past the last point that happened to align with `eval_interval`), and that iteration is never evaluated, it never gets compared against `best_val_loss` — so the actual best model of the run might never get saved. The fix: force an evaluation on the final iteration regardless of the modulo arithmetic:

```python
if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    ...
```

This guarantees the last snapshot you print (and check against `best_val_loss`) always reflects the model exactly as training left it, no matter how `max_iters` and `eval_interval` happen to relate to each other.

## Early stopping: don't guess `max_iters`, let the val loss tell you when to stop

Instead of trial-and-erroring `max_iters` to find where the loss bottoms out, track how many **consecutive evaluations** have failed to beat the best `val_loss` so far, and stop once that streak reaches a chosen "patience":

```python
patience = 4          # how many evaluations in a row without improvement we tolerate
patience_counter = 0

if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss(eval_iters)
    print(...)

    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        patience_counter = 0      # reset: we found a new best
        # ... save best.pt (unchanged from before)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at iter {iter}: no val loss improvement for {patience} evaluations.")
            break
```

**Compare against the all-time best, not against the previous reading.** The naive version of this idea checks "did `val_loss` go up compared to last time?" — but since each evaluation averages a fresh set of `eval_iters` random batches, the number can wobble slightly upward by pure sampling luck even while the model is still improving overall, triggering a false stop. Comparing against `best_val_loss` (and only resetting the counter on an actual new record) is far less sensitive to that noise, and is the standard implementation used everywhere (Keras' `EarlyStopping`, PyTorch Lightning's, etc. all work this way).

**`patience` counts evaluations, not training iterations.** With `patience=4` and `eval_interval=100`, that's tolerating up to `4 × 100 = 400` training iterations without a new best before giving up — worth thinking about `patience` at that real scale, not just as a small-looking number.

**The in-memory model at the moment of `break` is *not* the best one.** By the time the patience streak triggers, training has continued for `patience × eval_interval` iterations past the actual best point — so `model`'s current weights are already a bit past their peak. The genuinely-best weights were already written to `best.pt` the moment that record was set. Early stopping here isn't protecting the model (it's already safe on disk) — it's purely about not wasting compute continuing to train once it's clear nothing more will be gained. Always load `best.pt` for later use (e.g. in `generate.py`), never just trust the live `model` object after the loop ends.

## Checkpoints: self-describing, without a config system yet

Saving only `model.state_dict()` isn't enough — whoever loads the checkpoint later (`generate.py`) needs to know *which* model class to instantiate and with what constructor arguments before the weights can even be loaded into it. So each checkpoint carries its own recipe:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_args': {'vocab_size': vocab_size},   # whatever this model's __init__ needs
    'iter': iter,
}
os.makedirs('checkpoints', exist_ok=True)
torch.save(checkpoint, 'checkpoints/bigram.pt')
```

To load it later: reconstruct the model with the saved `model_args`, then load the weights into it:

```python
checkpoint = torch.load('checkpoints/bigram.pt')
model = BigramLanguageModel(**checkpoint['model_args'])
model.load_state_dict(checkpoint['model_state_dict'])
```

### What `model_state_dict` actually is

Not a bare tensor — it's a dictionary (`OrderedDict`) mapping `"layer name"` → `"that layer's tensor"`. For the bigram, since only one layer is registered (`self.token_embedding_table = nn.Embedding(...)`), it has exactly one entry:

```python
>>> model.state_dict()
OrderedDict([('token_embedding_table.weight', tensor of shape (vocab_size, vocab_size))])
```

The key name comes directly from the attribute name used in `__init__` (`token_embedding_table`) plus `.weight`, which is `nn.Embedding`'s internal name for its own weight tensor. Once the model has more layers (multi-head attention, feedforward, several stacked blocks), `state_dict()` grows to one entry per learnable tensor, each named after its place in the module tree (e.g. `blocks.0.attn.query.weight`).

**Could `vocab_size` be read back from this tensor's shape instead of storing it in `model_args`?** Technically yes, *only* for the bigram — its table happens to be square, `(vocab_size, vocab_size)`, so `state_dict['token_embedding_table.weight'].shape[0]` would work. But this is a coincidence of this specific architecture, not a general technique:

- Once the real GPT exists, its embedding table is `(vocab_size, n_embd)` — no longer square, and hyperparameters like `n_head`, `n_layer`, or `dropout` leave no shape footprint anywhere to reverse-engineer from at all.
- Even where shape-sniffing *would* work, it's fragile and implicit — it silently breaks if a layer gets renamed or restructured, and it forces whoever loads the checkpoint to understand the model's internals just to reconstruct it. Storing `model_args` explicitly is a stable, self-documenting contract that doesn't depend on any of that.

### How it's physically stored on disk

`torch.save(checkpoint, path)` serializes the whole nested structure (the `checkpoint` dict, containing the `model_state_dict` dict, containing tensors) using Python's `pickle` mechanism under the hood, with an efficient binary format specifically for the tensors themselves (not naive text or plain Python lists). `torch.load(path)` reconstructs the exact same nested structure back — same dict keys, same tensor shapes/dtypes/values.

**Why the filename is `bigram.pt`, not just `checkpoint.pt`**: naming it after the architecture avoids collisions once `model/gpt.py` exists and produces its own `checkpoints/gpt.pt` with its own `model_args` (`n_embd`, `n_head`, `n_layer`, `seq_length`, ...) — each checkpoint is self-contained and doesn't depend on any shared configuration.

**Why there's no `settings.py` / CLI args yet, on purpose**: bigram and GPT have almost entirely different hyperparameters, so building a shared config/CLI abstraction now, with only one real model in existence, would mean redesigning it as soon as the second model shows up — wasted work. That abstraction gets built once both models exist and it's clear what they actually need to share (this mirrors the earlier decision to keep hyperparameters as plain inline variables during this build-up phase, rather than centralizing them prematurely).
