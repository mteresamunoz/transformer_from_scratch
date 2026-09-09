# Data loader

Bridges the pre-processed dataset (`.npy` files in `data/shakespeare_char/`) and the model: samples random mini-batches of token IDs and converts them into PyTorch tensors ready to be fed into training.

This is also where `numpy` "hands off" to PyTorch: everything before this point (`prepare_data.py`) only used `numpy`; from here on, tensors take over.

## Two independent hyperparameters: `batch_size` vs `block_size`

These are easy to mix up, but they control two completely different axes of the data:

| | Controls | Example value |
|---|---|---|
| `batch_size` | How many independent sequences are processed **in parallel** in one training step | 64 |
| `block_size` | How many tokens are in **each** sequence — the context window | 256 |

Neither depends on the other, and `block_size` doesn't depend on the length of the raw text either — it's a design choice: how much context the model is allowed to look at when predicting the next character. The full dataset is one continuous stream of ~1,000,000 characters with no sentence boundaries, so `block_size` is just an arbitrary window length we cut out of that stream.

## `get_batch(split)`

The core function of this module. Given a split (`'train'` or `'val'`), it returns a pair of 2D tensors:

```python
x, y = get_batch('train')
x.shape  # (batch_size, block_size)
y.shape  # (batch_size, block_size)
```

- **`x`**: the input — `batch_size` chunks of `block_size` consecutive tokens, cut from random positions in the dataset.
- **`y`**: the target — the same chunks shifted one position to the right, i.e. `y[b, t] == x[b, t+1]`. At every position `t`, `y` tells the model "this is the character that actually comes next."

Both `x` and `y` are processed **at once**, in a single forward pass — the model sees the whole `(batch_size, block_size)` matrix, not one sequence at a time. This is what makes training on a GPU efficient: it's one batched matrix operation instead of `batch_size` sequential ones.

### Steps inside `get_batch`

1. Pick the right array (`train_data` or `val_data`) depending on `split`.
2. Draw `batch_size` random starting indices with `torch.randint`.
3. Slice out a chunk of length `block_size` starting at each index, for both `x` (`data[i : i+block_size]`) and `y` (`data[i+1 : i+block_size+1]`).
4. Stack all the chunks into single 2D tensors with `torch.stack`.

## What's actually random: rows, not characters

It's easy to picture the `(batch_size, block_size)` matrix as tokens scattered randomly all over the place, but that's not quite it — there are two different levels of randomness here:

- **Within one row (one sequence): not random at all.** Each row is a **contiguous** run of `block_size` characters, taken exactly as they appear in the original text — nothing is shuffled or reordered. If a chunk starts at position 5000 of the text, that row is literally `data[5000:5256]`, character by character, in its real order.
- **Across rows: yes, random.** What's random is only *where each row starts*. Row 0 might start at position 5000, row 1 at position 300,000, row 2 at position 812 — unrelated points scattered across the whole ~1,000,000-character text.

```
full text:        [ ...all of Shakespeare, ~1,000,000 characters in a row... ]
                          ↑ row 0 starts here (5000)   ↑ row 1 starts here (300000)   ↑ row 2 starts here (812)

row 0 (x[0]):      "ROMEO: But soft, what light th"   ← 256 CONSECUTIVE chars from position 5000
row 1 (x[1]):      "hall I compare thee to a summ"    ← 256 CONSECUTIVE chars from position 300000
row 2 (x[2]):      "e king is dead, long live the"    ← 256 CONSECUTIVE chars from position 812
...
```

Each row is a coherent, readable fragment of the real text (a genuine "window" into it), but different rows in the same batch are usually unrelated to each other — random fragments from completely different parts of the play.

Why sample random starting points instead of cutting the text into fixed, non-overlapping chunks (`0-256`, `256-512`, ...)? Because with `block_size=256` over ~1,000,000 characters there are almost 1,000,000 possible (overlapping) starting positions, so the model sees a different combination of context on every call to `get_batch`, instead of always the same few thousand fixed partitions.

## Why the random starting index is capped at `len(data) - block_size`

`batch_size` and this cap have nothing to do with each other — that's a common mix-up. `torch.randint(low, high, size)` has two *independent* arguments:

- `size` (here `(batch_size,)`) — **how many** random numbers to generate.
- `low`/`high` — the **range of values** each of those numbers can take.

The range has to be capped so that every chunk we cut actually has `block_size` elements — cutting one that runs past the end of the array would silently return a shorter chunk instead of raising an error, which then breaks `torch.stack` (it requires every tensor to have the same shape).

**Small example to see why:** say `len(data) = 20` and `block_size = 5`.

- If we allowed a starting index of `i = 18`, then `data[18:23]` doesn't error — Python slicing just returns whatever exists, so we'd get only `data[18]` and `data[19]`: **2 elements instead of 5**.
- The last *safe* starting index is `len(data) - block_size = 15`, because `data[15:20]` reaches exactly the last element (`data[19]`) without running out.

So the valid range for the starting index is `0` to `len(data) - block_size` — guaranteeing every chunk is exactly `block_size` long, regardless of `batch_size` (which only decides how many such indices we draw).

With the real numbers here (`len(data) ≈ 1,000,000`, `block_size = 256`), the valid range is roughly `0` to `999,744` — still enormous; `batch_size = 64` just means we draw 64 random values from within that range.

## A note on dtypes

The arrays loaded from disk (`train.npy`/`val.npy`) are `uint16` (see `data/shakespeare_char/README.md` for why). PyTorch doesn't support `uint16` in most operations, and embedding lookups specifically require `long` (`int64`) indices — so the data is cast to `torch.long` when converted to tensors here.
