# Shakespeare (character-level)

Tiny Shakespeare dataset tokenized **at the character level** (not words or sub-words/BPE). Every unique character in the text is a token.

Source: [`karpathy/char-rnn`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Text stats

| | |
|---|---|
| Total length | 1,115,394 characters |
| Unique characters (vocab size) | 65 |

**Vocabulary:**
```
\n   !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

## Tokenization

Direct character ↔ integer mapping, no BPE or external tokenization libraries:

- **`stoi`** (string → int): `{'\n': 0, ' ': 1, '!': 2, ..., 'z': 64}`
- **`itos`** (int → string): the reverse mapping, `{0: '\n', 1: ' ', ..., 64: 'z'}`

```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

## Train / val split

Simple split by index (not random): the first 90% of the text for training, the last 10% for validation.

| Split | Shape | % of text |
|---|---|---|
| `train.npy` | `(1,003,854,)` | 90% |
| `val.npy` | `(111,540,)` | 10% |

## Storage format

Encoded tokens are saved as `numpy` arrays of type **`uint16`** (unsigned integer, 2 bytes):

- `uint16` represents the range `0` to `2^16 - 1` = **0 to 65,535**.
- Since tokens are indices (never negative), we don't need a sign bit → `uint16` instead of `int16` (which would waste half its range on negative numbers we never use).
- With `vocab_size = 65`, there's plenty of headroom, but `uint16` is used instead of `uint8` because the same pipeline is reused for larger vocabularies too (e.g. GPT-2's BPE tokenizer, `vocab_size = 50,257`, which no longer fits in `uint8` but does fit in `uint16`).

## Why numpy here, and PyTorch later

Data preparation and training are split into two independent stages, each using the tool that fits its job:

1. **`prepare_data.py` (this stage) → `numpy` only, no PyTorch.**
   Text is read, encoded to token IDs, and saved to disk as plain `uint16` arrays. This step runs **once**, offline, and has no dependency on PyTorch at all — it's just data wrangling.

2. **`train.py` (later) → PyTorch, loaded on demand.**
   During training, `train.bin`/`train.npy` is opened with `np.memmap`, which maps the file on disk without loading it fully into RAM. For every training step, a small random **mini-batch** of token IDs is sliced out of that memory-mapped array and only *that slice* is converted into a `torch.tensor` (moved to GPU if available) to feed the model.

Why not just save everything as a `torch.tensor` directly, like in the original *"Let's build GPT"* notebook? There, the whole dataset lives in one Jupyter session and is small enough to fit in memory, so encoding straight into a tensor is simplest. Here, the goal is a pipeline that also scales to huge datasets (e.g. OpenWebText, ~9B tokens / ~17GB) where loading everything as a tensor upfront isn't feasible. `np.memmap` + on-the-fly tensor creation means only the current mini-batch ever needs to exist as a PyTorch tensor in memory — the rest stays on disk until it's needed.

For tiny Shakespeare (~1MB) this distinction barely matters in practice, but keeping the same two-stage pattern (`numpy` prep → `torch` batching) makes the code work unchanged on much larger datasets later.

## Generated files

Running [`prepare_data.py`](prepare_data.py) on [`input.txt`](input.txt) produces:

| File | Contents | Tracked in git |
|---|---|---|
| `input.txt` | Raw, unprocessed text | ✅ |
| `train.npy` | Training tokens (`uint16`) | ❌ (regenerable) |
| `val.npy` | Validation tokens (`uint16`) | ❌ (regenerable) |
| `meta.pkl` | `{'vocab_size', 'stoi', 'itos'}` — needed to decode the model's output | ❌ (regenerable) |

Generated files are not pushed to the repo (see `.gitignore`): they can be recreated in seconds by running `prepare_data.py`.
