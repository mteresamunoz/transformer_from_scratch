# Model

The neural network itself, built up in stages — starting from a model with no attention at all, then adding self-attention piece by piece until it becomes a small decoder-only GPT. Each stage has its own doc, since the internal mechanics genuinely differ from one to the next (unlike training, which stays the same across all of them — see [`TRAINING.md`](../TRAINING.md)).

| File | Doc | What it adds |
|---|---|---|
| `bigram.py` | [`BIGRAM.md`](BIGRAM.md) | The baseline: predicts the next token from only the current one, no context at all. |
| `attention_head.py`, `multi_head_attention.py` | [`ATTENTION.md`](ATTENTION.md) | Gives the model real memory: each position can attend to every earlier position (never the future — causal/masked). |
| `feedforward.py`, `block.py` | [`BLOCK.md`](BLOCK.md) | Packages attention + a per-token MLP + residual connections + layer norm into one repeatable unit. |
| `gpt.py` | [`GPT.md`](GPT.md) | Stacks `n_layer` blocks, adds positional embeddings, and becomes the full decoder-only GPT. |

Every stage — no matter how different internally — exposes the same interface: `forward(idx, targets=None) -> (logits, loss)` and `generate(idx, max_new_tokens) -> idx`. That's what lets `train.py` and `generate.py` stay identical regardless of which model is plugged in (see `TRAINING.md`'s phases walkthrough).
