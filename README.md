# Transformer from Scratch

A decoder-only transformer (GPT-style) built from the ground up in PyTorch, without relying on high-level libraries like `transformers` — every component (tokenization, attention, training loop) is implemented by hand for learning purposes.

This project follows the ideas from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) walkthrough, but is written independently, split into small, self-contained, heavily-commented modules instead of a single notebook — the goal is for each piece to be understandable on its own before it's wired into the next.

The model is trained on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) using a character-level vocabulary.

## Project structure

```text
nanoGPT/
├── data/
│   └── shakespeare_char/
│       ├── input.txt          # raw training text
│       ├── prepare_data.py    # reads the text, builds the char vocab, encodes it, saves it to disk
│       ├── inspect_data.py    # loads the generated files back to inspect/verify them
│       ├── tokenizer.py       # reusable get_tokenizer() -> (encode, decode, vocab_size)
│       └── README.md          # dataset stats, tokenization scheme, storage format
│
├── dataloader/
│   ├── batch_loader.py        # samples random (x, y) mini-batches from the encoded dataset
│   └── README.md              # explains batch_size, seq_length and the sampling strategy
│
├── model/
│   ├── bigram.py               # simplest possible baseline language model (no attention)
│   ├── attention_head.py       # a single self-attention head, isolated
│   ├── multi_head_attention.py # several attention heads in parallel, concatenated
│   ├── feedforward.py          # the per-token MLP inside each transformer block
│   ├── block.py                 # one transformer block: attention + feedforward + residuals + layer norm
│   ├── gpt.py                   # stacks N blocks into the full model
│   └── README.md                # architecture overview, tensor shapes at each stage
│
├── train.py                    # training loop: forward pass, loss, backward pass, optimizer step
├── generate.py                 # autoregressive text generation from a trained model
└── README.md                   # this file
```

## How the pieces fit together

1. **`data/`** turns raw text into integer token IDs and stores them on disk as compact `numpy` arrays (`.npy`), together with the vocabulary mapping needed to decode predictions back into text.
2. **`dataloader/`** reads those arrays and produces the mini-batches (`x`, `y`) that the model trains on — this is also where PyTorch tensors first appear, converting from disk-stored `numpy` data into tensors just before they're needed.
3. **`model/`** defines the neural network itself, built up in stages: starting from a minimal baseline with no attention, then adding self-attention piece by piece (single head → multi-head → full transformer block), until it matches a small decoder-only GPT.
4. **`train.py`** ties the data loader and the model together: it runs the training loop, evaluates the loss on train/validation splits, and saves checkpoints.
5. **`generate.py`** loads a trained model and samples new text from it, one token at a time, using the same tokenizer/vocab defined in `data/`.

## Why not use `transformers` / `tiktoken`

Libraries like Hugging Face's `transformers` already implement the transformer architecture and provide pretrained weights — great for using models in production, but they hide exactly the internals this project is meant to teach. Here, the architecture (attention, blocks, embeddings) and the training loop are written from scratch in plain PyTorch, and tokenization is a simple hand-written character-level mapping instead of a BPE tokenizer — no external tokenization or modeling libraries are used.
