"""
Small helper to avoid repeating "load meta.pkl + rebuild encode/decode"
in every script that needs to tokenize/detokenize shakespeare_char text.

We don't pickle the encode/decode functions themselves (see README.md for why:
lambdas aren't reliably picklable, and shipping serialized code is bad practice).
Instead, meta.pkl stores only raw data (vocab_size, stoi, itos), and this module
rebuilds the same encode/decode one-liners from prepare_data.py, once, for reuse.
"""
import os
import pickle

_DATA_DIR = os.path.dirname(__file__)


def load_meta(data_dir=_DATA_DIR):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def get_tokenizer(data_dir=_DATA_DIR):
    """Returns (encode, decode, vocab_size) rebuilt from meta.pkl."""
    meta = load_meta(data_dir)
    stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)

    return encode, decode, vocab_size
