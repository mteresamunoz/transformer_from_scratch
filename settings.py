"""
Per-model configuration: which model class to use, its architecture arguments
(model_args), and its own training hyperparameters (batch_size, seq_length, lr...).

`vocab_size` is deliberately NOT stored here: it's only known once the tokenizer
is loaded (data/shakespeare_char/tokenizer.py), so train.py/generate.py add it to
model_args themselves at runtime, right before constructing the model.

Why one plain dict per model instead of a shared config class: the bigram and the
GPT don't have the same hyperparameters at all (the bigram has no notion of
n_embd/num_heads/n_layer/context_window) — forcing them into one shared shape
would mean padding one model's config with fields it doesn't use. See TRAINING.md
and model/README.md for why train.py/generate.py can stay generic across models
despite this: they only ever read the handful of keys every config here provides.
"""
from model.bigram import BigramLanguageModel
from model.gpt import GPT

CONFIGS = {
    'bigram': {
        'model_class': BigramLanguageModel,
        'model_args': {},          # BigramLanguageModel(vocab_size) — vocab_size added at runtime
        'batch_size': 8,
        'seq_length': 32,          # bigram has no real context window, kept small just for fast batching
        'lr': 1e-3,
        'max_iters': 100_000,      # upper bound only — early stopping decides the real stopping point
        'eval_interval': 200,
        'eval_iters': 200,
        'patience': 4,
    },
    'gpt': {
        'model_class': GPT,
        'model_args': {
            'n_embd': 384,
            'num_heads': 6,
            'n_layer': 6,
            'context_window': 256,  # must match 'seq_length' below — see dataloader/README.md
            'dropout': 0.2
        },
        'batch_size': 64,
        'seq_length': 256,
        'lr': 1e-3,
        'max_iters': 100_000,      # upper bound only — early stopping decides the real stopping point
        'eval_interval': 200,
        'eval_iters': 200,
        'patience': 4,
    },
}
