# A bigram is the simplest form of a language model that predicts the next word based just on the previous word. It uses a two-word sequence to estimate the probability of a word given its predecessor.

import torch

class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The model is a single (vocab_size, vocab_size) lookup table: given a token index,
        # it directly returns the logits for the next token (no separate embedding step here).
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is a tensor of shape (batch_size, seq_length) containing the indices of the input words.
        # We pass these indices through the embedding layer to get the logits for the next word predictions.
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # If targets are provided, we compute the cross-entropy loss between the predicted logits and the true targets.
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # This method generates new text given an initial context (idx).
        for _ in range(max_new_tokens):
            # Get the logits for the current context
            logits, _ = self.forward(idx)
            # Focus on the last time step's logits
            logits = logits[:, -1, :]
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from the distribution to get the next word index
            next_idx = torch.multinomial(probs, num_samples=1)
            # Append the new index to the context
            idx = torch.cat((idx, next_idx), dim=1)
        return idx