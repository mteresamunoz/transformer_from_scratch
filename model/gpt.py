import torch
import torch.nn as nn
from model.block import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, num_heads, n_layer, context_window, dropout):
        super().__init__()

        # 1. embedding table initialize randomly
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. positional embedding with max context window
        self.position_embedding_table = nn.Embedding(context_window, n_embd)
        # 3. block
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads, context_window, dropout) for _ in range(n_layer)]
        )
        # 4. Final layer tonNormalize 
        self.ln_f = nn.LayerNorm(n_embd)
        # 5. Logits projection
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # *saving context_window to generate
        self.context_window = context_window
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        batch_size, seq_length = idx.shape
        # idx is a tensor of shape (batch_size, seq_length) with index of input words
        # builf token embed of idx (batch_size, seq_length, num_embd)
        token_emb = self.token_embedding_table(idx)
        # adding positional embedding, tensor of shape (seq_length, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(seq_length, device=idx.device))
        x = token_emb + pos_emb #input x (batch_size, seq_length, num_embd)
        x = self.dropout(x)
        # input x througth block , (batch_size, seq_length, num_embd)
        x = self.blocks(x)
        # final layer norm, (batch_size, seq_length, num_embd)
        x = self.ln_f(x)
        # logits projection, (batch_size, seq_length, vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else: 
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # This method generates new text given an initial context (idx).
        for _ in range(max_new_tokens):
            # during training, the model doesnt see sequencies larger than context_window
            # so we need to cut actual input to see the last context_window sequence of this input
            idx_cond = idx[:, -self.context_window:]
            # get logits for current context
            logits, _ = self.forward(idx_cond)
            # Focus on the last time step's logits
            logits = logits[:, -1, :]
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from the distribution to get the next word index
            next_idx = torch.multinomial(probs, num_samples=1)
            # Append the new index to the context
            idx = torch.cat((idx, next_idx), dim=1)
        return idx           