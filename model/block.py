# a block is n layer with sub-alyers Multihead and FFN

import torch.nn as nn
from model.multi_head_attention import MultiAttentionHead
from model.feedforward import FeedForward

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, context_window):
        super().__init__()

        # pre-norm (before each sub-layer)
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiAttentionHead(n_embd, num_heads, context_window)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        # residual stream --> adding input x to output of sub-layer
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x