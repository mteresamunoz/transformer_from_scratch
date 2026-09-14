# num_heads independent AttentionHead's (each with its own Q/K/V)
# head_size =n_embd / num_heads

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention_head import AttentionHead

class MultiAttentionHead(nn.Module):
    def __init__(self, n_embd, num_heads, context_window, dropout):
        super().__init__()
        # num_heads instances of AttentionHead --> each one with head_size
        head_size = n_embd // num_heads # // for int (2, no 2.0)

        # run num_heads in parallel of the same input x
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size, context_window, dropout) for _ in range(num_heads)])
        # outputs are concatenated side by side, so nn.Linear mixes them together
        # learns to combine info of different heads 
        self.proj = nn.Linear(n_embd, n_embd)

        # adding dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate outputs of the different heads, next to each other
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        #adding dropout
        out = self.dropout(out)
        return out