# we are going to create MASKED attention head
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module): # as in bigram, our model is torch.nn.Module
    def __init__(self, n_embd, head_size, context_window, dropout):
        super().__init__()
        # 1. attributes
        self.head_size = head_size

        # 2. 3 linear matrix indep
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

        #d3. masked 
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

        # 4. dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2,-1) # how much position i query matches position j key??
        scores = scores / math.sqrt(self.head_size) 
        scores = scores.masked_fill(self.tril[:seq_length, :seq_length]==0, float('-inf'))
        #self.tril is the matrix (context_window, context_window)
        # [:seq_length, :seq_length] cause the actual input could be smaller tahn context_window
        # and fill the future positions with -inf --> i no match with j > i
        weights = F.softmax(scores, dim = -1) # for each row i, weigths on j sum 1
        #adding dropout
        weights = self.dropout(weights)
        out = weights @ v
        return out
