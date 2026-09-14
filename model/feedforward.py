# adding no-lineal, indepent for each position, no mix info of positions/tokens

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential( # save attributes
        nn.Linear(n_embd, 4 * n_embd), # expand, 4x as in original paper
        nn.GELU(), # no-lineal
        nn.Linear(4 * n_embd, n_embd), # compress
        nn.Dropout(dropout) # add dropout
        )

    def forward(self, x):
        return self.net(x)