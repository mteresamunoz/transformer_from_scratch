# adding no-lineal, indepent for each position, no mix info of positions/tokens

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        nn.Linear(n_embd, 4 * n_embd) # expand, 4x as in original paper
        nn.GELU() # no-lineal
        nn.Linear(4 * n_embd, n_embd) # compress