# %%
# Quick smoke test for BigramLanguageModel: no training yet, just checking
# that shapes/loss/generation all work as expected on an untrained model.
import os
import sys
import math
import pickle
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloader.batch_loader import get_batch
from model.bigram import BigramLanguageModel

# %%
# load vocab_size and itos (needed to decode generated ids back to text)
meta_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_char', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
itos = meta['itos']

model = BigramLanguageModel(vocab_size)

# %%
# sanity check #1: loss on a real batch should be close to -log(1/vocab_size),
# since the model is untrained (weights are random -> ~uniform predictions)
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
print("loss:", loss.item())
print("expected (untrained):", math.log(vocab_size))

# %%
# sanity check #2: generate from an untrained model — this WILL be garbage,
# the point is only to check that generate() runs and produces valid characters
context = torch.zeros((1, 1), dtype=torch.long)  # start from token 0
generated = model.generate(context, max_new_tokens=200)
print(''.join(itos[i] for i in generated[0].tolist()))
