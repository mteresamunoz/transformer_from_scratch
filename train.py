import argparse
import os
import torch
from data.shakespeare_char.tokenizer import get_tokenizer
from dataloader.batch_loader import get_batch
from settings import CONFIGS

dir_path = os.path.dirname(os.path.realpath(__file__))

# 0. pick which model to train from the command line, e.g.:
#    python train.py --model bigram
#    python train.py --model gpt
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=CONFIGS.keys(), default='bigram')
args = parser.parse_args()
config = CONFIGS[args.model]

# 1. load data and tokenize
encode, decode, vocab_size = get_tokenizer()

# *we can move it to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. initialize model, with this model's own architecture args + the vocab_size we just learned
model_args = {**config['model_args'], 'vocab_size': vocab_size}
model = config['model_class'](**model_args)
model.to(device)  # move the model to the device (GPU or CPU)

# 3. initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

# this model's own training hyperparameters, read from settings.py instead of hardcoded
batch_size = config['batch_size']
seq_length = config['seq_length']
max_iters = config['max_iters']
eval_interval = config['eval_interval']
eval_iters = config['eval_iters']
patience = config['patience']

best_val_loss = float('inf')  # we will keep track of the best validation loss so far
patience_counter = 0


@torch.no_grad()  # we don't need gradients during evaluation
def estimate_loss(eval_iters):
    out = {}
    model.eval()  # eval mode: matters once dropout/batchnorm exist, no-op for now
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xy, by = get_batch(split, batch_size=batch_size, seq_length=seq_length)
            xy, by = xy.to(device), by.to(device)
            logits, loss = model.forward(xy, by)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 4. training loop
for iter in range(max_iters):
    xy, by = get_batch('train', batch_size=batch_size, seq_length=seq_length)
    xy, by = xy.to(device), by.to(device)

    logits, loss = model.forward(xy, by)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(eval_iters)
        print(f"[{args.model}] Iteration {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_args': model_args,
            'iter': iter,
            'loss': losses['train'].item(),
        }
        ckpt_dir = os.path.join(dir_path, 'checkpoints', args.model)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(ckpt_dir, f'checkpoint_{iter}.pt'))

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(checkpoint, os.path.join(ckpt_dir, 'best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iter} due to no improvement in validation loss for {patience} evaluations.")
                break
