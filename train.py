import os
import torch
from data.shakespeare_char.tokenizer import get_tokenizer
from model.bigram import BigramLanguageModel
from dataloader.batch_loader import get_batch

dir_path = os.path.dirname(os.path.realpath(__file__)) # get the directory path of the current file

# 1. load data and tokenize
encode, decode, vocab_size = get_tokenizer()

# *we can move it to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. initialize model
model = BigramLanguageModel(vocab_size)
model.to(device) # move the model to the device (GPU or CPU)

# 3. initialize optimizer
# parameters() is a method inherited from torch.nn.Module that returns an iterator over the model's parameters (weights and biases). We can use this iterator to access the model's parameters and update them during training.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # initialize the optimizer, we will use Adam optimizer with learning rate 1e-3

max_iters = 100000 # we will train the model for MAX_ITERS iterations
eval_interval = 200 # we will evaluate the model every EVAL_INTERVAL iterations
eval_iters = 200 # we will evaluate the model on EVAL_ITERS batches of data
best_val_loss = float('inf') # we will keep track of the best validation loss so far, we will save the model checkpoint if the validation loss is the best so far
patience = 4 # we will stop training if the validation loss does not improve for PATIENCE evaluations in a row
patience_counter = 0 
@torch.no_grad() # we don't need to compute gradients during EVALUATION, so we can use this decorator to disable gradient computation
def estimate_loss(eval_iters):
    out = {}
    model.eval() # set the model to evaluation mode, this is necessary because some layers like dropout and batchnorm behave differently during training and evaluation
    for split in ['train', 'val']: # we will evaluate the model on both training and validation data
        losses = torch.zeros(eval_iters) # we will store the losses for each batch of data in this tensor
        for k in range(eval_iters): # we will evaluate the model on EVAL_ITERS batches of data
            xy, by = get_batch(split) # return a batch of data for training or validation, xy is the input and by is the target (batch_size, seq_length)
            xy, by = xy.to(device), by.to(device) # move also the data to the device (GPU or CPU)
            logits, loss = model.forward(xy, by) #return logits and loss, logits is the output of the model (batch_size, seq_length, vocab_size) and loss is the cross entropy loss between the logits and the target (batch_size, seq_length)
            losses[k] = loss.item() # store the loss for this batch of data in the losses tensor
        out[split] = losses.mean() # compute the mean loss for this split (train or val) and store it in the out dictionary
    model.train() # set the model back to training mode
    return out
    

# 3. training loop
for iter in range(max_iters): 
    # each iteration of the training loop will get a new batch of data, compute the loss, and update the model parameters based on the gradients computed from the loss.    
    xy, by = get_batch('train') # return a batch of data for training, xy is the input and by is the target (batch_size, seq_length)
    # move also the data to the device (GPU or CPU)
    xy, by = xy.to(device), by.to(device)

    logits, loss = model.forward(xy, by) #return logits and loss, logits is the output of the model (batch_size, seq_length, vocab_size) and loss is the cross entropy loss between the logits and the target (batch_size, seq_length)

    optimizer.zero_grad(set_to_none=True) # set the gradients of all model parameters to zero, this is necessary because by default, gradients are accumulated in PyTorch, so we need to clear them before computing the new gradients for the current batch

    loss.backward() # compute the gradients of the loss with respect to the model parameters

    optimizer.step() # update the model parameters based on the gradients computed in loss.backward()

    # print the loss every 100 iterations to monitor the training progress
    #if iter % 100 == 0:
    #    print(f"Iteration {iter}: loss = {loss.item()}") # loss.item() returns the scalar value of the loss tensor
    # but just doing this will print the just the training loss, we also want to print the validation loss, so we need estimate_loss()
    if iter % eval_interval == 0 or iter == max_iters - 1: # we will evaluate the model every EVAL_INTERVAL iterations and also at the last iteration
        losses = estimate_loss(eval_iters) # compute the mean loss for both training and validation data
        print(f"Iteration {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}") # print the mean loss for both training and validation data
    # save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_args': {'vocab_size': vocab_size},
            'iter': iter,
            'loss': losses['train'].item(),
        }
        # save the checkpoint to a file checkpoints/(bigram o gpt)/checkpoint_{iter}.pt, create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/checkpoints/bigram', exist_ok=True)
        torch.save(checkpoint, f'{dir_path}/checkpoints/bigram/checkpoint_{iter}.pt') # save the checkpoint to a file
        # save best checkpoint, if the validation loss is the best so far, we will save it as best.pt
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0 # reset the patience counter if the validation loss improves
            torch.save(checkpoint, f'{dir_path}/checkpoints/bigram/best.pt') # save the best checkpoint to a file
        else:
            patience_counter += 1 # increment the patience counter if the validation loss does not improve
            if patience_counter >= patience: # if the patience counter reaches the patience limit, we will stop training
                print(f"Early stopping at iteration {iter} due to no improvement in validation loss for {patience} evaluations.")
                break