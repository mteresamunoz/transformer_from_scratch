import os
import torch 
import numpy as np

# 1. Load the data from the .npy files
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_char')

train_data = np.load(os.path.join(data_dir, 'train.npy'))
val_data = np.load(os.path.join(data_dir, 'val.npy'))

# 2. choosing batch size and block size
batch_size = 8 # how many independent sequences will we process in parallel 
seq_length = 32 # length of each sequence (a.k.a. context window, for now — see dataloader/README.md)

# 3. We need to cut the data into batches of sequences of length seq_length
#x = train_data[:seq_length] # x is input FOR TRAINING, which is the first seq_length characters of the training data
#y = train_data[1:seq_length+1] # y is the target, which is the next character in the sequence, so we shift the input by one character
# BUT we have just one chunk of data, but we need to create a batch of data, so we will repeat this process batch_size times
# a function that will generate a batch of data for training, and return x and y as torch tensors 2D (batch_size, seq_length)

def get_batch(split):
    # 1. we will choose the data based on the split (train or val)
    data = train_data if split == 'train' else val_data

    # 2. we will randomly choose batch_size starting indices for the sequences (o sea cogemos batch_size (en nuestro caso 8) indices aleatorios de la data)
    ix = np.random.randint(0, len(data) - seq_length, size=batch_size)
    # max range is len(data) - seq_length because we need to make sure that we have enough characters to create a sequence of length seq_length
    # we are saying "give me batch_size (8) random int between 0 and len(data) - seq_length, and store them in ix"

    # 3. we will create the input and target sequences based on the starting indices
    x = np.stack([data[i:i+seq_length] for i in ix]) 
    y = np.stack([data[i+1:i+seq_length+1] for i in ix]) # y is the target, which is the next character in the sequence, so we shift the input by one character

    # 4. we will convert x and y to torch tensors and return them
    x = torch.tensor(x, dtype=torch.long) # torch.long is the data type for integer tensors, which is what we need for our model
    y = torch.tensor(y, dtype=torch.long)

    return x, y # we return x and y as torch tensors 2D (batch_size, seq_length)

if __name__ == '__main__':
    # this block only runs when you execute this file directly (python batch_loader.py),
    # not when another file does `from dataloader.batch_loader import get_batch`
    xb, yb = get_batch('train') # we will test the function by getting a batch of data for training
    print(f"xb.shape: {xb.shape}, yb.shape: {yb.shape}")
    print(xb[0, :10])
    print(yb[0, :10]) # we will print the first 10 characters of the first sequence in the batch for both x and y to see if they are shifted by one character