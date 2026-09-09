import numpy as np
import pickle
import os

data_dir = os.path.join(os.path.dirname(__file__))

train_data = np.load(os.path.join(data_dir, 'train.npy'))
val_data = np.load(os.path.join(data_dir, 'val.npy'))

print(f"Train data shape: {train_data.shape}")
print(f"Val data shape: {val_data.shape}")
print(f"Train data type: {train_data.dtype}")
print(f"Val data type: {val_data.dtype}")
print(f"Train data first 10 elements: {train_data[:10]}")
print(f"Val data first 10 elements: {val_data[:10]}")

#Train data shape: (1003854,)
#Val data shape: (111540,)
#Train data type: uint16
#Val data type: uint16
#Train data first 10 elements: [18 47 56 57 58  1 15 47 58 47]
#Val data first 10 elements: [12  0  0 19 30 17 25 21 27 10]

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

decode = lambda l: ''.join([itos[i] for i in l])

#to see if it matches the original text, we will decode the first elements of train_data

decoded_train_data = decode(train_data[:100])
print(f"Decoded train data first 100 elements: {decoded_train_data}")

#First Citizen:
#Before we proceed any further, hear me speak.

#All:
#Speak, speak.

#First Citizen:
#You