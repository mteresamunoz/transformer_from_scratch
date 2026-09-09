import numpy as np
import pickle
import os

data_dir = os.path.join(os.path.dirname(__file__))

with open(os.path.join(data_dir, 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#same as viewing_data.py, now we will split the data into train and test sets 
#90% of the data will be used for training and 10% for testing, from index 0 to 90% of the data will be used for training and from index 90% to the end of the data will be used for testing
train_data = text[:int(len(text)*0.9)]
val_data = text[int(len(text)*0.9):]

encoded_train_data = encode(train_data)
encoded_val_data = encode(val_data)

#save the encoded data to binary files with numpy, we will use the .npy format which is a binary format for storing numpy arrays
np.save(os.path.join(data_dir, 'train.npy'), np.array(encoded_train_data, dtype=np.uint16)) #dtype is tokens IDs (not the same as weigths and activactions)
np.save(os.path.join(data_dir, 'val.npy'), np.array(encoded_val_data, dtype=np.uint16))

#np.save will save the numpy array to a binary file with the .npy extension, we will use the np.uint16 data type which is an unsigned 16-bit integer, this is because the maximum number of unique characters in the text is 65,536 which is less than 2^16, so we can use 16 bits to represent each character.
#we will be using np.load to load the data back into memory when we need it, this is because loading the data from a binary file is much faster than loading it from a text file.

#save it in meta.pkl file using pickle, we will use the pickle module to save the vocab size and the stoi and itos dictionaries to a binary file with the .pkl extension, this is because we want to be able to load the data back into memory when we need it, and we want to be able to use the data in other programming languages as well.
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f) 
