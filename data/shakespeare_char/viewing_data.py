# %%
# read data 
with open('C:\\Users\\UJA\\dev\\nanoGPT\\data\\shakespeare_char\\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of text: {len(text)} characters")

# %%
#unique  characters of the text
chars = list(set(text))
print(f"Unique characters: {len(chars)}")
#sorted unique characters
chars = sorted(chars)
#vocab size defined as the number of unique characters sorted
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Unique characters: {''.join(chars)}")
# %%
# encoder is something that takes a string and turns it into a list of integers
# decoder is something that takes a list of integers and turns it into a string
stoi = { ch:i for i,ch in enumerate(chars) } #stoi (string to int) is a dictionary that maps each character to its index in the sorted list of unique characters
itos = { i:ch for i,ch in enumerate(chars) } #itos (int to string) is a dictionary that maps each index in the sorted list of unique characters to its character

print(f"stoi: {stoi}")
print(f"itos: {itos}")
# %%
encode = lambda s: [stoi[c] for c in s] #encode is a function that takes a string and returns a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decode is a function that takes a list of integers and returns a string

print(f"Encode for 'hello': {encode('hello')}")
print(f"Decode for [46, 43, 50, 50, 53]: {decode([46, 43, 50, 50, 53])}")
# %%
