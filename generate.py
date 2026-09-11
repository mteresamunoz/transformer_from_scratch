# we are going to use the model to generate text
import os
import torch
from model.bigram import BigramLanguageModel
from data.shakespeare_char.tokenizer import get_tokenizer

dir_path = os.path.dirname(os.path.realpath(__file__)) # get the directory path of the current file

# 1. load tokenizer
encode, decode, vocab_size = get_tokenizer()

# 2. load model checkpoint
# 2.1 devide cuda?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 2.2 load best checkpoint
best_checkpoint_path = os.path.join(dir_path, 'checkpoints/bigram/best.pt')
best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
# 2.3 load state dict
model = BigramLanguageModel(**best_checkpoint['model_args'])
model.load_state_dict(best_checkpoint['model_state_dict'])
model.to(device)

# 3. encode prompt
prompt = "ROMEO: "
max_new_tokens = 300
# remember: idx (batch_size, seq_length) --> bigram (1, context_window)
idx = torch.tensor([encode(prompt)], dtype=torch.long)
idx = idx.to(device)

# 3. generate
output = decode(model.generate(idx, max_new_tokens)[0].tolist())
print(f'Prompt: {prompt} \n')
print(f'output: ¨{output}')
