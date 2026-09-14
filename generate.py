import argparse
import os
import torch
from data.shakespeare_char.tokenizer import get_tokenizer
from settings import CONFIGS

dir_path = os.path.dirname(os.path.realpath(__file__))

# 0. pick which trained model to load and generate from, e.g.:
#    python generate.py --model bigram --prompt "ROMEO: "
#    python generate.py --model gpt --prompt "ROMEO: " --max_new_tokens 500
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=CONFIGS.keys(), default='bigram')
parser.add_argument('--prompt', default='ROMEO: ')
parser.add_argument('--max_new_tokens', type=int, default=300)
args = parser.parse_args()

# 1. load tokenizer
encode, decode, vocab_size = get_tokenizer()

# 2. load model checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_checkpoint_path = os.path.join(dir_path, 'checkpoints', args.model, 'best.pt')
best_checkpoint = torch.load(best_checkpoint_path, map_location=device)

model_class = CONFIGS[args.model]['model_class']
model = model_class(**best_checkpoint['model_args'])
model.load_state_dict(best_checkpoint['model_state_dict'])
model.to(device)

# 3. encode prompt
idx = torch.tensor([encode(args.prompt)], dtype=torch.long)
idx = idx.to(device)

# 4. generate
output = decode(model.generate(idx, args.max_new_tokens)[0].tolist())
print(f'Model: {args.model}')
print(f'Prompt: {args.prompt} \n')
print(f'output: "{output}"')
