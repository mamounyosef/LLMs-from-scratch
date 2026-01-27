import warnings
warnings.filterwarnings('ignore')

import torch
from gpt import GPTLanguageModel, decode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel()
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval().to(device)

start_text = "Cooper"
generated_text = model.generate_streaming(start_text=start_text, max_new_tokens=3000, print_every=1)
open('generated_sample.txt', 'w', encoding='utf-8').write(generated_text)