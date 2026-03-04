import torch
import tiktoken
from gpt2 import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = "gpt2/checkpoints/best_model.pt"
print(f"Loading model from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
model = GPT().to(device)

state_dict = checkpoint['model_state_dict']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

enc = tiktoken.get_encoding('gpt2')

prompt = "When do you think we will get AGI? I think it will be around"
max_new_tokens = 100
temperature = 0.3
top_k = 10

print(f"Prompt: {prompt}")

tokens = enc.encode(prompt)
generated_tokens = list(tokens)
tokens_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)

print("Generating: ", end="", flush=True)
with torch.no_grad():
    for i in range(max_new_tokens):
        logits, _ = model(tokens_tensor)
        logits = logits[0, -1, :]
        logits = logits / temperature

        penalty = 1.2  # >1 discourages repetition, try 1.2–1.5
        for token_id in set(generated_tokens[-50:]):  # look at last 50 tokens
            logits[token_id] /= penalty

        # Forbid specific tokens
        # banned_tokens = [198, 628]  # \n and \n\n
        # for token_id in banned_tokens:
        #     logits[token_id] = float('-inf')

        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        next_token_idx = torch.multinomial(top_k_probs, num_samples=1).item()
        next_token = top_k_indices[next_token_idx].item()

        generated_tokens.append(next_token)
        next_token_text = enc.decode([next_token])
        print(next_token_text, end="", flush=True)

        if next_token == 50256:
            print(f"\nEOS at step {i + 1}")
            break

        ctx_len = model.hyperparamiters['max_seq_len']
        tokens_tensor = torch.tensor([generated_tokens[-ctx_len:]], dtype=torch.long, device=device)

print()

generated_text = enc.decode(generated_tokens[len(tokens):])

output_path = "gpt2/generated_sample.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(generated_text)

print(f"\nSaved to {output_path}")
print(f"Generated ({len(generated_tokens) - len(tokens)} tokens):")