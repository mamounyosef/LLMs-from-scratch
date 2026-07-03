import torch
from transformers import AutoTokenizer
from llama3 import llama

RUN_ON_MODAL = True

if RUN_ON_MODAL:
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "transformers")
        .add_local_python_source("llama3")
    )

    app = modal.App("llama3-generate")
    volume = modal.Volume.from_name("llama3-checkpoints", create_if_missing=True)

def run_generate():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_seq_len = 512

    print(f"Running {'on Modal' if RUN_ON_MODAL else 'Locally'}")
    checkpoint_path = "/checkpoints/checkpoint_step_290.pt" if RUN_ON_MODAL else "llama3/checkpoints/checkpoint_step_290.pt"
    output_path = "/checkpoints/generated_sample.txt" if RUN_ON_MODAL else "llama3/generated_sample.txt"


    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Loaded successfully")
    model = llama(vocab_size=128_256, max_seq_len=max_seq_len).to(device)

    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    enc = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    print("Loaded the tokenizer")
    prompt = "When do you think we will have AGI?"
    prefill = True

    max_new_tokens = 150
    temperature = 0.3
    top_k = 100

    print(f"Prompt: {prompt}")

    tokens = enc.encode(prompt)
    generated_tokens = list(tokens)
    tokens_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)

    print("Generating: ", end="", flush=True)
    with torch.no_grad():
        for i in range(max_new_tokens):
            if prefill:
                logits, _ = model(tokens_tensor, is_inference=True, is_prefill=True)
                prefill = False
            else:
                logits, _ = model(torch.tensor([[next_token]], device=device), is_inference=True)

            logits = logits[0, -1, :]
            logits = logits / temperature

            penalty = 1.4  # >1 discourages repetition
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

            if next_token == 128001:
                print(f"\nEOS at step {i + 1}")
                break

            tokens_tensor = torch.tensor([generated_tokens[-max_seq_len:]], dtype=torch.long, device=device)

    print()

    generated_text = enc.decode(generated_tokens[len(tokens):])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    if RUN_ON_MODAL:
        volume.commit()

    print(f"\nSaved to {output_path}")
    print(f"Generated ({len(generated_tokens) - len(tokens)} tokens):")

if RUN_ON_MODAL:
    run_generate = app.function(
        gpu="H200",
        image=image,
        volumes={"/checkpoints": volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
        cpu=4.0,
        memory=16 * 1024,
        timeout=30 * 60,
    )(run_generate)

if RUN_ON_MODAL:
    @app.local_entrypoint()
    def main():
        run_generate.remote()
else:
    run_generate()