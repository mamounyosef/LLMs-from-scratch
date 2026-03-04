import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os
import time
import csv

from gpt2 import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters and configs
num_iters=80
effective_batch_size_in_tokens = 524288 #2**19, ~0.5M number of tokens, which is 524288 // T samples
batch_size = 4
seq_len = 1024
gradient_accumulation_steps = effective_batch_size_in_tokens // (batch_size * seq_len)
learning_rate = 3e-4

eval_interval = 1 #
eval_steps = 30 # number of validation steps to run
checkpoint_interval = 10 # save checkpoint every N steps

print(f"Effective Total batch size in tokens: {effective_batch_size_in_tokens}, Batch Size (in steps): {batch_size}, Gradient Accumulation Steps: {gradient_accumulation_steps}")

model = GPT().to(device) # default parameters
model = torch.compile(model)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    
class DataLoaderLite:
    def __init__(self, batch_size, seq_len=1024, split='train', dataset_path=r'gpt2\dataset.txt'):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)

        n = int(0.9 * len(self.data))

        if split == 'train':
            self.data = self.data[:n]
        else:
            self.data = self.data[n:]
        
        self.current_position = 0

    def next_batch(self):
        B, T = self.batch_size, self.seq_len

        # random sampling
        ix = torch.randint(0, len(self.data) - T - 1, (B,))
        x = torch.stack([self.data[i:i+T] for i in ix])
        y = torch.stack([self.data[i+1:i+T+1] for i in ix])

        return x, y
    

train_loader = DataLoaderLite(batch_size=batch_size, split='train')
val_loader = DataLoaderLite(batch_size=batch_size, split='val')

print(f"Train number of tokens: {len(train_loader.data)}, Val number of tokens: {len(val_loader.data)}")
num_tokens_in_dataset = len(train_loader.data)
num_epochs = (num_iters * effective_batch_size_in_tokens) / num_tokens_in_dataset
print(f"Total Epochs (how many times the dataset is seen): {num_epochs:.2f}")
    
save_dir = r"gpt2\checkpoints"
os.makedirs(save_dir, exist_ok=True)

# csv logging
log_file = os.path.join(save_dir, "training_log.csv")
csv_header = ["step", "loss", "lr", "norm", "dt_ms", "tok_sec", "val_loss"]
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_iters, eta_min=learning_rate*0.1)

best_loss = float('inf')

t0 = time.time()
total_starting_time = time.time()
print("Starting training...")
print("*" * 80)

for iter in range(num_iters):

    optimizer.zero_grad()
    loss_accum = 0.0

    # Actual training
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / gradient_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping

    optimizer.step()
    scheduler.step()
    # timing and stats
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    tokens_processed = effective_batch_size_in_tokens
    tokens_per_sec = tokens_processed / dt
    current_lr = scheduler.get_last_lr()[0]

    # validation
    val_loss = 0.0
    if iter % eval_interval == 0 or iter == num_iters - 1:
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for _ in range(eval_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach()
        val_loss = val_loss_accum / eval_steps
        model.train()

    # checkpointing
    if iter >= int(num_iters * 0.5) and iter % checkpoint_interval == 0:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{iter}.pt")
        torch.save({
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_accum,
            'val_loss': val_loss,
        }, checkpoint_path)

    # save best model based on training loss
    if loss_accum < best_loss:
        best_loss = loss_accum
        best_model_path = os.path.join(save_dir, "best_model.pt")
        torch.save({
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, best_model_path)

    print(f"step {iter:4d} | loss: {loss_accum:.4f} | lr: {current_lr:.6e} | norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.0f} | val_loss: {val_loss:.4f}", flush=True)

    # log to csv
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iter, float(loss_accum), current_lr, float(grad_norm), dt*1000, tokens_per_sec, float(val_loss)])

total_time = time.time() - total_starting_time
print("*" * 80)
print(f"\nTraining complete! Total time: {total_time / 60:.2f} min")