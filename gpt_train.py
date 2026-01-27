import torch
import torch.nn as nn
import os
from torch.amp import GradScaler
from torch.amp import autocast
import time
from gpt import GPTLanguageModel, get_batch

# hyperparameters
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


torch.manual_seed(1337)


model = GPTLanguageModel()
model = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

def train_model(max_iters=5000, learning_rate=3e-4, save_dir='checkpoints'):
    
    os.makedirs(save_dir, exist_ok=True)

    # Mixed precision scaler
    scaler = GradScaler()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, 
                                                           patience=25, verbose=True)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with autocast(device):
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    best_val_loss = float('inf')
    metrics_file = os.path.join(save_dir, 'training_metrics.txt')
    
    # Initialize metrics file
    with open(metrics_file, 'w') as f:
        f.write("Step\tTrain_Loss\tVal_Loss\tLearning_Rate\tEval_Time\tAvg_Step_Time\tTotal_Time\n")
    
    print(f"Training: max_iters={max_iters}, lr={learning_rate}, eval_interval={eval_interval}, device={device}")
    print("=" * 120)
    
    training_start_time = time.time()
    iter_times = []
    
    for iter in range(max_iters):
        iter_start_time = time.time()
        
        # Evaluation
        if iter % eval_interval == 0 or iter == max_iters - 1:
            eval_start_time = time.time()
            losses = estimate_loss()
            current_lr = optimizer.param_groups[0]['lr']
            eval_time = time.time() - eval_start_time
            avg_step_time = sum(iter_times) / len(iter_times) if iter_times else 0
            total_time = time.time() - training_start_time
            
            # Calculate ETA
            if iter > 0:
                eta_minutes = (avg_step_time * (max_iters - iter)) / 60
            else:
                eta_minutes = 0

            # Single line print
            improvement = f"(↓{best_val_loss - losses['val']:.6f})" if losses['val'] < best_val_loss else ""
            print(f"iter:{iter:5d}/{max_iters} ({100*iter/max_iters:5.1f}%) | train_loss:{losses['train']:.6f} | val_loss:{losses['val']:.6f} {improvement} | best:{best_val_loss:.6f} | lr:{current_lr:.2e} | eval_time:{eval_time:.2f}s | step_time:{avg_step_time:.4f}s | total:{total_time/60:.1f}m | ETA:{eta_minutes:.1f}m")
            
            # Save metrics
            with open(metrics_file, 'a') as f:
                f.write(f"{iter}\t{losses['train']:.6f}\t{losses['val']:.6f}\t{current_lr:.2e}\t{eval_time:.2f}\t{avg_step_time:.4f}\t{total_time:.2f}\n")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train'],
                }, os.path.join(save_dir, 'best_model.pt'))
            
            # Update scheduler
            scheduler.step(losses['val'])
            
            # Reset iter_times after evaluation
            iter_times = []
        
        # Training step
        xb, yb = get_batch('train')
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device):
            logits, loss = model(xb, yb)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track iteration time (excluding evaluation iterations)
        if iter % eval_interval != 0:
            iter_time = time.time() - iter_start_time
            iter_times.append(iter_time)
            
            # Keep only last 100 measurements for rolling average
            if len(iter_times) > 100:
                iter_times.pop(0)

    total_training_time = time.time() - training_start_time
    print("=" * 120)
    print(f"Training complete! Best val_loss: {best_val_loss:.6f} | Total time: {total_training_time/60:.2f}m ({total_training_time/3600:.2f}h)")
    
    return model


if __name__ == "__main__":
    train_model()