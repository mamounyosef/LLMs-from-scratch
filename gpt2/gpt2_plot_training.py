import matplotlib.pyplot as plt
import csv

log_file = "gpt2/checkpoints/training_log.csv"

steps = []
train_losses = []
val_losses = []
learning_rates = []
grad_norms = []

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row['step']))
        train_losses.append(float(row['loss']))
        val_losses.append(float(row['val_loss']))
        learning_rates.append(float(row['lr']))
        grad_norms.append(float(row['norm']))

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Training loss
axes[0].plot(steps, train_losses, label='Train Loss', color='blue', linewidth=2)
axes[0].plot(steps, val_losses, label='Val Loss', color='orange', linewidth=2)
axes[0].set_title('Training & Validation Loss')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Learning rate
axes[1].plot(steps, learning_rates, label='Learning Rate', color='green', linewidth=2)
axes[1].set_title('Learning Rate Schedule')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Learning Rate')

plt.tight_layout()
plt.savefig('gpt2/checkpoints/training_plots.png', dpi=300)
print("Training plots saved to gpt2/checkpoints/training_plots.png")
