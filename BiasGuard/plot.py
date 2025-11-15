import re
import matplotlib.pyplot as plt
import pandas as pd

# Parse the training log file
log_file = "nohup_rl.out"

steps = []
train_losses = []
eval_losses = []
learning_rates = []
eval_accuracies = []

with open(log_file, 'r') as f:
    for line in f:
        # Extract training loss
        # Example: {'loss': 0.6234, 'grad_norm': 1.234, 'learning_rate': 0.0002, 'epoch': 0.65}
        if "'loss':" in line and "'grad_norm':" in line and "'epoch':" in line:
            loss_match = re.search(r"'loss':\s*([\d.]+)", line)
            lr_match = re.search(r"'learning_rate':\s*([\d.e-]+)", line)
            
            if loss_match:
                steps.append(len(train_losses))
                train_losses.append(float(loss_match.group(1)))
                if lr_match:
                    learning_rates.append(float(lr_match.group(1)))
        
        # Extract evaluation metrics
        # Example: {'eval_loss': 0.5234, 'eval_runtime': 55.5, 'eval_mean_token_accuracy': 0.8684}
        if "'eval_loss':" in line and "'eval_mean_token_accuracy':" in line:
            eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
            eval_acc_match = re.search(r"'eval_mean_token_accuracy':\s*([\d.]+)", line)
            
            if eval_loss_match:
                eval_losses.append(float(eval_loss_match.group(1)))
            if eval_acc_match:
                eval_accuracies.append(float(eval_acc_match.group(1)))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BiasGuard SFT Training Metrics', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
if train_losses:
    axes[0, 0].plot(train_losses, linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss over Steps', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Evaluation Loss
if eval_losses:
    eval_steps = [i * (len(train_losses) // len(eval_losses)) for i in range(len(eval_losses))]
    axes[0, 1].plot(eval_steps, eval_losses, linewidth=2, color='#A23B72', marker='o', markersize=5)
    axes[0, 1].set_xlabel('Training Steps', fontsize=12)
    axes[0, 1].set_ylabel('Evaluation Loss', fontsize=12)
    axes[0, 1].set_title('Evaluation Loss', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
if learning_rates:
    axes[1, 0].plot(learning_rates, linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Training Steps', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Plot 4: Evaluation Token Accuracy
if eval_accuracies:
    eval_steps = [i * (len(train_losses) // len(eval_accuracies)) for i in range(len(eval_accuracies))]
    axes[1, 1].plot(eval_steps, eval_accuracies, linewidth=2, color='#06A77D', marker='o', markersize=5)
    axes[1, 1].set_xlabel('Training Steps', fontsize=12)
    axes[1, 1].set_ylabel('Token Accuracy', fontsize=12)
    axes[1, 1].set_title('Evaluation Token Accuracy', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('biasguard_training_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Training metrics plot saved to: biasguard_training_metrics.png")

# Print summary statistics
if train_losses:
    print(f"\n📊 Training Summary:")
    print(f"Total training steps: {len(train_losses)}")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"Initial eval loss: {eval_losses[0]:.4f}")
        print(f"Final eval loss: {eval_losses[-1]:.4f}")
    if eval_accuracies:
        print(f"Initial token accuracy: {eval_accuracies[0]:.4f}")
        print(f"Final token accuracy: {eval_accuracies[-1]:.4f}")
else:
    print("⚠️ No training data found in log file")