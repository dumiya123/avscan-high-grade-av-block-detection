"""
Research Visualization Utilities.
Generates publication-quality plots for ECG analysis and model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Professional plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def save_publication_plots(history, test_results, plot_dir):
    """Generates all 10+ required research plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    plt.title("AtrionNet Hybrid: Training & Validation Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_dir}/loss_curves.png", dpi=300)
    plt.close()

    # 2. Metric Curves (P/R/F1/mAP)
    plt.figure(figsize=(12, 8))
    if 'val_precision' in history:
        plt.plot(epochs, history['val_precision'], label='Precision', linestyle='--')
    if 'val_recall' in history:
        plt.plot(epochs, history['val_recall'], label='Recall', linestyle='--')
    if 'val_f1' in history:
        plt.plot(epochs, history['val_f1'], label='F1 Score', linewidth=3)
    if 'val_map' in history:
        plt.plot(epochs, history['val_map'], label='mAP @ 0.5', linewidth=3, color='gold')
    plt.title("Evolution of Detection Performance on Validation Set")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_dir}/val_metrics.png", dpi=300)
    plt.close()

    # 3. Learning Rate Curve
    if 'lr' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, history['lr'], color='green')
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.savefig(f"{plot_dir}/lr_schedule.png", dpi=300)
        plt.close()

def plot_confusion_matrix(tp, fp, fn, save_path):
    """Generates detection confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap([[tp, fp], [fn, 0]], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.title("Instance Detection Confusion Matrix")
    # Ground Truth vs Prediction
    plt.xlabel("Model Prediction")
    plt.ylabel("Ground Truth (P-Wave)")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pr_curve(recalls, precisions, ap, save_path):
    """Generates Precision-Recall curve."""
    plt.figure(figsize=(8, 8))
    plt.step(recalls, precisions, where='post', color='b', alpha=0.8, label=f'AP={ap:.4f}')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Instance Level)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
