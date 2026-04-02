"""
AtrionNet Research Visualization Utility v4.0 (Official Thesis Edition)
=====================================================================
Generates high-resolution academic plots (Loss, Accuracy, ROC, CM).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Standard Academic Aesthetics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.style.use('seaborn-v0_8-paper') # Robust, clean style

def save_research_plots(history, test_results, save_dir, model_name, epochs, batch_size):
    """Generates the full suite of plots for a specific experiment setup."""
    os.makedirs(save_dir, exist_ok=True)
    e_range = range(1, len(history['train_loss']) + 1)
    title_suffix = f"({epochs} epochs, batch {batch_size})"

    # 1. ACCURACY & LOSS DUAL PLOT (Screenshot 1 Style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss Column
    ax1.plot(e_range, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(e_range, history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_title(f"{model_name} - Loss Evolution\n{title_suffix}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy Column 
    # (Note: In instance detection, we use F1-Score as the Accuracy proxy)
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(e_range, history['train_acc'], label='Train Accuracy', color='#1f77b4', linewidth=2)
        ax2.plot(e_range, history['val_acc'], label='Validation Accuracy', color='#d62728', linewidth=2)
        ax2.set_title(f"{model_name} - Accuracy Evolution\n{title_suffix}")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Percentage (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curves.png", dpi=300)
    plt.close()

    # 2. CONFUSION MATRIX (Appendix F Style)
    if 'y_true' in test_results and 'y_pred' in test_results:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(test_results['y_true'], test_results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f"Confusion Matrix: {model_name}\n {epochs} Epochs - Batch Size {batch_size}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
        plt.close()

    # 3. ROC CURVES (Professional Clinical Style)
    if 'y_scores' in test_results:
        plt.figure(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(test_results['y_true'], test_results['y_scores'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AtrionNet (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Analysis: {model_name}\n({title_suffix})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.2)
        plt.savefig(f"{save_dir}/roc_analysis.png", dpi=300)
        plt.close()

    print(f"INFO: Research visualizations for {title_suffix} exported successfully.")
