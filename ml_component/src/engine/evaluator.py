"""
Evaluation metrics and visualization for ECG model
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.ecg_unet import ECGUNet
from src.data_pipeline.loader import create_dataloaders
from src.utils import get_device, load_checkpoint


def dice_coefficient(pred, target, num_classes):
    """
    Calculate Dice coefficient per class
    
    Args:
        pred: Predictions (batch, seq_len)
        target: Targets (batch, seq_len)
        num_classes: Number of classes
        
    Returns:
        Dice scores per class
    """
    dice_scores = []
    
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union
        
        dice_scores.append(dice.item())
    
    return dice_scores


def iou_score(pred, target, num_classes):
    """Calculate IoU per class"""
    iou_scores = []
    
    for class_id in range(num_classes):
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_scores.append(iou.item())
    
    return iou_scores


def evaluate_model(checkpoint: str, data_dir: str = "data/processed", output_dir: str = "results"):
    """
    Evaluate model on test set
    
    Args:
        checkpoint: Path to model checkpoint
        data_dir: Directory with processed data
        output_dir: Directory to save results
    """
    # Setup
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_file = Path(data_dir) / "ecg_data.h5"
    _, _, test_loader = create_dataloaders(data_file, batch_size=16)
    
    # Load model
    print("📥 Loading model...")
    model = ECGUNet(
        in_channels=1,
        num_seg_classes=5,
        num_clf_classes=6,
        use_attention=True
    ).to(device)
    
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    
    # Evaluation
    print("\n📊 Evaluating on test set...")
    
    all_seg_preds = []
    all_seg_targets = []
    all_clf_preds = []
    all_clf_targets = []
    all_clf_probs = []
    
    with torch.no_grad():
        for signals, seg_masks, clf_labels in tqdm(test_loader):
            signals = signals.to(device)
            seg_masks = seg_masks.to(device)
            clf_labels = clf_labels.to(device)
            
            # Forward pass
            seg_pred, clf_pred = model(signals)
            
            # Get predictions
            seg_pred_class = torch.argmax(seg_pred, dim=1)
            clf_pred_class = torch.argmax(clf_pred, dim=1)
            clf_probs = torch.softmax(clf_pred, dim=1)
            
            # Store
            all_seg_preds.append(seg_pred_class.cpu())
            all_seg_targets.append(seg_masks.cpu())
            all_clf_preds.append(clf_pred_class.cpu())
            all_clf_targets.append(clf_labels.cpu())
            all_clf_probs.append(clf_probs.cpu())
    
    # Concatenate
    all_seg_preds = torch.cat(all_seg_preds, dim=0)
    all_seg_targets = torch.cat(all_seg_targets, dim=0)
    all_clf_preds = torch.cat(all_clf_preds, dim=0)
    all_clf_targets = torch.cat(all_clf_targets, dim=0)
    all_clf_probs = torch.cat(all_clf_probs, dim=0)
    
    # Segmentation metrics
    print("\n" + "=" * 80)
    print("SEGMENTATION METRICS")
    print("=" * 80)
    
    class_names = ['Background', 'P-associated', 'P-dissociated', 'QRS', 'T-wave']
    
    # Dice scores
    dice_scores = dice_coefficient(all_seg_preds, all_seg_targets, num_classes=5)
    print("\nDice Coefficient per class:")
    for i, (name, score) in enumerate(zip(class_names, dice_scores)):
        print(f"  {name:15s}: {score:.4f}")
    print(f"  {'Mean':15s}: {np.mean(dice_scores):.4f}")
    
    # IoU scores
    iou_scores = iou_score(all_seg_preds, all_seg_targets, num_classes=5)
    print("\nIoU per class:")
    for i, (name, score) in enumerate(zip(class_names, iou_scores)):
        print(f"  {name:15s}: {score:.4f}")
    print(f"  {'Mean':15s}: {np.mean(iou_scores):.4f}")
    
    # Classification metrics
    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS")
    print("=" * 80)
    
    av_block_names = ['Normal', '1st degree', '2nd Type I', '2nd Type II', '3rd degree', 'VT w/ dissoc']
    
    # Accuracy
    accuracy = accuracy_score(all_clf_targets, all_clf_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_clf_targets, all_clf_preds, 
        labels=range(len(av_block_names)),
        average=None, zero_division=0
    )
    
    print("\nPer-class metrics:")
    print(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 70)
    for i, name in enumerate(av_block_names):
        print(f"{name:20s} {precision[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f} {int(support[i]):10d}")
    
    # Macro averages
    print("-" * 70)
    print(f"{'Macro Average':20s} {np.mean(precision):10.4f} {np.mean(recall):10.4f} {np.mean(f1):10.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_clf_targets, all_clf_preds, labels=range(len(av_block_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=av_block_names, yticklabels=av_block_names)
    plt.title('Confusion Matrix - AV Block Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300)
    print(f"\n💾 Saved confusion matrix to {output_path / 'confusion_matrix.png'}")
    
    # ROC-AUC (if multi-class)
    try:
        roc_auc = roc_auc_score(all_clf_targets, all_clf_probs, multi_class='ovr', average='macro')
        print(f"\nROC-AUC (macro): {roc_auc:.4f}")
    except:
        print("\n⚠️  ROC-AUC calculation skipped (insufficient class representation)")
    
    # Final Summary Table
    mean_dice = np.mean(dice_scores)
    mean_f1 = np.mean(f1)
    
    print("\n" + "=" * 80)
    print("UNIFIED SYSTEM PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Main Component':30s} | {'Primary Metric':25s} | {'Score':10s}")
    print("-" * 80)
    print(f"{'ECG Wave Segmentation':30s} | {'Mean Dice Coefficient':25s} | {mean_dice:10.4f}")
    print(f"{'AV Block Detection':30s} | {'Overall Classification Acc':25s} | {accuracy:10.4f}")
    print(f"{'Clinical Explainer Confidence':30s} | {'Macro Average F1-Score':25s} | {mean_f1:10.4f}")
    print("-" * 80)
    
    # Combined score (weighted toward classification as it's the primary task)
    system_reliability = (0.4 * mean_dice) + (0.6 * accuracy)
    print(f"{'OVERALL SYSTEM RELIABILITY':30s} | {'(Weighted Average)':25s} | {system_reliability:10.4f}")
    print("=" * 80)
    
    print("\nEvaluation complete!")
    print("Results saved to:", output_path)
    print("-" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.data_dir, args.output_dir)
