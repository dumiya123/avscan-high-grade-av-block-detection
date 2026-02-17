"""
Medical-Grade Evaluation Suite for Multi-Label ECG Classification.

Reports all metrics required for peer-reviewed medical AI papers:
- Macro/Micro F1
- AUROC per class + Mean AUROC
- Balanced Accuracy
- Sensitivity & Specificity per class
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- 95% Confidence Intervals via bootstrap
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, balanced_accuracy_score,
    multilabel_confusion_matrix,
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    class_names: List[str] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics for multi-label classification.

    Args:
        y_true: (N, C) ground truth multi-hot labels
        y_prob: (N, C) predicted probabilities (after sigmoid)
        threshold: Classification threshold
        class_names: List of class names

    Returns:
        Dictionary with all metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = y_true.shape[1]
    y_pred = (y_prob >= threshold).astype(int)

    results = {}

    # ----------------------------------------------------------
    # 1. F1 Scores
    # ----------------------------------------------------------
    results['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    results['micro_f1'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    results['per_class_f1'] = {}
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        results['per_class_f1'][name] = float(f1_per[i])

    # ----------------------------------------------------------
    # 2. AUROC
    # ----------------------------------------------------------
    results['per_class_auroc'] = {}
    auroc_values = []
    for i, name in enumerate(class_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            results['per_class_auroc'][name] = float(auc)
            auroc_values.append(auc)
        except ValueError:
            results['per_class_auroc'][name] = float('nan')

    results['mean_auroc'] = float(np.nanmean(auroc_values)) if auroc_values else float('nan')

    # ----------------------------------------------------------
    # 3. Sensitivity & Specificity per class
    # ----------------------------------------------------------
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    results['per_class_sensitivity'] = {}
    results['per_class_specificity'] = {}

    for i, name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results['per_class_sensitivity'][name] = float(sens)
        results['per_class_specificity'][name] = float(spec)

    # ----------------------------------------------------------
    # 4. Balanced Accuracy (per class, then average)
    # ----------------------------------------------------------
    bal_acc_per_class = []
    for i in range(num_classes):
        ba = balanced_accuracy_score(y_true[:, i], y_pred[:, i])
        bal_acc_per_class.append(ba)
    results['balanced_accuracy'] = float(np.mean(bal_acc_per_class))

    # ----------------------------------------------------------
    # 5. Matthews Correlation Coefficient (per class average)
    # ----------------------------------------------------------
    mcc_values = []
    for i in range(num_classes):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        except Exception:
            mcc = 0.0
        mcc_values.append(mcc)
    results['mean_mcc'] = float(np.mean(mcc_values))

    # ----------------------------------------------------------
    # 6. Cohen's Kappa (per class average)
    # ----------------------------------------------------------
    kappa_values = []
    for i in range(num_classes):
        try:
            kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])
        except Exception:
            kappa = 0.0
        kappa_values.append(kappa)
    results['mean_kappa'] = float(np.mean(kappa_values))

    # ----------------------------------------------------------
    # 7. Subset Accuracy (exact match ratio)
    # ----------------------------------------------------------
    results['subset_accuracy'] = float(np.mean(
        np.all(y_pred == y_true, axis=1)
    ))

    # ----------------------------------------------------------
    # 8. Hamming Loss
    # ----------------------------------------------------------
    results['hamming_loss'] = float(np.mean(y_pred != y_true))

    return results


def compute_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a given metric.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric_fn: Callable(y_true, y_prob) -> scalar
        n_bootstrap: Number of bootstrap iterations
        confidence_level: CI level (0.95 = 95%)
        seed: Random seed

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            continue

    scores = np.array(scores)
    alpha = 1 - confidence_level
    lower = np.percentile(scores, alpha / 2 * 100)
    upper = np.percentile(scores, (1 - alpha / 2) * 100)
    mean = np.mean(scores)

    return float(mean), float(lower), float(upper)


def full_evaluation_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Full evaluation with 95% confidence intervals for key metrics.

    Returns:
        Dictionary with point estimates and CIs.
    """
    # Point estimates
    results = compute_all_metrics(y_true, y_prob, threshold)

    # CIs for key metrics
    y_pred = (y_prob >= threshold).astype(int)

    # Macro F1 CI
    def macro_f1_fn(yt, yp):
        yp_bin = (yp >= threshold).astype(int)
        return f1_score(yt, yp_bin, average='macro', zero_division=0)

    mean, lower, upper = compute_confidence_intervals(y_true, y_prob, macro_f1_fn, n_bootstrap)
    results['macro_f1_ci'] = {'mean': mean, 'lower': lower, 'upper': upper}

    # Mean AUROC CI
    def auroc_fn(yt, yp):
        aucs = []
        for i in range(yt.shape[1]):
            try:
                aucs.append(roc_auc_score(yt[:, i], yp[:, i]))
            except ValueError:
                pass
        return np.mean(aucs) if aucs else 0.0

    mean, lower, upper = compute_confidence_intervals(y_true, y_prob, auroc_fn, n_bootstrap)
    results['mean_auroc_ci'] = {'mean': mean, 'lower': lower, 'upper': upper}

    return results


@torch.no_grad()
def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Run model inference on a DataLoader and compute all metrics.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader to evaluate on.
        device: torch.device.
        threshold: Classification threshold.

    Returns:
        Dictionary of all metrics + raw predictions.
    """
    model.eval()

    all_probs = []
    all_labels = []

    for signals, labels in data_loader:
        signals = signals.to(device)
        logits = model(signals)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    # Full evaluation with CIs
    results = full_evaluation_with_ci(y_true, y_prob, threshold)
    results['y_true'] = y_true
    results['y_prob'] = y_prob

    return results


def print_evaluation_report(results: Dict, title: str = "Evaluation Report"):
    """
    Pretty-print evaluation results to console.
    Publication-ready format.
    """
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 50)
    print(f"{'Macro F1-Score':<35} {results['macro_f1']:>10.4f}")
    if 'macro_f1_ci' in results:
        ci = results['macro_f1_ci']
        print(f"{'  95% CI':<35} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"{'Micro F1-Score':<35} {results['micro_f1']:>10.4f}")
    print(f"{'Mean AUROC':<35} {results['mean_auroc']:>10.4f}")
    if 'mean_auroc_ci' in results:
        ci = results['mean_auroc_ci']
        print(f"{'  95% CI':<35} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"{'Balanced Accuracy':<35} {results['balanced_accuracy']:>10.4f}")
    print(f"{'Mean MCC':<35} {results['mean_mcc']:>10.4f}")
    print(f"{'Mean Cohen Kappa':<35} {results['mean_kappa']:>10.4f}")
    print(f"{'Subset Accuracy':<35} {results['subset_accuracy']:>10.4f}")
    print(f"{'Hamming Loss':<35} {results['hamming_loss']:>10.4f}")

    # Per-class breakdown
    print(f"\n{'Class':<8} {'F1':>8} {'AUROC':>8} {'Sens':>8} {'Spec':>8}")
    print("-" * 42)
    for name in CLASS_NAMES:
        f1 = results['per_class_f1'].get(name, 0)
        auc = results['per_class_auroc'].get(name, 0)
        sens = results['per_class_sensitivity'].get(name, 0)
        spec = results['per_class_specificity'].get(name, 0)
        print(f"{name:<8} {f1:>8.4f} {auc:>8.4f} {sens:>8.4f} {spec:>8.4f}")

    print("=" * 80)


def save_results(results: Dict, filepath: str):
    """Save results to JSON (excludes numpy arrays)."""
    serializable = {k: v for k, v in results.items()
                    if not isinstance(v, np.ndarray)}
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {filepath}")
