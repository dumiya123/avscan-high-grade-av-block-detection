"""
Classical ML Baselines for ECG Classification.

Implements Logistic Regression, Random Forest, and SVM using scikit-learn.
These serve as comparison baselines to demonstrate deep learning improvement.

Feature extraction: Statistical features from each lead are used as input
(mean, std, max, min, skew, kurtosis, energy, zero-crossings).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from typing import Dict, Tuple, List
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

from src.engine.evaluator_v2 import compute_all_metrics, print_evaluation_report


def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a multi-lead ECG signal.

    Args:
        signal: (num_leads, seq_len) numpy array

    Returns:
        (num_features,) feature vector
    """
    features = []

    for lead_idx in range(signal.shape[0]):
        lead = signal[lead_idx]

        # Statistical features
        features.extend([
            np.mean(lead),
            np.std(lead),
            np.max(lead),
            np.min(lead),
            np.max(lead) - np.min(lead),                # peak-to-peak
            scipy_stats.skew(lead),
            scipy_stats.kurtosis(lead),
            np.sqrt(np.mean(lead**2)),                   # RMS energy
            np.sum(np.abs(np.diff(np.sign(lead)))) / 2,  # zero crossings
            np.mean(np.abs(np.diff(lead))),               # mean absolute diff
        ])

    return np.array(features, dtype=np.float32)


def extract_features_from_loader(data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all samples in a DataLoader.

    Returns:
        X: (N, num_features) feature matrix
        y: (N, num_classes) multi-hot labels
    """
    all_features = []
    all_labels = []

    for signals, labels in data_loader:
        signals = signals.numpy()  # (batch, leads, seq_len)
        labels = labels.numpy()    # (batch, num_classes)

        for i in range(signals.shape[0]):
            feat = extract_features(signals[i])
            all_features.append(feat)

        all_labels.append(labels)

    X = np.stack(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def train_and_evaluate_baseline(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict = None,
) -> Dict:
    """
    Train a classical ML baseline and evaluate on val+test sets.

    Args:
        model_name: 'logistic_regression', 'random_forest', or 'svm'
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        config: Model-specific config dict

    Returns:
        Dict with val and test metrics.
    """
    if config is None:
        config = {}

    print(f"\n  Baseline: {model_name}")
    print(f"  {'─'*40}")

    # 1. Feature extraction
    print("  Extracting features...")
    t0 = time.time()
    X_train, y_train = extract_features_from_loader(train_loader)
    X_val, y_val = extract_features_from_loader(val_loader)
    X_test, y_test = extract_features_from_loader(test_loader)
    print(f"  Features extracted in {time.time()-t0:.1f}s")
    print(f"  Feature dim: {X_train.shape[1]}")

    # 2. Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 3. Create model
    if model_name == 'logistic_regression':
        base_model = LogisticRegression(
            max_iter=config.get('max_iter', 1000),
            C=config.get('C', 1.0),
            solver='lbfgs',
        )
    elif model_name == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            n_jobs=config.get('n_jobs', -1),
            random_state=42,
        )
    elif model_name == 'svm':
        base_model = LinearSVC(
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 5000),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown baseline: {model_name}")

    # Wrap in OneVsRestClassifier for multi-label
    model = OneVsRestClassifier(base_model)

    # 4. Train
    print("  Training...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    # 5. Predict
    # For proba-based metrics (AUROC), use predict_proba if available
    try:
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)
    except AttributeError:
        # SVM doesn't have predict_proba by default
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        val_probs = val_preds.astype(float)
        test_probs = test_preds.astype(float)

    # 6. Evaluate
    val_metrics = compute_all_metrics(y_val, val_probs)
    test_metrics = compute_all_metrics(y_test, test_probs)

    print(f"  Val  Macro F1: {val_metrics['macro_f1']:.4f} | AUROC: {val_metrics['mean_auroc']:.4f}")
    print(f"  Test Macro F1: {test_metrics['macro_f1']:.4f} | AUROC: {test_metrics['mean_auroc']:.4f}")

    return {
        'model_name': model_name,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_time': train_time,
        'feature_dim': X_train.shape[1],
    }


def run_all_baselines(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    configs: Dict = None,
) -> Dict:
    """
    Run all classical baselines and return results.

    Args:
        configs: Dict of model_name -> config_dict

    Returns:
        Dict of model_name -> results
    """
    if configs is None:
        configs = {
            'logistic_regression': {'max_iter': 1000, 'C': 1.0},
            'random_forest': {'n_estimators': 200, 'max_depth': 20, 'n_jobs': -1},
            'svm': {'C': 1.0, 'max_iter': 5000},
        }

    all_results = {}

    print("\n" + "=" * 60)
    print("  CLASSICAL ML BASELINES")
    print("=" * 60)

    for model_name, config in configs.items():
        result = train_and_evaluate_baseline(
            model_name, train_loader, val_loader, test_loader, config
        )
        all_results[model_name] = result

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Model':<25} {'Val F1':>8} {'Val AUC':>8} {'Test F1':>8} {'Test AUC':>8}")
    print("-" * 70)
    for name, res in all_results.items():
        print(
            f"  {name:<25} "
            f"{res['val_metrics']['macro_f1']:>8.4f} "
            f"{res['val_metrics']['mean_auroc']:>8.4f} "
            f"{res['test_metrics']['macro_f1']:>8.4f} "
            f"{res['test_metrics']['mean_auroc']:>8.4f}"
        )
    print("=" * 70)

    return all_results
