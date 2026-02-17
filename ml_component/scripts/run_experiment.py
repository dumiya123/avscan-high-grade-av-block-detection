"""
Main Experiment Runner for Publication-Level ECG Classification.

Usage:
    # Full pipeline (recommended split: folds 1-8 train, 9 val, 10 test)
    python scripts/run_experiment.py --model resnet1d

    # All models comparison
    python scripts/run_experiment.py --model all

    # With ablation study
    python scripts/run_experiment.py --model resnet1d --ablation

    # Classical baselines only
    python scripts/run_experiment.py --baselines-only

    # 10-fold cross validation
    python scripts/run_experiment.py --model resnet1d --cross-validate
"""

import sys
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment import (
    RAW_DIR, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    TRAIN_CONFIG, LOSS_CONFIG, MODELS, AUG_CONFIG,
    BASELINE_CONFIG, ABLATION_EXPERIMENTS,
    TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS, CLASS_NAMES,
)
from src.data_pipeline.ptbxl_dataset import PTBXLDataset, create_ptbxl_loaders
from src.data_pipeline.augmentations import get_train_augmentation
from src.modeling.model_factory import create_model
from src.engine.trainer_v2 import train_model, set_seed
from src.engine.evaluator_v2 import (
    evaluate_model, full_evaluation_with_ci,
    print_evaluation_report, save_results,
)
from src.baselines.classical import run_all_baselines


def run_single_experiment(
    model_name: str,
    data_dir: str,
    use_augmentation: bool = True,
    use_class_weight: bool = True,
    loss_type: str = 'bce',
    experiment_tag: str = '',
    train_folds=None,
    val_folds=None,
):
    """
    Run a single training + evaluation experiment.

    Returns:
        Dict with training results and test metrics.
    """
    if train_folds is None:
        train_folds = TRAIN_FOLDS
    if val_folds is None:
        val_folds = VAL_FOLDS

    experiment_name = f"{model_name}_{experiment_tag}" if experiment_tag else model_name

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")

    # 1. Set seed
    set_seed(TRAIN_CONFIG['seed'])

    # 2. Create augmentation
    train_transform = get_train_augmentation(AUG_CONFIG) if use_augmentation else None

    # 3. Create data loaders
    train_loader, val_loader, test_loader = create_ptbxl_loaders(
        data_dir=data_dir,
        batch_size=TRAIN_CONFIG['batch_size'],
        train_transform=train_transform,
        train_folds=train_folds,
        val_folds=val_folds,
        test_folds=TEST_FOLDS,
    )

    # 4. Create model
    model_config = MODELS.get(model_name, {})
    model = create_model(model_name, **model_config)

    # 5. Compute class weights
    train_config = dict(TRAIN_CONFIG)
    train_config['loss_type'] = loss_type

    if use_class_weight:
        weights = train_loader.dataset.compute_class_weights()
        train_config['pos_weight'] = weights
        print(f"  Class weights: {weights.numpy().round(2)}")
    else:
        train_config['pos_weight'] = None

    # 6. Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        config=train_config,
        checkpoint_dir=str(CHECKPOINT_DIR),
        experiment_name=experiment_name,
    )

    # 7. Load best model for test evaluation
    best_ckpt = Path(train_results['checkpoint_dir']) / 'best_model.pth'
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best model from epoch {checkpoint['epoch']}")

    # 8. Test evaluation with CIs
    print("\n  Evaluating on test set (fold 10)...")
    test_results = evaluate_model(model, test_loader, DEVICE)
    print_evaluation_report(test_results, title=f"TEST RESULTS: {experiment_name}")

    # 9. Save results
    results_dir = RESULTS_DIR / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    save_results(test_results, str(results_dir / 'test_results.json'))
    save_results(train_results, str(results_dir / 'train_results.json'))

    return {
        'experiment_name': experiment_name,
        'train_results': train_results,
        'test_results': {k: v for k, v in test_results.items()
                        if not isinstance(v, np.ndarray)},
    }


def run_cross_validation(model_name: str, data_dir: str, n_folds: int = 10):
    """
    10-fold cross validation for statistical rigor.

    Returns:
        Dict with mean ± std across folds.
    """
    print(f"\n{'='*70}")
    print(f"  {n_folds}-FOLD CROSS VALIDATION: {model_name}")
    print(f"{'='*70}")

    fold_results = []

    for fold_idx in range(1, n_folds + 1):
        print(f"\n  ─── Fold {fold_idx}/{n_folds} (test fold = {fold_idx}) ───")

        # Use fold_idx as test, fold_idx-1 as val, rest as train
        test_fold = [fold_idx]
        val_fold = [(fold_idx % n_folds) + 1]
        train_folds_cv = [f for f in range(1, n_folds + 1)
                          if f not in test_fold and f not in val_fold]

        result = run_single_experiment(
            model_name=model_name,
            data_dir=data_dir,
            experiment_tag=f'cv_fold{fold_idx}',
            train_folds=train_folds_cv,
            val_folds=val_fold,
        )

        fold_results.append(result['test_results'])

    # Aggregate across folds
    print(f"\n{'='*70}")
    print(f"  CROSS VALIDATION SUMMARY: {model_name}")
    print(f"{'='*70}")

    metrics_to_aggregate = ['macro_f1', 'micro_f1', 'mean_auroc',
                            'balanced_accuracy', 'mean_mcc', 'mean_kappa']

    cv_summary = {}
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in fold_results if metric in r]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            cv_summary[metric] = {'mean': float(mean), 'std': float(std)}
            print(f"  {metric:<25} {mean:.4f} ± {std:.4f}")

    # Save CV results
    cv_dir = RESULTS_DIR / f'{model_name}_cv'
    cv_dir.mkdir(parents=True, exist_ok=True)
    with open(cv_dir / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)

    return cv_summary


def run_ablation_study(model_name: str, data_dir: str):
    """
    Ablation study: systematically disable components to measure contribution.
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY: {model_name}")
    print(f"{'='*70}")

    ablation_results = {}

    for exp_name, settings in ABLATION_EXPERIMENTS.items():
        print(f"\n  ─── Ablation: {exp_name} ───")
        print(f"  Settings: {settings}")

        result = run_single_experiment(
            model_name=model_name,
            data_dir=data_dir,
            use_augmentation=settings['augmentation'],
            use_class_weight=settings['class_weight'],
            experiment_tag=f'ablation_{exp_name}',
        )

        ablation_results[exp_name] = result['test_results']

    # Summary table
    print(f"\n{'='*80}")
    print(f"  ABLATION SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"  {'Experiment':<25} {'Macro F1':>10} {'AUROC':>10} {'Bal Acc':>10}")
    print(f"  {'─'*55}")

    for exp_name, res in ablation_results.items():
        print(
            f"  {exp_name:<25} "
            f"{res.get('macro_f1', 0):>10.4f} "
            f"{res.get('mean_auroc', 0):>10.4f} "
            f"{res.get('balanced_accuracy', 0):>10.4f}"
        )

    # Save ablation results
    abl_dir = RESULTS_DIR / f'{model_name}_ablation'
    abl_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in ablation_results.items():
        serializable[k] = {mk: mv for mk, mv in v.items()
                          if not isinstance(mv, np.ndarray)}
    with open(abl_dir / 'ablation_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    return ablation_results


def run_model_comparison(data_dir: str):
    """Run all deep learning models and compare."""
    print(f"\n{'='*70}")
    print(f"  DEEP LEARNING MODEL COMPARISON")
    print(f"{'='*70}")

    all_results = {}
    model_names = ['resnet1d', 'inception_time', 'transformer']

    for model_name in model_names:
        result = run_single_experiment(
            model_name=model_name,
            data_dir=data_dir,
        )
        all_results[model_name] = result['test_results']

    # Summary table
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<20} {'Macro F1':>10} {'AUROC':>10} {'Bal Acc':>10} {'MCC':>10}")
    print(f"  {'─'*60}")

    for name, res in all_results.items():
        print(
            f"  {name:<20} "
            f"{res.get('macro_f1', 0):>10.4f} "
            f"{res.get('mean_auroc', 0):>10.4f} "
            f"{res.get('balanced_accuracy', 0):>10.4f} "
            f"{res.get('mean_mcc', 0):>10.4f}"
        )

    # Save comparison
    comp_dir = RESULTS_DIR / 'model_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {mk: mv for mk, mv in v.items()
                          if not isinstance(mv, np.ndarray)}
    with open(comp_dir / 'comparison_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="ECG Classification Experiment Runner")

    parser.add_argument('--model', type=str, default='resnet1d',
                        choices=['resnet1d', 'inception_time', 'transformer', 'all'],
                        help='Model to train')
    parser.add_argument('--data-dir', type=str, default=str(RAW_DIR),
                        help='Path to PTB-XL dataset root')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Only run classical baselines')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Run 10-fold cross validation')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'focal'],
                        help='Loss function')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')

    args = parser.parse_args()

    # Override config if specified
    if args.epochs:
        TRAIN_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        TRAIN_CONFIG['batch_size'] = args.batch_size
    if args.lr:
        TRAIN_CONFIG['lr'] = args.lr

    data_dir = args.data_dir
    print(f"\n  Data directory: {data_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Config: epochs={TRAIN_CONFIG['epochs']}, "
          f"batch_size={TRAIN_CONFIG['batch_size']}, "
          f"lr={TRAIN_CONFIG['lr']}")

    # ──────────────────────────────────────
    # Classical Baselines
    # ──────────────────────────────────────
    if args.baselines_only or args.model == 'all':
        print("\n  Running classical baselines...")
        train_loader, val_loader, test_loader = create_ptbxl_loaders(
            data_dir=data_dir,
            batch_size=TRAIN_CONFIG['batch_size'],
            num_workers=0,  # baselines don't need GPU
        )
        baseline_results = run_all_baselines(
            train_loader, val_loader, test_loader, BASELINE_CONFIG
        )
        # Save baseline results
        bl_dir = RESULTS_DIR / 'baselines'
        bl_dir.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in baseline_results.items():
            serializable[k] = {
                'val_metrics': {mk: mv for mk, mv in v['val_metrics'].items()
                               if not isinstance(mv, np.ndarray)},
                'test_metrics': {mk: mv for mk, mv in v['test_metrics'].items()
                                if not isinstance(mv, np.ndarray)},
                'train_time': v['train_time'],
            }
        with open(bl_dir / 'baseline_results.json', 'w') as f:
            json.dump(serializable, f, indent=2)

        if args.baselines_only:
            return

    # ──────────────────────────────────────
    # Deep Learning Models
    # ──────────────────────────────────────
    if args.model == 'all':
        run_model_comparison(data_dir)
    else:
        # Single model
        run_single_experiment(
            model_name=args.model,
            data_dir=data_dir,
            loss_type=args.loss,
        )

        # Optional cross-validation
        if args.cross_validate:
            run_cross_validation(args.model, data_dir)

        # Optional ablation
        if args.ablation:
            run_ablation_study(args.model, data_dir)

    print("\n  ✅ All experiments complete!")
    print(f"  Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
