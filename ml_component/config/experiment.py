"""
Central Experiment Configuration.
All hyperparameters, paths, and experiment settings live here.
No magic numbers anywhere else in the codebase.
"""

from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ptbxl"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# DATASET
# ============================================================
SAMPLING_RATE = 500          # Hz (PTB-XL native)
SIGNAL_LENGTH = 5000         # 10s * 500Hz
NUM_LEADS = 12               # Standard 12-lead ECG
NUM_CLASSES = 5              # NORM, MI, STTC, CD, HYP

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# PTB-XL recommended folds (patient-wise stratified)
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# ============================================================
# AUGMENTATION
# ============================================================
AUG_CONFIG = {
    'time_shift_max': 50,        # samples
    'amplitude_scale_range': (0.8, 1.2),
    'gaussian_noise_std': 0.05,
    'baseline_wander_freq': 0.5, # Hz
    'baseline_wander_amp': 0.1,
    'probability': 0.5,          # probability of applying each aug
}

# ============================================================
# MODEL
# ============================================================
MODELS = {
    'resnet1d': {
        'in_channels': NUM_LEADS,
        'num_classes': NUM_CLASSES,
    },
    'inception_time': {
        'in_channels': NUM_LEADS,
        'num_classes': NUM_CLASSES,
        'num_blocks': 6,
        'num_filters': 32,
        'bottleneck_channels': 32,
        'use_residual': True,
    },
    'transformer': {
        'in_channels': NUM_LEADS,
        'num_classes': NUM_CLASSES,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
}

# ============================================================
# TRAINING
# ============================================================
TRAIN_CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stop_patience': 10,
    'scheduler': 'cosine',       # 'cosine' or 'plateau'
    'label_smoothing': 0.0,
    'gradient_clip': 1.0,
    'seed': 42,
}

# Loss function options
LOSS_CONFIG = {
    'type': 'bce',               # 'bce' or 'focal'
    'use_class_weights': True,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
}

# ============================================================
# EVALUATION
# ============================================================
EVAL_CONFIG = {
    'threshold': 0.5,            # sigmoid threshold for predictions
    'confidence_level': 0.95,    # for confidence intervals
    'bootstrap_iterations': 1000,
}

# ============================================================
# BASELINE MODELS
# ============================================================
BASELINE_CONFIG = {
    'logistic_regression': {'max_iter': 1000, 'C': 1.0},
    'random_forest': {'n_estimators': 200, 'max_depth': 20, 'n_jobs': -1},
    'svm': {'C': 1.0, 'kernel': 'rbf', 'max_iter': 5000},
}

# ============================================================
# ABLATION
# ============================================================
ABLATION_EXPERIMENTS = {
    'full':            {'augmentation': True,  'attention': True,  'class_weight': True},
    'no_aug':          {'augmentation': False, 'attention': True,  'class_weight': True},
    'no_attention':    {'augmentation': True,  'attention': False, 'class_weight': True},
    'no_class_weight': {'augmentation': True,  'attention': True,  'class_weight': False},
    'minimal':         {'augmentation': False, 'attention': False, 'class_weight': False},
}

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
