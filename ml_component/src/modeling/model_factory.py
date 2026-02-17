"""
Model Factory: Create any architecture by name.
Single entry point for all model creation in the pipeline.
"""

from .resnet1d import resnet18_1d
from .inception_time import InceptionTime
from .transformer_ecg import ECGTransformer


MODEL_REGISTRY = {
    'resnet1d': resnet18_1d,
    'inception_time': InceptionTime,
    'transformer': ECGTransformer,
}


def create_model(model_name: str, **kwargs):
    """
    Factory function to create a model by name.

    Args:
        model_name: One of 'resnet1d', 'inception_time', 'transformer'.
        **kwargs:   Model-specific hyperparameters.

    Returns:
        nn.Module instance.

    Raises:
        ValueError if model_name is not recognized.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    constructor = MODEL_REGISTRY[model_name]

    # ResNet uses a factory function, others are classes
    if model_name == 'resnet1d':
        model = constructor(
            in_channels=kwargs.get('in_channels', 12),
            num_classes=kwargs.get('num_classes', 5),
        )
    elif model_name == 'inception_time':
        model = constructor(
            in_channels=kwargs.get('in_channels', 12),
            num_classes=kwargs.get('num_classes', 5),
            num_blocks=kwargs.get('num_blocks', 6),
            num_filters=kwargs.get('num_filters', 32),
            bottleneck_channels=kwargs.get('bottleneck_channels', 32),
            use_residual=kwargs.get('use_residual', True),
        )
    elif model_name == 'transformer':
        model = constructor(
            in_channels=kwargs.get('in_channels', 12),
            num_classes=kwargs.get('num_classes', 5),
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 4),
            dim_feedforward=kwargs.get('dim_feedforward', 256),
            dropout=kwargs.get('dropout', 0.1),
        )

    # Print param count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name}")
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    return model
