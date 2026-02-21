"""
AtrionNet Model Factory.
Single entry point for creating research architectures.
"""
from .atrion_net import AtrionNetHybrid, AtrionNetBaseline

MODEL_REGISTRY = {
    'atrion_hybrid': AtrionNetHybrid,
    'atrion_baseline': AtrionNetBaseline,
}

def create_model(model_name: str, in_channels=12, **kwargs):
    """
    Creates a model by name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Model '{model_name}' created with {total:,} parameters.")
    return model
