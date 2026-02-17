
from .cifarmlp import CifarMLP
from .resnet18 import ResNet18_CIFAR10_Wrapper

MODEL_REGISTRY = {
    "cifarmlp": CifarMLP,
    "resnet18": ResNet18_CIFAR10_Wrapper,
}

def get_model(name):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available_models}")
    
    return MODEL_REGISTRY[name]

def list_models():
    return list(MODEL_REGISTRY.keys())

__all__ = ["CifarMLP", "ResNet18_CIFAR10_Wrapper", "MODEL_REGISTRY", "get_model", "list_models"] 