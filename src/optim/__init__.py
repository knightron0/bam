from .bam import BAM
from .muon import Muon
from .svd import SVD
from .halfbam import HalfBAM

OPTIMIZER_REGISTRY = {
    "bam": BAM,
    "muon": Muon,
    "svd": SVD,
    "halfbam": HalfBAM,
}

def get_optimizer(name):
    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        available_optimizers = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer '{name}'. Available optimizers: {available_optimizers}")
    
    return OPTIMIZER_REGISTRY[name]

def list_optimizers():
    return list(OPTIMIZER_REGISTRY.keys())

__all__ = ["BAM", "Muon", "SVD", "HalfBAM", "OPTIMIZER_REGISTRY", "get_optimizer", "list_optimizers"] 
