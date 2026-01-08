"""Model adapters for AWS Bedrock image generation models."""

from typing import Dict, Type

from .base import ModelAdapter
from .titan import TitanImageAdapter
from .sdxl import SDXLAdapter
from .sd35 import SD35LargeAdapter

_ADAPTERS: Dict[str, Type[ModelAdapter]] = {
    "titan": TitanImageAdapter,
    "sdxl": SDXLAdapter,
    "sd35": SD35LargeAdapter,
}


def get_model_adapter(model_name: str) -> ModelAdapter:
    """Get a model adapter instance by name.

    Args:
        model_name: One of "titan", "sdxl", or "sd35".

    Returns:
        An instance of the appropriate ModelAdapter.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in _ADAPTERS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[model_name]()


__all__ = [
    "ModelAdapter",
    "get_model_adapter",
    "TitanImageAdapter",
    "SDXLAdapter",
    "SD35LargeAdapter",
]
