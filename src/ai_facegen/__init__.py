"""ai-facegen: Generate consistent character portraits with Amazon Bedrock.

This library provides a simple API for generating character portraits
using AWS Bedrock image models. Define your world context once, then
generate consistent portraits for your entire character roster.

Example:
    >>> from ai_facegen import PortraitClient, WorldSpec, CharacterSpec, VariantSpec
    >>>
    >>> world = WorldSpec(
    ...     context="A sci-fi space station orbiting Mars.",
    ...     style="Clean digital art, bold colors.",
    ...     negative="blurry, low quality"
    ... )
    >>>
    >>> character = CharacterSpec(
    ...     name="Captain Nova",
    ...     role="Station Commander",
    ...     description="Confident leader, silver hair, blue uniform."
    ... )
    >>>
    >>> variants = [
    ...     VariantSpec(name="icon", size=64, prompt_frame="Face icon, bold silhouette."),
    ...     VariantSpec(name="bust", size=256, prompt_frame="Head and shoulders."),
    ... ]
    >>>
    >>> client = PortraitClient(region_name="us-east-1")
    >>> results = client.generate(world=world, character=character, variants=variants)
    >>> results["icon"].save("captain_icon.png")
"""

from ._version import __version__
from .cache import FileCache, MemoryCache, NoOpCache
from .client import PortraitClient
from .exceptions import (
    CacheError,
    ContentModerationError,
    FaceGenError,
    GenerationError,
    InvalidSpecError,
    ModelNotFoundError,
)
from .result import PortraitResult
from .specs import CharacterSpec, VariantSpec, WorldSpec

__all__ = [
    # Core specs
    "WorldSpec",
    "CharacterSpec",
    "VariantSpec",
    # Client
    "PortraitClient",
    # Results
    "PortraitResult",
    # Caching
    "FileCache",
    "MemoryCache",
    "NoOpCache",
    # Exceptions
    "FaceGenError",
    "GenerationError",
    "ContentModerationError",
    "ModelNotFoundError",
    "InvalidSpecError",
    "CacheError",
    # Version
    "__version__",
]
