"""Data classes for world, character, and variant specifications."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldSpec:
    """Shared world/style context for consistent visual generation.

    Attributes:
        context: Setting description, lore, and world details.
        style: Visual style rules and art direction.
        negative: Elements to avoid in generation (passed to negative_prompt).
    """

    context: str
    style: str
    negative: str = ""

    def __post_init__(self) -> None:
        if not self.context.strip():
            raise ValueError("WorldSpec.context cannot be empty")
        if not self.style.strip():
            raise ValueError("WorldSpec.style cannot be empty")


@dataclass(frozen=True)
class CharacterSpec:
    """Per-character details for portrait generation.

    Attributes:
        name: Character name.
        role: Role, faction, or job title.
        description: Physical appearance, clothing, expression details.
    """

    name: str
    role: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CharacterSpec.name cannot be empty")


@dataclass(frozen=True)
class VariantSpec:
    """Output variant configuration.

    Attributes:
        name: Variant identifier (e.g., "icon", "bust", "full").
        size: Target output size in pixels (square, e.g., 64, 256, 1024).
        prompt_frame: Framing and composition instructions for this variant.
    """

    name: str
    size: int
    prompt_frame: str
    _generation_size: int = field(default=1024, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"VariantSpec.size must be positive, got {self.size}")
        if self.size > 1024:
            raise ValueError(f"VariantSpec.size max is 1024, got {self.size}")
        if not self.name.strip():
            raise ValueError("VariantSpec.name cannot be empty")
        # Always generate at 1024 for best quality, then downscale
        object.__setattr__(self, "_generation_size", 1024)
