"""Result container for generated portraits."""

import io
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class PortraitResult:
    """Container for a generated portrait variant.

    Provides access to the image as both raw PNG bytes and a PIL Image,
    with lazy loading for efficient memory usage.

    Attributes:
        name: The variant name (e.g., "icon", "bust").
        png_bytes: Raw PNG image data.
        seed: The seed used for generation (for reproducibility).
    """

    name: str
    png_bytes: bytes
    seed: int | None = None
    _pil_image: Image.Image | None = field(default=None, repr=False, compare=False)

    @property
    def image(self) -> Image.Image:
        """Get the image as a PIL Image (lazy-loaded).

        Returns:
            PIL Image object.
        """
        if self._pil_image is None:
            self._pil_image = Image.open(io.BytesIO(self.png_bytes))
        return self._pil_image

    def save(self, path: str) -> None:
        """Write PNG to disk.

        Args:
            path: File path to write to.
        """
        with open(path, "wb") as f:
            f.write(self.png_bytes)

    @property
    def size(self) -> tuple[int, int]:
        """Get the image dimensions.

        Returns:
            Tuple of (width, height).
        """
        return self.image.size

    def __len__(self) -> int:
        """Return the size of the PNG data in bytes."""
        return len(self.png_bytes)
