"""Image processing pipeline for crop and downscale operations."""

import io

from PIL import Image


class ImageProcessor:
    """Handles image post-processing: center crop and high-quality downscale.

    Strategy for crisp output:
    1. Generate at a larger size (e.g., 1024x1024)
    2. Center crop to square (if not already square)
    3. Downscale using LANCZOS filter for high quality
    4. Export as PNG for lossless output
    """

    def process(
        self,
        raw_bytes: bytes,
        target_size: int,
        source_size: int = 1024,
    ) -> bytes:
        """Process raw image bytes to target size.

        Args:
            raw_bytes: Input image as bytes (PNG/JPEG from model).
            target_size: Target square size (e.g., 64, 256, 1024).
            source_size: Expected source size (for validation).

        Returns:
            PNG bytes at target_size x target_size.
        """
        # Load image
        img = Image.open(io.BytesIO(raw_bytes))

        # Convert to RGB if necessary (handles RGBA, palette modes)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Center crop to square if not already
        img = self._center_crop_square(img)

        # Downscale if needed
        if img.size[0] != target_size:
            img = self._high_quality_resize(img, target_size)

        # Export as PNG
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _center_crop_square(self, img: Image.Image) -> Image.Image:
        """Center crop an image to square aspect ratio.

        Takes the center square region based on the shorter dimension.

        Args:
            img: Input PIL Image.

        Returns:
            Square-cropped PIL Image.
        """
        width, height = img.size

        if width == height:
            return img

        # Determine crop box
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim

        return img.crop((left, top, right, bottom))

    def _high_quality_resize(self, img: Image.Image, target_size: int) -> Image.Image:
        """High-quality resize using LANCZOS filter.

        LANCZOS provides the best quality for downscaling,
        especially important for small sizes like 64x64.

        Args:
            img: Input PIL Image.
            target_size: Target dimension (square).

        Returns:
            Resized PIL Image.
        """
        return img.resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )
