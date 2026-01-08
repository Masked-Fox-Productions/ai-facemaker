"""Tests for image processing."""

import io

import pytest
from PIL import Image

from ai_facegen.processing import ImageProcessor


class TestImageProcessor:
    """Tests for ImageProcessor."""

    @pytest.fixture
    def processor(self):
        """Create an ImageProcessor instance."""
        return ImageProcessor()

    @pytest.fixture
    def square_image_bytes(self):
        """Create a 1024x1024 test image."""
        img = Image.new("RGB", (1024, 1024), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def landscape_image_bytes(self):
        """Create a 1024x768 landscape test image."""
        img = Image.new("RGB", (1024, 768), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def portrait_image_bytes(self):
        """Create a 768x1024 portrait test image."""
        img = Image.new("RGB", (768, 1024), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_center_crop_already_square(self, processor):
        """Test that square images are not cropped."""
        img = Image.new("RGB", (100, 100), "red")
        result = processor._center_crop_square(img)
        assert result.size == (100, 100)

    def test_center_crop_landscape(self, processor):
        """Test center cropping a landscape image."""
        img = Image.new("RGB", (200, 100), "red")
        result = processor._center_crop_square(img)
        assert result.size == (100, 100)

    def test_center_crop_portrait(self, processor):
        """Test center cropping a portrait image."""
        img = Image.new("RGB", (100, 200), "red")
        result = processor._center_crop_square(img)
        assert result.size == (100, 100)

    def test_high_quality_resize(self, processor):
        """Test high-quality resize."""
        img = Image.new("RGB", (1024, 1024), "blue")
        result = processor._high_quality_resize(img, 64)
        assert result.size == (64, 64)

    def test_process_square_to_64(self, processor, square_image_bytes):
        """Test full pipeline: 1024x1024 -> 64x64."""
        result_bytes = processor.process(square_image_bytes, target_size=64)

        # Verify output
        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (64, 64)
        assert result_img.format == "PNG"

    def test_process_square_to_256(self, processor, square_image_bytes):
        """Test full pipeline: 1024x1024 -> 256x256."""
        result_bytes = processor.process(square_image_bytes, target_size=256)

        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (256, 256)

    def test_process_landscape_to_64(self, processor, landscape_image_bytes):
        """Test pipeline with landscape input."""
        result_bytes = processor.process(landscape_image_bytes, target_size=64)

        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (64, 64)

    def test_process_portrait_to_256(self, processor, portrait_image_bytes):
        """Test pipeline with portrait input."""
        result_bytes = processor.process(portrait_image_bytes, target_size=256)

        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (256, 256)

    def test_process_no_resize_needed(self, processor):
        """Test that 1024->1024 doesn't resize unnecessarily."""
        img = Image.new("RGB", (1024, 1024), "purple")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        result_bytes = processor.process(buffer.getvalue(), target_size=1024)

        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (1024, 1024)

    def test_process_rgba_image(self, processor):
        """Test that RGBA images are handled correctly."""
        img = Image.new("RGBA", (1024, 1024), (255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        result_bytes = processor.process(buffer.getvalue(), target_size=64)

        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (64, 64)
