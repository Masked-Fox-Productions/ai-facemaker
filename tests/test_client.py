"""Tests for PortraitClient."""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ai_facegen import (
    CharacterSpec,
    PortraitClient,
    PortraitResult,
    VariantSpec,
    WorldSpec,
)
from ai_facegen.cache import MemoryCache
from ai_facegen.exceptions import ModelNotFoundError


class TestPortraitClient:
    """Tests for PortraitClient."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("boto3.client"):
            client = PortraitClient()
            assert client._model_name == "titan"
            assert client._region is None

    def test_init_custom_region(self):
        """Test initialization with custom region."""
        with patch("boto3.client"):
            client = PortraitClient(region_name="us-west-2")
            assert client._region == "us-west-2"

    def test_init_custom_model(self):
        """Test initialization with different models."""
        with patch("boto3.client"):
            client = PortraitClient(model="sdxl")
            assert client._model_name == "sdxl"

            client2 = PortraitClient(model="sd35")
            assert client2._model_name == "sd35"

    def test_init_invalid_model(self):
        """Test that invalid model raises error."""
        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            PortraitClient(model="invalid_model")

    def test_lazy_bedrock_init(self):
        """Test that Bedrock client is lazily initialized."""
        with patch("boto3.client") as mock_boto:
            client = PortraitClient()
            # Client not created yet
            mock_boto.assert_not_called()

            # Access bedrock property
            _ = client.bedrock
            mock_boto.assert_called_once()

    def test_generate_returns_dict(
        self,
        sample_world,
        sample_character,
        sample_variants,
        mock_bedrock_client,
    ):
        """Test that generate returns a dict of results."""
        with patch("boto3.client", return_value=mock_bedrock_client):
            client = PortraitClient()
            results = client.generate(
                world=sample_world,
                character=sample_character,
                variants=sample_variants,
            )

            assert isinstance(results, dict)
            assert "icon" in results
            assert "bust" in results

    def test_generate_result_has_png_bytes(
        self,
        sample_world,
        sample_character,
        sample_variants,
        mock_bedrock_client,
    ):
        """Test that results have PNG bytes."""
        with patch("boto3.client", return_value=mock_bedrock_client):
            client = PortraitClient()
            results = client.generate(
                world=sample_world,
                character=sample_character,
                variants=sample_variants,
            )

            icon_result = results["icon"]
            assert isinstance(icon_result, PortraitResult)
            assert icon_result.png_bytes is not None
            assert len(icon_result.png_bytes) > 0

    def test_generate_result_correct_size(
        self,
        sample_world,
        sample_character,
        sample_variants,
        mock_bedrock_client,
    ):
        """Test that results are at the correct target size."""
        with patch("boto3.client", return_value=mock_bedrock_client):
            client = PortraitClient()
            results = client.generate(
                world=sample_world,
                character=sample_character,
                variants=sample_variants,
            )

            # Check icon is 64x64
            icon_img = Image.open(io.BytesIO(results["icon"].png_bytes))
            assert icon_img.size == (64, 64)

            # Check bust is 256x256
            bust_img = Image.open(io.BytesIO(results["bust"].png_bytes))
            assert bust_img.size == (256, 256)

    def test_generate_with_cache(
        self,
        sample_world,
        sample_character,
        sample_variants,
        mock_bedrock_client,
    ):
        """Test that caching works."""
        cache = MemoryCache()

        with patch("boto3.client", return_value=mock_bedrock_client):
            client = PortraitClient(cache=cache)

            # First call
            results1 = client.generate(
                world=sample_world,
                character=sample_character,
                variants=sample_variants,
                seed=42,
            )

            # Should have called invoke_model twice (2 variants)
            assert mock_bedrock_client.invoke_model.call_count == 2

            # Second call with same params should use cache
            results2 = client.generate(
                world=sample_world,
                character=sample_character,
                variants=sample_variants,
                seed=42,
            )

            # Should not have called invoke_model again
            assert mock_bedrock_client.invoke_model.call_count == 2

            # Results should be equivalent
            assert results1["icon"].png_bytes == results2["icon"].png_bytes

    def test_generate_count_greater_than_one(
        self,
        sample_world,
        sample_character,
        mock_bedrock_client,
    ):
        """Test generating multiple images per variant."""
        variant = VariantSpec(name="icon", size=64, prompt_frame="Test")

        with patch("boto3.client", return_value=mock_bedrock_client):
            client = PortraitClient()
            results = client.generate(
                world=sample_world,
                character=sample_character,
                variants=[variant],
                count=3,
            )

            # Should return list of results
            assert "icon" in results
            assert isinstance(results["icon"], list)
            assert len(results["icon"]) == 3

    def test_generate_invalid_count(
        self,
        sample_world,
        sample_character,
        sample_variants,
    ):
        """Test that count < 1 raises error."""
        with patch("boto3.client"):
            client = PortraitClient()
            with pytest.raises(ValueError, match="count must be at least 1"):
                client.generate(
                    world=sample_world,
                    character=sample_character,
                    variants=sample_variants,
                    count=0,
                )
