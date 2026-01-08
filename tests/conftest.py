"""Pytest fixtures for ai-facegen tests."""

import base64
import io
import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ai_facegen import CharacterSpec, VariantSpec, WorldSpec


@pytest.fixture
def sample_world():
    """Sample WorldSpec for testing."""
    return WorldSpec(
        context="A sci-fi space station orbiting Mars.",
        style="Clean digital art, bold colors, sharp lines.",
        negative="blurry, low quality, text",
    )


@pytest.fixture
def sample_character():
    """Sample CharacterSpec for testing."""
    return CharacterSpec(
        name="Test Character",
        role="Engineer",
        description="Tall person with blue eyes and short hair.",
    )


@pytest.fixture
def sample_variants():
    """Sample VariantSpecs for testing."""
    return [
        VariantSpec(name="icon", size=64, prompt_frame="Small icon, face only."),
        VariantSpec(name="bust", size=256, prompt_frame="Portrait, shoulders up."),
    ]


@pytest.fixture
def mock_image_bytes():
    """Create test PNG image bytes (1024x1024 red square)."""
    img = Image.new("RGB", (1024, 1024), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_bedrock_response(mock_image_bytes):
    """Mock Bedrock API response for Titan."""
    b64_image = base64.b64encode(mock_image_bytes).decode("ascii")
    return {
        "images": [b64_image],
    }


@pytest.fixture
def mock_bedrock_client(mock_bedrock_response):
    """Mock boto3 Bedrock runtime client."""
    mock_client = MagicMock()

    # Mock response body
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(mock_bedrock_response).encode()

    mock_client.invoke_model.return_value = {
        "body": mock_body,
    }

    return mock_client
