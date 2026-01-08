"""Adapter for Amazon Titan Image Generator."""

import base64
import json
import random
from typing import Any

from ..exceptions import ContentModerationError, GenerationError
from .base import ModelAdapter


class TitanImageAdapter(ModelAdapter):
    """Adapter for Amazon Titan Image Generator V1.

    Titan Image Generator is Amazon's native image generation model,
    well-integrated with Bedrock and supporting various sizes up to 1408px.
    """

    MAX_PROMPT_LENGTH = 512
    MAX_NEGATIVE_LENGTH = 512
    SUPPORTED_SIZES = [512, 768, 1024, 1152, 1408]
    MODEL_ID = "amazon.titan-image-generator-v1"

    @property
    def max_prompt_length(self) -> int:
        return self.MAX_PROMPT_LENGTH

    @property
    def supported_sizes(self) -> list[int]:
        return self.SUPPORTED_SIZES

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    def generate(
        self,
        client: Any,
        model_id: str,
        prompt: str,
        negative_prompt: str,
        seed: int | None,
        width: int,
        height: int,
    ) -> tuple[bytes, int | None]:
        """Generate an image using Amazon Titan Image Generator.

        Args:
            client: boto3 bedrock-runtime client.
            model_id: The Bedrock model ID (default: amazon.titan-image-generator-v1).
            prompt: The positive prompt (max 512 chars).
            negative_prompt: Elements to avoid (max 512 chars).
            seed: Optional seed (0-2147483646).
            width: Image width (will be clamped to supported sizes).
            height: Image height (will be clamped to supported sizes).

        Returns:
            Tuple of (PNG image bytes, seed used).
        """
        # Clamp to max dimension
        max_dim = max(self.SUPPORTED_SIZES)
        width = min(width, max_dim)
        height = min(height, max_dim)

        # Truncate prompts if needed
        prompt = prompt[: self.MAX_PROMPT_LENGTH]
        negative_prompt = negative_prompt[: self.MAX_NEGATIVE_LENGTH]

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2147483646)

        # Build request body
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": height,
                "width": width,
                "cfgScale": 8.0,
                "seed": seed,
            },
        }

        # Add negative prompt if provided
        if negative_prompt.strip():
            request_body["textToImageParams"]["negativeText"] = negative_prompt

        # Invoke model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
        )

        # Parse response
        response_body = json.loads(response["body"].read())

        # Check for errors
        if "error" in response_body:
            error_msg = response_body["error"]
            if "blocked" in error_msg.lower() or "moderat" in error_msg.lower():
                raise ContentModerationError(error_msg)
            raise GenerationError(error_msg)

        if "images" not in response_body or not response_body["images"]:
            raise GenerationError("No images in response")

        # Decode base64 image
        image_b64 = response_body["images"][0]
        image_bytes = base64.b64decode(image_b64)

        return image_bytes, seed
