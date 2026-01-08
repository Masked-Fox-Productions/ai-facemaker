"""Adapter for Stability AI SDXL model."""

import base64
import json
import random
from typing import Any

from ..exceptions import ContentModerationError, GenerationError
from .base import ModelAdapter


class SDXLAdapter(ModelAdapter):
    """Adapter for Stability AI Stable Diffusion XL 1.0.

    SDXL is a high-quality image generation model from Stability AI,
    producing 1024x1024 images with excellent detail and coherence.
    """

    MAX_PROMPT_LENGTH = 2000
    SUPPORTED_SIZES = [1024]
    MODEL_ID = "stability.stable-diffusion-xl-v1"

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
        """Generate an image using Stability SDXL.

        Args:
            client: boto3 bedrock-runtime client.
            model_id: The Bedrock model ID.
            prompt: The positive prompt (max 2000 chars).
            negative_prompt: Elements to avoid.
            seed: Optional seed (0-4294967294).
            width: Image width (1024 for SDXL).
            height: Image height (1024 for SDXL).

        Returns:
            Tuple of (PNG image bytes, seed used).
        """
        # SDXL outputs 1024x1024
        width = 1024
        height = 1024

        # Truncate prompts if needed
        prompt = prompt[: self.MAX_PROMPT_LENGTH]
        negative_prompt = negative_prompt[: self.MAX_PROMPT_LENGTH]

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 4294967294)

        # Build request body for SDXL
        request_body = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
            ],
            "cfg_scale": 7,
            "seed": seed,
            "steps": 50,
            "width": width,
            "height": height,
        }

        # Add negative prompt if provided
        if negative_prompt.strip():
            request_body["text_prompts"].append(
                {"text": negative_prompt, "weight": -1.0}
            )

        # Invoke model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json",
        )

        # Parse response
        response_body = json.loads(response["body"].read())

        # Check for content filter
        artifacts = response_body.get("artifacts", [])
        if not artifacts:
            raise GenerationError("No artifacts in response")

        artifact = artifacts[0]
        finish_reason = artifact.get("finishReason", "")

        if finish_reason == "CONTENT_FILTERED":
            raise ContentModerationError("Content blocked by safety filter")

        if finish_reason not in ("SUCCESS", "END_OF_TEXT", ""):
            raise GenerationError(f"Generation failed: {finish_reason}")

        # Decode base64 image
        image_b64 = artifact.get("base64", "")
        if not image_b64:
            raise GenerationError("No image data in response")

        image_bytes = base64.b64decode(image_b64)

        return image_bytes, seed
