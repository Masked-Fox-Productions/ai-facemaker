"""Adapter for Stability AI SD3.5 Large model."""

import base64
import json
import random
from typing import Any, List, Optional, Tuple

from ..exceptions import ContentModerationError, GenerationError
from .base import ModelAdapter


class SD35LargeAdapter(ModelAdapter):
    """Adapter for Stability AI Stable Diffusion 3.5 Large.

    SD3.5 Large is the latest high-quality model from Stability AI,
    supporting longer prompts (up to 10k chars) and aspect ratio control.
    """

    MAX_PROMPT_LENGTH = 10000
    SUPPORTED_SIZES = [1024]  # SD3.5 uses aspect ratio, outputs ~1MP
    MODEL_ID = "stability.sd3-5-large-v1:0"

    # Supported aspect ratios
    ASPECT_RATIOS = ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]

    @property
    def max_prompt_length(self) -> int:
        return self.MAX_PROMPT_LENGTH

    @property
    def supported_sizes(self) -> List[int]:
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
        seed: Optional[int],
        width: int,
        height: int,
    ) -> Tuple[bytes, Optional[int]]:
        """Generate an image using Stability SD3.5 Large.

        Args:
            client: boto3 bedrock-runtime client.
            model_id: The Bedrock model ID.
            prompt: The positive prompt (max 10000 chars).
            negative_prompt: Elements to avoid.
            seed: Optional seed (0-4294967294).
            width: Ignored (SD3.5 uses aspect ratio).
            height: Ignored (SD3.5 uses aspect ratio).

        Returns:
            Tuple of (PNG image bytes, seed used).
        """
        # SD3.5 uses aspect ratio, we use 1:1 for portraits
        aspect_ratio = "1:1"

        # Truncate prompts if needed
        prompt = prompt[: self.MAX_PROMPT_LENGTH]

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 4294967294)

        # Build request body for SD3.5
        request_body = {
            "prompt": prompt,
            "mode": "text-to-image",
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "seed": seed,
        }

        # Add negative prompt if provided
        if negative_prompt.strip():
            request_body["negative_prompt"] = negative_prompt[: self.MAX_PROMPT_LENGTH]

        # Invoke model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )

        # Parse response
        response_body = json.loads(response["body"].read())

        # Check for content filter / errors
        finish_reasons = response_body.get("finish_reasons", [])
        if finish_reasons and finish_reasons[0] is not None:
            reason = finish_reasons[0]
            if "filter" in reason.lower() or "blocked" in reason.lower():
                raise ContentModerationError(reason)
            raise GenerationError(f"Generation failed: {reason}")

        # Get image data
        images = response_body.get("images", [])
        if not images:
            raise GenerationError("No images in response")

        # Decode base64 image
        image_b64 = images[0]
        image_bytes = base64.b64decode(image_b64)

        # Get actual seed used
        seeds = response_body.get("seeds", [])
        used_seed = int(seeds[0]) if seeds else seed

        return image_bytes, used_seed
