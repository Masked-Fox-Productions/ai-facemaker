"""Abstract base class for Bedrock model adapters."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any


class ModelAdapter(ABC):
    """Abstract base class for Bedrock model adapters.

    Each adapter handles the specifics of invoking a particular image
    generation model through AWS Bedrock, including request formatting,
    response parsing, and error handling.
    """

    @abstractmethod
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
        """Generate an image via the model.

        Args:
            client: boto3 bedrock-runtime client.
            model_id: The Bedrock model ID to invoke.
            prompt: The positive prompt describing what to generate.
            negative_prompt: Elements to avoid in generation.
            seed: Optional seed for reproducibility.
            width: Desired image width in pixels.
            height: Desired image height in pixels.

        Returns:
            Tuple of (image_bytes, seed_used) where image_bytes is the
            raw image data (PNG or JPEG) and seed_used is the actual
            seed used for generation (for reproducibility).

        Raises:
            GenerationError: If generation fails.
            ContentModerationError: If content is blocked by filters.
        """
        pass

    @property
    @abstractmethod
    def max_prompt_length(self) -> int:
        """Maximum prompt length in characters."""
        pass

    @property
    @abstractmethod
    def supported_sizes(self) -> List[int]:
        """List of supported image sizes."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The default Bedrock model ID for this adapter."""
        pass
