"""Main client for generating character portraits via AWS Bedrock."""

from collections.abc import Sequence

import boto3
from botocore.config import Config

from .cache import NoOpCache, ResultCache
from .exceptions import ModelNotFoundError
from .models import ModelAdapter, get_model_adapter
from .processing import ImageProcessor
from .prompt import PromptComposer
from .result import PortraitResult
from .specs import CharacterSpec, VariantSpec, WorldSpec


class PortraitClient:
    """Main client for generating character portraits via AWS Bedrock.

    Orchestrates prompt composition, image generation, and post-processing
    to produce consistent character portraits from specs.

    Example:
        >>> from ai_facegen import PortraitClient, WorldSpec, CharacterSpec, VariantSpec
        >>> client = PortraitClient(region_name="us-east-1")
        >>> world = WorldSpec(context="...", style="...", negative="...")
        >>> character = CharacterSpec(name="Hero", role="Knight", description="...")
        >>> variants = [VariantSpec(name="icon", size=64, prompt_frame="...")]
        >>> results = client.generate(world=world, character=character, variants=variants)
        >>> results["icon"].save("hero_icon.png")
    """

    # Supported models with their Bedrock IDs
    SUPPORTED_MODELS = {
        "titan": "amazon.titan-image-generator-v1",
        "sdxl": "stability.stable-diffusion-xl-v1",
        "sd35": "stability.sd3-5-large-v1:0",
    }
    DEFAULT_MODEL = "titan"

    def __init__(
        self,
        region_name: str | None = None,
        model: str = DEFAULT_MODEL,
        cache: ResultCache | None = None,
        boto_config: Config | None = None,
    ):
        """Initialize the portrait client.

        Args:
            region_name: AWS region (None = use default resolution).
            model: Model shortname ("titan", "sdxl", "sd35").
            cache: Optional result cache implementation.
            boto_config: Optional botocore Config for timeout/retry settings.

        Raises:
            ModelNotFoundError: If the specified model is not supported.
        """
        self._region = region_name
        self._model_name = model
        self._cache = cache or NoOpCache()
        self._boto_config = boto_config or Config(
            read_timeout=120, retries={"max_attempts": 3}
        )

        # Lazy-init Bedrock client
        self._bedrock_client = None

        # Validate and get model adapter
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Unknown model '{model}'. Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self._model_id = self.SUPPORTED_MODELS[model]
        self._adapter: ModelAdapter = get_model_adapter(model)

        # Initialize prompt composer and image processor
        self._composer = PromptComposer()
        self._processor = ImageProcessor()

    @property
    def bedrock(self):
        """Lazy-initialize Bedrock runtime client."""
        if self._bedrock_client is None:
            kwargs = {"service_name": "bedrock-runtime"}
            if self._region:
                kwargs["region_name"] = self._region
            if self._boto_config:
                kwargs["config"] = self._boto_config
            self._bedrock_client = boto3.client(**kwargs)
        return self._bedrock_client

    def generate(
        self,
        world: WorldSpec,
        character: CharacterSpec,
        variants: Sequence[VariantSpec],
        seed: int | None = None,
        count: int = 1,
    ) -> dict[str, PortraitResult] | dict[str, list[PortraitResult]]:
        """Generate portrait variants for a character.

        Args:
            world: Shared world/style context.
            character: Character details.
            variants: List of variant specs to generate.
            seed: Optional seed for reproducibility.
            count: Number of images to generate per variant (default 1).
                   When count > 1, returns Dict[str, List[PortraitResult]].

        Returns:
            When count == 1: Dict mapping variant name -> PortraitResult.
            When count > 1: Dict mapping variant name -> List[PortraitResult].
        """
        if count < 1:
            raise ValueError("count must be at least 1")

        if count == 1:
            return self._generate_single(world, character, variants, seed)
        else:
            return self._generate_multiple(world, character, variants, seed, count)

    def _generate_single(
        self,
        world: WorldSpec,
        character: CharacterSpec,
        variants: Sequence[VariantSpec],
        seed: int | None,
    ) -> dict[str, PortraitResult]:
        """Generate a single image per variant."""
        results: dict[str, PortraitResult] = {}

        for variant in variants:
            # Compose the prompt
            prompt, negative = self._composer.compose(world, character, variant)

            # Check cache
            cache_key = self._cache.make_key(
                prompt=prompt,
                negative=negative,
                model=self._model_name,
                size=variant.size,
                seed=seed,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Update name to match variant (cache stores original name)
                results[variant.name] = PortraitResult(
                    name=variant.name,
                    png_bytes=cached.png_bytes,
                    seed=cached.seed,
                )
                continue

            # Generate via Bedrock
            raw_image_bytes, used_seed = self._adapter.generate(
                client=self.bedrock,
                model_id=self._model_id,
                prompt=prompt,
                negative_prompt=negative,
                seed=seed,
                width=variant._generation_size,
                height=variant._generation_size,
            )

            # Process (center crop if needed, then downscale)
            final_bytes = self._processor.process(
                raw_bytes=raw_image_bytes,
                target_size=variant.size,
                source_size=variant._generation_size,
            )

            result = PortraitResult(
                name=variant.name,
                png_bytes=final_bytes,
                seed=used_seed,
            )

            # Cache result
            self._cache.put(cache_key, result)

            results[variant.name] = result

        return results

    def _generate_multiple(
        self,
        world: WorldSpec,
        character: CharacterSpec,
        variants: Sequence[VariantSpec],
        seed: int | None,
        count: int,
    ) -> dict[str, list[PortraitResult]]:
        """Generate multiple images per variant."""
        results: dict[str, list[PortraitResult]] = {}

        for variant in variants:
            variant_results: list[PortraitResult] = []

            # Compose the prompt once
            prompt, negative = self._composer.compose(world, character, variant)

            for i in range(count):
                # Use different seed for each image
                # If seed provided, increment it; otherwise use None (random)
                current_seed = (seed + i) if seed is not None else None

                # Check cache
                cache_key = self._cache.make_key(
                    prompt=prompt,
                    negative=negative,
                    model=self._model_name,
                    size=variant.size,
                    seed=current_seed,
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    variant_results.append(
                        PortraitResult(
                            name=variant.name,
                            png_bytes=cached.png_bytes,
                            seed=cached.seed,
                        )
                    )
                    continue

                # Generate via Bedrock
                raw_image_bytes, used_seed = self._adapter.generate(
                    client=self.bedrock,
                    model_id=self._model_id,
                    prompt=prompt,
                    negative_prompt=negative,
                    seed=current_seed,
                    width=variant._generation_size,
                    height=variant._generation_size,
                )

                # Process
                final_bytes = self._processor.process(
                    raw_bytes=raw_image_bytes,
                    target_size=variant.size,
                    source_size=variant._generation_size,
                )

                result = PortraitResult(
                    name=variant.name,
                    png_bytes=final_bytes,
                    seed=used_seed,
                )

                # Cache result
                self._cache.put(cache_key, result)

                variant_results.append(result)

            results[variant.name] = variant_results

        return results
