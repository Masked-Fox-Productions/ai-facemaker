"""Command-line interface for ai-facegen."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from . import CharacterSpec, PortraitClient, VariantSpec, WorldSpec
from .cache import FileCache
from .exceptions import FaceGenError


@click.group()
@click.version_option()
def cli():
    """ai-facegen: Generate consistent character portraits with AWS Bedrock."""
    pass


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for generated images",
)
@click.option(
    "--characters",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to character JSON file or directory of character JSONs",
)
@click.option(
    "--region",
    "-r",
    type=str,
    default=None,
    help="AWS region (default: use AWS default resolution)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["titan", "sdxl", "sd35"]),
    default="titan",
    help="Bedrock model to use",
)
@click.option("--seed", "-s", type=int, default=None, help="Seed for reproducible generation")
@click.option(
    "--count",
    "-n",
    type=int,
    default=1,
    help="Number of images to generate per variant",
)
@click.option("--cache/--no-cache", default=True, help="Enable/disable result caching")
@click.option("--cache-dir", type=click.Path(), default=None, help="Custom cache directory")
def generate(
    config: str,
    output: str,
    characters: Optional[str],
    region: Optional[str],
    model: str,
    seed: Optional[int],
    count: int,
    cache: bool,
    cache_dir: Optional[str],
):
    """Generate portraits from a JSON configuration file.

    CONFIG is a JSON file with world and variants configuration.
    Characters can be in the config file or specified separately via --characters.
    """
    # Load config
    with open(config, "r") as f:
        cfg = json.load(f)

    # Parse world spec
    world_cfg = cfg.get("world", {})
    world = WorldSpec(
        context=world_cfg.get("context", ""),
        style=world_cfg.get("style", ""),
        negative=world_cfg.get("negative", ""),
    )

    # Parse variants
    variants_cfg = cfg.get("variants", [])
    variants = [
        VariantSpec(
            name=v["name"],
            size=v.get("size", 256),
            prompt_frame=v.get("prompt_frame", ""),
        )
        for v in variants_cfg
    ]

    if not variants:
        click.echo("Error: No variants specified in config", err=True)
        sys.exit(1)

    # Load characters
    characters_list = []

    # Characters from config file
    for char_cfg in cfg.get("characters", []):
        characters_list.append(
            CharacterSpec(
                name=char_cfg["name"],
                role=char_cfg.get("role", ""),
                description=char_cfg.get("description", ""),
            )
        )

    # Characters from --characters option
    if characters:
        char_path = Path(characters)
        if char_path.is_file():
            # Single character file
            with open(char_path, "r") as f:
                char_cfg = json.load(f)
            characters_list.append(
                CharacterSpec(
                    name=char_cfg["name"],
                    role=char_cfg.get("role", ""),
                    description=char_cfg.get("description", ""),
                )
            )
        elif char_path.is_dir():
            # Directory of character files
            for json_file in sorted(char_path.glob("*.json")):
                with open(json_file, "r") as f:
                    char_cfg = json.load(f)
                characters_list.append(
                    CharacterSpec(
                        name=char_cfg["name"],
                        role=char_cfg.get("role", ""),
                        description=char_cfg.get("description", ""),
                    )
                )

    if not characters_list:
        click.echo("Error: No characters specified", err=True)
        sys.exit(1)

    # Setup cache
    cache_impl = FileCache(cache_dir) if cache else None

    # Create client
    client = PortraitClient(
        region_name=region,
        model=model,
        cache=cache_impl,
    )

    # Output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each character
    total_images = 0
    for character in characters_list:
        click.echo(f"Generating portraits for {character.name}...")

        try:
            results = client.generate(
                world=world,
                character=character,
                variants=variants,
                seed=seed,
                count=count,
            )

            # Save results
            safe_name = character.name.lower().replace(" ", "_")

            if count == 1:
                # Single image per variant
                for variant_name, result in results.items():
                    filename = f"{safe_name}_{variant_name}.png"
                    filepath = output_dir / filename
                    result.save(str(filepath))
                    click.echo(f"  Saved: {filepath}")
                    total_images += 1
            else:
                # Multiple images per variant
                for variant_name, result_list in results.items():
                    for i, result in enumerate(result_list):
                        filename = f"{safe_name}_{variant_name}_{i + 1}.png"
                        filepath = output_dir / filename
                        result.save(str(filepath))
                        click.echo(f"  Saved: {filepath}")
                        total_images += 1

        except FaceGenError as e:
            click.echo(f"  Error: {e}", err=True)
            continue

    click.echo(f"Done! Generated {total_images} images.")


@cli.command()
@click.option("--region", "-r", type=str, default=None, help="AWS region to test")
def test_credentials(region: Optional[str]):
    """Test AWS credentials and Bedrock access."""
    import boto3

    try:
        kwargs = {"service_name": "bedrock"}
        if region:
            kwargs["region_name"] = region

        client = boto3.client(**kwargs)

        # List foundation models to test access
        response = client.list_foundation_models(byOutputModality="IMAGE")

        models = response.get("modelSummaries", [])
        click.echo(f"Found {len(models)} image generation models:")
        for m in models[:10]:  # Show first 10
            click.echo(f"  - {m['modelId']}")

        click.echo("\nCredentials OK!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--cache-dir", type=click.Path(), default=None, help="Cache directory to clear")
def clear_cache(cache_dir: Optional[str]):
    """Clear the result cache."""
    cache = FileCache(cache_dir)
    cache.clear()
    click.echo("Cache cleared.")


@cli.command()
def init_config():
    """Generate a sample configuration file."""
    sample = {
        "world": {
            "context": "A fantasy medieval kingdom with magic and dragons.",
            "style": "Painterly fantasy art, rich colors, dramatic lighting.",
            "negative": "text, watermark, blurry, modern elements",
        },
        "characters": [
            {
                "name": "Elena Stormblade",
                "role": "Knight Captain",
                "description": "Stern woman in plate armor, silver hair, battle scars, determined expression.",
            },
            {
                "name": "Finn Quickfingers",
                "role": "Thief",
                "description": "Young man with mischievous grin, hooded cloak, daggers at belt.",
            },
        ],
        "variants": [
            {
                "name": "icon",
                "size": 64,
                "prompt_frame": "Small square icon, face closeup, bold silhouette, readable at tiny size.",
            },
            {
                "name": "bust",
                "size": 256,
                "prompt_frame": "Portrait from shoulders up, neutral background, clear details.",
            },
            {
                "name": "full",
                "size": 1024,
                "prompt_frame": "Full body portrait, character in environment, detailed.",
            },
        ],
    }

    click.echo(json.dumps(sample, indent=2))


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
