# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ai-facemaker (package name: `ai-facegen`) is a Python library for generating consistent character profile portraits using Amazon Bedrock image models. It combines shared world/style context with per-character prompts to produce portrait variants at different sizes (64×64, 256×256, 1024×1024).

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ai_facegen --cov-report=term-missing

# Run single test file
pytest tests/test_specs.py

# Run single test
pytest tests/test_specs.py::TestWorldSpec::test_valid_creation

# Lint
ruff check src/ tests/

# Type check
mypy src/ai_facegen/ --ignore-missing-imports

# CLI usage
ai-facegen init-config > config.json
ai-facegen generate config.json -o ./output
ai-facegen test-credentials --region us-east-1
```

## Architecture

The library uses a spec-based configuration pattern:

```
src/ai_facegen/
├── specs.py          # WorldSpec, CharacterSpec, VariantSpec (frozen dataclasses)
├── client.py         # PortraitClient - main orchestrator
├── result.py         # PortraitResult - container with png_bytes and PIL Image
├── prompt.py         # PromptComposer - combines specs into prompts
├── processing.py     # ImageProcessor - crop and LANCZOS downscale
├── cache.py          # FileCache, MemoryCache, NoOpCache
├── cli.py            # Click-based CLI
└── models/
    ├── base.py       # ModelAdapter ABC
    ├── titan.py      # Amazon Titan adapter
    ├── sdxl.py       # Stability SDXL adapter
    └── sd35.py       # Stability SD3.5 adapter
```

**Generation flow**: PortraitClient → PromptComposer → ModelAdapter → ImageProcessor → PortraitResult

## Key Implementation Details

- Images always generated at 1024×1024, then center-cropped and downscaled with LANCZOS for crisp output
- Titan has 512-char prompt limit; SD3.5 has 10k-char limit
- Cache keys are SHA-256 hashes of (prompt, negative, model, size, seed)
- Bedrock client is lazy-initialized (not created until first generate call)

## AWS Integration

Uses standard AWS credential resolution (environment variables, `~/.aws/credentials`, AWS SSO profiles). Requires `bedrock:InvokeModel` permission.

## Naming Conventions

- Package import: `ai_facegen` (underscore)
- Repository/CLI: `ai-facemaker` (hyphen)
- Classes: PascalCase
- Variants: string identifiers ("icon", "bust", "full")
