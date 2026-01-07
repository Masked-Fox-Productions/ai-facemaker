# ai-facemaker

Generate consistent character profile portraits with Amazon Bedrock image models — from a shared **world/style context** + per-character prompts — using **local AWS credentials**.

> Perfect for game casts: define your setting once (art direction + world lore), then generate a whole roster of consistent portraits at sizes like **64×64**, **256×256**, and **1024×1024**.

---

## ✨ What it does

`ai-facegen` is an OSS Python library (and optional CLI) that:

- Uses **standard AWS credential resolution** (env vars, `~/.aws/credentials`, AWS SSO profiles) — no hosted service, no secrets shipped
- Builds consistent prompts from:
  - **World context** (setting + visual style rules)
  - **Character context** (name, role/faction, traits, etc.)
  - **Variant framing** (icon vs bust vs full body, etc.)
- Generates one or more **variants** per character (e.g. `icon`, `bust`, `full`)
- Accepts **output size parameters** (e.g. `64×64`) and handles **crop + downscale** for crisp avatars
- Returns images as **bytes** and/or **PIL Images**, and can write PNGs to disk
- Supports **seeded generation** for repeatability (when supported by the chosen model)
- (Optional) caches results to avoid regenerating the same prompt over and over

---

## Use cases

- RPG / visual novel portraits
- “Cast headshots” for a wiki or codex
- NPC icons for inventory/dialog UI
- Batch generation pipelines for games (CI builds, asset pipelines)

---

## Quick example

```python
from ai_facegen import PortraitClient, WorldSpec, CharacterSpec, VariantSpec

world = WorldSpec(
    context="""
Moon Hard Lemonade is a far-future sci-fi cartoon adventure.
Humans joined an alien alliance because they love Moon Hard Lemonade.
The light side of the Moon is a huge industrial manufacturing hub.
""",
    style="""
Clean, readable sci-fi cartoon portraits. Crisp linework, bold shapes, simple shading.
Friendly character design. Consistent cast style. Uncluttered backgrounds.
""",
    negative="text, watermark, logo, signature, blurry, low-res"
)

character = CharacterSpec(
    name="Tessa Quark",
    role="Digger",
    description="Optimistic smuggler-mechanic, mischievous grin, short hair, grease-stained gloves."
)

variants = [
    VariantSpec(name="icon", size=64, prompt_frame="Centered face icon, readable at tiny size, bold silhouette."),
    VariantSpec(name="bust", size=256, prompt_frame="Head and shoulders portrait, light background."),
]

client = PortraitClient(region_name="us-east-1")  # or omit to use AWS defaults
results = client.generate(world=world, character=character, variants=variants)

# Save the 64×64 icon
with open("tessa_icon.png", "wb") as f:
    f.write(results["icon"].png_bytes)
