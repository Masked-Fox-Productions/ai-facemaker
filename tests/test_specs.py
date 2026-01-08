"""Tests for spec dataclasses."""

import pytest

from ai_facegen import CharacterSpec, VariantSpec, WorldSpec


class TestWorldSpec:
    """Tests for WorldSpec."""

    def test_valid_creation(self):
        """Test creating a valid WorldSpec."""
        world = WorldSpec(
            context="Fantasy world",
            style="Painterly",
            negative="blurry",
        )
        assert world.context == "Fantasy world"
        assert world.style == "Painterly"
        assert world.negative == "blurry"

    def test_default_negative(self):
        """Test that negative defaults to empty string."""
        world = WorldSpec(context="Test context", style="Test style")
        assert world.negative == ""

    def test_empty_context_raises(self):
        """Test that empty context raises ValueError."""
        with pytest.raises(ValueError, match="context cannot be empty"):
            WorldSpec(context="", style="Some style")

    def test_whitespace_context_raises(self):
        """Test that whitespace-only context raises ValueError."""
        with pytest.raises(ValueError, match="context cannot be empty"):
            WorldSpec(context="   ", style="Some style")

    def test_empty_style_raises(self):
        """Test that empty style raises ValueError."""
        with pytest.raises(ValueError, match="style cannot be empty"):
            WorldSpec(context="Some context", style="")

    def test_whitespace_style_raises(self):
        """Test that whitespace-only style raises ValueError."""
        with pytest.raises(ValueError, match="style cannot be empty"):
            WorldSpec(context="Some context", style="   ")

    def test_frozen(self):
        """Test that WorldSpec is immutable."""
        world = WorldSpec(context="Test", style="Test")
        with pytest.raises(AttributeError):
            world.context = "Changed"


class TestCharacterSpec:
    """Tests for CharacterSpec."""

    def test_valid_creation(self):
        """Test creating a valid CharacterSpec."""
        char = CharacterSpec(
            name="Hero",
            role="Warrior",
            description="Tall and strong",
        )
        assert char.name == "Hero"
        assert char.role == "Warrior"
        assert char.description == "Tall and strong"

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            CharacterSpec(name="", role="Test", description="Test")

    def test_whitespace_name_raises(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            CharacterSpec(name="   ", role="Test", description="Test")

    def test_frozen(self):
        """Test that CharacterSpec is immutable."""
        char = CharacterSpec(name="Hero", role="Test", description="Test")
        with pytest.raises(AttributeError):
            char.name = "Changed"


class TestVariantSpec:
    """Tests for VariantSpec."""

    def test_valid_creation(self):
        """Test creating a valid VariantSpec."""
        variant = VariantSpec(name="icon", size=64, prompt_frame="Face only")
        assert variant.name == "icon"
        assert variant.size == 64
        assert variant.prompt_frame == "Face only"

    def test_generation_size_always_1024(self):
        """Test that _generation_size is always 1024."""
        variant = VariantSpec(name="icon", size=64, prompt_frame="Test")
        assert variant._generation_size == 1024

        variant2 = VariantSpec(name="full", size=1024, prompt_frame="Test")
        assert variant2._generation_size == 1024

    def test_invalid_size_zero(self):
        """Test that size=0 raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            VariantSpec(name="test", size=0, prompt_frame="Test")

    def test_invalid_size_negative(self):
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            VariantSpec(name="test", size=-1, prompt_frame="Test")

    def test_invalid_size_too_large(self):
        """Test that size > 1024 raises ValueError."""
        with pytest.raises(ValueError, match="size max is 1024"):
            VariantSpec(name="test", size=2048, prompt_frame="Test")

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            VariantSpec(name="", size=64, prompt_frame="Test")

    def test_frozen(self):
        """Test that VariantSpec is immutable."""
        variant = VariantSpec(name="icon", size=64, prompt_frame="Test")
        with pytest.raises(AttributeError):
            variant.size = 128
