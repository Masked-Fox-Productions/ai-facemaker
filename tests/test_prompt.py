"""Tests for prompt composition."""

import pytest

from ai_facegen import CharacterSpec, VariantSpec, WorldSpec
from ai_facegen.prompt import PromptComposer


class TestPromptComposer:
    """Tests for PromptComposer."""

    @pytest.fixture
    def composer(self):
        """Create a PromptComposer instance."""
        return PromptComposer()

    def test_compose_basic(self, composer, sample_world, sample_character, sample_variants):
        """Test basic prompt composition."""
        variant = sample_variants[0]  # icon
        prompt, negative = composer.compose(sample_world, sample_character, variant)

        # Check prompt contains key elements
        assert sample_character.name in prompt
        assert sample_character.role in prompt
        assert sample_character.description in prompt
        assert variant.prompt_frame in prompt
        assert "Style:" in prompt

        # Check negative comes from world
        assert negative == sample_world.negative

    def test_compose_preserves_variant_frame(self, composer):
        """Test that variant frame appears at the start."""
        world = WorldSpec(context="Test world", style="Test style")
        char = CharacterSpec(name="Hero", role="Knight", description="Brave warrior")
        variant = VariantSpec(name="icon", size=64, prompt_frame="UNIQUE_FRAME_TEXT")

        prompt, _ = composer.compose(world, char, variant)
        assert prompt.startswith("UNIQUE_FRAME_TEXT")

    def test_compose_handles_empty_negative(self, composer):
        """Test that empty negative prompt is handled."""
        world = WorldSpec(context="Test world", style="Test style", negative="")
        char = CharacterSpec(name="Hero", role="Knight", description="Test")
        variant = VariantSpec(name="icon", size=64, prompt_frame="Test")

        _, negative = composer.compose(world, char, variant)
        assert negative == ""

    def test_normalize_whitespace(self, composer):
        """Test whitespace normalization."""
        text = "  Multiple   spaces   here  "
        result = composer._normalize_whitespace(text)
        assert "  " not in result
        assert result == "Multiple spaces here"

    def test_normalize_whitespace_preserves_newlines(self, composer):
        """Test that meaningful newlines are preserved."""
        text = "Line one\n\nLine two"
        result = composer._normalize_whitespace(text)
        assert "Line one" in result
        assert "Line two" in result

    def test_summarize_context_short(self, composer):
        """Test that short context is not truncated."""
        context = "Short context here."
        result = composer._summarize_context(context)
        assert result == context

    def test_summarize_context_long(self, composer):
        """Test that long context is truncated."""
        context = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = composer._summarize_context(context, max_sentences=2)
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Fourth" not in result
