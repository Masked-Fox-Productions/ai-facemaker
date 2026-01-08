"""Prompt composition for character portrait generation."""


from .specs import CharacterSpec, VariantSpec, WorldSpec


class PromptComposer:
    """Composes effective prompts from specs for portrait generation.

    Combines world context, character details, and variant framing into
    a coherent prompt that produces consistent, high-quality portraits.
    """

    PROMPT_TEMPLATE = """{variant_frame}

Character: {character_name}, {character_role}.
{character_description}

Setting: {world_context}

Style: {world_style}"""

    def compose(
        self,
        world: WorldSpec,
        character: CharacterSpec,
        variant: VariantSpec,
    ) -> tuple[str, str]:
        """Compose a prompt and negative prompt from specs.

        Args:
            world: The shared world/style context.
            character: The character to generate.
            variant: The output variant configuration.

        Returns:
            Tuple of (prompt, negative_prompt).
        """
        # Build main prompt
        prompt = self.PROMPT_TEMPLATE.format(
            variant_frame=variant.prompt_frame.strip(),
            character_name=character.name.strip(),
            character_role=character.role.strip(),
            character_description=character.description.strip(),
            world_context=self._summarize_context(world.context),
            world_style=world.style.strip(),
        )

        # Clean up whitespace
        prompt = self._normalize_whitespace(prompt)

        # Negative prompt comes directly from WorldSpec
        negative = world.negative.strip()

        return prompt, negative

    def _summarize_context(self, context: str, max_sentences: int = 3) -> str:
        """Summarize context to keep prompts concise.

        For very long contexts, takes the first few sentences to stay
        within prompt length limits while preserving key information.

        Args:
            context: The full context string.
            max_sentences: Maximum sentences to include.

        Returns:
            Summarized context string.
        """
        context = context.strip()
        # Split on sentence boundaries
        sentences = []
        current = ""
        for char in context:
            current += char
            if char in ".!?" and current.strip():
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        if len(sentences) <= max_sentences:
            return context

        return " ".join(sentences[:max_sentences])

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure.

        Args:
            text: Input text with potential excess whitespace.

        Returns:
            Cleaned text with normalized whitespace.
        """
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces within line
            cleaned = " ".join(line.split())
            if cleaned:
                cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines)
