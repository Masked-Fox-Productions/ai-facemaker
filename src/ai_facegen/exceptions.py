"""Custom exceptions for ai-facegen."""


class FaceGenError(Exception):
    """Base exception for ai-facegen errors."""

    pass


class GenerationError(FaceGenError):
    """Error during image generation."""

    pass


class ContentModerationError(FaceGenError):
    """Content was blocked by moderation filters."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Content blocked: {reason}")


class ModelNotFoundError(FaceGenError):
    """Requested model is not available."""

    pass


class CacheError(FaceGenError):
    """Error reading/writing cache."""

    pass


class InvalidSpecError(FaceGenError):
    """Invalid specification provided."""

    pass
