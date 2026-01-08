"""Caching implementations for generated portraits."""

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

from .result import PortraitResult


class ResultCache(ABC):
    """Abstract base class for result caching."""

    @abstractmethod
    def make_key(
        self,
        prompt: str,
        negative: str,
        model: str,
        size: int,
        seed: int | None,
    ) -> str:
        """Generate a cache key from generation parameters.

        Args:
            prompt: The generation prompt.
            negative: The negative prompt.
            model: The model name.
            size: The target size.
            seed: The generation seed.

        Returns:
            A unique cache key string.
        """
        pass

    @abstractmethod
    def get(self, key: str) -> PortraitResult | None:
        """Retrieve a cached result.

        Args:
            key: The cache key.

        Returns:
            The cached PortraitResult, or None if not found.
        """
        pass

    @abstractmethod
    def put(self, key: str, result: PortraitResult) -> None:
        """Store a result in the cache.

        Args:
            key: The cache key.
            result: The result to cache.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached results."""
        pass


class NoOpCache(ResultCache):
    """No-op cache that doesn't actually cache anything.

    Useful as a default when caching is disabled.
    """

    def make_key(
        self,
        prompt: str,
        negative: str,
        model: str,
        size: int,
        seed: int | None,
    ) -> str:
        return ""

    def get(self, key: str) -> PortraitResult | None:
        return None

    def put(self, key: str, result: PortraitResult) -> None:
        pass

    def clear(self) -> None:
        pass


class FileCache(ResultCache):
    """File-based cache that stores results on disk.

    Structure:
        cache_dir/
            ab/
                abcd1234...json  (metadata)
                abcd1234...png   (image)
    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize file cache.

        Args:
            cache_dir: Directory for cache files.
                       Defaults to ~/.ai_facegen_cache
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.ai_facegen_cache")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        prompt: str,
        negative: str,
        model: str,
        size: int,
        seed: int | None,
    ) -> str:
        """Generate a deterministic hash key from parameters."""
        key_data = {
            "prompt": prompt,
            "negative": negative,
            "model": model,
            "size": size,
            "seed": seed,
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Get paths for metadata and image files."""
        # Use first 2 chars as subdirectory for better file distribution
        subdir = self._cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return (
            subdir / f"{key}.json",
            subdir / f"{key}.png",
        )

    def get(self, key: str) -> PortraitResult | None:
        """Retrieve a cached result."""
        if not key:
            return None

        meta_path, img_path = self._get_paths(key)

        if not meta_path.exists() or not img_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            with open(img_path, "rb") as f:
                png_bytes = f.read()

            return PortraitResult(
                name=meta["name"],
                png_bytes=png_bytes,
                seed=meta.get("seed"),
            )
        except (OSError, json.JSONDecodeError, KeyError):
            # Cache corrupted, return None
            return None

    def put(self, key: str, result: PortraitResult) -> None:
        """Store a result in the cache."""
        if not key:
            return

        meta_path, img_path = self._get_paths(key)

        meta = {
            "name": result.name,
            "seed": result.seed,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with open(img_path, "wb") as f:
            f.write(result.png_bytes)

    def clear(self) -> None:
        """Clear all cached files."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)


class MemoryCache(ResultCache):
    """In-memory cache for short-lived sessions.

    Uses LRU eviction when the cache reaches max_size.
    Useful for batch processing where results may be reused.
    """

    def __init__(self, max_size: int = 100):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of results to cache (LRU eviction).
        """
        self._cache: OrderedDict[str, PortraitResult] = OrderedDict()
        self._max_size = max_size

    def make_key(
        self,
        prompt: str,
        negative: str,
        model: str,
        size: int,
        seed: int | None,
    ) -> str:
        """Generate a hash key from parameters."""
        key_data = f"{prompt}|{negative}|{model}|{size}|{seed}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> PortraitResult | None:
        """Retrieve from cache with LRU update."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, result: PortraitResult) -> None:
        """Store in cache with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # Remove oldest
                self._cache.popitem(last=False)
            self._cache[key] = result

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
