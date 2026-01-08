"""Tests for caching implementations."""

import os
import tempfile

import pytest

from ai_facegen import PortraitResult
from ai_facegen.cache import FileCache, MemoryCache, NoOpCache


class TestNoOpCache:
    """Tests for NoOpCache."""

    @pytest.fixture
    def cache(self):
        """Create a NoOpCache instance."""
        return NoOpCache()

    def test_make_key_returns_empty(self, cache):
        """Test that make_key returns empty string."""
        key = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        assert key == ""

    def test_get_returns_none(self, cache):
        """Test that get always returns None."""
        assert cache.get("any_key") is None

    def test_put_does_nothing(self, cache):
        """Test that put doesn't raise errors."""
        result = PortraitResult(name="test", png_bytes=b"test")
        cache.put("key", result)  # Should not raise

    def test_clear_does_nothing(self, cache):
        """Test that clear doesn't raise errors."""
        cache.clear()  # Should not raise


class TestMemoryCache:
    """Tests for MemoryCache."""

    @pytest.fixture
    def cache(self):
        """Create a MemoryCache instance."""
        return MemoryCache(max_size=3)

    @pytest.fixture
    def sample_result(self):
        """Create a sample PortraitResult."""
        return PortraitResult(name="test", png_bytes=b"test_image_data", seed=42)

    def test_make_key_deterministic(self, cache):
        """Test that make_key produces consistent keys."""
        key1 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        key2 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex length

    def test_make_key_different_params(self, cache):
        """Test that different params produce different keys."""
        key1 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        key2 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=43
        )
        assert key1 != key2

    def test_put_and_get(self, cache, sample_result):
        """Test storing and retrieving a result."""
        key = cache.make_key(
            prompt="test", negative="", model="titan", size=64, seed=42
        )
        cache.put(key, sample_result)

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.name == sample_result.name
        assert retrieved.png_bytes == sample_result.png_bytes

    def test_get_missing_returns_none(self, cache):
        """Test that getting a missing key returns None."""
        assert cache.get("nonexistent_key") is None

    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache (max_size=3)
        for i in range(3):
            result = PortraitResult(name=f"test_{i}", png_bytes=f"data_{i}".encode())
            cache.put(f"key_{i}", result)

        # Add one more, should evict key_0
        result = PortraitResult(name="test_new", png_bytes=b"data_new")
        cache.put("key_new", result)

        # key_0 should be evicted
        assert cache.get("key_0") is None
        # Others should still be there
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None
        assert cache.get("key_new") is not None

    def test_lru_access_updates_order(self, cache):
        """Test that accessing a key moves it to end."""
        # Fill cache
        for i in range(3):
            result = PortraitResult(name=f"test_{i}", png_bytes=f"data_{i}".encode())
            cache.put(f"key_{i}", result)

        # Access key_0 to make it recently used
        cache.get("key_0")

        # Add new item, should evict key_1 (now oldest)
        result = PortraitResult(name="test_new", png_bytes=b"data_new")
        cache.put("key_new", result)

        # key_0 should still be there (was accessed recently)
        assert cache.get("key_0") is not None
        # key_1 should be evicted
        assert cache.get("key_1") is None

    def test_clear(self, cache, sample_result):
        """Test clearing the cache."""
        key = cache.make_key(
            prompt="test", negative="", model="titan", size=64, seed=42
        )
        cache.put(key, sample_result)
        assert cache.get(key) is not None

        cache.clear()
        assert cache.get(key) is None


class TestFileCache:
    """Tests for FileCache."""

    @pytest.fixture
    def cache_dir(self):
        """Create a temporary directory for cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, cache_dir):
        """Create a FileCache instance."""
        return FileCache(cache_dir=cache_dir)

    @pytest.fixture
    def sample_result(self):
        """Create a sample PortraitResult."""
        return PortraitResult(name="test", png_bytes=b"test_image_data", seed=42)

    def test_make_key_deterministic(self, cache):
        """Test that make_key produces consistent keys."""
        key1 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        key2 = cache.make_key(
            prompt="test", negative="none", model="titan", size=64, seed=42
        )
        assert key1 == key2

    def test_put_and_get(self, cache, sample_result):
        """Test storing and retrieving a result."""
        key = cache.make_key(
            prompt="test", negative="", model="titan", size=64, seed=42
        )
        cache.put(key, sample_result)

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.name == sample_result.name
        assert retrieved.png_bytes == sample_result.png_bytes
        assert retrieved.seed == sample_result.seed

    def test_get_missing_returns_none(self, cache):
        """Test that getting a missing key returns None."""
        assert cache.get("nonexistent_key") is None

    def test_get_empty_key_returns_none(self, cache):
        """Test that getting an empty key returns None."""
        assert cache.get("") is None

    def test_put_empty_key_ignored(self, cache, sample_result):
        """Test that putting with empty key is ignored."""
        cache.put("", sample_result)  # Should not raise

    def test_files_created_on_put(self, cache, cache_dir, sample_result):
        """Test that cache files are created."""
        key = cache.make_key(
            prompt="test", negative="", model="titan", size=64, seed=42
        )
        cache.put(key, sample_result)

        # Check files exist
        subdir = os.path.join(cache_dir, key[:2])
        assert os.path.exists(os.path.join(subdir, f"{key}.json"))
        assert os.path.exists(os.path.join(subdir, f"{key}.png"))

    def test_clear(self, cache, sample_result, cache_dir):
        """Test clearing the cache."""
        key = cache.make_key(
            prompt="test", negative="", model="titan", size=64, seed=42
        )
        cache.put(key, sample_result)
        assert cache.get(key) is not None

        cache.clear()
        assert cache.get(key) is None

        # Cache dir should still exist but be empty of cache files
        assert os.path.exists(cache_dir)
