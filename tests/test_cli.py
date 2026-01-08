"""Tests for CLI."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ai_facegen.cli import cli


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config(self):
        """Create a sample config dict."""
        return {
            "world": {
                "context": "Test world context.",
                "style": "Test style.",
                "negative": "blurry",
            },
            "characters": [
                {
                    "name": "Test Hero",
                    "role": "Knight",
                    "description": "Brave warrior with a sword.",
                }
            ],
            "variants": [
                {
                    "name": "icon",
                    "size": 64,
                    "prompt_frame": "Small icon.",
                }
            ],
        }

    @pytest.fixture
    def config_file(self, sample_config):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_init_config(self, runner):
        """Test init-config command."""
        result = runner.invoke(cli, ["init-config"])
        assert result.exit_code == 0

        # Output should be valid JSON
        config = json.loads(result.output)
        assert "world" in config
        assert "characters" in config
        assert "variants" in config

    def test_clear_cache(self, runner):
        """Test clear-cache command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["clear-cache", "--cache-dir", tmpdir])
            assert result.exit_code == 0
            assert "cleared" in result.output.lower()

    def test_generate_no_variants(self, runner):
        """Test generate with empty variants."""
        config = {
            "world": {"context": "Test", "style": "Test"},
            "characters": [{"name": "Test", "role": "Test", "description": "Test"}],
            "variants": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            result = runner.invoke(cli, ["generate", config_path])
            assert result.exit_code != 0
            assert "no variants" in result.output.lower()
        finally:
            os.unlink(config_path)

    def test_generate_no_characters(self, runner):
        """Test generate with no characters."""
        config = {
            "world": {"context": "Test", "style": "Test"},
            "characters": [],
            "variants": [{"name": "icon", "size": 64, "prompt_frame": "Test"}],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            result = runner.invoke(cli, ["generate", config_path])
            assert result.exit_code != 0
            assert "no characters" in result.output.lower()
        finally:
            os.unlink(config_path)

    def test_generate_with_mock_client(self, runner, config_file, mock_image_bytes):
        """Test generate command with mocked Bedrock."""
        import base64

        mock_response = {"images": [base64.b64encode(mock_image_bytes).decode()]}

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(mock_response).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("boto3.client", return_value=mock_client):
                result = runner.invoke(
                    cli,
                    [
                        "generate",
                        config_file,
                        "-o",
                        output_dir,
                        "--no-cache",
                    ],
                )

                assert result.exit_code == 0
                assert "done" in result.output.lower()

                # Check that files were created
                files = os.listdir(output_dir)
                assert len(files) > 0
                assert any(f.endswith(".png") for f in files)

    def test_generate_with_characters_file(
        self, runner, mock_image_bytes
    ):
        """Test generate with separate characters file."""
        import base64

        # Create config without characters
        config = {
            "world": {"context": "Test", "style": "Test"},
            "variants": [{"name": "icon", "size": 64, "prompt_frame": "Test"}],
        }

        # Create character file
        char = {"name": "External Char", "role": "Test", "description": "Test desc"}

        mock_response = {"images": [base64.b64encode(mock_image_bytes).decode()]}
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(mock_response).encode()
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            char_path = os.path.join(tmpdir, "char.json")
            output_dir = os.path.join(tmpdir, "output")

            with open(config_path, "w") as f:
                json.dump(config, f)
            with open(char_path, "w") as f:
                json.dump(char, f)

            with patch("boto3.client", return_value=mock_client):
                result = runner.invoke(
                    cli,
                    [
                        "generate",
                        config_path,
                        "-c",
                        char_path,
                        "-o",
                        output_dir,
                        "--no-cache",
                    ],
                )

                assert result.exit_code == 0
                assert "external_char" in result.output.lower()
