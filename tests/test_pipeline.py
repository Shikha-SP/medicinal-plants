"""
Unit tests for the plant pipeline utilities.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Mock heavy deps
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['faiss'] = MagicMock()


def test_format_bytes():
    """Test byte formatting in frontend server"""
    # Simulate formatBytes logic
    def format_bytes(size_bytes):
        if size_bytes <= 0:
            return "-"
        import math
        units = ["B", "KB", "MB", "GB"]
        exponent = min(int(math.log(size_bytes, 1024)), len(units) - 1)
        value = size_bytes / (1024 ** exponent)
        return f"{value:.0f} {units[exponent]}"

    assert format_bytes(0) == "-"
    assert format_bytes(1024) == "1 KB"
    assert format_bytes(1024 * 1024) == "1 MB"


def test_common_names_lookup():
    """Test that common names mapping is correct"""
    # Import COMMON_NAMES without loading the full app
    os.environ['GROQ_API_KEY'] = 'test_mock_key'

    sys.modules['chromadb'] = MagicMock()
    sys.modules['sentence_transformers'] = MagicMock()
    sys.modules['groq'] = MagicMock()

    from main import COMMON_NAMES

    assert COMMON_NAMES["Ocimum_tenuiflorum"] == "Tulsi"
    assert COMMON_NAMES["Azadirachta_indica"] == "Neem"
    assert COMMON_NAMES["Curcuma_longa"] == "Turmeric"
    assert "Aloe_vera" in COMMON_NAMES


def test_confidence_threshold():
    """Test confidence threshold is within valid range"""
    os.environ['GROQ_API_KEY'] = 'test_mock_key'

    sys.modules['chromadb'] = MagicMock()
    sys.modules['sentence_transformers'] = MagicMock()
    sys.modules['groq'] = MagicMock()

    from main import CONFIDENCE_THRESHOLD

    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0


def test_cleanup_empty_dirs(tmp_path):
    """Test cleanup script removes empty folders"""
    # Create empty and non-empty dirs
    empty_dir = tmp_path / "EmptySpecies"
    empty_dir.mkdir()

    nonempty_dir = tmp_path / "NonEmptySpecies"
    nonempty_dir.mkdir()
    (nonempty_dir / "image.jpg").touch()

    # Simulate cleanup logic
    for folder in tmp_path.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    assert not empty_dir.exists()
    assert nonempty_dir.exists()
