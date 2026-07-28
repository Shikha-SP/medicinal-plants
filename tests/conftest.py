"""
conftest.py — runs before all tests.
Mocks native/heavy packages that are not installed in CI.
"""

import os
import sys
from unittest.mock import MagicMock

# ── Environment ───────────────────────────────────────────
os.environ.setdefault('GROQ_API_KEY', 'test_mock_key')

# ── Mock packages with native extensions ─────────────────
# These are not installed in the CI test environment because
# they require GPU drivers, large downloads, or C extensions.
# The actual app code imports them at module level so they
# must be stubbed before any project module is imported.

_mocks = [
    'torch',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    'faiss',
    'chromadb',
    'sentence_transformers',
    'groq',
]

for mod in _mocks:
    sys.modules[mod] = MagicMock()
