"""
Shared test configuration and fixtures.
Adds the backend directory to sys.path so all backend modules are importable.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Add both backend/ and backend/tests/ to sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _dir in (BACKEND_DIR, TESTS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store():
    """A MagicMock that stands in for VectorStore."""
    store = MagicMock()
    store.get_lesson_link.return_value = None
    return store


@pytest.fixture
def mock_anthropic_client():
    """A MagicMock that stands in for anthropic.Anthropic()."""
    return MagicMock()
