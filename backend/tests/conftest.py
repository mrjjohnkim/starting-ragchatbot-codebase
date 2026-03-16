"""
Shared test configuration and fixtures.
Adds the backend directory to sys.path so all backend modules are importable.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# Add both backend/ and backend/tests/ to sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _dir in (BACKEND_DIR, TESTS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


# ─── Pydantic models (mirror app.py) ──────────────────────────────────────────

class _QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class _QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str

class _CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ─── Test app factory ─────────────────────────────────────────────────────────

def _build_test_app(rag) -> FastAPI:
    """
    Create a minimal FastAPI app whose API routes mirror app.py but use an
    injected mock RAGSystem.  No static-file mounting — avoids the missing
    ../frontend directory that would break imports in CI / unit tests.
    """
    _app = FastAPI(title="Test RAG App")

    @_app.post("/api/query", response_model=_QueryResponse)
    async def query_documents(request: _QueryRequest):
        try:
            session_id = request.session_id or rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return _QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/api/courses", response_model=_CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return _CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/")
    async def root():
        return {"status": "ok"}

    return _app


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


@pytest.fixture
def mock_rag_system():
    """Pre-configured RAGSystem mock with sensible default return values."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = (
        "Here is an answer about the course material.",
        [{"text": "Lesson 1 content", "url": "https://example.com/lesson1"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Intro to ML", "Deep Learning", "NLP Fundamentals"],
    }
    return rag


@pytest.fixture
def client(mock_rag_system):
    """TestClient wrapping a test FastAPI app backed by mock_rag_system."""
    app = _build_test_app(mock_rag_system)
    return TestClient(app)
