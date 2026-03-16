"""
Tests for the FastAPI API endpoints: POST /api/query, GET /api/courses, GET /.

A test-specific FastAPI app (built in conftest.py) is used so that the
static-file mount in the real app.py (which requires ../frontend to exist)
does not break the test environment.
"""
import pytest


# ─── POST /api/query ──────────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_returns_200_with_answer(self, client, mock_rag_system):
        """Successful query returns HTTP 200 and an answer string."""
        response = client.post("/api/query", json={"query": "What is deep learning?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Here is an answer about the course material."

    def test_returns_sources_list(self, client, mock_rag_system):
        """Response must include a non-empty sources list from the RAG system."""
        response = client.post("/api/query", json={"query": "Explain backpropagation."})

        data = response.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_returns_session_id_when_none_provided(self, client, mock_rag_system):
        """When no session_id is supplied, a new one is returned in the response."""
        response = client.post("/api/query", json={"query": "What is a transformer?"})

        data = response.json()
        assert data["session_id"] == "test-session-id"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client, mock_rag_system):
        """When a session_id is provided it is forwarded to rag.query() and returned."""
        response = client.post(
            "/api/query",
            json={"query": "Follow-up question.", "session_id": "existing-session"},
        )

        data = response.json()
        assert data["session_id"] == "existing-session"
        # create_session must NOT be called — we already have one
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_correct_args(self, client, mock_rag_system):
        """rag.query() must receive the exact query text and session_id."""
        client.post(
            "/api/query",
            json={"query": "What topics are covered?", "session_id": "sess-42"},
        )

        mock_rag_system.query.assert_called_once_with("What topics are covered?", "sess-42")

    def test_returns_500_when_rag_raises(self, client, mock_rag_system):
        """If rag.query() raises an exception the endpoint returns HTTP 500."""
        mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "Will this fail?"})

        assert response.status_code == 500
        assert "vector store unavailable" in response.json()["detail"]

    def test_missing_query_field_returns_422(self, client):
        """Omitting the required 'query' field yields a 422 Unprocessable Entity."""
        response = client.post("/api/query", json={"session_id": "s1"})

        assert response.status_code == 422

    def test_empty_sources_list_is_valid(self, client, mock_rag_system):
        """An empty sources list is a valid response (no relevant chunks found)."""
        mock_rag_system.query.return_value = ("No relevant content found.", [])

        response = client.post("/api/query", json={"query": "Obscure topic."})

        assert response.status_code == 200
        assert response.json()["sources"] == []


# ─── GET /api/courses ─────────────────────────────────────────────────────────

class TestCoursesEndpoint:

    def test_returns_200(self, client):
        """GET /api/courses returns HTTP 200."""
        response = client.get("/api/courses")

        assert response.status_code == 200

    def test_returns_correct_course_count(self, client, mock_rag_system):
        """total_courses must match what get_course_analytics() reports."""
        response = client.get("/api/courses")

        assert response.json()["total_courses"] == 3

    def test_returns_course_titles(self, client, mock_rag_system):
        """course_titles must be the list returned by get_course_analytics()."""
        response = client.get("/api/courses")

        titles = response.json()["course_titles"]
        assert titles == ["Intro to ML", "Deep Learning", "NLP Fundamentals"]

    def test_calls_get_course_analytics(self, client, mock_rag_system):
        """The endpoint must delegate to rag.get_course_analytics()."""
        client.get("/api/courses")

        mock_rag_system.get_course_analytics.assert_called_once()

    def test_returns_500_when_analytics_raises(self, client, mock_rag_system):
        """If get_course_analytics() raises, the endpoint returns HTTP 500."""
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "db error" in response.json()["detail"]

    def test_zero_courses_is_valid(self, client, mock_rag_system):
        """An empty catalog (0 courses, empty list) is a valid response."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


# ─── GET / ────────────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        """GET / on the test app returns HTTP 200."""
        response = client.get("/")

        assert response.status_code == 200
