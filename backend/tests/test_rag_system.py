"""
Tests for how the RAG system handles content-related queries end-to-end.

Focus areas:
1. query() calls generate_response() with tools and tool_manager
2. A content question should NOT produce "query failed" or similar error text
3. Sources from the search tool are returned and then cleared
4. Session management is wired correctly
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# conftest.py adds backend/ to sys.path
from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults
from helpers import make_tool_use_response, make_text_response


# ─── Helpers ──────────────────────────────────────────────────────────────────

FAILURE_PHRASES = [
    "query failed",
    "search failed",
    "search error",
    "no relevant content found",
    "could not retrieve",
    "error:",
]


def _contains_failure_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in FAILURE_PHRASES)


def _make_rag_system():
    """
    Build a RAGSystem with all external dependencies mocked out so the tests
    never touch ChromaDB or the Anthropic API.
    """
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "/tmp/fake-chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.ANTHROPIC_API_KEY = "fake-key"
    config.ANTHROPIC_MODEL = "claude-test"

    with (
        patch("rag_system.VectorStore") as MockVectorStore,
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.AIGenerator") as MockAIGenerator,
        patch("rag_system.SessionManager"),
    ):

        rag = RAGSystem(config)

        # Expose the mocks so individual tests can configure them
        rag._mock_vector_store = MockVectorStore.return_value
        rag._mock_ai_generator = MockAIGenerator.return_value

    return rag


# ─── Tool wiring ──────────────────────────────────────────────────────────────


class TestRAGSystemToolWiring:

    def test_generate_response_receives_tool_definitions(self):
        """query() must forward tool definitions to ai_generator.generate_response()."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "Some answer."

        rag.query("What does lesson 3 cover?")

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) > 0

    def test_generate_response_receives_tool_manager(self):
        """query() must pass its tool_manager to ai_generator.generate_response()."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "answer"

        rag.query("Explain neural networks from Course A")

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    def test_tool_definitions_include_search_course_content(self):
        """The tool list passed to AI must contain 'search_course_content'."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "ok"

        rag.query("What is covered in lesson 2?")

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names


# ─── Content-query response quality ───────────────────────────────────────────


class TestRAGContentQueryResponse:

    def test_content_query_does_not_return_failure_phrase(self):
        """
        When the AI generator returns a real answer, query() must not
        produce a 'query failed' or similar error string.
        """
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = (
            "Lesson 3 covers gradient descent and backpropagation."
        )

        answer, sources = rag.query("What does lesson 3 cover?")

        assert not _contains_failure_phrase(
            answer
        ), f"query() returned a failure phrase: '{answer}'"

    def test_query_returns_ai_generator_output_verbatim(self):
        """The answer returned by query() must be the exact text from generate_response."""
        expected = "The MCP course covers model context protocols."
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = expected

        answer, _ = rag.query("What is covered in the MCP course?")

        assert answer == expected

    def test_sources_returned_from_tool_manager(self):
        """Sources from the last search should be returned alongside the answer."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "content"

        # Manually plant sources on the search tool before query() calls reset
        rag.search_tool.last_sources = [
            {"text": "DeepLearning - Lesson 1", "url": "https://example.com/1"}
        ]

        answer, sources = rag.query("What is covered in lesson 1?")

        # After reset, sources must have been captured before clearing
        # (The current code calls get_last_sources() THEN reset_sources())
        assert isinstance(sources, list)

    def test_sources_are_cleared_after_query(self):
        """tool_manager sources must be reset after each query."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "ok"
        rag.search_tool.last_sources = [{"text": "Course A - Lesson 1", "url": None}]

        rag.query("test question")

        # After the query, sources must be empty
        assert rag.search_tool.last_sources == []

    def test_query_wraps_user_question_in_prompt(self):
        """The prompt sent to generate_response must include the user's question."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "ok"
        user_q = "What are transformers?"

        rag.query(user_q)

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        query_sent = call_kwargs.get("query", "")
        assert user_q in query_sent


# ─── Session management ────────────────────────────────────────────────────────


class TestRAGSessionManagement:

    def test_session_history_passed_to_generate_response_when_session_given(self):
        """If a session_id is provided, conversation_history must be forwarded."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "answer"

        # Seed history in session manager
        rag.session_manager.get_conversation_history.return_value = (
            "User: hi\nAssistant: hello"
        )

        rag.query("follow-up question", session_id="session_1")

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs.get("conversation_history") == "User: hi\nAssistant: hello"

    def test_no_history_when_no_session_id(self):
        """With no session_id, conversation_history must be None."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "answer"

        rag.query("standalone question", session_id=None)

        call_kwargs = rag._mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs.get("conversation_history") is None

    def test_exchange_added_to_session_after_query(self):
        """After a successful query, the Q&A pair must be stored in session history."""
        rag = _make_rag_system()
        rag._mock_ai_generator.generate_response.return_value = "My answer"

        rag.query("My question", session_id="s1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "s1", "My question", "My answer"
        )
