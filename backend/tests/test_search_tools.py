"""
Tests for CourseSearchTool.execute() and ToolManager.

Focus areas:
1. CourseSearchTool.execute() output for various VectorStore responses
2. Error and empty-result paths
3. Result formatting (source tracking, lesson links, headers)
4. ToolManager routing and source management
"""

import pytest
from unittest.mock import MagicMock, patch

# conftest.py adds backend/ to sys.path
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_search_results(docs, metas, distances=None):
    """Build a SearchResults with the given documents and metadata lists."""
    if distances is None:
        distances = [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=distances)


def _make_error_results(msg):
    return SearchResults.empty(msg)


# ─── CourseSearchTool.execute() ───────────────────────────────────────────────


class TestCourseSearchToolExecute:

    def test_returns_formatted_string_for_valid_results(self, mock_vector_store):
        """execute() with real results must return a non-empty formatted string."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["Neural networks are..."],
            metas=[{"course_title": "Deep Learning", "lesson_number": 2}],
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="neural networks")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Deep Learning" in result
        assert "Lesson 2" in result
        assert "Neural networks are" in result

    def test_returns_error_string_when_store_returns_error(self, mock_vector_store):
        """execute() must propagate the error message from SearchResults.error."""
        mock_vector_store.search.return_value = _make_error_results(
            "No course found matching 'XYZ'"
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="neural networks", course_name="XYZ")

        assert "No course found matching" in result

    def test_returns_no_results_message_for_empty_results(self, mock_vector_store):
        """execute() with empty (but not error) results should say no content found."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="quantum computing")

        assert "No relevant content found" in result

    def test_empty_results_with_course_filter_mentions_course(self, mock_vector_store):
        """Empty results with course_name should include the course in the message."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="something", course_name="Python Basics")

        assert "Python Basics" in result

    def test_empty_results_with_lesson_filter_mentions_lesson(self, mock_vector_store):
        """Empty results with lesson_number should include the lesson in the message."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="something", lesson_number=3)

        assert "lesson 3" in result.lower()

    def test_passes_query_to_store_search(self, mock_vector_store):
        """execute() must pass its query argument through to VectorStore.search()."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="backpropagation")

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args
        assert (
            call_kwargs.kwargs.get("query") == "backpropagation"
            or call_kwargs.args[0] == "backpropagation"
        )

    def test_passes_course_name_to_store_search(self, mock_vector_store):
        """execute() must pass course_name through to VectorStore.search()."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="x", course_name="MCP Course")

        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs.get("course_name") == "MCP Course"

    def test_passes_lesson_number_to_store_search(self, mock_vector_store):
        """execute() must pass lesson_number through to VectorStore.search()."""
        mock_vector_store.search.return_value = _make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="x", lesson_number=5)

        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs.get("lesson_number") == 5

    def test_multiple_results_are_separated(self, mock_vector_store):
        """Multiple results should appear in the output string."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["Content A", "Content B"],
            metas=[
                {"course_title": "Course 1", "lesson_number": 1},
                {"course_title": "Course 1", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="something")

        assert "Content A" in result
        assert "Content B" in result

    def test_result_without_lesson_number_in_metadata(self, mock_vector_store):
        """Chunks without a lesson_number in metadata should still format correctly."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["Intro content"],
            metas=[{"course_title": "Intro Course"}],  # no lesson_number key
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="intro")

        assert "Intro Course" in result
        assert "Intro content" in result

    def test_stores_last_sources_after_successful_search(self, mock_vector_store):
        """After execute(), last_sources should contain one entry per result."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["Doc A", "Doc B"],
            metas=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course X - Lesson 1"

    def test_last_sources_empty_on_error(self, mock_vector_store):
        """An error result must not populate last_sources with stale data."""
        mock_vector_store.search.return_value = _make_error_results(
            "Search error: boom"
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        # last_sources should remain at its default (empty list)
        assert tool.last_sources == []

    def test_store_search_exception_is_not_raised(self, mock_vector_store):
        """If the store raises an exception, execute() should not propagate it."""
        mock_vector_store.search.side_effect = Exception("DB connection lost")
        tool = CourseSearchTool(mock_vector_store)
        # Should NOT raise
        try:
            result = tool.execute(query="something")
            # If the exception leaks, this test records the failure explicitly
        except Exception as exc:
            pytest.fail(
                f"CourseSearchTool.execute() leaked an exception from the store: {exc}"
            )


# ─── ToolManager ──────────────────────────────────────────────────────────────


class TestToolManager:

    def test_register_and_execute_search_tool(self, mock_vector_store):
        """ToolManager should route 'search_course_content' to CourseSearchTool."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["Result text"],
            metas=[{"course_title": "TestCourse", "lesson_number": 1}],
        )
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))

        result = manager.execute_tool("search_course_content", query="something")
        assert "Result text" in result

    def test_execute_unknown_tool_returns_error_message(self, mock_vector_store):
        """ToolManager should return an error string for unknown tool names."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="x")
        assert "not found" in result.lower()

    def test_get_tool_definitions_returns_list(self, mock_vector_store):
        """get_tool_definitions() should return a list with correct tool names."""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        defs = manager.get_tool_definitions()
        assert isinstance(defs, list)
        assert any(d["name"] == "search_course_content" for d in defs)

    def test_get_last_sources_returns_sources_from_search_tool(self, mock_vector_store):
        """get_last_sources() should expose sources set by CourseSearchTool."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["text"], metas=[{"course_title": "MyCourse", "lesson_number": 1}]
        )
        mock_vector_store.get_lesson_link.return_value = None
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        manager.execute_tool("search_course_content", query="x")
        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "MyCourse - Lesson 1"

    def test_reset_sources_clears_all_tool_sources(self, mock_vector_store):
        """reset_sources() should clear last_sources on all tools that track them."""
        mock_vector_store.search.return_value = _make_search_results(
            docs=["text"], metas=[{"course_title": "C", "lesson_number": 1}]
        )
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.execute_tool("search_course_content", query="x")

        assert len(manager.get_last_sources()) == 1

        manager.reset_sources()
        assert manager.get_last_sources() == []
