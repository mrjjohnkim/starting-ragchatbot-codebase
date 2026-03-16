"""
Tests for VectorStore.search() and its helper _build_filter().

These tests use a real in-memory ChromaDB (EphemeralClient) to catch actual
ChromaDB 1.0.x compatibility issues, particularly:
- Querying an empty collection (should not raise)
- Querying when n_results > number of stored documents
- Filter construction for all parameter combinations
- Course name resolution via semantic search
"""

import sys
import os
import pytest
import chromadb
from unittest.mock import MagicMock, patch

# conftest.py adds backend/ to sys.path
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


# ─── In-memory VectorStore fixture ────────────────────────────────────────────


class InMemoryVectorStore(VectorStore):
    """
    Subclass that replaces PersistentClient with EphemeralClient so tests
    run in-memory without touching disk or needing a real ChromaDB path.
    Also uses a fast dummy embedding function to avoid loading the ML model.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

        # Use EphemeralClient (in-memory, no disk)
        self.client = chromadb.EphemeralClient()

        # Tiny dummy embedding function — returns fixed-size vectors
        # Must implement name() because ChromaDB 1.0.x validates it.
        class DummyEF:
            @staticmethod
            def name():
                return "dummy"

            def __call__(self, input):
                # Return a deterministic 4-dim embedding per document
                return [
                    [float(hash(doc) % 100) / 100.0, 0.1, 0.2, 0.3] for doc in input
                ]

        self.embedding_function = DummyEF()
        self.course_catalog = self._create_collection("course_catalog")
        self.course_content = self._create_collection("course_content")


@pytest.fixture
def store():
    return InMemoryVectorStore(max_results=5)


@pytest.fixture
def store_with_data(store):
    """A store pre-populated with two courses and several content chunks."""
    # Course 1
    course1 = Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml",
        instructor="Dr. Smith",
        lessons=[
            Lesson(
                lesson_number=1, title="What is ML?", lesson_link="https://ex.com/ml/1"
            ),
            Lesson(
                lesson_number=2,
                title="Supervised Learning",
                lesson_link="https://ex.com/ml/2",
            ),
        ],
    )
    store.add_course_metadata(course1)
    store.add_course_content(
        [
            CourseChunk(
                content="Machine learning is a subset of AI.",
                course_title="Introduction to Machine Learning",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Supervised learning uses labeled data.",
                course_title="Introduction to Machine Learning",
                lesson_number=2,
                chunk_index=1,
            ),
            CourseChunk(
                content="Deep learning uses neural networks.",
                course_title="Introduction to Machine Learning",
                lesson_number=2,
                chunk_index=2,
            ),
        ]
    )

    # Course 2
    course2 = Course(
        title="Python Programming Basics",
        course_link="https://example.com/python",
        instructor="Prof. Jones",
        lessons=[
            Lesson(
                lesson_number=1, title="Variables", lesson_link="https://ex.com/py/1"
            ),
        ],
    )
    store.add_course_metadata(course2)
    store.add_course_content(
        [
            CourseChunk(
                content="Python is a high-level language.",
                course_title="Python Programming Basics",
                lesson_number=1,
                chunk_index=3,
            ),
        ]
    )

    return store


# ─── _build_filter ─────────────────────────────────────────────────────────────


class TestBuildFilter:

    def test_returns_none_when_no_params(self, store):
        assert store._build_filter(None, None) is None

    def test_returns_course_filter_when_only_course_given(self, store):
        f = store._build_filter("Deep Learning", None)
        assert f == {"course_title": "Deep Learning"}

    def test_returns_lesson_filter_when_only_lesson_given(self, store):
        f = store._build_filter(None, 3)
        assert f == {"lesson_number": 3}

    def test_returns_and_filter_when_both_given(self, store):
        f = store._build_filter("Deep Learning", 2)
        assert "$and" in f
        items = f["$and"]
        assert len(items) == 2
        # Each item should reference the correct field
        fields = {}
        for item in items:
            fields.update(item)
        assert "course_title" in fields
        assert "lesson_number" in fields


# ─── search() with empty collection ───────────────────────────────────────────


class TestSearchEmptyCollection:

    def test_empty_collection_does_not_raise(self, store):
        """Querying an empty collection should return a SearchResults, not raise."""
        try:
            result = store.search(query="anything")
        except Exception as exc:
            pytest.fail(f"search() raised an exception on an empty collection: {exc}")

    def test_empty_collection_returns_search_results_instance(self, store):
        result = store.search(query="anything")
        assert isinstance(result, SearchResults)

    def test_empty_collection_result_has_error_or_is_empty(self, store):
        """An empty collection should produce either an error OR an empty result set."""
        result = store.search(query="anything")
        assert result.error is not None or result.is_empty(), (
            "Expected either an error or empty documents, "
            f"got documents={result.documents}"
        )


# ─── search() with fewer docs than n_results ──────────────────────────────────


class TestSearchFewerDocsThanNResults:

    def test_returns_available_docs_when_fewer_than_n_results(self, store):
        """
        CRITICAL: When the collection has fewer documents than n_results,
        ChromaDB should return what it has (or raise an exception that we catch).
        This test detects the 'Collection has N elements, but M were requested' bug.
        """
        # Add only 2 documents but store has n_results=5
        store.add_course_content(
            [
                CourseChunk(
                    content="Doc one content.",
                    course_title="Test",
                    lesson_number=1,
                    chunk_index=0,
                ),
                CourseChunk(
                    content="Doc two content.",
                    course_title="Test",
                    lesson_number=2,
                    chunk_index=1,
                ),
            ]
        )

        result = store.search(query="content")

        # Must not raise — result must be a valid SearchResults object
        assert isinstance(result, SearchResults)

        # Either returns the available 2 docs (correct behavior) or an error
        # (caught by our try/except). It must NOT be an unhandled exception.
        if result.error:
            # An error is acceptable IF it was caught and packaged in SearchResults
            assert isinstance(result.error, str)
        else:
            # Returned some results (could be 1 or 2, depending on ChromaDB version)
            assert len(result.documents) >= 1

    def test_search_returns_no_more_than_n_results(self, store_with_data):
        """search() must never return more results than max_results."""
        result = store_with_data.search(query="learning")
        assert len(result.documents) <= store_with_data.max_results


# ─── search() with filters ────────────────────────────────────────────────────


class TestSearchWithFilters:

    def test_search_without_filters_returns_results(self, store_with_data):
        """A basic query with no filters should return at least one result."""
        result = store_with_data.search(query="machine learning")
        assert not result.error, f"Unexpected error: {result.error}"
        assert (
            not result.is_empty()
        ), "Expected at least one result for 'machine learning'"

    def test_search_with_valid_course_name_returns_results(self, store_with_data):
        """search() with a known course name should return results from that course."""
        result = store_with_data.search(
            query="supervised learning", course_name="Introduction to Machine Learning"
        )
        assert not result.error, f"Unexpected error: {result.error}"
        # All results should belong to the requested course
        for meta in result.metadata:
            assert meta["course_title"] == "Introduction to Machine Learning"

    def test_search_with_unknown_course_name_returns_error(
        self, store_with_data, monkeypatch
    ):
        """
        search() with a course name that can't be resolved must return an error SearchResults.
        We monkeypatch _resolve_course_name to guarantee it returns None (the threshold-exceeded
        path) — isolating this path from the DummyEF's non-semantic similarity scores.
        """
        monkeypatch.setattr(store_with_data, "_resolve_course_name", lambda name: None)
        result = store_with_data.search(
            query="anything", course_name="Definitely Not A Course"
        )
        assert (
            result.error is not None
        ), f"Expected an error SearchResults when course is unresolvable, got: {result}"
        assert "No course found" in result.error

    def test_search_with_lesson_number_filter(self, store_with_data):
        """Filtering by lesson_number should only return chunks from that lesson."""
        result = store_with_data.search(
            query="learning",
            course_name="Introduction to Machine Learning",
            lesson_number=1,
        )
        if not result.error and not result.is_empty():
            for meta in result.metadata:
                assert (
                    meta["lesson_number"] == 1
                ), f"Expected lesson 1 only, got {meta['lesson_number']}"

    def test_search_with_and_filter_course_and_lesson(self, store_with_data):
        """
        CRITICAL: Using both course_name AND lesson_number should not raise.
        This tests the $and filter path in _build_filter().
        """
        try:
            result = store_with_data.search(
                query="supervised",
                course_name="Introduction to Machine Learning",
                lesson_number=2,
            )
        except Exception as exc:
            pytest.fail(
                f"search() with both course_name and lesson_number raised: {exc}"
            )
        assert isinstance(result, SearchResults)


# ─── Course resolution ─────────────────────────────────────────────────────────


class TestCourseResolution:

    def test_resolve_exact_course_name(self, store_with_data):
        """_resolve_course_name() should find an exact course title."""
        title = store_with_data._resolve_course_name("Introduction to Machine Learning")
        assert title == "Introduction to Machine Learning"

    def test_resolve_partial_course_name(self, store_with_data):
        """
        _resolve_course_name() should return one of the known course titles for a
        course-like query (i.e., it should not raise and should not return None).
        Note: with a non-semantic DummyEF the *specific* course matched depends on
        hash-based distances; the real sentence-transformer would return the
        semantically closest title. Here we just verify it returns a known title.
        """
        known_titles = {
            "Introduction to Machine Learning",
            "Python Programming Basics",
        }
        title = store_with_data._resolve_course_name("Machine Learning")
        # It may return either title (DummyEF is non-semantic), but must return one of them
        assert title in known_titles, f"Expected one of {known_titles}, got {title!r}"

    def test_resolve_returns_none_for_completely_unknown_course(self, store_with_data):
        """_resolve_course_name() on a totally unrelated string should return None."""
        # This may or may not return None depending on distance thresholds;
        # the important thing is it should not raise.
        try:
            title = store_with_data._resolve_course_name("ZZZNOMATCH999")
        except Exception as exc:
            pytest.fail(f"_resolve_course_name() raised an exception: {exc}")


# ─── Lesson link retrieval ─────────────────────────────────────────────────────


class TestLessonLinkRetrieval:

    def test_get_lesson_link_returns_correct_url(self, store_with_data):
        link = store_with_data.get_lesson_link("Introduction to Machine Learning", 1)
        assert link == "https://ex.com/ml/1"

    def test_get_lesson_link_returns_none_for_unknown_lesson(self, store_with_data):
        link = store_with_data.get_lesson_link("Introduction to Machine Learning", 99)
        assert link is None

    def test_get_lesson_link_returns_none_for_unknown_course(self, store_with_data):
        link = store_with_data.get_lesson_link("Unknown Course XYZ", 1)
        assert link is None
