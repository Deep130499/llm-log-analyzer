"""
Tests for VectorStore.

The SentenceTransformer model is mocked to avoid downloading ~22 MB
of model weights during CI — making the tests fast and network-free.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def make_mock_encoder(dim: int = 384):
    """Return a mock SentenceTransformer that emits deterministic vectors."""
    encoder = MagicMock()

    def fake_encode(text, show_progress_bar=False, **_):
        # Derive a seed from the text so identical inputs → identical vectors
        seed = hash(text) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype("float32")
        # Normalise so cosine similarity behaves correctly
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    encoder.encode.side_effect = fake_encode
    return encoder


@pytest.fixture
def vector_store(tmp_path):
    """VectorStore with a mocked encoder and a temp ChromaDB directory."""
    with patch("src.vector_store.SentenceTransformer") as mock_st:
        mock_st.return_value = make_mock_encoder()
        from src.vector_store import VectorStore
        store = VectorStore(persist_dir=str(tmp_path / "chroma"))
        yield store


# --------------------------------------------------------------------------- #
#  Basic CRUD                                                                  #
# --------------------------------------------------------------------------- #

class TestAddAndCount:
    def test_starts_empty(self, vector_store):
        assert vector_store.count() == 0

    def test_add_single_issue(self, vector_store):
        vector_store.add_issue("ImportError: no module named requests",
                               {"resolution": "pip install requests"})
        assert vector_store.count() == 1

    def test_add_returns_id(self, vector_store):
        issue_id = vector_store.add_issue("some error text", {})
        assert isinstance(issue_id, str)
        assert len(issue_id) > 0

    def test_add_with_explicit_id(self, vector_store):
        returned_id = vector_store.add_issue("error text", {}, issue_id="my-id-001")
        assert returned_id == "my-id-001"
        assert vector_store.count() == 1

    def test_upsert_does_not_duplicate(self, vector_store):
        vector_store.add_issue("same text", {}, issue_id="dup-001")
        vector_store.add_issue("same text", {}, issue_id="dup-001")
        assert vector_store.count() == 1


class TestBulkIngest:
    def test_bulk_adds_all(self, vector_store):
        issues = [
            {"id": f"bulk-{i}", "text": f"Error number {i}", "metadata": {}}
            for i in range(5)
        ]
        ids = vector_store.add_issues_bulk(issues)
        assert len(ids) == 5
        assert vector_store.count() == 5

    def test_bulk_returns_ids(self, vector_store):
        issues = [{"id": "b1", "text": "error one", "metadata": {}}]
        ids = vector_store.add_issues_bulk(issues)
        assert ids == ["b1"]


# --------------------------------------------------------------------------- #
#  Search                                                                      #
# --------------------------------------------------------------------------- #

class TestSearch:
    def _seed(self, store, n=5):
        issues = [
            {"id": f"s{i}", "text": f"Sample issue {i}: ConnectionRefusedError port 543{i}", "metadata": {"resolution": f"Fix {i}"}}
            for i in range(n)
        ]
        store.add_issues_bulk(issues)

    def test_returns_list(self, vector_store):
        self._seed(vector_store)
        results = vector_store.search_similar("connection refused", n_results=3)
        assert isinstance(results, list)

    def test_returns_correct_count(self, vector_store):
        self._seed(vector_store, n=5)
        results = vector_store.search_similar("connection error", n_results=3)
        assert len(results) == 3

    def test_results_have_required_keys(self, vector_store):
        self._seed(vector_store)
        result = vector_store.search_similar("error", n_results=1)[0]
        for key in ("id", "document", "metadata", "distance", "similarity"):
            assert key in result

    def test_similarity_between_0_and_1(self, vector_store):
        self._seed(vector_store)
        for r in vector_store.search_similar("error", n_results=3):
            assert 0.0 <= r["similarity"] <= 1.0

    def test_empty_store_returns_empty_list(self, vector_store):
        results = vector_store.search_similar("any query")
        assert results == []

    def test_n_results_capped_at_store_size(self, vector_store):
        self._seed(vector_store, n=2)
        results = vector_store.search_similar("error", n_results=10)
        assert len(results) == 2


# --------------------------------------------------------------------------- #
#  Utility methods                                                             #
# --------------------------------------------------------------------------- #

class TestUtilities:
    def test_list_ids(self, vector_store):
        vector_store.add_issue("error A", {}, issue_id="id-a")
        vector_store.add_issue("error B", {}, issue_id="id-b")
        ids = vector_store.list_ids()
        assert "id-a" in ids
        assert "id-b" in ids

    def test_clear_resets_count(self, vector_store):
        vector_store.add_issue("some error", {})
        assert vector_store.count() == 1
        vector_store.clear()
        assert vector_store.count() == 0

    def test_metadata_sanitization(self, vector_store):
        """Non-scalar metadata values are coerced to strings."""
        vector_store.add_issue("error", {
            "tags": ["a", "b"],          # list → should be converted
            "count": 42,                  # int → fine
            "nested": {"key": "val"},     # dict → should be converted
        })
        assert vector_store.count() == 1
