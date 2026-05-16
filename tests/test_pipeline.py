"""
End-to-end pipeline tests.

Runs the full stack (Parser → VectorStore → RAGRetriever → LLMAnalyzer)
using demo mode — no LLM or real embeddings required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.log_parser import LogParser, LogEntry
from src.rag_retriever import RAGRetriever, RetrievalResult


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def make_mock_encoder(dim=384):
    encoder = MagicMock()
    def fake_encode(text, show_progress_bar=False, **_):
        seed = hash(text) % (2 ** 31)
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v
    encoder.encode.side_effect = fake_encode
    return encoder


@pytest.fixture
def seeded_vector_store(tmp_path):
    with patch("src.vector_store.SentenceTransformer") as mock_st:
        mock_st.return_value = make_mock_encoder()
        from src.vector_store import VectorStore
        store = VectorStore(persist_dir=str(tmp_path / "chroma"))
        # Seed with representative past issues
        store.add_issues_bulk([
            {"id": "pi-1", "text": "ConnectionRefusedError: PostgreSQL not running on port 5432",
             "metadata": {"resolution": "Add services: postgres: to workflow YAML"}},
            {"id": "pi-2", "text": "ImportError: No module named requests pip install missing",
             "metadata": {"resolution": "Add requests to requirements.txt"}},
            {"id": "pi-3", "text": "AssertionError expected 201 got 500 database not seeded",
             "metadata": {"resolution": "Add DB fixture in conftest.py"}},
        ])
        yield store


SAMPLE_PYTEST_LOG = """\
============================= test session starts ==============================
collected 4 items

tests/test_db.py::test_connect FAILED
tests/test_api.py::test_create FAILED

================================= FAILURES ====================================
E   psycopg2.OperationalError: could not connect to server: Connection refused
E   Is the server running on host localhost and accepting TCP/IP connections on port 5432?

=================== 2 failed, 2 passed in 5.12s ================================
"""

SAMPLE_JEST_LOG = """\
 FAIL  src/api/users.test.js

  ● fetchUser › returns data

    TypeError: Cannot read properties of null (reading 'data')

      at fetchUser (src/api/users.js:12:20)

Tests:  1 failed, 3 passed, 4 total
npm ERR! Test failed.
"""


# --------------------------------------------------------------------------- #
#  Parser → Retriever integration                                              #
# --------------------------------------------------------------------------- #

class TestParserToRetriever:
    def test_pytest_log_retrieves_similar_issues(self, seeded_vector_store):
        entry = LogParser().parse(SAMPLE_PYTEST_LOG)
        retriever = RAGRetriever(seeded_vector_store, n_results=2)
        result = retriever.retrieve(entry)

        assert isinstance(result, RetrievalResult)
        assert result.count <= 2
        assert isinstance(result.issues, list)

    def test_jest_log_retrieves_results(self, seeded_vector_store):
        entry = LogParser().parse(SAMPLE_JEST_LOG)
        retriever = RAGRetriever(seeded_vector_store, n_results=2)
        result = retriever.retrieve(entry)

        assert result.count <= 2

    def test_empty_store_returns_zero_results(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.vector_store import VectorStore
            empty_store = VectorStore(persist_dir=str(tmp_path / "empty"))
            entry = LogParser().parse(SAMPLE_PYTEST_LOG)
            retriever = RAGRetriever(empty_store, n_results=3)
            result = retriever.retrieve(entry)
            assert result.count == 0

    def test_retrieval_result_formats_for_prompt(self, seeded_vector_store):
        entry = LogParser().parse(SAMPLE_PYTEST_LOG)
        retriever = RAGRetriever(seeded_vector_store, n_results=2)
        result = retriever.retrieve(entry)
        prompt_text = result.format_for_prompt()

        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0

    def test_empty_retrieval_formats_gracefully(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.vector_store import VectorStore
            empty_store = VectorStore(persist_dir=str(tmp_path / "empty2"))
            entry = LogParser().parse(SAMPLE_PYTEST_LOG)
            retriever = RAGRetriever(empty_store)
            result = retriever.retrieve(entry)
            text = result.format_for_prompt()
            assert "No similar" in text


# --------------------------------------------------------------------------- #
#  Full pipeline (demo mode)                                                   #
# --------------------------------------------------------------------------- #

class TestFullPipelineDemo:
    def test_run_analysis_returns_dict(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(
                SAMPLE_PYTEST_LOG,
                provider="demo",
                db_path=str(tmp_path / "chroma"),
            )
        assert "log_entry" in result
        assert "retrieval" in result
        assert "analysis" in result

    def test_analysis_is_non_empty_string(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(SAMPLE_PYTEST_LOG, provider="demo",
                                  db_path=str(tmp_path / "chroma"))
        assert isinstance(result["analysis"], str)
        assert len(result["analysis"]) > 100

    def test_analysis_contains_root_cause_section(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(SAMPLE_PYTEST_LOG, provider="demo",
                                  db_path=str(tmp_path / "chroma"))
        assert "Root Cause" in result["analysis"]

    def test_jest_log_full_pipeline(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(SAMPLE_JEST_LOG, provider="demo",
                                  db_path=str(tmp_path / "chroma"))
        assert result["log_entry"].log_type == "jest"
        assert "Fix Recommendations" in result["analysis"]

    def test_log_entry_is_correct_type(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(SAMPLE_PYTEST_LOG, provider="demo",
                                  db_path=str(tmp_path / "chroma"))
        assert isinstance(result["log_entry"], LogEntry)

    def test_retrieval_is_correct_type(self, tmp_path):
        with patch("src.vector_store.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_encoder()
            from src.analyzer import run_analysis
            result = run_analysis(SAMPLE_PYTEST_LOG, provider="demo",
                                  db_path=str(tmp_path / "chroma"))
        assert isinstance(result["retrieval"], RetrievalResult)


# --------------------------------------------------------------------------- #
#  LLMAnalyzer demo mode                                                       #
# --------------------------------------------------------------------------- #

class TestLLMAnalyzerDemo:
    def test_demo_returns_string(self):
        from src.llm_analyzer import LLMAnalyzer
        from src.rag_retriever import RetrievalResult
        analyzer = LLMAnalyzer(provider="demo")
        entry = LogParser().parse(SAMPLE_PYTEST_LOG)
        retrieval = RetrievalResult(query="test", issues=[], count=0)
        result = analyzer.analyze(entry, retrieval)
        assert isinstance(result, str)

    def test_demo_analysis_includes_sections(self):
        from src.llm_analyzer import LLMAnalyzer
        from src.rag_retriever import RetrievalResult
        analyzer = LLMAnalyzer(provider="demo")
        entry = LogParser().parse(SAMPLE_PYTEST_LOG)
        retrieval = RetrievalResult(query="test", issues=[], count=0)
        result = analyzer.analyze(entry, retrieval)
        for section in ("Root Cause", "Fix Recommendations", "Priority"):
            assert section in result, f"Missing section: {section}"

    def test_demo_with_retrieval_context(self):
        from src.llm_analyzer import LLMAnalyzer
        from src.rag_retriever import RetrievalResult
        analyzer = LLMAnalyzer(provider="demo")
        entry = LogParser().parse(SAMPLE_PYTEST_LOG)
        retrieval = RetrievalResult(
            query="connection refused",
            issues=[{
                "id": "pi-1",
                "document": "ConnectionRefusedError: PostgreSQL not running",
                "metadata": {"resolution": "Add services: postgres: to workflow"},
                "similarity": 0.91,
                "distance": 0.09,
            }],
            count=1,
        )
        result = analyzer.analyze(entry, retrieval)
        # Past issue context should appear somewhere in the output
        assert "knowledge base" in result.lower() or "past" in result.lower()

    def test_invalid_provider_raises(self):
        from src.llm_analyzer import LLMAnalyzer
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMAnalyzer(provider="fakeProvider")
