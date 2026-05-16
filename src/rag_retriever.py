"""
RAG Retriever — builds a semantic search query from a parsed log entry
and retrieves the most similar past issues from the vector store.

The retrieved context is later injected into the LLM prompt so the
model can reference known resolutions when generating its analysis.

Usage:
    retriever = RAGRetriever(vector_store)
    result    = retriever.retrieve(log_entry)
    print(result.format_for_prompt())
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .log_parser import LogEntry
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    """
    The output of a RAG retrieval step.

    Attributes
    ----------
    query    : the search string that was embedded
    issues   : list of similar past issue dicts from VectorStore
    count    : number of results returned
    """

    query: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    count: int = 0

    # ------------------------------------------------------------------ #
    #  Formatting helpers                                                  #
    # ------------------------------------------------------------------ #

    def format_for_prompt(self) -> str:
        """
        Render the retrieved issues as a compact Markdown block
        suitable for injection into an LLM prompt.
        """
        if not self.issues:
            return "No similar past issues found in the knowledge base."

        lines = []
        for i, issue in enumerate(self.issues, start=1):
            sim_pct = int(issue.get("similarity", 0) * 100)
            doc = issue["document"][:280].replace("\n", " ")
            lines.append(f"{i}. [{sim_pct}% match] {doc}")

            meta = issue.get("metadata", {})
            if meta.get("resolution"):
                lines.append(f"   ↳ Resolution: {meta['resolution']}")
            if meta.get("tags"):
                lines.append(f"   ↳ Tags: {meta['tags']}")

        return "\n".join(lines)

    def has_high_confidence_match(self, threshold: float = 0.80) -> bool:
        """Return True if at least one result exceeds the similarity threshold."""
        return any(r.get("similarity", 0) >= threshold for r in self.issues)


class RAGRetriever:
    """
    Constructs a rich semantic query from a LogEntry and retrieves
    similar documents from a VectorStore.

    Parameters
    ----------
    vector_store : VectorStore
        The persistent ChromaDB-backed store to query.
    n_results : int
        Maximum number of similar issues to retrieve (default 3).
    """

    def __init__(self, vector_store: VectorStore, n_results: int = 3):
        self.vector_store = vector_store
        self.n_results = n_results

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(self, log_entry: LogEntry) -> RetrievalResult:
        """
        Build a query from `log_entry`, run a similarity search, and
        return a RetrievalResult with the top matching past issues.
        """
        query = self._build_query(log_entry)

        if self.vector_store.count() == 0:
            return RetrievalResult(query=query, issues=[], count=0)

        issues = self.vector_store.search_similar(query, n_results=self.n_results)

        return RetrievalResult(query=query, issues=issues, count=len(issues))

    # ------------------------------------------------------------------ #
    #  Query construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_query(self, entry: LogEntry) -> str:
        """
        Combine the most diagnostic fields of a LogEntry into a
        single string for embedding.

        Priority order (most → least informative):
          1. Explicit error messages   (closest to the root cause)
          2. Failure names             (identify which tests broke)
          3. First stack trace frame   (pinpoint the crash site)
          4. Log type                  (technology context)
          5. Raw log fallback          (when nothing else is available)
        """
        parts: List[str] = []

        if entry.error_messages:
            parts.append("Errors: " + " | ".join(entry.error_messages[:4]))

        if entry.failures:
            parts.append("Failed: " + ", ".join(entry.failures[:4]))

        if entry.stack_traces:
            # Use only the last (most specific) line of the first trace
            last_line = entry.stack_traces[0].strip().splitlines()[-1]
            parts.append(f"Trace: {last_line}")

        if entry.log_type:
            parts.append(f"Type: {entry.log_type}")

        if not parts:
            # Fall back to raw log excerpt
            parts.append(entry.raw_log[:300])

        return " | ".join(parts)
