"""
Vector Store — wraps ChromaDB with sentence-transformers embeddings.

All embeddings are computed locally using the `all-MiniLM-L6-v2` model
(~22 MB).  No API key required.

Usage:
    store = VectorStore()
    store.add_issue("test failed with ImportError", {"resolution": "pip install requests"})
    results = store.search_similar("ImportError no module named requests")
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Parameters
    ----------
    persist_dir : str
        Directory where ChromaDB will save its data between runs.
    model_name : str
        HuggingFace sentence-transformers model to use for embeddings.
    collection_name : str
        ChromaDB collection name.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_COLLECTION = "past_issues"

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._encoder = SentenceTransformer(model_name)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_issue(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        issue_id: Optional[str] = None,
    ) -> str:
        """
        Embed `text` and upsert it into the collection.

        Returns the ID used to store the document.
        """
        if issue_id is None:
            issue_id = "issue-" + hashlib.md5(text.encode()).hexdigest()[:12]

        embedding = self._encoder.encode(text, show_progress_bar=False).tolist()
        clean_meta = self._sanitize_metadata(metadata or {})

        self._collection.upsert(
            ids=[issue_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_meta],
        )
        return issue_id

    def add_issues_bulk(self, issues: List[Dict[str, Any]]) -> List[str]:
        """
        Batch-ingest a list of issue dicts.

        Each dict must have at least a ``text`` key.
        Optional keys: ``id``, ``metadata``.
        """
        ids, embeddings, documents, metadatas = [], [], [], []

        for issue in issues:
            text = issue["text"]
            issue_id = issue.get("id") or "issue-" + hashlib.md5(text.encode()).hexdigest()[:12]
            meta = self._sanitize_metadata(issue.get("metadata", {}))

            ids.append(issue_id)
            embeddings.append(self._encoder.encode(text, show_progress_bar=False).tolist())
            documents.append(text)
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def search_similar(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Return the top-`n_results` most similar documents to `query`.

        Each result dict contains:
            id, document, metadata, distance  (lower distance = more similar)
        """
        total = self.count()
        if total == 0:
            return []

        n = min(n_results, total)
        embedding = self._encoder.encode(query, show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": round(1 - results["distances"][0][i], 4),
                }
            )
        return output

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Return the number of documents currently stored."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection (keeps the collection)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_ids(self) -> List[str]:
        """Return all stored document IDs."""
        result = self._collection.get(include=[])
        return result["ids"]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only allows str/int/float/bool values in metadata."""
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[str(k)] = v
            elif isinstance(v, list):
                clean[str(k)] = ", ".join(str(x) for x in v)
            else:
                clean[str(k)] = str(v)
        return clean
