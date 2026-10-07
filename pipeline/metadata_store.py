"""
SQLite Metadata Store
=====================
Handles chunk and passage metadata lookup, keeping SQL queries out of core RAG logic.
"""

import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any

from core.tracing import traceable

_COLUMNS = (
    "vector_id", "query_id", "chunk_id", "passage_id", "part_id",
    "text", "word_count", "token_count", "is_selected", "language",
)
# Stay well below SQLITE_MAX_VARIABLE_NUMBER on old SQLite builds.
_MAX_SQL_VARS = 500


def make_doc_key(query_id: Any, passage_id: Any) -> str:
    """Relevance key shared by pipeline, API and evaluation: '<corpus query_id>:<passage_id>'."""
    return f"{query_id}:{passage_id}"


class MetadataStore:
    """
    Thread-safe, read-only SQLite metadata accessor for indexed chunks and passages.

    The database is opened with `mode=ro` (URI), so a wrong path can never create an empty
    database file, and every query runs under a lock because one connection is shared by
    the server's worker threads.
    """

    def __init__(self, db_path: Path, lang_code: str = "gu"):
        self.db_path = Path(db_path)
        self.lang_code = lang_code
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Metadata database not found: {self.db_path}")

        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

        has_chunks = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not has_chunks:
            self.conn.close()
            raise RuntimeError(f"Metadata database has no 'chunks' table (empty or wrong file?): {self.db_path}")

    def count_chunks(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def _row_to_doc(self, r) -> Dict[str, Any]:
        doc = dict(zip(_COLUMNS, r))
        doc["language"] = doc.get("language") or self.lang_code
        doc["doc_key"] = make_doc_key(doc["query_id"], doc["passage_id"])
        return doc

    @traceable(run_type="tool", name="fetch_metadata")
    def fetch_by_vector_ids(self, vector_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetches metadata for vector_ids and returns them preserving the order of vector_ids
        (duplicates are returned once, at their first position). Each doc carries `doc_key`.
        """
        if not vector_ids:
            return []

        ids = list(dict.fromkeys(int(x) for x in vector_ids))
        by_id: Dict[int, Dict[str, Any]] = {}
        cols = ", ".join(_COLUMNS)
        for start in range(0, len(ids), _MAX_SQL_VARS):
            batch = ids[start:start + _MAX_SQL_VARS]
            placeholders = ",".join("?" for _ in batch)
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT {cols} FROM chunks WHERE vector_id IN ({placeholders})",
                    batch,
                ).fetchall()
            for r in rows:
                by_id[r[0]] = self._row_to_doc(r)

        # Return documents strictly following the input ID order
        return [by_id[x] for x in ids if x in by_id]

    def close(self):
        conn = getattr(self, "conn", None)
        if conn is not None:
            with self._lock:
                conn.close()
            self.conn = None
