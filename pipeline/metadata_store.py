"""
SQLite Metadata Store
=====================
Handles chunk and passage metadata lookup, keeping SQL queries out of core RAG logic.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.tracing import traceable


class MetadataStore:
    """
    Thread-safe SQLite metadata accessor for indexed chunks and passages.
    """

    def __init__(self, db_path: Path, lang_code: str = "gu"):
        self.db_path = Path(db_path)
        self.lang_code = lang_code
        if not self.db_path.exists():
            raise FileNotFoundError(f"Metadata database not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

    def count_chunks(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        row = cursor.fetchone()
        return row[0] if row else 0

    @traceable(run_type="tool", name="fetch_metadata")
    def fetch_by_vector_ids(self, vector_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetches metadata for vector_ids and returns them preserving the exact order of vector_ids.
        """
        if not vector_ids:
            return []

        ids = [int(x) for x in vector_ids]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT vector_id, query_id, chunk_id, passage_id, part_id, text, word_count, token_count, is_selected, language
            FROM chunks WHERE vector_id IN ({placeholders})
            """,
            ids,
        ).fetchall()

        by_id = {
            r[0]: {
                "vector_id": r[0],
                "query_id": r[1],
                "chunk_id": r[2],
                "passage_id": r[3],
                "part_id": r[4],
                "text": r[5],
                "word_count": r[6],
                "token_count": r[7],
                "is_selected": r[8],
                "language": r[9] or self.lang_code,
            }
            for r in rows
        }

        # Return documents strictly following the input ID order
        return [by_id[x] for x in ids if x in by_id]

    def close(self):
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
