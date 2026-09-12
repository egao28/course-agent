"""What every retriever returns, so RRF can fuse two lists without knowing where they came from.

Dense (#5) and BM25 (#6) agree on this shape; fusion (#7) merges them; ``retrieve(query, mode)``
(#8) hands it to the caller. Keeping it in one module -- rather than each retriever returning its
own dict -- is what makes ``mode`` a parameter instead of three code paths.

The one thing to know about ``score``: it is **not** comparable across retrievers. A cosine
similarity of 0.42 and a BM25 score of 11.3 say nothing about each other, and even within one
retriever the scale shifts with the query. That is exactly why RRF fuses on ``rank``, and why
``rank`` is carried here rather than recomputed from a sorted list downstream.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol


class Hit(NamedTuple):
    """One retrieved chunk, as one retriever ranked it."""

    #: ``{term}:{SUBJ}-{NUM}``, minted by ``ingest.chunk.doc_id`` and nowhere else.
    doc_id: str
    #: Higher is better, whatever the retriever's own scale is. Only meaningful within one list.
    score: float
    #: 1-based position in that list.
    rank: int
    #: The chunk text that was indexed, i.e. what the score is about.
    text: str
    #: ``ingest.chunk.chunk_metadata`` shape: lists are lists, on both retrievers.
    metadata: dict[str, Any]


class Retriever(Protocol):
    """A thing that ranks chunks for a query.

    ``k`` defaults to ``fusion.top_k_each`` from the config, so a caller that does not care about
    recall depth gets the depth the fusion step was tuned for.
    """

    def search(self, query: str, k: int | None = None) -> list[Hit]: ...
