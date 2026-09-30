"""Cross-encoder reranking over a list of hits: ``rerank(query, hits, k, score)`` (#79).

    from retrieval import retrieve
    from retrieval.rerank import load_scorer, rerank

    pool = retrieve(q, "hybrid", k=cfg["rerank"]["pool"], cfg=cfg).hits
    top5 = rerank(q, pool, 5, load_scorer(cfg))

**A step after retrieve(), not a mode.** A fourth entry in :data:`retrieval.MODES` would change the
list #75's run files, #76's checks and the three-mode table are keyed on, and ``hybrid`` would stop
being the only thing that decides what ``hybrid`` returns. Here nothing in ``retrieval/__init__``
moves: the pool is an ordinary hybrid call, and this module only reorders it.

**It can only reorder.** The top ``k`` it returns is a subset of the pool, so its Recall@k is at
most the pool's Recall@pool -- the ceiling the report prints beside the row.

**Scores are the cross-encoder's**, on its own scale (sentence-transformers applies a sigmoid to
a one-label model, so bge-reranker-base scores lie in (0, 1)). A tie keeps the pool's order, so a
reranker that cannot tell two chunks apart leaves them where hybrid put them. ``sources`` is kept
from the pool: where dense and bm25 had the document is still true after reranking.

``load_scorer`` imports ``sentence_transformers`` (and torch) at call time, for the reason
``retrieval._build_dense`` imports chromadb late: it is the ``local`` extra, and nothing else in
the package needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from retrieval.fusion import check_limit
from retrieval.types import Hit

#: ``(query, chunk text)`` pairs in, one score per pair out, higher is better.
Scorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


def rerank(query: str, hits: Sequence[Hit], k: int, score: Scorer) -> list[Hit]:
    """``hits`` rescored by ``score`` on ``(query, hit.text)``, best first, cut to ``k``."""
    k = check_limit(k)
    if not hits:
        return []
    scores = [float(s) for s in score([(query, hit.text) for hit in hits])]
    if len(scores) != len(hits):
        raise ValueError(f"the scorer returned {len(scores)} scores for {len(hits)} pairs")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return [hits[i]._replace(score=scores[i], rank=rank) for rank, i in enumerate(order[:k], 1)]


def load_scorer(cfg: Mapping[str, Any]) -> Scorer:
    """The ``rerank`` section's cross-encoder, loaded once, as a :data:`Scorer`."""
    section = cfg["rerank"]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise ImportError("reranking needs the local extra: `uv sync --extra local`") from error
    model = CrossEncoder(section["model"], revision=section["revision"], device=section["device"])
    batch_size = int(section["batch_size"])

    def score(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False).tolist()

    return score
