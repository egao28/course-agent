"""Reciprocal Rank Fusion: several rankings of one corpus, combined on rank alone.

``score(d) = Σ_i 1 / (k + rank_i(d))`` over the retrievers that returned ``d``, ranks 1-based,
``k = 60``. Nothing else goes in -- no scores, no weights, no normalization -- and that is the
point rather than a simplification. A cosine similarity of 0.42 and a BM25 score of 11.3 are not
on one scale and cannot be put on one: both distributions move with the query, so an
``α·dense + (1-α)·bm25`` tunes a constant against a moving target. Rank is the one thing the two
retrievers report in the same unit.

What the formula buys, in the numbers it actually produces: a document one retriever put **first**
scores ``1/61 = 0.0164``; a document both retrievers put **tenth** scores ``2/70 = 0.0286`` and
wins. Agreement across two different notions of relevance outranks one confident vote, which is
what a hybrid is for. ``k`` is the knob on that trade: it damps the head of each list, so the gap
between rank 1 and rank 2 (0.0164 vs 0.0161) stays small beside the gain from being found twice.
At ``k = 0`` this degenerates to plain reciprocal rank: a document one retriever put first
(1/1 = 1.0) merely ties one both put second (1/2 + 1/2 = 1.0) and beats one both put third
(2/3 = 0.67), so the fusion becomes mostly "whoever was most confident". 60 is Cormack et al.'s
constant and ``config.yaml``'s default; it lives in the config so Day 4 can ablate it.

This module is **pure**: no index, no API key, no I/O, and ``rrf`` takes no config. It is testable
(and ablatable) on a handful of hand-written hits, which is why it is not part of either
retriever. :func:`fuse` is the one function that reads the ``fusion`` section, for
``retrieve(query, "hybrid")`` (#8).

    from retrieval.fusion import rrf

    rrf({"dense": dense_hits, "bm25": bm25_hits})[:5]

Each fused hit reports where it came from::

    Hit(doc_id="2026-sp:CS-374", score=0.0321, rank=1, ..., sources={"dense": 3, "bm25": 1})

``sources`` names every retriever that took part, with ``None`` where that retriever did not
return the document at all -- so the Day 4 ablation can say which half of the hybrid a result came
from, and a result that only one half ever finds is visible rather than inferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from retrieval.types import Hit, fusion_int, fusion_section

#: Cormack et al. (2009). Here so the pure function has its documented behaviour with no config;
#: the live value comes from ``fusion.rrf_k``, which is what an ablation moves.
DEFAULT_RRF_K = 60

#: The only ``fusion.method`` this module implements.
METHOD = "rrf"


def check_k(k: int) -> int:
    """Refuse a damping constant that is not one.

    Zero is allowed -- plain reciprocal rank, the degenerate end of the ablation -- and cannot
    divide by zero, because ranks start at 1. Negative is refused: ``k = -1`` divides by zero at
    rank 1, and below it ranks the list backwards.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"rrf k must be an integer >= 0, got {k!r}")
    return k


def check_rankings(rankings: Mapping[str, Sequence[Hit]]) -> None:
    """Refuse input that would fuse into a well-formed, wrong ranking.

    Every check here is for something that cannot happen to a list a retriever in this repo
    produced, and that RRF cannot detect afterwards. A document listed twice by one retriever is
    counted twice by a sum over its ranks -- the fused score says two retrievers agreed when one
    did. Two documents sharing a rank means the list is not a ranking. A rank below 1 makes
    ``1/(k + rank)`` larger than any real rank can be, and at ``k = 0`` it divides by zero.
    """
    if not isinstance(rankings, Mapping):
        raise TypeError(
            "rankings must be a mapping of retriever name -> hits, e.g. "
            '{"dense": dense_hits, "bm25": bm25_hits}; those names are what a fused hit reports '
            "in `sources`, and a bare sequence of lists has none to report"
        )
    for name, hits in rankings.items():
        at_rank: dict[int, str] = {}
        seen: set[str] = set()
        for hit in hits:
            if isinstance(hit.rank, bool) or not isinstance(hit.rank, int) or hit.rank < 1:
                raise ValueError(
                    f"retriever {name!r} ranked {hit.doc_id!r} {hit.rank!r}; ranks are 1-based "
                    "positions in a list"
                )
            if hit.doc_id in seen:
                raise ValueError(
                    f"retriever {name!r} returned {hit.doc_id!r} twice; RRF sums one term per "
                    "retriever per document, so a repeat would read as two retrievers agreeing"
                )
            if hit.rank in at_rank:
                raise ValueError(
                    f"retriever {name!r} gave rank {hit.rank} to both {at_rank[hit.rank]!r} and "
                    f"{hit.doc_id!r}; a position in a ranking holds one document"
                )
            seen.add(hit.doc_id)
            at_rank[hit.rank] = hit.doc_id


def rrf(rankings: Mapping[str, Sequence[Hit]], k: int = DEFAULT_RRF_K) -> list[Hit]:
    """Fuse ``rankings`` into one ranking, best first.

    ``rankings`` maps a retriever's name to the hits it returned; a document that any of them
    returned appears exactly once in the result, with ``score`` its RRF score, ``rank`` its
    1-based position here, and ``sources`` the rank each retriever gave it (``None`` for one that
    did not return it).

    Ranks come from ``hit.rank``, not from the position in the list, so a caller may pass a
    filtered or re-sliced ranking and still fuse the ranks the retriever assigned -- that is why
    :class:`~retrieval.types.Hit` carries ``rank`` at all.

    Ties are broken by ``doc_id`` ascending. They are not rare: any two documents found by the
    same set of retrievers at the same ranks score identically, and with 20 hits per retriever
    that happens on nearly every query. Dict order would make the fused list depend on which
    retriever happened to be asked first, and a Day 4 ablation would measure that instead of the
    change it made.

    Text and metadata come from the **best-ranked** copy of the document (ties going to the
    first-named retriever). They are the same chunk on every retriever -- both indexes are built
    from ``data/chunks.jsonl`` and each refuses a corpus with another ``chunk_key`` -- so this
    picks a representative, it does not resolve a disagreement.

    The full fused list is returned; ``fuse`` applies ``fusion.top_k_final``. Fusing an empty
    mapping, or rankings that are all empty, returns ``[]``.
    """
    check_rankings(rankings)
    check_k(k)

    names = list(rankings)
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, int | None]] = {}
    best: dict[str, Hit] = {}
    for name, hits in rankings.items():
        for hit in hits:
            doc_id = hit.doc_id
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = dict.fromkeys(names, None)
                best[doc_id] = hit
            scores[doc_id] += 1.0 / (k + hit.rank)
            sources[doc_id][name] = hit.rank
            if hit.rank < best[doc_id].rank:
                best[doc_id] = hit

    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [
        best[doc_id]._replace(score=scores[doc_id], rank=rank, sources=sources[doc_id])
        for rank, doc_id in enumerate(order, start=1)
    ]


# --- the config-driven entry point ----------------------------------------------------------------


def rrf_k(cfg: Mapping[str, Any]) -> int:
    """``fusion.rrf_k`` -- the damping constant, ``60`` in ``config.yaml``."""
    return fusion_int(
        cfg, "rrf_k", "the RRF damping constant, 60 unless an ablation says otherwise", minimum=0
    )


def top_k_final(cfg: Mapping[str, Any]) -> int:
    """``fusion.top_k_final`` -- how many hits survive fusion, ``5`` in ``config.yaml``."""
    return fusion_int(cfg, "top_k_final", "how many hits survive fusion")


def check_method(cfg: Mapping[str, Any]) -> str:
    """Refuse a config asking for a fusion this repo does not have.

    ``method`` exists so Day 4 can name the alternative it compares against. Until something is
    written beside :func:`rrf`, an unknown value is a request that cannot be honoured -- running
    RRF anyway would file the result under the method that was asked for.
    """
    method = fusion_section(cfg).get("method")
    if method != METHOD:
        raise ValueError(
            f"config fusion.method is {method!r}, but the only fusion implemented is {METHOD!r}; "
            "set it back, or add the method beside rrf() in retrieval/fusion.py"
        )
    return str(method)


def fuse(rankings: Mapping[str, Sequence[Hit]], cfg: Mapping[str, Any]) -> list[Hit]:
    """:func:`rrf` with the config's parameters, cut to ``fusion.top_k_final``.

    What ``retrieve(query, "hybrid")`` (#8) calls, and the only function here that reads a config
    -- the truncation lives on this side of the line so that :func:`rrf` stays the formula and
    nothing else.
    """
    check_method(cfg)
    return rrf(rankings, rrf_k(cfg))[: top_k_final(cfg)]
