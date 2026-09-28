"""Recall@k and MRR@k over doc ids, per question -- the caller takes the mean.

Hand-written on purpose (#74): the two functions below are the whole definition, short enough to
check against a reported number in one screen, and :mod:`tests.test_metrics` holds them to 20
lines between them. No ``ranx``, no ``pytrec_eval``.

**Per question, not averaged.** The paired bootstrap resamples a vector of per-question scores,
so nothing here takes a mean.

**Truncate, then count once.** ``ranked`` is cut to its first ``k`` entries before anything else,
so a gold id at rank 6 scores nothing even when an earlier id repeats; a repeated id inside the
top ``k`` counts once toward recall, and MRR only ever sees the first occurrence.

**Multi-gold recall is fractional.** A multihop row carries 2 to 4 gold ids and single-shot
retrieval is not expected to find the later hops, so Recall@5 there is not Hit@5. That is a
property of the pipeline, not of the metric.

**Empty gold raises.** A row without gold ids has no recall; the caller filters it out rather
than scoring it 0.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int = 5) -> float:
    """|gold ∩ top-k| / |gold|."""
    if not gold:
        raise ValueError("empty gold: filter this row out, do not score it")
    return len(set(gold) & set(ranked[:k])) / len(set(gold))


def mrr_at_k(ranked: Sequence[str], gold: Collection[str], k: int = 5) -> float:
    """1 / rank of the first gold id in the top k, and 0 if none is there."""
    if not gold:
        raise ValueError("empty gold: filter this row out, do not score it")
    return next((1 / r for r, d in enumerate(ranked[:k], 1) if d in gold), 0.0)
