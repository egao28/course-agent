"""Tests for eval/metrics.py: every expected value is computed by hand, not by the code under test.

Ids follow the real ``{term}:{SUBJ}-{NUM}`` shape so a reader can map a case onto a dataset row,
and the multi-gold case is #74's own example: FR 104 -> FR 103 -> FR 102 -> FR 101.
"""

from __future__ import annotations

import inspect

import pytest

from eval import metrics
from eval.metrics import mrr_at_k, recall_at_k

T = "2026-sp"
A, B, C, D, E, F = (f"{T}:CS-{n}" for n in (101, 124, 128, 173, 225, 233))
FR = [f"{T}:FR-{n}" for n in (104, 103, 102, 101)]


@pytest.mark.parametrize(
    ("ranked", "recall", "mrr"),
    [
        ([A, B, C, D, E], 1.0, 1.0),  # gold at rank 1
        ([B, C, A, D, E], 1.0, 1 / 3),  # gold at rank 3
        ([B, C, D, E, F], 0.0, 0.0),  # gold absent
    ],
)
def test_single_gold(ranked, recall, mrr):
    assert recall_at_k(ranked, {A}) == recall
    assert mrr_at_k(ranked, {A}) == pytest.approx(mrr)


def test_multi_gold_partial_hit_is_fractional_recall_and_full_mrr():
    ranked = [FR[0], FR[1], A, B, C]
    assert recall_at_k(ranked, FR) == 0.5
    assert mrr_at_k(ranked, FR) == 1.0


def test_multi_gold_first_hit_decides_mrr():
    assert mrr_at_k([A, B, FR[2], FR[0], C], FR) == pytest.approx(1 / 3)


def test_duplicate_ids_in_ranked_count_once():
    assert recall_at_k([FR[0], FR[0], FR[0], A, B], FR) == 0.25
    assert recall_at_k([A, A, B, C, D], {A}) == 1.0


def test_duplicate_gold_ids_count_once():
    assert recall_at_k([A, B, C, D, E], [A, A, F]) == 0.5


def test_ranked_longer_than_k_is_truncated():
    ranked = [B, C, D, E, F, A]  # gold at rank 6
    assert recall_at_k(ranked, {A}) == 0.0
    assert mrr_at_k(ranked, {A}) == 0.0
    assert recall_at_k(ranked, {A}, k=6) == 1.0
    assert mrr_at_k(ranked, {A}, k=6) == pytest.approx(1 / 6)


def test_truncation_happens_before_deduplication():
    # Five slots, four distinct ids: the gold id in slot 6 is still outside the top 5.
    assert recall_at_k([B, B, C, D, E, A], {A}) == 0.0


def test_ranked_shorter_than_k_is_fine():
    assert recall_at_k([B, A], {A}) == 1.0
    assert mrr_at_k([B, A], {A}) == 0.5
    assert recall_at_k([], {A}) == 0.0
    assert mrr_at_k([], {A}) == 0.0


@pytest.mark.parametrize("fn", [recall_at_k, mrr_at_k])
@pytest.mark.parametrize("gold", [set(), [], ()])
def test_empty_gold_raises(fn, gold):
    with pytest.raises(ValueError, match="empty gold"):
        fn([A, B, C], gold)


def test_returns_floats():
    assert isinstance(recall_at_k([A], {A}), float)
    assert isinstance(mrr_at_k([B], {A}), float)


def test_the_two_metrics_fit_in_twenty_lines():
    """#74's budget: the definitions, signature to return, blank lines excluded."""
    lines = [
        ln
        for fn in (metrics.recall_at_k, metrics.mrr_at_k)
        for ln in inspect.getsource(fn).splitlines()
        if ln.strip()
    ]
    assert len(lines) <= 20, len(lines)
