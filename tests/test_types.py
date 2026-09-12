"""Tests for retrieval/types.py: the contract every retriever meets.

``Hit`` itself is exercised where it is produced (``tests/test_dense.py``); what needs its own
tests is ``top_k_each``, since it is the one place three retrievers read the same config value.
"""

from __future__ import annotations

import pytest

from retrieval.types import Hit, top_k_each

CFG = {"fusion": {"method": "rrf", "rrf_k": 60, "top_k_each": 20, "top_k_final": 5}}


def test_top_k_each_reads_the_recall_depth_fusion_was_tuned_for():
    assert top_k_each(CFG) == 20


def test_a_config_with_no_fusion_section_is_a_sentence_not_a_key_error():
    with pytest.raises(ValueError, match="no `fusion` section"):
        top_k_each({"seed": 42})


def test_a_missing_top_k_each_names_itself():
    # Dense (#5), BM25 (#6) and fusion (#7) all read this value; they must all report it the same
    # way, naming config.yaml rather than surfacing a bare KeyError from a constructor.
    with pytest.raises(ValueError, match="fusion section is missing top_k_each"):
        top_k_each({"fusion": {"method": "rrf", "rrf_k": 60}})


@pytest.mark.parametrize("value", [0, -1, 2.5, "20", True, None])
def test_a_recall_depth_that_is_not_a_positive_integer_is_refused(value):
    # True is an int in Python and would silently mean "one hit per retriever".
    with pytest.raises(ValueError, match="positive integer"):
        top_k_each({"fusion": {"top_k_each": value}})


def test_hit_is_a_tuple_of_the_five_fields_fusion_reads():
    hit = Hit("2026-sp:CS-446", 0.87, 1, "CS 446 - Machine Learning", {"code": "CS 446"})
    assert hit.doc_id == "2026-sp:CS-446" and hit.rank == 1
    assert tuple(hit) == (hit.doc_id, hit.score, hit.rank, hit.text, hit.metadata)
