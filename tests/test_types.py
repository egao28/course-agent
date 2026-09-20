"""Tests for retrieval/types.py: the contract every retriever meets.

``Hit`` itself is exercised where it is produced (``tests/test_dense.py``) and where it is fused
(``tests/test_fusion.py``); what needs its own tests here is the ``fusion`` section guard, since it
is the one place three modules read the same four config values.
"""

from __future__ import annotations

import pytest

from retrieval.types import Hit, fusion_int, top_k_each

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


def test_a_setting_with_a_floor_of_zero_says_so_rather_than_asking_for_a_positive_one():
    # fusion.rrf_k passes minimum=0: zero damping is degenerate but meaningful, so the message
    # cannot be the "positive integer" one every other fusion setting gets.
    with pytest.raises(ValueError, match=r"fusion\.rrf_k must be an integer >= 0"):
        fusion_int({"fusion": {"rrf_k": -1}}, "rrf_k", "the RRF damping constant", minimum=0)


def test_hit_is_a_tuple_of_the_fields_fusion_reads():
    hit = Hit("2026-sp:CS-446", 0.87, 1, "CS 446 - Machine Learning", {"code": "CS 446"})
    assert hit.doc_id == "2026-sp:CS-446" and hit.rank == 1
    assert tuple(hit) == (hit.doc_id, hit.score, hit.rank, hit.text, hit.metadata, hit.sources)


def test_a_retriever_builds_a_hit_without_sources_and_fusion_fills_them_in():
    # `sources` is the one field a retriever leaves alone, so it has a default -- both retrievers
    # construct a Hit by keyword and neither knows what the other returned.
    hit = Hit("2026-sp:CS-446", 0.87, 1, "CS 446 - Machine Learning", {})
    assert hit.sources is None
    assert hit._replace(sources={"dense": 1, "bm25": None}).sources == {"dense": 1, "bm25": None}
