"""Tests for retrieval/fusion.py: the formula, the tie-break, and what a fused hit reports.

Fusion is the one retrieval step with no index behind it, so these run on hand-written rankings --
no corpus, no API key, nothing built. That is the point of keeping ``rrf`` pure: the arithmetic
that decides the final order is checkable against numbers written out by hand, rather than against
whatever two indexes happened to return.

The numbers below are the ones in the issue: ``1/61 = 0.01639`` for a single first place,
``2/70 = 0.02857`` for two tenth places.
"""

from __future__ import annotations

import pytest

from config import load_config
from retrieval.fusion import (
    DEFAULT_RRF_K,
    check_method,
    fuse,
    rrf,
    rrf_k,
    top_k_final,
)
from retrieval.types import Hit

CFG = {"fusion": {"method": "rrf", "rrf_k": 60, "top_k_each": 20, "top_k_final": 5}}


def ranking(*doc_ids: str) -> list[Hit]:
    """One retriever's output: these doc_ids in this order, ranks 1..n, scores descending."""
    return [
        Hit(doc_id=doc_id, score=1.0 / i, rank=i, text=f"{doc_id} chunk", metadata={"code": doc_id})
        for i, doc_id in enumerate(doc_ids, start=1)
    ]


def filler(prefix: str, n: int) -> list[str]:
    """``n`` doc_ids no other ranking will mention, to push a document down a list."""
    return [f"{prefix}-{i}" for i in range(1, n + 1)]


# --- the formula ----------------------------------------------------------------------------------


def test_the_default_k_is_the_one_the_issue_specifies():
    assert DEFAULT_RRF_K == 60


def test_a_document_both_retrievers_rank_first_scores_two_sixty_firsts():
    fused = rrf({"dense": ranking("2026-sp:CS-374"), "bm25": ranking("2026-sp:CS-374")})
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_agreement_at_rank_ten_beats_one_confident_first_place():
    # The check that RRF is doing its job: 1/61 = 0.0164 for a document only one retriever found,
    # 2/70 = 0.0286 for one both found tenth. A hybrid exists for the second kind of result.
    dense = ranking("SOLO", *filler("D", 8), "AGREED")
    bm25 = ranking(*filler("B", 9), "AGREED")
    fused = rrf({"dense": dense, "bm25": bm25})

    by_id = {hit.doc_id: hit for hit in fused}
    assert by_id["AGREED"].score == pytest.approx(2 / 70)
    assert by_id["SOLO"].score == pytest.approx(1 / 61)
    assert fused[0].doc_id == "AGREED"


def test_k_damps_the_head_of_each_list():
    rankings = {"dense": ranking("SOLO", "AGREED"), "bm25": ranking("OTHER", "AGREED")}

    # At k = 60 the gap between rank 1 and rank 2 is small (1/61 vs 1/62), so being found twice
    # decides the order.
    fused = rrf(rankings)
    assert fused[0].doc_id == "AGREED"
    assert fused[0].score == pytest.approx(2 / 62)

    # At k = 0 the head is not damped at all: one first place (1/1) exactly ties two second
    # places (1/2 + 1/2), and the tie-break -- not the fusion -- decides the order.
    flat = rrf(rankings, k=0)
    assert [hit.score for hit in flat] == [1.0, 1.0, 1.0]
    assert [hit.doc_id for hit in flat] == ["AGREED", "OTHER", "SOLO"]


def test_scores_never_increase_down_the_fused_list():
    fused = rrf({"dense": ranking("A", "B", "C", "D"), "bm25": ranking("D", "B", "E")})
    scores = [hit.score for hit in fused]
    assert scores == sorted(scores, reverse=True)


def test_fusing_a_single_ranking_returns_it_in_its_own_order():
    # 1/(k+rank) is monotonic in rank, so one list in is the same list out -- which is what makes
    # `retrieve(mode="dense")` and `mode="hybrid"` comparable in the Day 4 ablation.
    order = ["2026-sp:CS-374", "2026-sp:ECE-374", "2026-sp:CS-473"]
    assert [hit.doc_id for hit in rrf({"dense": ranking(*order)})] == order


def test_fusion_uses_the_rank_the_retriever_assigned_not_the_position_in_the_list():
    # A caller may hand over a sliced ranking; the hits still carry the ranks they were given,
    # and those are what the formula is about.
    tail = ranking("A", "B", "C")[2:]
    assert rrf({"dense": tail})[0].score == pytest.approx(1 / 63)


def test_rank_is_the_position_in_the_fused_list_counted_from_one():
    fused = rrf({"dense": ranking("A", "B", "C"), "bm25": ranking("C", "D")})
    assert [hit.rank for hit in fused] == [1, 2, 3, 4]


# --- ties -----------------------------------------------------------------------------------------


def test_ties_are_broken_by_doc_id_not_by_which_retriever_came_first():
    # Both documents are ranked 1 by one retriever and 2 by the other, so their scores are equal
    # to the bit. Iteration order of the input must not leak into the result: an ablation that
    # reordered the mapping would otherwise measure that instead of the change it made.
    rankings = {"dense": ranking("Z", "A"), "bm25": ranking("A", "Z")}
    assert [hit.doc_id for hit in rrf(rankings)] == ["A", "Z"]

    flipped = {"bm25": rankings["bm25"], "dense": rankings["dense"]}
    assert [hit.doc_id for hit in rrf(flipped)] == ["A", "Z"]
    assert [hit.score for hit in rrf(flipped)] == [hit.score for hit in rrf(rankings)]


# --- what a fused hit carries ---------------------------------------------------------------------


def test_sources_name_every_retriever_and_none_means_it_did_not_return_the_document():
    fused = rrf({"dense": ranking("A", "B"), "bm25": ranking("B", "C")})
    assert {hit.doc_id: hit.sources for hit in fused} == {
        "B": {"dense": 2, "bm25": 1},
        "A": {"dense": 1, "bm25": None},
        "C": {"dense": None, "bm25": 2},
    }


def test_a_retriever_that_returned_nothing_is_still_named_in_sources():
    # BM25 drops documents no query term touched, so an empty lexical list is a real case -- and
    # "bm25 found nothing" is the interesting half of that result, not an absence to hide.
    fused = rrf({"dense": ranking("A"), "bm25": []})
    assert fused[0].sources == {"dense": 1, "bm25": None}


def test_a_hit_from_a_retriever_carries_no_sources():
    assert ranking("A")[0].sources is None


def test_a_fused_hit_carries_the_text_and_metadata_of_its_best_ranked_copy():
    dense = [Hit("A", 0.42, 2, "dense copy", {"from": "dense"})]
    bm25 = [Hit("A", 11.3, 1, "bm25 copy", {"from": "bm25"})]
    fused = rrf({"dense": dense, "bm25": bm25})
    assert (fused[0].text, fused[0].metadata) == ("bm25 copy", {"from": "bm25"})


def test_when_both_retrievers_rank_a_document_the_same_the_first_named_represents_it():
    dense = [Hit("A", 0.42, 1, "dense copy", {})]
    bm25 = [Hit("A", 11.3, 1, "bm25 copy", {})]
    assert rrf({"dense": dense, "bm25": bm25})[0].text == "dense copy"
    assert rrf({"bm25": bm25, "dense": dense})[0].text == "bm25 copy"


# --- empty and malformed input --------------------------------------------------------------------


def test_no_rankings_at_all_is_an_empty_result_not_a_crash():
    assert rrf({}) == []


def test_rankings_that_are_all_empty_are_an_empty_result():
    assert rrf({"dense": [], "bm25": []}) == []


def test_a_bare_sequence_of_lists_names_the_mapping_it_should_have_been():
    with pytest.raises(TypeError, match="mapping of retriever name"):
        rrf([ranking("A"), ranking("B")])


def test_a_document_returned_twice_by_one_retriever_is_refused():
    # It would add two terms to a sum that is supposed to have one per retriever, and read as
    # agreement between retrievers that never happened.
    doubled = [*ranking("A", "B"), Hit("A", 0.1, 3, "A chunk", {})]
    with pytest.raises(ValueError, match="returned '?A'? twice"):
        rrf({"dense": doubled})


def test_two_documents_at_the_same_rank_are_refused():
    clashing = [Hit("A", 0.9, 1, "", {}), Hit("B", 0.8, 1, "", {})]
    with pytest.raises(ValueError, match="a position in a ranking holds one document"):
        rrf({"dense": clashing})


@pytest.mark.parametrize("rank", [0, -1])
def test_a_rank_below_one_is_refused(rank):
    # 1/(k + 0) is bigger than any real rank can score, and at k = 0 it does not divide.
    with pytest.raises(ValueError, match="ranks are 1-based"):
        rrf({"dense": [Hit("A", 0.9, rank, "", {})]})


@pytest.mark.parametrize("k", [-1, 1.5, "60", True, None])
def test_a_k_that_is_not_a_whole_number_at_least_zero_is_refused(k):
    with pytest.raises(ValueError, match="integer >= 0"):
        rrf({"dense": ranking("A")}, k=k)


# --- the config-driven entry point ----------------------------------------------------------------


def test_fuse_reads_k_and_the_final_depth_from_the_config():
    rankings = {"dense": ranking(*filler("D", 8)), "bm25": ranking(*filler("B", 8))}
    fused = fuse(rankings, CFG)
    assert len(fused) == CFG["fusion"]["top_k_final"]
    assert [hit.rank for hit in fused] == [1, 2, 3, 4, 5]
    assert fused[0].score == pytest.approx(1 / 61)


def test_fuse_keeps_the_head_of_the_full_fusion_not_of_either_ranking():
    # AGREED is 9th in each list on its own, but both retrievers found it, so it survives a
    # top-3 cut that drops every document only one of them ranked above it.
    dense = ranking(*filler("D", 8), "AGREED")
    bm25 = ranking(*filler("B", 8), "AGREED")
    cfg = {"fusion": {**CFG["fusion"], "top_k_final": 3}}
    assert fuse({"dense": dense, "bm25": bm25}, cfg)[0].doc_id == "AGREED"


def test_fuse_refuses_a_method_this_repo_does_not_implement():
    cfg = {"fusion": {**CFG["fusion"], "method": "weighted_sum"}}
    with pytest.raises(ValueError, match="only fusion implemented is 'rrf'"):
        fuse({"dense": ranking("A")}, cfg)


def test_check_method_accepts_the_one_method_there_is():
    assert check_method(CFG) == "rrf"


def test_rrf_k_may_be_zero_because_zero_damping_is_a_real_ablation():
    assert rrf_k({"fusion": {"rrf_k": 0}}) == 0


@pytest.mark.parametrize("value", [-1, 2.5, "60", True, None])
def test_an_rrf_k_that_is_not_a_whole_number_at_least_zero_is_refused(value):
    with pytest.raises(ValueError, match=r"fusion\.rrf_k must be an integer >= 0"):
        rrf_k({"fusion": {"rrf_k": value}})


@pytest.mark.parametrize("value", [0, -1, 2.5, "5", True, None])
def test_a_final_depth_that_is_not_a_positive_integer_is_refused(value):
    with pytest.raises(ValueError, match=r"fusion\.top_k_final must be a positive integer"):
        top_k_final({"fusion": {"top_k_final": value}})


def test_the_missing_keys_name_themselves():
    with pytest.raises(ValueError, match="fusion section is missing rrf_k"):
        rrf_k({"fusion": {"method": "rrf"}})
    with pytest.raises(ValueError, match="fusion section is missing top_k_final"):
        top_k_final({"fusion": {"method": "rrf"}})
    with pytest.raises(ValueError, match="no `fusion` section"):
        rrf_k({"seed": 42})


def test_the_shipped_config_holds_the_parameters_day_1_was_specified_with():
    # config.yaml is where the ablation will move these; this is the baseline they move from.
    live = load_config()["fusion"]
    assert (live["method"], live["rrf_k"], live["top_k_each"], live["top_k_final"]) == (
        "rrf",
        60,
        20,
        5,
    )
