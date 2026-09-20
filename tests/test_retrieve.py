"""Tests for retrieval/__init__.py: the one interface, the cache behind it, and the table.

The retrievers are fakes, injected through ``BUILDERS``. That is deliberate and not a shortcut:
what this module does is choose *which* retrievers run, with what depth, how often they are built,
and what the result says about the run -- none of which is a fact about Chroma or rank_bm25, both
of which have their own tests against real indexes. A fake also lets a test count builds and
record the ``k`` each retriever was asked for, which is most of what #8 specifies.

The fake corpus is arranged so the two rankings disagree: ``CS 374`` is first on the lexical side
and third on the dense one, ``CS 441`` is only ever found by one of them. That is the shape RRF
exists for, and the shape the ``sources`` column has to be able to show.
"""

from __future__ import annotations

from collections import Counter
from time import sleep
from types import SimpleNamespace

import pytest

import retrieval
from config import Config, load_config
from retrieval import (
    BUILDERS,
    MODE_SOURCES,
    MODES,
    RetrievalResult,
    check_mode,
    clear_cache,
    format_result,
    format_table,
    get_retriever,
    hit_name,
    hit_sources,
    result_k,
    retrieve,
)
from retrieval.__main__ import duration
from retrieval.__main__ import main as cli
from retrieval.types import Hit

CFG = Config(
    {
        "seed": 42,
        "fusion": {"method": "rrf", "rrf_k": 60, "top_k_each": 20, "top_k_final": 5},
    }
)

TITLES = {
    "2026-sp:CS-446": "Machine Learning",
    "2026-sp:CS-441": "Applied Machine Learning",
    "2026-sp:CS-374": "Introduction to Algorithms & Models of Computation",
    "2026-sp:MATH-213": "Basic Discrete Mathematics",
    "2026-sp:ENGL-104": "Introduction to Film",
}

#: Two rankings of the same corpus that agree on some documents and not on others.
DENSE_ORDER = ["2026-sp:CS-446", "2026-sp:CS-441", "2026-sp:CS-374", "2026-sp:ENGL-104"]
BM25_ORDER = ["2026-sp:CS-374", "2026-sp:CS-446", "2026-sp:MATH-213", "2026-sp:ENGL-104"]


def code_of(doc_id: str) -> str:
    """``2026-sp:CS-374`` -> ``CS 374``, the way ``ingest.chunk.doc_id`` built it."""
    return doc_id.split(":", 1)[1].replace("-", " ")


class FakeRetriever:
    """Returns a fixed ranking, and remembers every ``(query, k)`` it was asked for."""

    def __init__(self, name: str, doc_ids: list[str], search_ms: float = 0.0) -> None:
        self.name = name
        self.doc_ids = doc_ids
        self.search_ms = search_ms
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        self.calls.append((query, k))
        if self.search_ms:
            sleep(self.search_ms / 1000)
        doc_ids = self.doc_ids if k is None else self.doc_ids[:k]
        return [
            Hit(
                doc_id=doc_id,
                score=1.0 / rank,
                rank=rank,
                text=f"{code_of(doc_id)} — {TITLES[doc_id]}\nCredits: 3",
                metadata={"code": code_of(doc_id)},
            )
            for rank, doc_id in enumerate(doc_ids, start=1)
        ]


@pytest.fixture(autouse=True)
def fakes(monkeypatch) -> SimpleNamespace:
    """Both retrievers replaced by fakes, with the process-wide cache emptied around each test."""
    clear_cache()
    state = SimpleNamespace(built=Counter(), made={})

    def builder(name: str, doc_ids: list[str]):
        def build(cfg):
            state.built[name] += 1
            state.made[name] = FakeRetriever(name, doc_ids)
            return state.made[name]

        return build

    monkeypatch.setitem(BUILDERS, "dense", builder("dense", DENSE_ORDER))
    monkeypatch.setitem(BUILDERS, "bm25", builder("bm25", BM25_ORDER))
    yield state
    clear_cache()


# --- one interface, three modes -------------------------------------------------------------------


def test_all_three_modes_answer_the_same_query_with_the_same_shape():
    # #8's acceptance: the ablation switches `mode` and nothing else about how it reads the result.
    results = [retrieve("machine learning", mode, cfg=CFG) for mode in MODES]
    for result, mode in zip(results, MODES, strict=True):
        assert isinstance(result, RetrievalResult)
        assert result.query == "machine learning"
        assert result.mode == mode
        assert result.config_hash == CFG.hash
        assert all(isinstance(hit, Hit) for hit in result.hits)
        assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))


def test_hybrid_is_the_default_mode():
    assert retrieve("machine learning", cfg=CFG).mode == "hybrid"


def test_a_single_retriever_mode_returns_that_retriever_s_own_ranking():
    assert [hit.doc_id for hit in retrieve("q", "dense", cfg=CFG).hits] == DENSE_ORDER
    assert [hit.doc_id for hit in retrieve("q", "bm25", cfg=CFG).hits] == BM25_ORDER


def test_a_single_retriever_mode_leaves_the_scores_the_retriever_gave():
    hits = retrieve("q", "bm25", cfg=CFG).hits
    assert [hit.score for hit in hits] == [1.0, 0.5, 1 / 3, 0.25]
    assert all(hit.sources is None for hit in hits)


def test_hybrid_ranks_what_both_retrievers_found_above_what_one_did():
    hits = retrieve("machine learning", "hybrid", cfg=CFG).hits
    # CS 446 (1st + 2nd) and CS 374 (3rd + 1st) are found twice; CS 441 only by the dense side.
    assert [hit.doc_id for hit in hits[:3]] == [
        "2026-sp:CS-446",
        "2026-sp:CS-374",
        "2026-sp:ENGL-104",
    ]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0].sources == {"dense": 1, "bm25": 2}


def test_hybrid_reports_the_retriever_that_did_not_find_a_document():
    hits = retrieve("machine learning", "hybrid", cfg=CFG).hits
    only_dense = next(hit for hit in hits if hit.doc_id == "2026-sp:CS-441")
    assert only_dense.sources == {"dense": 2, "bm25": None}


# --- depth ----------------------------------------------------------------------------------------


def test_k_defaults_to_the_final_depth_in_every_mode():
    # Four documents per ranking, five distinct between them, against fusion.top_k_final = 5:
    # each mode returns everything it found rather than a number of its own choosing.
    found = {mode: len(retrieve("q", mode, cfg=CFG).hits) for mode in MODES}
    assert found == {"dense": 4, "bm25": 4, "hybrid": 5}


def test_a_single_retriever_mode_asks_for_exactly_what_the_caller_wants(fakes):
    retrieve("q", "bm25", k=2, cfg=CFG)
    assert fakes.made["bm25"].calls == [("q", 2)]
    assert len(retrieve("q", "bm25", k=2, cfg=CFG).hits) == 2


def test_hybrid_asks_each_retriever_for_the_fusion_depth_not_the_final_one(fakes):
    result = retrieve("q", "hybrid", cfg=CFG)
    assert fakes.made["dense"].calls == [("q", 20)]  # fusion.top_k_each
    assert fakes.made["bm25"].calls == [("q", 20)]
    assert len(result.hits) == 5  # ... cut to fusion.top_k_final


def test_a_k_past_the_fusion_depth_deepens_recall_instead_of_returning_short(fakes):
    # 20 hits each can fuse to at most 40 documents, so a k of 50 has to ask for more than 20.
    retrieve("q", "hybrid", k=50, cfg=CFG)
    assert fakes.made["dense"].calls == [("q", 50)]


def test_a_k_inside_the_fusion_depth_leaves_the_tuned_recall_alone(fakes):
    retrieve("q", "hybrid", k=3, cfg=CFG)
    assert fakes.made["bm25"].calls == [("q", 20)]
    assert len(retrieve("q", "hybrid", k=3, cfg=CFG).hits) == 3


def test_result_k_falls_back_to_the_config():
    assert result_k(CFG, None) == 5
    assert result_k(CFG, 12) == 12


# --- the cache ------------------------------------------------------------------------------------


def test_a_retriever_is_built_once_and_reused_for_every_later_query(fakes):
    for _ in range(3):
        retrieve("machine learning", "hybrid", cfg=CFG)
    assert fakes.built == {"dense": 1, "bm25": 1}
    assert len(fakes.made["dense"].calls) == 3


def test_a_mode_builds_only_the_retriever_it_uses(fakes):
    # The reason bm25 mode needs no API key and no dense index: nothing dense is ever constructed.
    retrieve("CS 374", "bm25", cfg=CFG)
    assert fakes.built == {"bm25": 1}


def test_a_different_config_does_not_reuse_the_retriever_built_for_another(fakes):
    other = Config({**CFG.to_dict(), "seed": 43})
    retrieve("q", "bm25", cfg=CFG)
    retrieve("q", "bm25", cfg=other)
    assert other.hash != CFG.hash
    assert fakes.built["bm25"] == 2


def test_clear_cache_makes_the_next_query_reopen_the_index(fakes):
    retrieve("q", "bm25", cfg=CFG)
    clear_cache()
    retrieve("q", "bm25", cfg=CFG)
    assert fakes.built["bm25"] == 2


def test_a_build_that_failed_is_not_remembered_as_a_failure(monkeypatch, fakes):
    # Querying before `build_bm25` has been run, then running it, should work in the same process.
    calls = Counter()

    def flaky(cfg):
        calls["build"] += 1
        if calls["build"] == 1:
            raise FileNotFoundError("no BM25 index at indexes/bm25_877a6610.pkl")
        return FakeRetriever("bm25", BM25_ORDER)

    monkeypatch.setitem(BUILDERS, "bm25", flaky)
    with pytest.raises(FileNotFoundError):
        retrieve("q", "bm25", cfg=CFG)
    assert len(retrieve("q", "bm25", cfg=CFG).hits) == 4


def test_get_retriever_names_the_retrievers_there_are():
    with pytest.raises(KeyError, match="known retrievers: bm25, dense"):
        get_retriever("keyword", CFG)


def test_every_mode_names_retrievers_that_can_be_built():
    for names in MODE_SOURCES.values():
        assert set(names) <= set(BUILDERS)
    assert set(MODE_SOURCES) == set(MODES)


# --- what is refused, and when --------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_an_empty_query_is_refused_before_an_index_is_opened(query, fakes):
    with pytest.raises(ValueError, match="query is empty"):
        retrieve(query, "hybrid", cfg=CFG)
    assert fakes.built == {}


def test_an_unknown_mode_names_the_modes_there_are(fakes):
    with pytest.raises(ValueError, match="unknown mode 'keyword'; modes are dense, bm25, hybrid"):
        retrieve("q", "keyword", cfg=CFG)
    assert fakes.built == {}


@pytest.mark.parametrize("k", [0, -1, 2.5, "5", True])
def test_a_k_that_is_not_a_positive_whole_number_is_refused(k, fakes):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retrieve("q", "hybrid", k=k, cfg=CFG)
    assert fakes.built == {}


def test_check_mode_returns_the_mode_it_accepted():
    assert check_mode("hybrid") == "hybrid"


# --- the config -----------------------------------------------------------------------------------


def test_a_plain_mapping_is_frozen_so_a_later_edit_cannot_reach_a_cached_retriever(fakes):
    raw = CFG.to_dict()
    first = retrieve("q", "bm25", cfg=raw)
    raw["fusion"]["top_k_final"] = 1
    second = retrieve("q", "bm25", cfg=raw)
    assert first.config_hash == CFG.hash
    assert second.config_hash != first.config_hash  # the edit names a different run ...
    assert fakes.built["bm25"] == 2  # ... and gets its own retriever, not the one already built


def test_no_config_means_the_live_one(fakes):
    result = retrieve("machine learning", "bm25")
    assert result.config_hash == load_config().hash


# --- latency --------------------------------------------------------------------------------------


def test_latency_times_the_query_and_not_the_index_load(monkeypatch):
    def slow_build(cfg):
        sleep(0.05)
        return FakeRetriever("bm25", BM25_ORDER)

    monkeypatch.setitem(BUILDERS, "bm25", slow_build)
    result = retrieve("q", "bm25", cfg=CFG)
    assert 0 <= result.latency_ms < 50


def test_latency_covers_the_search_itself(monkeypatch):
    monkeypatch.setitem(BUILDERS, "bm25", lambda cfg: FakeRetriever("bm25", BM25_ORDER, 20))
    assert retrieve("q", "bm25", cfg=CFG).latency_ms >= 20


# --- the table ------------------------------------------------------------------------------------


def test_the_table_has_a_header_and_a_row_per_hit():
    table = format_table(retrieve("machine learning", "hybrid", cfg=CFG)).splitlines()
    assert table[0].split() == ["#", "doc_id", "score", "name", "sources"]
    assert len(table) == 6  # the header, and the five documents the two rankings hold between them
    assert table[1].startswith("1  2026-sp:CS-446")


def test_the_name_column_drops_the_code_the_doc_id_column_already_carries():
    hit = retrieve("q", "dense", cfg=CFG).hits[0]
    assert hit.text.startswith("CS 446 — ")
    assert hit_name(hit) == "Machine Learning"


def test_a_name_falls_back_to_the_doc_id_when_the_text_does_not_carry_one():
    assert hit_name(Hit("2026-sp:CS-374", 1.0, 1, "", {})) == "2026-sp:CS-374"


def test_a_title_too_long_for_the_column_is_elided():
    row = format_table(retrieve("q", "bm25", cfg=CFG)).splitlines()[1]
    assert "Introduction to Algorithms & Models of" in row
    assert TITLES["2026-sp:CS-374"] not in row
    assert "…" in row


def test_a_single_mode_fills_the_sources_column_from_the_mode():
    result = retrieve("q", "bm25", cfg=CFG)
    assert hit_sources(result.hits[2], "bm25") == "bm25 3"
    assert "bm25 1" in format_table(result).splitlines()[1]


def test_a_fused_row_shows_both_ranks_and_marks_the_retriever_that_missed_it():
    result = retrieve("machine learning", "hybrid", cfg=CFG)
    rows = {hit.doc_id: hit_sources(hit, result.mode) for hit in result.hits}
    assert rows["2026-sp:CS-446"] == "dense 1, bm25 2"
    assert rows["2026-sp:CS-441"] == "dense 2, bm25 —"


def test_scores_are_printed_deep_enough_to_tell_two_rrf_scores_apart():
    # 1/61 = 0.016393 and 1/62 = 0.016129 are the same number at three decimal places.
    table = format_table(retrieve("machine learning", "hybrid", cfg=CFG))
    assert "0.0325" in table and "0.0323" in table


def test_an_empty_result_says_so_rather_than_printing_a_bare_header(monkeypatch):
    monkeypatch.setitem(BUILDERS, "bm25", lambda cfg: FakeRetriever("bm25", []))
    assert format_table(retrieve("q", "bm25", cfg=CFG)) == "no hits for 'q' in bm25 mode"


def test_the_summary_line_says_what_ran_and_under_which_config():
    summary = format_result(retrieve("machine learning", "hybrid", cfg=CFG)).splitlines()[0]
    assert summary.startswith("'machine learning'  ->  5 hits, mode hybrid, config ")
    assert CFG.hash in summary and "ms" in summary


# --- the CLI --------------------------------------------------------------------------------------


def test_the_cli_prints_the_table_for_the_mode_it_was_given(capsys):
    cli(["CS", "374", "--mode", "bm25"])
    out = capsys.readouterr().out
    assert "mode bm25" in out
    assert "2026-sp:CS-374" in out.splitlines()[3]
    assert load_config().hash in out


def test_the_cli_honours_k(capsys):
    cli(["machine learning", "--mode", "dense", "-k", "2"])
    assert "2 hits" in capsys.readouterr().out


def test_the_cli_reports_the_index_load_apart_from_the_query_time(monkeypatch, capsys):
    monkeypatch.setitem(
        BUILDERS, "bm25", lambda cfg: sleep(0.01) or FakeRetriever("bm25", BM25_ORDER)
    )
    cli(["CS 374", "--mode", "bm25"])
    assert "opening the bm25 index, once per process" in capsys.readouterr().out


def test_the_cli_says_nothing_about_opening_an_index_it_found_already_open(capsys):
    retrieve("CS 374", "bm25", cfg=load_config())
    cli(["CS 374", "--mode", "bm25"])
    assert "opening" not in capsys.readouterr().out


def test_the_cli_refuses_a_mode_that_is_not_one(capsys):
    with pytest.raises(SystemExit):
        cli(["q", "--mode", "keyword"])
    assert "invalid choice" in capsys.readouterr().err


def test_durations_read_in_the_unit_a_person_would_have_said():
    assert duration(84.2) == "84 ms"
    assert duration(1400) == "1.4 s"


def test_the_package_docstring_example_is_the_signature_the_ablation_was_promised():
    # Day 4 calls this directly, so the parameter names are part of the contract, not a detail.
    assert retrieval.retrieve.__code__.co_varnames[:4] == ("query", "mode", "k", "cfg")
