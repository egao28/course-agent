"""Tests for retrieval/bm25.py: the acceptance query, the guards, and what a Hit carries.

The index is a real one, built in-process by ``ingest.build_bm25`` over the fixture corpus below --
so ``search`` runs through rank_bm25's own scoring rather than against a stub that would hand back
whatever ranking the test expected.

The corpus is small but not toy: it carries the hazard the live catalog carries. ``ECE 374``'s
whole description is "Same as CS 374. See CS 374." and ``CS 477`` cites CS 374 twice, so the query
``CS 374`` has three documents matching its terms and only one of them *is* CS 374. That is the
case ``bm25.code_boost`` exists for, and the case the acceptance test below asserts.

``redirect_paths`` is autouse: the corpus and ``INDEXES_DIR`` live in tmp_path, so no test reads
the live ``data/chunks.jsonl`` or writes into the real ``indexes/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from config import Config, bm25_path, load_config
from ingest import build_bm25 as bb
from ingest import chunk as ch
from ingest.chunk import chunk_key
from retrieval import bm25 as rb
from retrieval.bm25 import BM25Retriever
from retrieval.types import Hit

BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"

CHUNKS = [
    {
        "doc_id": "2026-sp:CS-374",
        "text": (
            "CS 374 — Introduction to Algorithms & Models of Computation\n"
            "Credits: 4\nTerm: 2026-sp\n"
            "Description: Analysis of algorithms, major paradigms of algorithm design including "
            "recursive algorithms, dynamic programming and graph algorithms. Formal models of "
            "computation. Same as ECE 374. Prerequisite: One of CS 173, MATH 213; CS 225.\n"
            "Prerequisites: One of CS 173, MATH 213; CS 225.\nDegree Attributes: None"
        ),
        "metadata": {
            "code": "CS 374",
            "credits": 4.0,
            "prereq_codes": ["CS 173", "MATH 213", "CS 225"],
            "gened_ids": [],
        },
    },
    {
        "doc_id": "2026-sp:ECE-374",
        "text": (
            "ECE 374 — Introduction to Algorithms & Models of Computation\n"
            "Credits: 4\nTerm: 2026-sp\n"
            "Description: Same as CS 374. See CS 374.\n"
            "Prerequisites: None listed\nDegree Attributes: None"
        ),
        "metadata": {"code": "ECE 374", "credits": 4.0, "prereq_codes": [], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:CS-477",
        "text": (
            "CS 477 — Formal Software Development Methods\nCredits: 3\nTerm: 2026-sp\n"
            "Description: Mathematical models for software specification. Prerequisite: CS 225; "
            "one of CS 374, ECE 374.\n"
            "Prerequisites: CS 225; one of CS 374, ECE 374.\nDegree Attributes: None"
        ),
        "metadata": {"code": "CS 477", "credits": 3.0, "prereq_codes": ["CS 225"], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:ENGL-104",
        "text": (
            "ENGL 104 — Poetry\nCredits: 3\nTerm: 2026-sp\n"
            "Description: Reading and writing poetry.\n"
            "Prerequisites: None listed\nDegree Attributes: Humanities - Lit & Arts"
        ),
        "metadata": {
            "code": "ENGL 104",
            "credits": 3.0,
            "prereq_codes": [],
            "gened_ids": ["humlit"],
        },
    },
    {
        "doc_id": "2026-sp:IS-407",
        "text": (
            "IS 407 — Introduction to Data Science\nCredits: 3\nTerm: 2026-sp\n"
            "Description: Data science with machine learning methods.\n"
            "Prerequisites: None listed\nDegree Attributes: None"
        ),
        "metadata": {"code": "IS 407", "credits": 3.0, "prereq_codes": [], "gened_ids": []},
    },
]


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch) -> dict[str, Path]:
    paths = {
        "CHUNKS_PATH": tmp_path / "chunks.jsonl",
        "CHUNKS_META_PATH": tmp_path / "chunks.meta.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(ch, name, path)
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    return paths


@pytest.fixture
def cfg() -> Config:
    raw = load_config(BASE_CONFIG).to_dict()
    raw["chunk"] = {
        "strategy": "one_course_one_chunk",
        "template_version": "v1",
        "fields": ["code", "name", "credits", "term", "desc", "prereqRaw", "attrs"],
    }
    raw["bm25"] = {"k1": 1.5, "b": 0.75, "tokenizer": "simple_v1", "code_boost": 2}
    raw["fusion"] = {**raw["fusion"], "top_k_each": 20}
    return Config(raw)


def write_corpus(paths, cfg) -> None:
    lines = [json.dumps(c, ensure_ascii=False) for c in CHUNKS]
    paths["CHUNKS_PATH"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"chunk_key": chunk_key(cfg), "chunks": len(CHUNKS)}
    paths["CHUNKS_META_PATH"].write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def built(cfg, redirect_paths) -> Config:
    """The fixture corpus, indexed where ``open_index`` will look for it."""
    write_corpus(redirect_paths, cfg)
    bb.build(cfg)
    return cfg


@pytest.fixture
def retriever(built) -> BM25Retriever:
    return BM25Retriever(built)


# --- the acceptance query -------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["CS 374", "cs374", "CS374", "cs 374"])
def test_a_course_code_finds_that_course_however_it_is_spelled(retriever, query):
    hits = retriever.search(query, 3)
    assert hits[0].doc_id == "2026-sp:CS-374"
    # Every spelling is the same query: the code is normalised before anything is scored.
    assert hits[0].score == pytest.approx(retriever.search("CS 374", 3)[0].score)


def test_the_course_outranks_the_stub_that_only_points_at_it(retriever):
    ranked = [h.doc_id for h in retriever.search("CS 374", 5)]
    assert ranked.index("2026-sp:CS-374") < ranked.index("2026-sp:ECE-374")


def test_without_code_boost_a_document_that_cites_the_course_wins_instead(cfg, redirect_paths):
    # The reason bm25.code_boost is not 0. On text alone, ECE 374 is a short document matching the
    # query twice and CS 477 cites the code twice in a longer one -- BM25 is right on its own
    # terms and ranks both above the course itself, because nothing in the text says which of them
    # *is* CS 374. Assert that failure, so a change that quietly disables the boost cannot pass
    # for a passing suite. (How much boost it takes is corpus-dependent: 1 is enough here, the
    # live 2026-sp corpus needs 2 -- see ingest/tokenize.py.)
    plain = Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "code_boost": 0}})
    write_corpus(redirect_paths, plain)
    bb.build(plain)
    ranked = [h.doc_id for h in BM25Retriever(plain).search("CS 374", 5)]
    assert ranked[0] != "2026-sp:CS-374"
    assert ranked.index("2026-sp:ECE-374") < ranked.index("2026-sp:CS-374")


# --- hits -----------------------------------------------------------------------------------------


def test_search_fills_in_every_field_of_a_hit(retriever):
    hit = retriever.search("CS 374", 1)[0]
    assert isinstance(hit, Hit)
    assert hit.rank == 1
    assert hit.score > 0
    assert hit.text == CHUNKS[0]["text"]
    assert hit.metadata["code"] == "CS 374"
    # Lists, like the dense side decodes them back into -- a Hit means one thing either way.
    assert hit.metadata["prereq_codes"] == ["CS 173", "MATH 213", "CS 225"]


def test_hits_are_ranked_from_one_by_descending_score(retriever):
    hits = retriever.search("algorithms", 5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_a_hit_carries_a_copy_of_its_metadata(retriever):
    retriever.search("CS 374", 1)[0].metadata["code"] = "mutated"
    assert retriever.search("CS 374", 1)[0].metadata["code"] == "CS 374"


def test_search_stops_at_k(retriever):
    assert len(retriever.search("algorithms", 1)) == 1


def test_search_defaults_to_the_recall_depth_fusion_asks_for(built):
    shallow = Config({**built.to_dict(), "fusion": {**built["fusion"], "top_k_each": 1}})
    assert len(BM25Retriever(shallow).search("algorithms")) == 1


def test_search_returns_only_documents_a_query_term_touched(retriever):
    # Not padded out to k: fusion ranks every list it is handed, and a document nothing matched
    # has not been retrieved.
    hits = retriever.search("poetry", 5)
    assert [h.doc_id for h in hits] == ["2026-sp:ENGL-104"]


def test_a_query_of_nothing_but_stopwords_retrieves_nothing(retriever):
    assert retriever.search("the and of") == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_search_refuses_an_empty_query(retriever, query):
    with pytest.raises(ValueError, match="query is empty"):
        retriever.search(query)


@pytest.mark.parametrize("k", [0, -1])
def test_search_refuses_a_non_positive_k(retriever, k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("algorithms", k)


def test_ties_are_broken_by_corpus_order(cfg, redirect_paths):
    # A duplicated document scores identically to its twin; the ranking must still come out the
    # same on every run and every machine, or a Day 4 ablation measures the shuffle.
    twin = dict(CHUNKS[3], doc_id="2026-sp:ENGL-105")
    lines = [json.dumps(c) for c in [*CHUNKS, twin]]
    redirect_paths["CHUNKS_PATH"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"chunk_key": chunk_key(cfg), "chunks": len(CHUNKS) + 1}
    redirect_paths["CHUNKS_META_PATH"].write_text(json.dumps(meta), encoding="utf-8")
    bb.build(cfg)
    hits = BM25Retriever(cfg).search("poetry", 2)
    assert hits[0].score == pytest.approx(hits[1].score)
    assert [h.doc_id for h in hits] == ["2026-sp:ENGL-104", "2026-sp:ENGL-105"]


# --- guards ---------------------------------------------------------------------------------------


def test_open_index_points_at_the_build_when_there_is_no_index(cfg):
    with pytest.raises(FileNotFoundError, match="ingest.build_bm25"):
        rb.open_index(cfg)


def test_open_index_does_not_create_what_it_could_not_find(cfg):
    # Naming an index must not bring its directory into existence -- the same rule chroma_path
    # follows, for the same reason: a failed query should leave nothing behind.
    with pytest.raises(FileNotFoundError):
        rb.open_index(cfg)
    assert not bm25_path(cfg, chunk_key(cfg)).parent.exists()


def test_a_retriever_refuses_an_index_built_under_another_key(built):
    index = {**bb.load_index(bm25_path(built, chunk_key(built))), "bm25_key": "deadbeef"}
    with pytest.raises(ValueError, match="bm25_key 'deadbeef'"):
        BM25Retriever(built, index=index)


def test_a_retriever_refuses_an_index_tokenized_differently(built):
    index = {**bb.load_index(bm25_path(built, chunk_key(built))), "tokenizer": "porter_v9"}
    with pytest.raises(ValueError, match="tokenized by 'porter_v9'"):
        BM25Retriever(built, index=index)


def test_a_retriever_refuses_columns_that_do_not_line_up(built):
    # A short column would not fail, it would answer with the wrong course.
    index = bb.load_index(bm25_path(built, chunk_key(built)))
    index["texts"] = index["texts"][:-1]
    with pytest.raises(ValueError, match="mismatched columns"):
        BM25Retriever(built, index=index)


def test_a_config_without_a_recall_depth_says_which_key_is_missing(built):
    fusion = {k: v for k, v in built["fusion"].items() if k != "top_k_each"}
    bare = Config({**built.to_dict(), "fusion": fusion})
    with pytest.raises(ValueError, match="fusion section is missing top_k_each"):
        BM25Retriever(bare)


# --- CLI ------------------------------------------------------------------------------------------


def test_main_prints_the_ranked_hits(built, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(built.to_dict()), encoding="utf-8")
    rb.main(["CS", "374", "-k", "2", "--config", str(path), "--terms"])
    out = capsys.readouterr().out
    assert "2 hits" in out
    assert "code:cs374" in out  # --terms shows what the query actually asked for
    assert "  1. " in out
    assert "CS 374 — Introduction to Algorithms" in out
