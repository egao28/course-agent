"""Tests for ingest/build_bm25.py: the config guards, what lands in the pickle, and reading it back.

The index built here is a real ``BM25Okapi`` over a real (tiny) corpus written to tmp_path -- there
is nothing expensive to fake, and the properties worth testing are the ones a stub would define
away: that the parallel columns stay in corpus order, that ``code_boost`` reaches the terms, and
that a file which cannot be read is answered with the command that rewrites it.

``redirect_paths`` is autouse and points the corpus, the sidecar and ``INDEXES_DIR`` into tmp_path:
no test can read the live ``data/chunks.jsonl`` or write into the real ``indexes/``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

import config
from config import Config, bm25_key, bm25_path, load_config
from ingest import build_bm25 as bb
from ingest import chunk as ch
from ingest.chunk import chunk_key

BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"

CHUNKS = [
    {
        "doc_id": "2026-sp:CS-374",
        "text": "CS 374 — Algorithms\nDescription: analysis of algorithms. Same as ECE 374.",
        "metadata": {"code": "CS 374", "credits": 4.0, "prereq_codes": ["CS 173"], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:ECE-374",
        "text": "ECE 374 — Algorithms\nDescription: Same as CS 374. See CS 374.",
        "metadata": {"code": "ECE 374", "credits": 4.0, "prereq_codes": [], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:ENGL-104",
        "text": "ENGL 104 — Poetry\nDescription: reading poetry",
        "metadata": {
            "code": "ENGL 104",
            "credits": 3.0,
            "prereq_codes": [],
            "gened_ids": ["humlit"],
        },
    },
]


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch) -> dict[str, Path]:
    """Corpus, sidecar and index directory, all inside tmp_path."""
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
    """A config with the sections this module reads, spelled out rather than read from the repo."""
    raw = load_config(BASE_CONFIG).to_dict()
    raw["chunk"] = {
        "strategy": "one_course_one_chunk",
        "template_version": "v1",
        "fields": ["code", "name", "credits", "term", "desc", "prereqRaw", "attrs"],
    }
    raw["bm25"] = {"k1": 1.5, "b": 0.75, "tokenizer": "simple_v1", "code_boost": 2}
    return Config(raw)


def write_corpus(paths, cfg, chunks=CHUNKS, key: str | None = None, count: int | None = None):
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    paths["CHUNKS_PATH"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "chunk_key": chunk_key(cfg) if key is None else key,
        "chunks": len(chunks) if count is None else count,
    }
    paths["CHUNKS_META_PATH"].write_text(json.dumps(meta), encoding="utf-8")
    return paths


@pytest.fixture
def corpus(cfg, redirect_paths) -> dict[str, Path]:
    return write_corpus(redirect_paths, cfg)


# --- the config guard -----------------------------------------------------------------------------


def test_check_bm25_config_returns_the_section(cfg):
    assert bb.check_bm25_config(cfg)["tokenizer"] == "simple_v1"


@pytest.mark.parametrize("key", ["k1", "b"])
def test_a_missing_scoring_parameter_is_named(cfg, key):
    bare = {k: v for k, v in cfg["bm25"].items() if k != key}
    with pytest.raises(ValueError, match=f"bm25 section is missing {key}"):
        bb.check_bm25_config(Config({**cfg.to_dict(), "bm25": bare}))


@pytest.mark.parametrize("value", [-0.5, "1.5", True, None])
def test_k1_must_be_a_non_negative_number(cfg, value):
    with pytest.raises(ValueError, match="bm25.k1 must be"):
        bb.check_bm25_config(Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "k1": value}}))


@pytest.mark.parametrize("value", [-0.1, 1.5, "0.75", True, None])
def test_b_must_be_a_fraction(cfg, value):
    # b scales length normalisation between none (0) and full (1); outside that it is not a
    # setting, it is a typo that would quietly reweight every document.
    with pytest.raises(ValueError, match="bm25.b must be"):
        bb.check_bm25_config(Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "b": value}}))


def test_the_tokenizer_and_the_boost_are_checked_by_the_same_call(cfg):
    named = Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "tokenizer": "porter_v9"}})
    with pytest.raises(ValueError, match="unknown bm25.tokenizer"):
        bb.check_bm25_config(named)
    bare = {k: v for k, v in cfg["bm25"].items() if k != "code_boost"}
    with pytest.raises(ValueError, match="missing code_boost"):
        bb.check_bm25_config(Config({**cfg.to_dict(), "bm25": bare}))


# --- building -------------------------------------------------------------------------------------


def test_build_writes_the_index_its_config_names(cfg, corpus):
    result = bb.build(cfg)
    assert result["path"] == bm25_path(cfg, chunk_key(cfg))
    assert result["path"].name == f"bm25_{bm25_key(cfg, chunk_key(cfg))}.pkl"
    assert result["path"].is_file()


def test_build_records_how_it_was_built(cfg, corpus):
    index = bb.load_index(bb.build(cfg)["path"])
    assert index["format"] == bb.FORMAT
    assert index["bm25_key"] == bm25_key(cfg, chunk_key(cfg))
    assert index["chunk_key"] == chunk_key(cfg)
    assert index["config_hash"] == cfg.hash
    assert (index["tokenizer"], index["code_boost"]) == ("simple_v1", 2)
    assert (index["k1"], index["b"]) == (1.5, 0.75)
    assert index["chunks"] == len(CHUNKS)
    assert index["built_at"].endswith("Z")


def test_build_stores_the_corpus_columns_in_order(cfg, corpus):
    index = bb.load_index(bb.build(cfg)["path"])
    assert index["doc_ids"] == [c["doc_id"] for c in CHUNKS]
    assert index["texts"] == [c["text"] for c in CHUNKS]
    # Lists stay lists: only Chroma needed them flattened, so a BM25 hit and a dense hit agree.
    assert index["metadatas"][0]["prereq_codes"] == ["CS 173"]


def test_build_indexes_every_document(cfg, corpus):
    result = bb.build(cfg)
    assert result["bm25"].corpus_size == len(CHUNKS)
    assert result["tally"].docs == len(CHUNKS)


def test_code_boost_reaches_the_terms(cfg, corpus):
    boosted = bb.build(cfg)["tally"]
    plain = bb.build(Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "code_boost": 0}}))["tally"]
    # Two extra terms per document, and the identity terms are in the vocabulary.
    assert boosted.tokens - plain.tokens == 2 * len(CHUNKS)
    assert boosted.terms - plain.terms == len(CHUNKS)


def test_a_boost_of_zero_leaves_a_separate_index(cfg, corpus):
    plain = Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "code_boost": 0}})
    assert bb.build(cfg)["path"] != bb.build(plain)["path"]


def test_building_twice_overwrites_in_place(cfg, corpus):
    first = bb.build(cfg)["path"]
    second = bb.build(cfg)["path"]
    assert first == second
    assert bb.load_index(second)["chunks"] == len(CHUNKS)
    assert sorted(p.name for p in first.parent.iterdir()) == [first.name]


def test_build_refuses_a_corpus_from_another_chunk_template(cfg, redirect_paths):
    write_corpus(redirect_paths, cfg, key="deadbeef")
    with pytest.raises(ValueError, match="chunk_key 'deadbeef'"):
        bb.build(cfg)
    assert not bm25_path(cfg, chunk_key(cfg)).exists()


def test_build_points_at_the_chunker_when_there_is_no_corpus(cfg):
    with pytest.raises(FileNotFoundError, match="ingest.chunk"):
        bb.build(cfg)


# --- the file -------------------------------------------------------------------------------------


def test_a_failed_write_leaves_the_previous_index_and_no_debris(cfg, corpus, monkeypatch):
    path = bb.build(cfg)["path"]
    good = path.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bb.pickle, "dump", explode)
    with pytest.raises(OSError, match="disk full"):
        bb.build(cfg)
    assert path.read_bytes() == good
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_load_index_names_the_rebuild_for_a_file_it_cannot_read(cfg, corpus):
    path = bb.build(cfg)["path"]
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="ingest.build_bm25"):
        bb.load_index(path)


def test_load_index_refuses_a_pickle_that_is_not_an_index(cfg, corpus):
    path = bb.build(cfg)["path"]
    path.write_bytes(pickle.dumps(["CS 374"]))
    with pytest.raises(ValueError, match="does not hold a BM25 index"):
        bb.load_index(path)


def test_load_index_names_the_key_a_reader_would_have_missed(cfg, corpus):
    path = bb.build(cfg)["path"]
    index = bb.load_index(path)
    del index["texts"]
    path.write_bytes(pickle.dumps(index))
    with pytest.raises(ValueError, match="missing texts"):
        bb.load_index(path)


def test_load_index_refuses_another_layout_version(cfg, corpus):
    path = bb.build(cfg)["path"]
    index = bb.load_index(path)
    index["format"] = bb.FORMAT + 1
    path.write_bytes(pickle.dumps(index))
    with pytest.raises(ValueError, match="is format"):
        bb.load_index(path)


# --- reporting ------------------------------------------------------------------------------------


def test_summarize_reports_the_corpus_and_where_it_landed(cfg, corpus):
    lines = "\n".join(bb.summarize(bb.build(cfg)))
    assert f"documents indexed      {len(CHUNKS)}" in lines
    assert "k1 / b                 1.5 / 0.75" in lines
    assert "indexes/" not in lines or "bm25_" in lines  # a repo-relative path, never an absolute


def test_main_prints_the_keys_it_built_under(cfg, corpus, tmp_path, capsys, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")  # JSON is valid YAML
    bb.main(["--config", str(path)])
    out = capsys.readouterr().out
    assert f"config hash {cfg.hash}" in out
    assert f"bm25 key {bm25_key(cfg, chunk_key(cfg))}" in out
