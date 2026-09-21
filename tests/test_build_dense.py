"""Tests for ingest/build_dense.py: the freshness guards, idempotency, and what lands in Chroma.

The store is real -- a ``PersistentClient`` under ``tmp_path`` -- because the properties worth
testing here are the ones a fake store would define away: that the collection is created in cosine
space, that ``count()`` is what the idempotency check reads, that the flattened metadata survives a
round trip. Only the *embedder* is fake, since that is the part that costs money.

``redirect_paths`` is autouse and points the corpus, the sidecar and ``INDEXES_DIR`` into tmp_path:
no test can read the live ``data/chunks.jsonl`` or write into the real ``indexes/chroma``. It
patches them on ``ingest.chunk``, which owns both the writing and the reading of the corpus.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import config
from config import Config, collection_name, load_config
from embedding import Vectors
from ingest import build_dense as bd
from ingest import chunk as ch
from ingest.chunk import chunk_key

BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"

#: A tiny vocabulary the fake embedder counts, so "machine learning" really is nearest the
#: machine-learning chunk and the ranking assertions mean something.
VOCAB = ("machine", "learning", "algorithm", "poetry")

CHUNKS = [
    {
        "doc_id": "2026-sp:CS-446",
        "text": "CS 446 - Machine Learning\nDescription: machine learning machine learning",
        "metadata": {"code": "CS 446", "credits": 3.0, "prereq_codes": ["CS 374"], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:CS-374",
        "text": "CS 374 - Algorithms\nDescription: algorithm algorithm",
        "metadata": {"code": "CS 374", "credits": 4.0, "prereq_codes": ["CS 173"], "gened_ids": []},
    },
    {
        "doc_id": "2026-sp:ENGL-104",
        "text": "ENGL 104 - Poetry\nDescription: poetry",
        "metadata": {
            "code": "ENGL 104",
            "credits": 3.0,
            "prereq_codes": [],
            "gened_ids": ["humlit"],
        },
    },
    {
        "doc_id": "2026-sp:STAT-100",
        "text": "STAT 100 - Statistics\nDescription: learning from data",
        "metadata": {"code": "STAT 100", "credits": 3.0, "prereq_codes": [], "gened_ids": ["qr1"]},
    },
    {
        "doc_id": "2026-sp:DANC-100",
        "text": "DANC 100 - Dance\nDescription: movement",
        "metadata": {"code": "DANC 100", "credits": 1.0, "prereq_codes": [], "gened_ids": []},
    },
]


def vector(text: str) -> list[float]:
    """Normalized word counts over VOCAB, plus a bias term so no vector is the zero vector."""
    words = text.lower().split()
    raw = [float(words.count(word)) for word in VOCAB] + [1.0]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class FakeEmbedder:
    """Deterministic, offline, and it records every batch it was handed."""

    model = "fake-embedding-3-small"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts) -> Vectors:
        texts = list(texts)
        self.batches.append(texts)
        return Vectors([vector(t) for t in texts], sum(len(t.split()) for t in texts))


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
    raw["embedding"] = {"provider": "openai", "model": "text-embedding-3-small", "batch_size": 256}
    return Config(raw)


@pytest.fixture
def corpus(cfg, redirect_paths) -> dict[str, Path]:
    """Write CHUNKS and a matching sidecar where the build will look for them."""
    write_corpus(redirect_paths, cfg, CHUNKS)
    return redirect_paths


def write_corpus(paths, cfg, chunks, key: str | None = None, count: int | None = None) -> None:
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    paths["CHUNKS_PATH"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "chunk_key": chunk_key(cfg) if key is None else key,
        "chunks": len(chunks) if count is None else count,
        "tokens": {"encoding": "cl100k_base", "total": 1_000_000, "limit": 8191},
    }
    paths["CHUNKS_META_PATH"].write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def store(cfg):
    return bd.open_store()


# --- reading the corpus ---------------------------------------------------------------------------


def test_load_chunks_reads_every_line(corpus):
    assert [c["doc_id"] for c in bd.load_chunks()] == [c["doc_id"] for c in CHUNKS]


def test_load_chunks_points_at_the_chunker_when_the_corpus_is_missing():
    with pytest.raises(FileNotFoundError, match="ingest.chunk"):
        bd.load_chunks()


def test_load_chunks_rejects_an_empty_corpus(corpus):
    corpus["CHUNKS_PATH"].write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        bd.load_chunks()


def test_load_chunks_names_the_line_that_is_missing_a_field(corpus, redirect_paths, cfg):
    write_corpus(redirect_paths, cfg, [CHUNKS[0], {"doc_id": "x", "text": "y"}])
    with pytest.raises(ValueError, match=r"chunks.jsonl:2 is missing metadata"):
        bd.load_chunks()


def test_load_chunks_refuses_a_repeated_doc_id(corpus, redirect_paths, cfg):
    # Chroma keeps the first of two rows with one id and drops the second without a word, so the
    # build would pay for 6 chunks and hold 5 -- and every later run would demand a --rebuild that
    # lands in exactly the same place.
    write_corpus(redirect_paths, cfg, [*CHUNKS, CHUNKS[0]])
    with pytest.raises(ValueError, match=r"chunks.jsonl:6 repeats doc_id '2026-sp:CS-446'"):
        bd.load_chunks()


def test_load_chunks_meta_points_at_the_chunker_when_the_sidecar_is_missing():
    with pytest.raises(FileNotFoundError, match="ingest.chunk"):
        bd.load_chunks_meta()


# --- freshness ------------------------------------------------------------------------------------


def test_check_chunks_accepts_a_corpus_the_config_describes(corpus, cfg):
    bd.check_chunks(bd.load_chunks(), bd.load_chunks_meta(), cfg)


def test_check_chunks_rejects_another_chunk_template(corpus, redirect_paths, cfg):
    # The template version is hashed into the collection name, so a corpus from another template
    # would be embedded under a name that describes chunks it does not contain.
    write_corpus(redirect_paths, cfg, CHUNKS, key="deadbeef")
    with pytest.raises(ValueError, match="chunk_key 'deadbeef'"):
        bd.check_chunks(bd.load_chunks(), bd.load_chunks_meta(), cfg)


def test_check_chunks_rejects_a_truncated_corpus(corpus, redirect_paths, cfg):
    write_corpus(redirect_paths, cfg, CHUNKS, count=len(CHUNKS) + 2)
    with pytest.raises(ValueError, match="records 7 chunks but"):
        bd.check_chunks(bd.load_chunks(), bd.load_chunks_meta(), cfg)


def test_build_checks_the_corpus_before_it_embeds_anything(corpus, redirect_paths, cfg, store):
    write_corpus(redirect_paths, cfg, CHUNKS, key="deadbeef")
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="chunk_key"):
        bd.build(cfg, client=store, embedder=embedder, progress=False)
    assert embedder.batches == []  # nothing was sent, so nothing was charged
    assert bd.get_collection(store, collection_name(cfg, chunk_key(cfg))) is None


# --- building -------------------------------------------------------------------------------------


def test_build_writes_every_chunk_into_the_named_collection(corpus, cfg, store):
    result = bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)

    assert result["status"] == "built"
    assert result["name"] == collection_name(cfg, chunk_key(cfg))
    collection = bd.get_collection(store, result["name"])
    assert collection.count() == len(CHUNKS)
    assert result["tally"].dim == len(VOCAB) + 1


def test_an_ablation_that_cannot_change_a_vector_reuses_the_collection(corpus, cfg, store):
    # The Day 4 case: tuning fusion / bm25 / seed moves cfg.hash, but not one number in the index.
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    ablation = Config({**cfg.to_dict(), "seed": 43, "fusion": {**cfg["fusion"], "rrf_k": 61}})
    assert ablation.hash != cfg.hash

    again = FakeEmbedder()
    result = bd.build(ablation, client=store, embedder=again, progress=False)
    assert result["status"] == "up-to-date"
    assert again.batches == []  # the whole point: the ablation does not re-pay for the corpus


def test_another_encoder_builds_its_own_collection(corpus, cfg, store):
    # The opposite case: a different model is a different vector space, and mixing two in one
    # collection would rank nonsense with no error anywhere.
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    large = Config(
        {**cfg.to_dict(), "embedding": {**cfg["embedding"], "model": "text-embedding-3-large"}}
    )
    result = bd.build(large, client=store, embedder=FakeEmbedder(), progress=False)

    assert result["status"] == "built"
    assert result["name"] != collection_name(cfg, chunk_key(cfg))


def test_build_stores_the_text_and_the_flattened_metadata(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    collection = bd.get_collection(store, collection_name(cfg, chunk_key(cfg)))

    row = collection.get(ids=["2026-sp:CS-446"], include=["documents", "metadatas"])
    assert row["documents"][0] == CHUNKS[0]["text"]
    # Chroma holds scalars; ingest.chunk owns the encoding and retrieval.dense reverses it.
    assert row["metadatas"][0]["prereq_codes"] == "CS 374"
    assert row["metadatas"][0]["gened_ids"] == ""
    assert row["metadatas"][0]["credits"] == 3.0


def test_build_creates_the_collection_in_cosine_space(corpus, cfg, store):
    # Chroma's default is squared L2, under which retrieval.dense's `1 - distance` is not a
    # similarity at all.
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    collection = bd.get_collection(store, collection_name(cfg, chunk_key(cfg)))
    assert collection.configuration_json["hnsw"]["space"] == "cosine"


def test_build_records_how_the_collection_was_built(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    metadata = bd.get_collection(store, collection_name(cfg, chunk_key(cfg))).metadata

    assert metadata["chunk_key"] == chunk_key(cfg)  # what DenseRetriever compares
    assert metadata["config_hash"] == cfg.hash
    assert metadata["model"] == "text-embedding-3-small"
    assert metadata["space"] == "cosine"
    assert metadata["chunks"] == len(CHUNKS)
    assert metadata["dim"] == len(VOCAB) + 1
    assert metadata["built_at"].endswith("Z")


def test_build_batches_by_the_configured_size(corpus, cfg, store):
    batched = Config({**cfg.to_dict(), "embedding": {**cfg["embedding"], "batch_size": 2}})
    embedder = FakeEmbedder()
    result = bd.build(batched, client=store, embedder=embedder, progress=False)

    assert [len(b) for b in embedder.batches] == [2, 2, 1]
    assert result["tally"].batches == 3
    name = collection_name(batched, chunk_key(batched))
    assert bd.get_collection(store, name).count() == len(CHUNKS)


def test_build_totals_the_tokens_every_batch_reported(corpus, cfg, store):
    result = bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    assert result["tally"].tokens == sum(len(c["text"].split()) for c in CHUNKS)


def test_build_reports_no_tokens_when_the_provider_reports_none(corpus, cfg, store):
    class Local(FakeEmbedder):
        def embed(self, texts):
            return Vectors(super().embed(texts).vectors, None)

    result = bd.build(cfg, client=store, embedder=Local(), progress=False)
    assert result["tally"].tokens is None


def test_build_refuses_vectors_of_two_different_dimensions(corpus, cfg, store):
    class Drifting(FakeEmbedder):
        def embed(self, texts):
            vectors = super().embed(texts).vectors
            if len(self.batches) > 1:  # every batch after the first comes back one shorter
                vectors = [v[:-1] for v in vectors]
            return Vectors(vectors, 0)

    small = Config({**cfg.to_dict(), "embedding": {**cfg["embedding"], "batch_size": 2}})
    with pytest.raises(RuntimeError, match="dimensional"):
        bd.build(small, client=store, embedder=Drifting(), progress=False)


def test_build_refuses_to_report_a_collection_that_is_short(corpus, cfg, store, monkeypatch):
    # count() is the signal every later run reads. A build that reported success over a short
    # collection would make each of those runs ask for a --rebuild that re-pays for the corpus and
    # ends in the identical state -- with a diagnosis ("interrupted build") naming the wrong cause.
    real = bd.embed_into

    def embed_then_lose_a_row(collection, chunks, *args, **kwargs):
        tally = real(collection, chunks, *args, **kwargs)
        collection.delete(ids=[chunks[0]["doc_id"]])
        return tally

    monkeypatch.setattr(bd, "embed_into", embed_then_lose_a_row)
    with pytest.raises(RuntimeError, match=r"holds 4 of 5"):
        bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)


def test_build_leaves_no_store_behind_when_it_cannot_embed(corpus, cfg, monkeypatch):
    # No client injected, so this is the path a real `uv run ... build_dense` takes with no key.
    def no_key(_cfg):
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(bd, "make_embedder", no_key)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        bd.build(cfg, progress=False)
    assert not config.chroma_path().exists()


# --- idempotency ----------------------------------------------------------------------------------


def test_a_second_build_embeds_nothing(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    again = FakeEmbedder()
    result = bd.build(cfg, client=store, embedder=again, progress=False)

    assert result["status"] == "up-to-date"
    assert again.batches == []  # the whole point: a re-run costs nothing


def test_rebuild_drops_the_collection_and_embeds_again(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    again = FakeEmbedder()
    result = bd.build(cfg, rebuild=True, client=store, embedder=again, progress=False)

    assert result["status"] == "built"
    assert sum(len(b) for b in again.batches) == len(CHUNKS)
    assert bd.get_collection(store, collection_name(cfg, chunk_key(cfg))).count() == len(CHUNKS)


def test_a_build_with_no_client_opens_the_store_itself_and_reuses_it(corpus, cfg):
    # The path a real `uv run ... build_dense` takes: nothing is injected, so the store is opened
    # here -- once to look, once to write -- and the second run has to see what the first wrote.
    first = bd.build(cfg, embedder=FakeEmbedder(), progress=False)
    assert first["status"] == "built"
    assert (config.chroma_path() / "chroma.sqlite3").is_file()

    again = FakeEmbedder()
    assert bd.build(cfg, embedder=again, progress=False)["status"] == "up-to-date"
    assert again.batches == []

    rebuilt = FakeEmbedder()
    assert bd.build(cfg, rebuild=True, embedder=rebuilt, progress=False)["status"] == "built"
    assert sum(len(b) for b in rebuilt.batches) == len(CHUNKS)


def test_an_incomplete_collection_asks_for_rebuild_instead_of_topping_up(corpus, cfg, store):
    # What an interrupted build leaves behind. Adding the missing rows would look like a repair
    # and produce a corpus that is silently a mix of two runs.
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    bd.get_collection(store, collection_name(cfg, chunk_key(cfg))).delete(ids=[CHUNKS[0]["doc_id"]])

    with pytest.raises(ValueError, match=r"holds 4 of the 5 chunks.*--rebuild"):
        bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)


def test_an_empty_collection_is_read_as_no_index_at_all(corpus, cfg, store):
    # What a run refused on its first API request leaves behind: created, never written to. It
    # was reached for real -- an OpenAI key with no credits on it -- and demanding --rebuild here
    # asks the reader to rebuild nothing.
    store.create_collection(
        collection_name(cfg, chunk_key(cfg)),
        configuration={"hnsw": {"space": bd.SPACE}},
        embedding_function=None,
    )
    result = bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)

    assert result["status"] == "built"
    assert bd.get_collection(store, result["name"]).count() == len(CHUNKS)


# --- reporting ------------------------------------------------------------------------------------


def test_cost_lines_quote_the_list_price():
    lines = bd.cost_lines("text-embedding-3-small", 522_658)
    assert "522658" in lines[0]
    assert "$0.0105" in lines[1] and "$0.02 / 1M tokens" in lines[1]


def test_cost_lines_say_unknown_rather_than_guess_a_price():
    assert "unknown" in bd.cost_lines("bge-m3", 1000)[1]


def test_cost_lines_say_nothing_was_billed_for_a_local_provider():
    assert bd.cost_lines("BAAI/bge-m3", None) == [
        "tokens billed          n/a  (provider reports no usage)"
    ]


def test_summarize_a_build_names_the_collection_and_the_cost(corpus, cfg, store):
    result = bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    text = "\n".join(bd.summarize(result))

    assert collection_name(cfg, chunk_key(cfg)) in text
    assert "cosine" in text and f"dim {len(VOCAB) + 1}" in text
    assert "tokens billed" in text and "elapsed" in text


def test_summarize_an_up_to_date_build_says_how_to_force_one(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    text = "\n".join(bd.summarize(bd.build(cfg, client=store, embedder=FakeEmbedder())))
    assert "--rebuild" in text


def test_plan_quotes_the_token_count_the_chunker_measured(corpus, cfg, store):
    lines = "\n".join(bd.plan(cfg, client=store))
    assert "does not exist yet" in lines
    assert "would embed            5 chunks via openai" in lines
    assert "1000000" in lines and "$0.0200" in lines  # the sidecar's total, at list price


def test_plan_embeds_nothing_and_creates_nothing(corpus, cfg, store):
    bd.plan(cfg, client=store)
    assert bd.get_collection(store, collection_name(cfg, chunk_key(cfg))) is None


def test_plan_reports_an_existing_collection(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    assert "holds 5 chunks" in "\n".join(bd.plan(cfg, client=store))


def test_plan_quotes_nothing_for_a_collection_that_is_already_complete(corpus, cfg, store):
    # The dry run is the command someone runs to decide whether to pay. build() would leave this
    # collection alone and spend $0, so quoting the whole corpus would be a bill for a no-op.
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    lines = "\n".join(bd.plan(cfg, client=store))

    assert "would embed            nothing" in lines
    assert "$0.0000" in lines and "$0.0200" not in lines


def test_plan_with_rebuild_quotes_the_re_embed(corpus, cfg, store):
    bd.build(cfg, client=store, embedder=FakeEmbedder(), progress=False)
    lines = "\n".join(bd.plan(cfg, rebuild=True, client=store))

    assert "drop and re-embed" in lines and "5 chunks via openai" in lines
    assert "$0.0200" in lines


def test_plan_does_not_create_the_store(corpus, cfg):
    # No client injected: plan() has to answer "does the collection exist" without opening a
    # persistent client, because opening one is what writes indexes/chroma/chroma.sqlite3.
    bd.plan(cfg)
    assert not config.chroma_path().exists()
