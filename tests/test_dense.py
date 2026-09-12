"""Tests for retrieval/dense.py: the distance-to-score conversion, the guards, and a real search.

The collection is a real Chroma one under ``tmp_path``, filled with toy vectors from a word-count
"model" -- so ``search`` is exercised through Chroma's own cosine distance rather than against a
stub that would just hand back whatever ranking the test expected. The embedder is fake, so no key
and no network are involved.

``redirect_paths`` is autouse: ``INDEXES_DIR`` points into tmp_path, so even a call that opens the
store by itself cannot touch the real ``indexes/chroma``.
"""

from __future__ import annotations

import math
from pathlib import Path

import chromadb
import pytest

import config
from config import Config, collection_name, load_config
from embedding import Vectors
from ingest.chunk import chunk_key
from retrieval import dense
from retrieval.dense import DenseRetriever
from retrieval.types import Hit

BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"

VOCAB = ("machine", "learning", "algorithm", "poetry")

CHUNKS = [
    {
        "doc_id": "2026-sp:CS-446",
        "text": "CS 446 - Machine Learning\nDescription: machine learning machine learning",
        "metadata": {
            "code": "CS 446",
            "credits": 3.0,
            "prereq_codes": "CS 374|CS 361",
            "gened_ids": "",
        },
    },
    {
        "doc_id": "2026-sp:CS-374",
        "text": "CS 374 - Algorithms\nDescription: algorithm algorithm",
        "metadata": {"code": "CS 374", "credits": 4.0, "prereq_codes": "CS 173", "gened_ids": ""},
    },
    {
        "doc_id": "2026-sp:ENGL-104",
        "text": "ENGL 104 - Poetry\nDescription: poetry poetry",
        "metadata": {"code": "ENGL 104", "credits": 3.0, "prereq_codes": "", "gened_ids": "humlit"},
    },
]


def vector(text: str) -> list[float]:
    """Normalized word counts over VOCAB, plus a bias term so no vector is the zero vector."""
    words = text.lower().split()
    raw = [float(words.count(word)) for word in VOCAB] + [1.0]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class FakeEmbedder:
    """The same word-count model the collection was filled with, for the query side."""

    model = "fake-embedding-3-small"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts) -> Vectors:
        texts = list(texts)
        self.queries += texts
        return Vectors([vector(t) for t in texts], len(texts))


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")


@pytest.fixture
def cfg() -> Config:
    raw = load_config(BASE_CONFIG).to_dict()
    raw["chunk"] = {
        "strategy": "one_course_one_chunk",
        "template_version": "v1",
        "fields": ["code", "name", "credits", "term", "desc", "prereqRaw", "attrs"],
    }
    raw["embedding"] = {"provider": "openai", "model": "text-embedding-3-small", "batch_size": 256}
    raw["fusion"] = {**raw["fusion"], "top_k_each": 20}
    return Config(raw)


@pytest.fixture
def client(tmp_path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def make_collection(client, cfg, space: str = "cosine", key: str | None = None):
    """A collection filled the way ``ingest.build_dense`` fills one."""
    collection = client.create_collection(
        collection_name(cfg),
        configuration={"hnsw": {"space": space}},
        metadata={"chunk_key": chunk_key(cfg) if key is None else key, "space": space},
        embedding_function=None,
    )
    collection.add(
        ids=[c["doc_id"] for c in CHUNKS],
        embeddings=[vector(c["text"]) for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["metadata"] for c in CHUNKS],
    )
    return collection


@pytest.fixture
def retriever(client, cfg) -> DenseRetriever:
    return DenseRetriever(cfg, collection=make_collection(client, cfg), embedder=FakeEmbedder())


# --- scores ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("distance", "score"), [(0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (2.0, -1.0)])
def test_similarity_turns_a_cosine_distance_around(distance, score):
    # Chroma ranks by ascending distance; every other retriever reports higher-is-better.
    assert dense.similarity(distance) == pytest.approx(score)


def test_similarity_refuses_a_space_it_cannot_convert():
    with pytest.raises(ValueError, match="'l2' distance"):
        dense.similarity(0.5, "l2")


def test_similarity_trusts_the_build_when_the_space_is_unreadable():
    # None means Chroma did not report a space, not that it is the wrong one; build_dense only
    # ever creates cosine collections.
    assert dense.similarity(0.25, None) == pytest.approx(0.75)


# --- hits -----------------------------------------------------------------------------------------


def test_to_hits_ranks_from_one_and_decodes_the_metadata():
    result = {
        "ids": [["2026-sp:CS-446", "2026-sp:CS-374"]],
        "documents": [["CS 446 - Machine Learning", "CS 374 - Algorithms"]],
        "metadatas": [[{"prereq_codes": "CS 374|CS 361"}, {"prereq_codes": ""}]],
        "distances": [[0.1, 0.4]],
    }
    hits = dense.to_hits(result)

    assert [h.rank for h in hits] == [1, 2]
    assert [h.doc_id for h in hits] == ["2026-sp:CS-446", "2026-sp:CS-374"]
    assert hits[0].score == pytest.approx(0.9)
    # The lists BM25 will return as lists, so a Hit means one thing whoever produced it.
    assert hits[0].metadata["prereq_codes"] == ["CS 374", "CS 361"]
    assert hits[1].metadata["prereq_codes"] == []


def test_to_hits_of_an_empty_result_is_an_empty_list():
    assert (
        dense.to_hits({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}) == []
    )


# --- guards ---------------------------------------------------------------------------------------


def test_open_collection_points_at_the_build_when_there_is_no_index(client, cfg):
    with pytest.raises(FileNotFoundError, match="ingest.build_dense"):
        dense.open_collection(cfg, client=client)


def test_check_collection_refuses_another_distance_space(client, cfg):
    collection = make_collection(client, cfg, space="l2")
    with pytest.raises(ValueError, match="built in 'l2' space"):
        dense.check_collection(collection, cfg)


def test_check_collection_refuses_a_collection_from_another_chunk_template(client, cfg):
    collection = make_collection(client, cfg, key="deadbeef")
    with pytest.raises(ValueError, match="chunk_key 'deadbeef'"):
        dense.check_collection(collection, cfg)


def test_check_collection_accepts_one_that_records_nothing(cfg):
    # An index built before the metadata existed is not wrong, just undescribed.
    class Bare:
        name, metadata, configuration_json = "courses_x", None, None

    dense.check_collection(Bare(), cfg)


def test_constructing_the_retriever_checks_the_collection(client, cfg):
    collection = make_collection(client, cfg, space="l2")
    with pytest.raises(ValueError, match="space"):
        DenseRetriever(cfg, collection=collection, embedder=FakeEmbedder())


# --- searching ------------------------------------------------------------------------------------


def test_search_returns_the_nearest_chunks_best_first(retriever):
    hits = retriever.search("machine learning", 3)

    assert [h.doc_id for h in hits] == ["2026-sp:CS-446", "2026-sp:CS-374", "2026-sp:ENGL-104"]
    assert hits[0].score > hits[1].score > hits[2].score
    # A cosine similarity, so a near-identical vector scores just under 1 -- not a raw distance,
    # which would have put the best hit nearest 0.
    assert 0.9 < hits[0].score <= 1.0


def test_search_fills_in_every_field_of_a_hit(retriever):
    hit = retriever.search("machine learning", 1)[0]

    assert isinstance(hit, Hit)
    assert hit.rank == 1
    assert hit.text == CHUNKS[0]["text"]
    assert hit.metadata["code"] == "CS 446"
    assert hit.metadata["prereq_codes"] == ["CS 374", "CS 361"]


def test_search_stops_at_k(retriever):
    assert len(retriever.search("machine learning", 1)) == 1


def test_search_defaults_to_the_recall_depth_fusion_asks_for(client, cfg):
    deep = Config({**cfg.to_dict(), "fusion": {**cfg["fusion"], "top_k_each": 2}})
    retriever = DenseRetriever(
        deep, collection=make_collection(client, deep), embedder=FakeEmbedder()
    )
    assert len(retriever.search("machine learning")) == 2


def test_search_returns_what_there_is_when_k_exceeds_the_corpus(retriever):
    assert len(retriever.search("machine learning", 50)) == len(CHUNKS)


def test_search_embeds_the_query_once(retriever):
    retriever.search("machine learning", 2)
    assert retriever.embedder.queries == ["machine learning"]


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_search_refuses_an_empty_query(retriever, query):
    with pytest.raises(ValueError, match="query is empty"):
        retriever.search(query)
    assert retriever.embedder.queries == []  # refused before anything was embedded


@pytest.mark.parametrize("k", [0, -1])
def test_search_refuses_a_non_positive_k(retriever, k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("machine learning", k)


# --- CLI ------------------------------------------------------------------------------------------


def test_format_hit_leads_with_the_rank_score_and_course_line():
    hit = Hit("2026-sp:CS-446", 0.8123, 1, "CS 446 - Machine Learning\nCredits: 3", {})
    assert dense.format_hit(hit) == "  1. 0.812  CS 446 - Machine Learning"


def test_format_hit_falls_back_to_the_doc_id_for_an_empty_chunk():
    assert "2026-sp:CS-446" in dense.format_hit(Hit("2026-sp:CS-446", 0.5, 2, "", {}))
