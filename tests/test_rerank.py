"""Tests for #79: retrieval/rerank.py (the reorder) and eval/scripts/rerank.py (the run on a half).

The reorder is tested on hand-made hits and a scorer that is a lookup table, so the right order is
known without a model. The run uses fake retrievers through ``retrieval.BUILDERS``, as
``tests/test_hybrid_failures.py`` does, over a small frozen dataset in ``tmp_path``; #75's runner
writes the run first, and the rerank run is paired with it. Two tests read committed files: the
rerank run against the test run it names, and BM25's lines against a live query, which is the
"three modes are unchanged" acceptance check that can be run (the dense side does not reproduce).
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from config import Config, bm25_path, file_sha256, load_config, rerank_key
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import ablation as ab
from eval.scripts import bootstrap as bs
from eval.scripts import rerank as rr
from eval.scripts.draft_items import write_json
from eval.scripts.split import load_split, meta_path
from ingest.chunk import chunk_key
from retrieval import BUILDERS, MODES, clear_cache
from retrieval.rerank import rerank
from retrieval.types import Hit

# --- the reorder --------------------------------------------------------------------------------


def hits(*doc_ids: str) -> list[Hit]:
    return [
        Hit(d, 1.0 / r, r, f"text of {d}", {}, {"dense": r, "bm25": None})
        for r, d in enumerate(doc_ids, 1)
    ]


def table(scores: dict[str, float]):
    """A scorer that looks the chunk text up: ``text of X`` scores ``scores[X]``, the rest 0."""
    return lambda pairs: [scores.get(text.removeprefix("text of "), 0.0) for _, text in pairs]


def test_the_pool_is_reordered_by_the_scorer_and_cut_to_k():
    out = rerank("q", hits("a", "b", "c", "d"), 2, table({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}))
    assert [(h.doc_id, h.rank, h.score) for h in out] == [("b", 1, 0.9), ("d", 2, 0.7)]


def test_a_tie_keeps_the_pools_order_and_sources_survive():
    out = rerank("q", hits("a", "b", "c"), 3, table({"a": 0.2, "b": 0.5, "c": 0.5}))
    assert [h.doc_id for h in out] == ["b", "c", "a"]
    assert out[0].sources == {"dense": 2, "bm25": None} and out[0].text == "text of b"


def test_the_scorer_sees_the_query_and_each_chunk_text_once():
    seen = []
    rerank("the query", hits("a", "b"), 5, lambda pairs: seen.extend(pairs) or [0.0, 0.0])
    assert seen == [("the query", "text of a"), ("the query", "text of b")]


def test_an_empty_pool_bad_k_and_a_short_scorer_are_handled():
    assert rerank("q", [], 5, table({})) == []
    with pytest.raises(ValueError, match="k must be a positive integer"):
        rerank("q", hits("a"), 0, table({"a": 1.0}))
    with pytest.raises(ValueError, match="1 scores for 2 pairs"):
        rerank("q", hits("a", "b"), 2, lambda pairs: [1.0])


# --- the run ------------------------------------------------------------------------------------

#: Twenty-five documents, in the same order on both sides, so hybrid's order is this order too.
ORDER = [f"2026-sp:CS-{100 + i}" for i in range(1, 26)]
KNOWN = frozenset(ORDER)

#: (id, bucket, gold). Gold at hybrid rank 7 (the reranker lifts it), 1, and 22 (past the pool).
ROWS = [
    ("v1-0001", "fact", [ORDER[6]]),
    ("v1-0002", "fact", [ORDER[0]]),
    ("v1-0003", "multihop", [ORDER[1], ORDER[21]]),
]


class FakeRetriever:
    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        ids = self.doc_ids if k is None else self.doc_ids[:k]
        return [Hit(d, 1.0 / r, r, f"text of {d}", {}) for r, d in enumerate(ids, 1)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    clear_cache()
    for name in ("dense", "bm25"):
        fake = FakeRetriever(list(ORDER))
        monkeypatch.setitem(BUILDERS, name, lambda cfg, fake=fake: fake)
    yield
    clear_cache()


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "v1.jsonl"
    items = [
        EvalItem(
            id=id_,
            bucket=bucket,
            question=f"question {id_}?",
            gold_answer="an answer",
            gold_doc_ids=gold,
            should_abstain=RULES[bucket].should_abstain,
            split="test",
        )
        for id_, bucket, gold in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


#: Prefers CS-107 (the first row's gold) over everything, and otherwise keeps hybrid's order.
PREFER = {d: (10.0 if d == ORDER[6] else -r) for r, d in enumerate(ORDER, 1)}


def run(cfg, dataset: Path, tmp_path: Path, scorer=None) -> tuple[list[dict], dict, str]:
    runs = tmp_path / "runs"
    ab.ablate(cfg, "test", StringIO(), dataset, runs, known=KNOWN)
    out = StringIO()
    path = rr.write_rerank(
        cfg, "test", out, dataset, runs, scorer=scorer or table(PREFER), known=KNOWN
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return lines, meta, out.getvalue()


def test_each_row_is_the_pool_reranked_and_scored_against_its_gold(cfg, dataset, tmp_path):
    lines, _, _ = run(cfg, dataset, tmp_path)
    first = lines[0]
    assert first["mode"] == "hybrid_rerank" and first["id"] == "v1-0001"
    assert first["ranked"] == [ORDER[6], *ORDER[:4]]
    assert first["recall_at_5"] == 1.0 and first["mrr_at_5"] == 1.0
    assert first["pool"] == ORDER[:20]
    assert first["pool_top5_matches_run"] is True
    assert isinstance(first["pool_ms"], float) and isinstance(first["rerank_ms"], float)


def test_the_ceiling_is_the_pools_recall_and_the_reranker_cannot_pass_it(cfg, dataset, tmp_path):
    lines, meta, printed = run(cfg, dataset, tmp_path)
    multihop = lines[2]
    assert multihop["pool_recall_at_20"] == 0.5  # ORDER[21] is past the pool
    assert multihop["recall_at_5"] <= multihop["pool_recall_at_20"]
    assert meta["pool_recall_at_20"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert "ceiling: hybrid Recall@20 = 0.833" in printed


def test_the_sidecar_names_the_model_the_pairing_and_the_latency(cfg, dataset, tmp_path):
    _, meta, _ = run(cfg, dataset, tmp_path)
    test_run = ab.run_path("test", cfg, tmp_path / "runs")
    assert meta["rerank_key"] == rerank_key(cfg) and meta["config_hash"] == cfg.hash
    assert meta["rerank"]["model"] == cfg["rerank"]["model"]
    assert meta["paired_with"]["sha256"] == file_sha256(test_run)
    assert meta["ids"] == ["v1-0001", "v1-0002", "v1-0003"] and meta["pool"] == 20
    assert meta["pool_top5_differs"] == []
    assert set(meta["latency_ms"]) == {"pool", "rerank"}
    assert set(meta["latency_ms"]["rerank"]) == {"median", "p90", "mean", "max"}


def test_the_scorer_is_warmed_up_before_the_first_question(cfg, dataset, tmp_path):
    calls = []

    def scorer(pairs):
        calls.append(len(pairs))
        return table(PREFER)(pairs)

    run(cfg, dataset, tmp_path, scorer)
    assert calls == [1, 20, 20, 20]


def test_a_pool_that_moved_since_the_run_is_flagged(cfg, dataset, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    ab.ablate(cfg, "test", StringIO(), dataset, runs, known=KNOWN)
    clear_cache()
    moved = FakeRetriever([ORDER[1], ORDER[0], *ORDER[2:]])
    for name in ("dense", "bm25"):
        monkeypatch.setitem(BUILDERS, name, lambda cfg, fake=moved: fake)
    path = rr.write_rerank(cfg, "test", StringIO(), dataset, runs, table(PREFER), known=KNOWN)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["pool_top5_differs"] == ["v1-0001", "v1-0002", "v1-0003"]


def test_the_run_reads_back_aligned_with_the_test_run(cfg, dataset, tmp_path):
    lines, _, _ = run(cfg, dataset, tmp_path)
    test_run = bs.load_run("test", cfg, tmp_path / "runs")
    loaded = rr.load_rerank(cfg, test_run, tmp_path / "runs")
    assert list(loaded.scores["mrr_at_5"]) == [line["mrr_at_5"] for line in lines]


def test_an_edited_rerank_run_is_refused(cfg, dataset, tmp_path):
    run(cfg, dataset, tmp_path)
    path = rr.run_path("test", cfg, tmp_path / "runs")
    path.write_text(path.read_text().replace('"mrr_at_5": 1.0', '"mrr_at_5": 0.5', 1))
    with pytest.raises(ValueError, match="sha256"):
        rr.load_rerank(cfg, bs.load_run("test", cfg, tmp_path / "runs"), tmp_path / "runs")


def test_a_rerank_run_paired_with_another_test_run_is_refused(cfg, dataset, tmp_path):
    run(cfg, dataset, tmp_path)
    runs = tmp_path / "runs"
    clear_cache()
    ab.ablate(cfg, "test", StringIO(), dataset, runs, known=KNOWN)  # same lines, same bytes
    assert rr.load_rerank(cfg, bs.load_run("test", cfg, runs), runs)
    test_run = ab.run_path("test", cfg, runs)
    lines = test_run.read_text().splitlines(keepends=True)
    test_run.write_text("".join(lines[3:] + lines[:3]))  # the same lines in another order
    meta = json.loads(test_run.with_suffix(".meta.json").read_text())
    write_json(test_run.with_suffix(".meta.json"), {**meta, "sha256": file_sha256(test_run)})
    with pytest.raises(ValueError, match="not paired with the test run on disk"):
        rr.load_rerank(cfg, bs.load_run("test", cfg, runs), runs)


def test_another_model_names_another_run_and_says_how_to_make_it(cfg, dataset, tmp_path):
    run(cfg, dataset, tmp_path)
    other = Config({**cfg.to_dict(), "rerank": {**cfg["rerank"], "model": "another/reranker"}})
    assert other.hash == cfg.hash and rerank_key(other) != rerank_key(cfg)
    test_run = bs.load_run("test", other, tmp_path / "runs")
    with pytest.raises(ValueError, match="--split test"):
        rr.load_rerank(other, test_run, tmp_path / "runs")


def test_a_pool_shallower_than_k_is_refused(cfg, dataset, tmp_path):
    shallow = Config({**cfg.to_dict(), "rerank": {**cfg["rerank"], "pool": 3}})
    ab.ablate(shallow, "test", StringIO(), dataset, tmp_path, known=KNOWN)
    with pytest.raises(ValueError, match="shallower than k = 5"):
        rr.write_rerank(shallow, "test", StringIO(), dataset, tmp_path, table(PREFER), known=KNOWN)


def test_reranking_is_not_a_retrieve_mode():
    assert MODES == ("dense", "bm25", "hybrid")


# --- the committed files --------------------------------------------------------------------------


def test_the_committed_rerank_run_is_paired_with_the_committed_test_run(cfg):
    loaded = rr.load_rerank(cfg, bs.load_run("test", cfg))
    assert loaded.meta["n"] == 41 and loaded.meta["pool_top5_differs"] == []
    assert loaded.meta["rerank"] == dict(cfg["rerank"])


@pytest.mark.skipif(
    not bm25_path(load_config(), chunk_key(load_config())).exists(),
    reason="the BM25 index has not been built here",
)
def test_bm25_still_returns_the_committed_test_lines_byte_for_byte(cfg, monkeypatch):
    """#79's acceptance: the existing modes are unchanged. BM25 is the one that reproduces."""
    from retrieval import _build_bm25

    monkeypatch.setitem(BUILDERS, "bm25", _build_bm25)
    clear_cache()
    committed = {
        line["id"]: line
        for line in map(json.loads, ab.run_path("test", cfg).read_text().splitlines())
        if line["mode"] == "bm25"
    }
    rows = ab.select(load_split("test")).rows
    fresh = {row.id: json.dumps(ab.score(row, "bm25", cfg)) for row in rows}
    assert fresh == {i: json.dumps(line) for i, line in committed.items()}
