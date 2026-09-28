"""Tests for eval/scripts/ablation.py: which rows run, that every (row, mode) runs once, the files.

The retrievers are fakes injected through ``retrieval.BUILDERS``, as in ``tests/test_retrieve.py``:
the runner's job is to call ``retrieve()`` the same way for every mode and write down what came
back, and that is checkable without an index or a key. The dataset is a small frozen fixture in
``tmp_path`` -- its own sidecar and digest, so it goes through the real :func:`load_split` -- and
never the committed ``v1.jsonl``, whose test half is read once, for the report.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from config import file_sha256, load_config
from eval.metrics import mrr_at_k, recall_at_k
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import ablation as ab
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path
from retrieval import BUILDERS, MODES, clear_cache, retrieve
from retrieval.types import Hit

KNOWN = frozenset(
    f"2026-sp:{code}" for code in ("CS-374", "CS-446", "CS-441", "MATH-213", "ENGL-104", "FR-104")
)

#: The two fake rankings disagree, so the three modes return three different lists.
DENSE_ORDER = ["2026-sp:CS-446", "2026-sp:CS-441", "2026-sp:CS-374", "2026-sp:ENGL-104"]
BM25_ORDER = ["2026-sp:CS-374", "2026-sp:MATH-213", "2026-sp:CS-446", "2026-sp:FR-104"]

#: (id, bucket, split, gold). Two halves, all four buckets, and a multihop row with two gold ids.
ROWS = [
    ("v1-0001", "fact", "test", ["2026-sp:CS-374"]),
    ("v1-0002", "fact", "calib", ["2026-sp:CS-446"]),
    ("v1-0003", "fact", "test", ["2026-sp:FR-104"]),
    ("v1-0004", "multihop", "test", ["2026-sp:MATH-213", "2026-sp:ENGL-104"]),
    ("v1-0005", "unanswerable", "test", []),
    ("v1-0006", "ambiguous", "test", []),
    ("v1-0007", "ambiguous", "calib", []),
]


class FakeRetriever:
    """A fixed ranking, and every ``(query, k)`` it was asked for."""

    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        self.calls.append((query, k))
        ids = self.doc_ids if k is None else self.doc_ids[:k]
        return [Hit(d, 1.0 / r, r, d, {}) for r, d in enumerate(ids, start=1)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch) -> dict[str, FakeRetriever]:
    clear_cache()
    made = {"dense": FakeRetriever(DENSE_ORDER), "bm25": FakeRetriever(BM25_ORDER)}
    for name, fake in made.items():
        monkeypatch.setitem(BUILDERS, name, lambda cfg, fake=fake: fake)
    yield made
    clear_cache()


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A frozen dataset with the sidecar #61's freeze would have written."""
    path = tmp_path / "v1.jsonl"
    items = [
        EvalItem(
            id=id_,
            bucket=bucket,
            question=f"question {id_}?",
            gold_answer="an answer",
            gold_doc_ids=gold,
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, gold in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


def ablate(cfg, dataset: Path, tmp_path: Path, split="test") -> tuple[Path, str]:
    out = StringIO()
    path = ab.ablate(cfg, split, out, dataset, tmp_path / "runs", known=KNOWN)
    return path, out.getvalue()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- which rows -----------------------------------------------------------------------------------


def test_only_gold_bearing_rows_of_the_asked_half_are_scored(cfg, dataset, tmp_path):
    lines = read(ablate(cfg, dataset, tmp_path)[0])
    assert sorted({line["id"] for line in lines}) == ["v1-0001", "v1-0003", "v1-0004"]


def test_skipped_rows_are_counted_by_bucket_and_printed(cfg, dataset, tmp_path):
    path, printed = ablate(cfg, dataset, tmp_path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"] == {"ambiguous": 1, "unanswerable": 1}
    assert meta["buckets"] == {"fact": 2, "multihop": 1}
    assert "scoring n = 3 (2 fact, 1 multihop)" in printed
    assert "skipped 2 without gold ids (1 ambiguous, 1 unanswerable)" in printed


def test_a_half_with_nothing_to_score_is_refused():
    row = EvalItem(
        id="v1-0001",
        bucket="ambiguous",
        question="q?",
        gold_answer="a",
        gold_doc_ids=[],
        should_abstain=True,
        split="test",
    )
    with pytest.raises(ValueError, match="nothing to score"):
        ab.select([row])


def test_an_unfrozen_dataset_is_refused_by_the_loader(cfg, dataset, tmp_path):
    dataset.write_text(dataset.read_text() + "\n")  # the digest no longer matches
    with pytest.raises(ValueError, match="changed after it was frozen"):
        ablate(cfg, dataset, tmp_path)


# --- every (row, mode) once, through retrieve(k=5) -----------------------------------------------


def test_every_row_and_mode_pair_is_present_exactly_once(cfg, dataset, tmp_path):
    lines = read(ablate(cfg, dataset, tmp_path)[0])
    pairs = [(line["id"], line["mode"]) for line in lines]
    assert len(pairs) == len(set(pairs)) == 3 * len(MODES)
    assert [line["mode"] for line in lines[:3]] == list(MODES)


def test_every_mode_asks_for_five_and_hybrid_fuses_twenty_from_each(cfg, dataset, tmp_path, fakes):
    lines = read(ablate(cfg, dataset, tmp_path)[0])
    assert all(len(line["ranked"]) <= ab.K == 5 for line in lines)
    # Per row: the dense mode asks dense for 5, hybrid asks it for top_k_each; same for bm25.
    assert {k for _, k in fakes["dense"].calls} == {5, cfg["fusion"]["top_k_each"]}
    assert {k for _, k in fakes["bm25"].calls} == {5, cfg["fusion"]["top_k_each"]}


def test_the_three_modes_return_their_own_rankings(cfg, dataset, tmp_path):
    lines = read(ablate(cfg, dataset, tmp_path)[0])
    first = {line["mode"]: line["ranked"] for line in lines if line["id"] == "v1-0001"}
    assert first["dense"] == DENSE_ORDER[:5]
    assert first["bm25"] == BM25_ORDER[:5]
    # Whatever retrieve() says hybrid is -- the tie-break between CS 374 and CS 446 is fuse's.
    fused = retrieve("question v1-0001?", "hybrid", k=5, cfg=cfg)
    assert first["hybrid"] == [hit.doc_id for hit in fused.hits]
    assert set(first["hybrid"][:2]) == {"2026-sp:CS-374", "2026-sp:CS-446"}


def test_scores_are_the_metrics_over_the_written_ranking(cfg, dataset, tmp_path):
    for line in read(ablate(cfg, dataset, tmp_path)[0]):
        assert line["recall_at_5"] == recall_at_k(line["ranked"], line["gold"], 5)
        assert line["mrr_at_5"] == mrr_at_k(line["ranked"], line["gold"], 5)
    multihop = [
        line
        for line in read(ablate(cfg, dataset, tmp_path)[0])
        if line["id"] == "v1-0004" and line["mode"] == "bm25"
    ]
    assert multihop[0]["recall_at_5"] == 0.5  # MATH 213 found, ENGL 104 not: fractional


def test_a_line_has_the_fields_the_issue_names_in_that_order(cfg, dataset, tmp_path):
    line = read(ablate(cfg, dataset, tmp_path)[0])[0]
    assert list(line) == ["id", "bucket", "mode", "ranked", "gold", "recall_at_5", "mrr_at_5"]


# --- the files ------------------------------------------------------------------------------------


def test_two_runs_are_byte_identical(cfg, dataset, tmp_path):
    path, _ = ablate(cfg, dataset, tmp_path)
    first, first_meta = path.read_bytes(), path.with_suffix(".meta.json").read_bytes()
    clear_cache()
    again, _ = ablate(cfg, dataset, tmp_path)
    assert again.read_bytes() == first
    assert again.with_suffix(".meta.json").read_bytes() == first_meta


def test_the_two_halves_do_not_overwrite_each_other(cfg, dataset, tmp_path):
    test_path, _ = ablate(cfg, dataset, tmp_path, "test")
    calib_path, _ = ablate(cfg, dataset, tmp_path, "calib")
    assert test_path.name == f"ablation_test_{cfg.hash}.jsonl"
    assert calib_path.name == f"ablation_calib_{cfg.hash}.jsonl"
    assert test_path.exists() and calib_path.exists()
    assert {line["id"] for line in read(calib_path)} == {"v1-0002"}


def test_the_sidecar_records_the_run(cfg, dataset, tmp_path):
    path, _ = ablate(cfg, dataset, tmp_path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["config_hash"] == cfg.hash
    assert len(meta["dense_key"]) == len(meta["bm25_key"]) == 8
    assert meta["dataset_sha256"] == file_sha256(dataset)
    assert meta["sha256"] == file_sha256(path)
    assert meta["split"] == "test" and meta["k"] == 5 and meta["modes"] == list(MODES)
    assert meta["ids"] == ["v1-0001", "v1-0003", "v1-0004"]
    assert meta["n"] == 3 and meta["lines"] == 9
    assert "latency" not in json.dumps(meta) and "time" not in meta


def test_the_run_file_carries_no_latency(cfg, dataset, tmp_path):
    assert "latency" not in ablate(cfg, dataset, tmp_path)[0].read_text()


# --- the command ----------------------------------------------------------------------------------


def test_split_is_required(capsys):
    with pytest.raises(SystemExit) as exit_:
        ab.main([])
    assert exit_.value.code == 2
    assert "--split" in capsys.readouterr().err


def test_there_is_no_all_split(capsys):
    with pytest.raises(SystemExit):
        ab.main(["--split", "all"])
