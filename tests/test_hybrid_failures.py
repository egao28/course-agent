"""Tests for eval/scripts/hybrid_failures.py: which questions are logged, and the ranks at depth.

As in ``tests/test_ablation.py``, the retrievers are fakes injected through ``retrieval.BUILDERS``
and the dataset is a small frozen fixture in ``tmp_path``. #75's runner writes the test run over
it first, and the log is then read off that run -- which is the order the real pipeline runs in.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from config import file_sha256, load_config
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import ablation as ab
from eval.scripts import hybrid_failures as hf
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path
from retrieval import BUILDERS, MODES, clear_cache
from retrieval.types import Hit

#: Ten documents, in the same order on both sides, so hybrid's order is this order too.
ORDER = [f"2026-sp:CS-{100 + i}" for i in range(1, 11)]
#: Known to the corpus, returned by neither retriever.
ABSENT = "2026-sp:FR-104"
KNOWN = frozenset([*ORDER, ABSENT])

#: (id, bucket, gold). v1-0001 has its gold at rank 7; v1-0002 at rank 1; v1-0003 half-found.
ROWS = [
    ("v1-0001", "fact", [ORDER[6]]),
    ("v1-0002", "fact", [ORDER[0]]),
    ("v1-0003", "multihop", [ORDER[1], ABSENT]),
]


class FakeRetriever:
    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        ids = self.doc_ids if k is None else self.doc_ids[:k]
        return [Hit(d, 1.0 / r, r, f"{d} — title of {d}\nbody", {}) for r, d in enumerate(ids, 1)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch) -> dict[str, FakeRetriever]:
    clear_cache()
    made = {"dense": FakeRetriever(list(ORDER)), "bm25": FakeRetriever(list(ORDER))}
    for name, fake in made.items():
        monkeypatch.setitem(BUILDERS, name, lambda cfg, fake=fake: fake)
    yield made
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


CHUNKS = [{"doc_id": d, "text": f"{d} — title of {d}\nbody", "metadata": {}} for d in KNOWN]


def log(cfg, dataset: Path, tmp_path: Path) -> tuple[list[dict], dict, str]:
    runs = tmp_path / "runs"
    ab.ablate(cfg, "test", StringIO(), dataset, runs, known=KNOWN)
    out = StringIO()
    path = hf.write_failures(cfg, out, dataset, runs, chunks=CHUNKS, known=KNOWN)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return lines, meta, out.getvalue()


def test_gold_at_hybrid_rank_7_is_logged_with_its_rank_at_depth(cfg, dataset, tmp_path):
    lines, _, _ = log(cfg, dataset, tmp_path)
    first = next(line for line in lines if line["id"] == "v1-0001")
    assert first["missing"] == [ORDER[6]]
    assert first["missing_ranks"] == {ORDER[6]: {"dense": 7, "bm25": 7, "hybrid": 7}}


def test_a_row_fully_in_the_top_5_is_not_logged(cfg, dataset, tmp_path):
    lines, _, _ = log(cfg, dataset, tmp_path)
    assert [line["id"] for line in lines] == ["v1-0001", "v1-0003"]


def test_multihop_records_which_gold_is_missing_and_null_where_no_mode_has_it(
    cfg, dataset, tmp_path
):
    lines, _, _ = log(cfg, dataset, tmp_path)
    multihop = next(line for line in lines if line["id"] == "v1-0003")
    assert multihop["gold"] == [ORDER[1], ABSENT]
    assert multihop["missing"] == [ABSENT]
    assert multihop["missing_ranks"] == {ABSENT: dict.fromkeys(MODES)}


def test_a_line_carries_the_question_and_top5_with_titles(cfg, dataset, tmp_path):
    line = log(cfg, dataset, tmp_path)[0][0]
    assert list(line) == [
        "id",
        "bucket",
        "question",
        "gold",
        "missing",
        "hybrid_top5",
        "missing_ranks",
        "live_top5_matches_run",
    ]
    assert line["question"] == "question v1-0001?"
    assert line["hybrid_top5"][0] == {
        "doc_id": ORDER[0],
        "title": f"{ORDER[0]} — title of {ORDER[0]}",
    }
    assert [hit["doc_id"] for hit in line["hybrid_top5"]] == ORDER[:5]
    assert line["live_top5_matches_run"] is True


def test_the_sidecar_counts_by_bucket_and_names_the_run(cfg, dataset, tmp_path):
    _, meta, printed = log(cfg, dataset, tmp_path)
    run = ab.run_path("test", cfg, tmp_path / "runs")
    assert meta["run_sha256"] == file_sha256(run)
    assert meta["config_hash"] == cfg.hash and meta["depth"] == cfg["fusion"]["top_k_each"]
    assert meta["scored"] == {"fact": 2, "multihop": 1}
    assert meta["failing"] == {"fact": 1, "multihop": 1}
    assert meta["live_top5_differs"] == []
    assert "on 2 of 3" in printed and "1 of 2 fact, 1 of 1 multihop" in printed


def test_a_live_pass_that_disagrees_with_the_run_is_flagged(cfg, dataset, tmp_path, fakes):
    runs = tmp_path / "runs"
    ab.ablate(cfg, "test", StringIO(), dataset, runs, known=KNOWN)
    for fake in fakes.values():  # the dense side jittered between the run and the second pass
        fake.doc_ids = [ORDER[1], ORDER[0], *ORDER[2:]]
    clear_cache()
    path = hf.write_failures(cfg, StringIO(), dataset, runs, chunks=CHUNKS, known=KNOWN)
    first = json.loads(path.read_text().splitlines()[0])
    assert first["live_top5_matches_run"] is False
    assert first["hybrid_top5"][0]["doc_id"] == ORDER[0]  # still the committed run's top 5
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["live_top5_differs"] == ["v1-0001", "v1-0003"]  # every logged row moved


def test_the_log_is_refused_once_the_run_it_was_read_from_changes(cfg, dataset, tmp_path):
    log(cfg, dataset, tmp_path)
    run = ab.run_path("test", cfg, tmp_path / "runs")
    assert hf.load_failures(cfg, run, tmp_path / "runs")["failing"] == {"fact": 1, "multihop": 1}
    run.write_text(run.read_text() + "\n")
    with pytest.raises(ValueError, match="not read from the test run on disk"):
        hf.load_failures(cfg, run, tmp_path / "runs")
