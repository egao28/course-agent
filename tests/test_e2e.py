"""Tests for eval/scripts/e2e.py: every row gets a line, the gate is on, and the file says so.

The chat client is a fake that answers by what it is asked -- the router's system prompt or the
answer node's -- so the whole graph runs with no network, through the real dispatch table, the real
gate and the real catalog for a tool row. BM25 is a fake injected through ``retrieval.BUILDERS``
as in ``tests/test_signals.py``, and its scores are chosen around the shipped 15.0 threshold so
that one row passes the gate and one is refused by it. The dataset is a small frozen fixture in
``tmp_path``, never ``v1.jsonl``.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pytest

from agent.nodes.check_evidence import thresholds
from agent.prompts import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from config import file_sha256, load_config
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import e2e
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path
from retrieval import BUILDERS, clear_cache
from retrieval.types import Hit

KNOWN = frozenset(f"2026-sp:{code}" for code in ("CS-374", "CS-446", "CS-441", "MATH-213"))

#: (id, bucket, split, question, gold ids, the router's label, the fake BM25 top score).
ROWS = [
    # A retrieval row whose evidence clears the 15.0 bm25 threshold: the gate lets it answer.
    ("v1-0001", "fact", "calib", "what is machine learning about?", ["2026-sp:CS-446"],
     "compare_recommend", 20.0),
    # A tool row: an exact lookup, no score to threshold, and the gate passes it untested.
    ("v1-0002", "fact", "calib", "how many hours is cs 374?", ["2026-sp:CS-374"],
     "course_fact", None),
    # Never reaches dispatch, so never reaches the gate either.
    ("v1-0003", "unanswerable", "calib", "what is the weather today?", [], "out_of_scope", None),
    # Reaches dispatch and comes back with nothing: a prereq question naming no course.
    ("v1-0004", "ambiguous", "calib", "what do i need before the course?", [],
     "prereq_chain", None),
    # A retrieval row under the threshold: this is the gate refusing, which #87's run bypassed.
    ("v1-0005", "fact", "calib", "what does the algorithms course cover?", ["2026-sp:CS-374"],
     "compare_recommend", 11.0),
    # The other half: must not be read by a calib run.
    ("v1-0006", "multihop", "test", "what comes after math 213?",
     ["2026-sp:MATH-213", "2026-sp:CS-374"], "compare_recommend", 20.0),
]  # fmt: skip

BY_QUESTION = {row[3]: row for row in ROWS}
DOC_ID_RE = re.compile(r"^doc_id: (.+)$", re.MULTILINE)


class FakeChat:
    """Answers by prompt type, and keeps every call as (kind, prompt, system)."""

    model = "fake-model"

    def __init__(self, cite: str | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.cite = cite

    def complete(self, prompt, *, system=None, temperature=None):
        question = next(q for q in BY_QUESTION if q in prompt)
        row = BY_QUESTION[question]
        if system == ROUTER_SYSTEM_PROMPT:
            kind, reply = "router", row[5]
        elif system == ANSWER_SYSTEM_PROMPT:
            doc_id = self.cite or DOC_ID_RE.search(prompt).group(1)
            kind, reply = "answer", f"The answer. [{doc_id}]"
        else:  # pragma: no cover - a prompt this fake does not know
            raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")
        self.calls.append((kind, prompt, system))
        return reply


class FakeRetriever:
    """Two hits per query, the top one scored by the row's own number."""

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        top = BY_QUESTION[query][6]
        ids = ["2026-sp:CS-446", "2026-sp:CS-441"]
        return [Hit(d, top / r, r, f"text of {d}", {}) for r, d in enumerate(ids, start=1)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    clear_cache()
    monkeypatch.setitem(BUILDERS, "bm25", lambda cfg: FakeRetriever())
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
            question=question,
            gold_answer="an answer",
            gold_doc_ids=gold,
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, question, gold, *_ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


def collect(cfg, dataset, tmp_path, chat=None, split="calib"):
    out = StringIO()
    chat = chat or FakeChat()
    path = e2e.collect_split(cfg, split, out, dataset, tmp_path / "runs", known=KNOWN, chat=chat)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return {line["id"]: line for line in lines}, meta, out.getvalue(), chat


# --- every row gets a line ------------------------------------------------------------------------


def test_every_row_of_the_half_gets_one_line_including_the_refusals(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    assert list(lines) == ["v1-0001", "v1-0002", "v1-0003", "v1-0004", "v1-0005"]
    assert meta["n_rows"] == meta["n_lines"] == 5
    assert all(tuple(line) == e2e.KEYS for line in lines.values())
    assert "v1-0006" not in printed  # the test half is not read by a calib run


def test_the_exit_counts_add_up_to_the_half(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    assert meta["exits"] == {"answer": 2, "refuse_handoff": 2, "out_of_scope": 1}
    assert meta["errors"] == 0
    assert sum(meta["exits"].values()) + meta["errors"] == meta["n_lines"] == len(lines)
    assert meta["exits_by_bucket"]["refuse_handoff"] == {"fact": 1, "ambiguous": 1}
    assert "= 5 of 5 rows" in printed


# --- the gate is on -------------------------------------------------------------------------------


def test_a_row_under_the_threshold_is_refused_by_the_gate_not_answered(cfg, dataset, tmp_path):
    lines, meta, printed, chat = collect(cfg, dataset, tmp_path)
    line = lines["v1-0005"]
    assert line["top_score"] == 11.0 < thresholds(cfg)["bm25"]
    assert line["gate_ok"] is False
    assert line["exit"] == "refuse_handoff"
    assert line["draft"] is None  # the answer node never ran, so there is nothing to discard
    assert "11.0" in line["answer"]  # the hand-off message quotes what it compared
    assert "what does the algorithms course cover?" not in [
        call[1] for call in chat.calls if call[0] == "answer"
    ]
    assert meta["gate_refused"] == {"fact": 1, "ambiguous": 1}
    # v1-0005 scored 11.0 against the 15.0 threshold; v1-0004's tool produced nothing at all.
    assert meta["gate_refused_reasons"] == {"no_evidence": 1, "under_threshold": 1}
    assert "gate refused: 2" in printed
    assert "1 had no evidence to judge and 1 scored under their threshold" in printed
    assert meta["gate"].startswith("on:")


def test_a_row_over_the_threshold_is_answered_with_its_citations(cfg, dataset, tmp_path):
    lines, _, _, _ = collect(cfg, dataset, tmp_path)
    line = lines["v1-0001"]
    assert (line["exit"], line["gate_ok"], line["top_score"]) == ("answer", True, 20.0)
    assert line["answer"] == "The answer. [2026-sp:CS-446]"
    assert line["draft"] is None  # the draft is what shipped, so it is not stored twice
    assert line["citations"] == ["2026-sp:CS-446"]
    assert line["citations_unknown"] is None
    # Every block the answer node was given, not only the cited one (#105 reads the whole list).
    assert line["evidence_doc_ids"] == ["2026-sp:CS-446", "2026-sp:CS-441"]
    assert line["evidence_source"] == "bm25"


def test_a_tool_row_has_a_null_top_score_and_passes_untested(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    line = lines["v1-0002"]
    assert line["top_score"] is None and line["gate_ok"] is True
    assert line["exit"] == "answer"
    assert line["evidence_source"] == "get_course"
    assert line["evidence_doc_ids"] == ["CS 374"]  # a tool's ids are codes, not chunk ids
    assert meta["tool_rows"] == {"fact": 1}
    assert "tool rows, top_score null: 1" in printed


def test_an_out_of_scope_row_never_reaches_the_gate(cfg, dataset, tmp_path):
    line = collect(cfg, dataset, tmp_path)[0]["v1-0003"]
    assert (line["intent"], line["exit"]) == ("out_of_scope", "out_of_scope")
    assert line["gate_ok"] is None and line["top_score"] is None
    assert line["evidence_doc_ids"] == [] and line["evidence_source"] is None
    assert line["citations"] is None and line["answer"]


def test_a_tool_that_produced_nothing_is_refused_with_no_score_to_show(cfg, dataset, tmp_path):
    line = collect(cfg, dataset, tmp_path)[0]["v1-0004"]
    assert (line["intent"], line["exit"]) == ("prereq_chain", "refuse_handoff")
    assert line["gate_ok"] is False and line["top_score"] is None
    assert line["evidence_doc_ids"] == []


def test_a_draft_the_citation_check_discarded_is_kept_beside_what_shipped(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path, chat=FakeChat(cite="NOPE 999"))
    line = lines["v1-0001"]
    assert line["exit"] == "refuse_handoff"
    assert line["gate_ok"] is True  # the gate passed it; the citation check is what discarded it
    assert line["draft"] == "The answer. [NOPE 999]"
    assert line["answer"] != line["draft"] and "NOPE 999" in line["answer"]
    assert line["citations_unknown"] == ["NOPE 999"]
    assert meta["citation_discarded"] == {"fact": 2}
    assert "citation check discarded a draft: 2" in printed


# --- the sidecar ----------------------------------------------------------------------------------


def test_the_sidecar_says_what_the_run_was(cfg, dataset, tmp_path):
    _, meta, _, _ = collect(cfg, dataset, tmp_path)
    path = tmp_path / "runs" / f"e2e_calib_{cfg.hash}.jsonl"
    assert meta["run"].endswith(path.name)
    assert meta["sha256"] == file_sha256(path)
    assert meta["config_hash"] == cfg.hash
    assert meta["dataset_sha256"] == file_sha256(dataset)
    assert meta["split"] == "calib"
    assert meta["model"] == "fake-model"
    assert meta["retrieval_mode"] == cfg["agent"]["retrieval_mode"]
    assert meta["evidence"] == {"method": "top_score", "thresholds": thresholds(cfg)}
    assert meta["timeouts"] == {"chat": 60.0, "embedding": 120.0}
    assert meta["embedding_model"] == cfg["embedding"]["model"]
    assert set(meta["prompts_sha256"]) == {"router_v2", "answer_v1"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in meta["prompts_sha256"].values())
    assert meta["ids"] == ["v1-0001", "v1-0002", "v1-0003", "v1-0004", "v1-0005"]


def test_the_timeouts_are_read_off_the_clients_not_typed(cfg):
    assert e2e.timeouts() == {"chat": 60.0, "embedding": 120.0}


def test_the_run_file_names_the_half_and_the_config(cfg, tmp_path):
    assert e2e.run_path("test", cfg, tmp_path).name == f"e2e_test_{cfg.hash}.jsonl"


# --- the discipline -------------------------------------------------------------------------------


def test_a_second_run_over_the_same_half_is_refused(cfg, dataset, tmp_path):
    collect(cfg, dataset, tmp_path)
    with pytest.raises(ValueError, match="exists"):
        collect(cfg, dataset, tmp_path)


def test_a_failed_row_keeps_its_line_and_the_run_says_it_is_not_final(cfg, dataset, tmp_path):
    class Truncating(FakeChat):
        def complete(self, prompt, *, system=None, temperature=None):
            if system == ANSWER_SYSTEM_PROMPT:
                raise RuntimeError("chat completion was cut off at the max_tokens ceiling (800)")
            return super().complete(prompt, system=system, temperature=temperature)

    lines, meta, printed, _ = collect(cfg, dataset, tmp_path, chat=Truncating())
    line = lines["v1-0001"]
    assert line["exit"] is None and line["answer"] is None
    assert line["error"].startswith("chat completion was cut off")
    assert meta["errors"] == 2  # the two rows the gate let through to the answer node
    assert sum(meta["exits"].values()) + meta["errors"] == meta["n_lines"] == 5
    assert [row["id"] for row in meta["error_rows"]] == ["v1-0001", "v1-0002"]
    assert "2 row(s) failed to run" in printed
    assert "is not the final run" in printed


def test_split_is_required(capsys):
    with pytest.raises(SystemExit):
        e2e.main([])
    assert "--split" in capsys.readouterr().err
