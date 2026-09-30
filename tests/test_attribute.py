"""Tests for eval/scripts/attribute.py: the rule order, the two alphabets, and who may write.

Everything is a fixture in ``tmp_path`` -- an eleven-row test half, a run over it, a verdict CSV,
a gold-intent sidecar and #104's metrics JSON, which this file's fixture produces by running
:mod:`eval.scripts.e2e_metrics` rather than hand-writing what it would have said. The rows are
chosen so that the failures cover all four classes, the ``other`` row, and the one case the rule
order is for: a row that fits rule 1 and rule 3 at once.

The Intent guidelines are the real ones, for the reason :mod:`tests.test_e2e_metrics` gives: the
sidecar is stamped with the digest of the section the labels were made under.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from config import file_sha256, load_config
from eval import intents as gold_intents
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import attribute, e2e, e2e_labels, e2e_metrics
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path

#: (id, bucket, split, intent, gold intents, exit, verdict, gold ids, evidence ids, source).
#: Read down the failures: one per class, plus the leftover the four do not cover.
ROWS: list[tuple[Any, ...]] = [
    # Answered and right. Not a failure, and in no table below.
    ("v1-0001", "fact", "test", "course_fact", ["course_fact"], "answer", True,
     ["2026-sp:CS-374"], ["2026-sp:CS-374"], "bm25"),
    # Rule 1 and rule 3 both fit: the router missed, and the gold was in the prompt anyway.
    # The order is what decides it, and this is the row that proves the order is applied.
    ("v1-0002", "fact", "test", "compare_recommend", ["course_fact"], "answer", False,
     ["2026-sp:CS-374"], ["2026-sp:CS-374"], "bm25"),
    # Rule 2: the retriever brought back a course, and not the one the row is about.
    ("v1-0003", "fact", "test", "course_fact", ["course_fact"], "answer", False,
     ["2026-sp:CS-374"], ["2026-sp:MATH-213"], "bm25"),
    # Rule 3, across the two alphabets: a tool's `CS 374` is the gold `2026-sp:CS-374`, so rule 2
    # does not fire and the wrong answer was written with the gold evidence in front of it.
    ("v1-0004", "fact", "test", "course_fact", ["course_fact"], "answer", False,
     ["2026-sp:CS-374"], ["CS 374"], "get_course"),
    # The leftover: a false block whose gold *was* retrieved. None of the four covers it.
    ("v1-0005", "fact", "test", "course_fact", ["course_fact"], "refuse_handoff", None,
     ["2026-sp:CS-374"], ["2026-sp:CS-374"], "bm25"),
    # Rule 3 again, on a multihop row with one of its two gold ids in the evidence.
    ("v1-1001", "multihop", "test", "prereq_chain", ["prereq_chain"], "answer", False,
     ["2026-sp:CS-374", "2026-sp:MATH-213"], ["CS 374"], "get_prereq_chain"),
    # Rule 4: it should have declined, it answered, and there is no gold evidence to misread.
    ("v1-2001", "unanswerable", "test", "course_fact", ["course_fact"], "answer", False,
     [], ["2026-sp:CS-374"], "bm25"),
    # Declined through an exit its bucket accepts: correct, not a failure.
    ("v1-2002", "unanswerable", "test", "out_of_scope", ["out_of_scope"], "out_of_scope", None,
     [], [], None),
    # The row the two readings disagree about: it answered, and the person who read the text said
    # it declined. A failure under exit-only only, so it is reported and never attributed.
    ("v1-2003", "unanswerable", "test", "course_fact", ["course_fact"], "answer", True,
     [], ["2026-sp:CS-374"], "bm25"),
    # Asked which course was meant: correct.
    ("v1-3001", "ambiguous", "test", "course_fact", ["course_fact"], "refuse_handoff", None,
     [], [], None),
    # The other half. In the dataset, in no run, in no number here.
    ("v1-0009", "fact", "calib", "course_fact", [], "answer", True,
     ["2026-sp:CS-374"], ["2026-sp:CS-374"], "bm25"),
]

TEST_ROWS = [row for row in ROWS if row[2] == "test"]
ANSWERED = [row for row in TEST_ROWS if row[5] == "answer"]

#: What the rules make of the six failures, before anyone confirms anything.
EXPECTED = {
    "v1-0002": "intent_misjudged",
    "v1-0003": "retrieval_failure",
    "v1-0004": "evidence_misread",
    "v1-0005": "other",
    "v1-1001": "evidence_misread",
    "v1-2001": "overconfident",
}


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
            question=f"question {id_}",
            gold_answer=f"the answer to {id_}",
            gold_doc_ids=gold,
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, _, _, _, _, gold, _, _ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


@pytest.fixture
def sidecar(tmp_path: Path, dataset: Path) -> Path:
    path = tmp_path / "v1_intents.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": row[0], "gold_intents": row[4], "note": ""}) + "\n"
            for row in TEST_ROWS
        ),
        encoding="utf-8",
    )
    written, violations = gold_intents.stamp("tester", path, dataset, gold_intents.GUIDELINES_PATH)
    assert written is not None, violations
    return path


def run_line(row: tuple[Any, ...]) -> dict[str, Any]:
    """One #103 run line, with the evidence this file's rows are about."""
    id_, bucket, split, intent, _, exit_, _, _, evidence, source = row
    answered = exit_ == "answer"
    return {
        **dict.fromkeys(e2e.KEYS),
        "id": id_,
        "bucket": bucket,
        "split": split,
        "question": f"question {id_}",
        "intent": intent,
        "exit": exit_,
        "answer": f"what the user saw for {id_}",
        "citations": list(evidence) if answered else None,
        "evidence_doc_ids": list(evidence),
        "evidence_source": source,
        "top_score": 14.0 if source == "bm25" else None,
        "gate_ok": answered or False,
        "error": None,
    }


def write_run(cfg, runs_dir: Path, dataset: Path, rows=TEST_ROWS, **meta: Any) -> Path:
    path = e2e.run_path("test", cfg, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({key: run_line(row)[key] for key in e2e.KEYS}) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    write_json(
        path.with_suffix(".meta.json"),
        {
            "run": str(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "split": "test",
            "dataset_sha256": file_sha256(dataset),
            "errors": 0,
            **meta,
        },
    )
    return path


def write_csv(cfg, runs_dir: Path, run: Path, rows=ANSWERED) -> Path:
    path = e2e_labels.csv_path("test", cfg, runs_dir)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(e2e_labels.CSV_COLUMNS)
        for id_, bucket, split, intent, _, exit_, verdict, *_ in rows:
            writer.writerow(
                [id_, bucket, split, intent, exit_, "", 1, int(verdict), int(verdict), "tester"]
            )
    write_json(
        path.with_name(path.name + ".meta.json"),
        {
            "csv": str(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "e2e_run": str(run),
            "e2e_run_sha256": file_sha256(run),
        },
    )
    return path


@pytest.fixture
def runs(cfg, tmp_path: Path, dataset: Path, sidecar: Path) -> Path:
    """A run, its verdicts and #104's metrics JSON over them -- everything this script reads."""
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run)
    e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)
    return runs_dir


@pytest.fixture
def loaded(cfg, runs: Path, dataset: Path, sidecar: Path):
    """The failures and their proposals, as every entry point sees them."""
    joined = attribute.load_joined(cfg, runs, dataset, sidecar)
    failures = attribute.failures_of(joined)
    return joined, failures, {f.id: attribute.propose(f) for f in failures}


def asker(answers: list[str]):
    """An ``ask`` that reads down a script and then behaves like a closed stdin."""
    queue = list(answers)
    return lambda _prompt: queue.pop(0) if queue else None


def review_all(failures, proposals, path: Path, answers: list[str], decided=None, **kw):
    """Run the loop over a scripted sitting, returning what it wrote and what it counted."""
    decided = {} if decided is None else decided
    made = attribute.review(
        failures, proposals, decided, kw.pop("reviewer", "tester"), path, {}, StringIO(),
        asker(answers), **kw,
    )
    return decided, made


# --- the rules ------------------------------------------------------------------------------------


def test_the_docstring_carries_the_rules_word_for_word():
    """#105's acceptance: the rules and their order are in the docstring, and in the report."""
    rendered = " ".join(attribute.rules_text().split())
    assert rendered in " ".join(attribute.__doc__.split())
    assert [rule.number for rule in attribute.RULES] == [1, 2, 3, 4]
    assert [rule.name for rule in attribute.RULES] == list(attribute.CLASSES)
    assert [rule.mechanical for rule in attribute.RULES] == [True, True, False, False]


def test_each_failure_gets_the_first_rule_that_fires(loaded):
    _, failures, proposals = loaded
    assert {f.id: proposals[f.id].cls for f in failures} == EXPECTED


def test_the_order_decides_a_row_that_fits_two_classes(loaded):
    """v1-0002 misjudged the intent *and* answered wrongly with the gold in the prompt."""
    _, failures, proposals = loaded
    row = next(f for f in failures if f.id == "v1-0002")
    assert proposals["v1-0002"].cls == "intent_misjudged"
    assert proposals["v1-0002"].rule == 1
    # Rule 3's own condition holds on the same row: the gold evidence was there.
    assert attribute.gold_in_evidence(row.item, row.line) == ["2026-sp:CS-374"]


def test_rules_one_and_two_decide_and_three_and_four_propose(loaded):
    _, _, proposals = loaded
    assert proposals["v1-0003"].mechanical and proposals["v1-0002"].mechanical
    assert not proposals["v1-0004"].mechanical and not proposals["v1-2001"].mechanical
    assert proposals["v1-0005"].rule is None and not proposals["v1-0005"].mechanical


def test_the_leftover_is_printed_and_never_folded_into_one_of_the_four(loaded, cfg, runs):
    """A false block whose gold was retrieved is `other`, with its id on screen."""
    _, failures, proposals = loaded
    assert proposals["v1-0005"].cls == attribute.OTHER
    decided = {
        f.id: {**{k: None for k in attribute.KEYS}, "id": f.id, "bucket": f.bucket,
               "proposed": proposals[f.id].cls, "final": proposals[f.id].cls,
               "reviewer": "tester", "action": "confirmed"}
        for f in failures
    }
    result = attribute.table(failures, decided)
    assert result["classes"]["other"]["ids"] == ["v1-0005"]
    printed = attribute.render_table(result, {"n_rows": 10, "n_correct": 4, "split": "test",
                                              "reading": attribute.READING})
    assert "v1-0005" in printed and "never folded" in printed


# --- the two alphabets ----------------------------------------------------------------------------


def test_a_tool_code_and_a_chunk_id_name_the_same_course():
    assert attribute.same_course("2026-sp:CS-374", "CS 374")
    assert attribute.same_course("2026-sp:MUS-090", "mus090")
    assert attribute.same_course("2026-sp:CS-374", "2026-sp:CS-374")
    assert not attribute.same_course("2026-sp:CS-374", "CS 375")
    # A program id from check_requirement is not a course, so it matches no gold id.
    assert not attribute.same_course("2026-sp:CS-374", "gened")


def test_two_chunk_ids_are_compared_whole_so_a_term_cannot_be_dropped():
    assert not attribute.same_course("2026-sp:CS-374", "2025-fa:CS-374")
    assert attribute.course_key("2026-sp:CS-374") == attribute.course_key("cs 374") == "CS 374"


def test_rule_two_reads_a_tool_row_through_the_code_matcher(loaded):
    """v1-0004's evidence is `CS 374` for gold `2026-sp:CS-374`: found, so rule 2 does not fire."""
    _, failures, proposals = loaded
    row = next(f for f in failures if f.id == "v1-0004")
    assert attribute.gold_in_evidence(row.item, row.line) == ["2026-sp:CS-374"]
    assert proposals["v1-0004"].cls == "evidence_misread"


# --- the failures are #104's ----------------------------------------------------------------------


def test_the_failures_are_the_rows_104_counted_wrong(cfg, runs, loaded):
    _, failures, _ = loaded
    checked = attribute.check_against_metrics(failures, cfg, runs)
    assert sorted(f.id for f in failures) == sorted(EXPECTED)
    assert checked["n_failures"] == len(failures) == checked["n_rows"] - checked["n_correct"]
    assert len({f.id for f in failures}) == len(failures)


def test_a_metrics_file_that_disagrees_is_refused(cfg, runs, loaded):
    _, failures, _ = loaded
    path = e2e_metrics.result_path(cfg, runs)
    result = json.loads(path.read_text(encoding="utf-8"))
    block = result["metrics"]["answer_correctness"][attribute.READING]
    block["numerator_ids"] = [*block["numerator_ids"], "v1-0003"]
    write_json(path, result)
    with pytest.raises(ValueError, match="disagree about which rows failed"):
        attribute.check_against_metrics(failures, cfg, runs)


def test_the_metrics_have_to_exist_first(cfg, runs, loaded):
    _, failures, _ = loaded
    e2e_metrics.result_path(cfg, runs).unlink()
    with pytest.raises(ValueError, match="no end-to-end metrics"):
        attribute.check_against_metrics(failures, cfg, runs)


def test_the_other_reading_is_reported_and_not_attributed(loaded):
    """v1-2003 answered and the reviewer read it as a decline: a failure under exit-only only."""
    joined, failures, _ = loaded
    assert "v1-2003" not in {f.id for f in failures}
    assert attribute.exit_only_extra(joined) == ["v1-2003"]


# --- who may write --------------------------------------------------------------------------------


def test_the_proposals_run_writes_nothing(cfg, runs, dataset, sidecar):
    out = StringIO()
    code = attribute.main(
        ["--runs-dir", str(runs), "--dataset", str(dataset), "--intents", str(sidecar)],
        out=out,
        runs_dir=runs,
    )
    assert code == 0
    assert not attribute.attribution_path(cfg, runs).exists()
    printed = out.getvalue()
    assert "nothing was written" in printed
    assert "retrieval_failure" in printed and "v1-0005" in printed


def test_the_table_refuses_until_every_failure_has_a_class(loaded):
    _, failures, _ = loaded
    decided = {"v1-0002": {"final": "intent_misjudged"}}
    with pytest.raises(ValueError, match="5 of 6 failures have no confirmed class"):
        attribute.table(failures, decided)


def test_a_sitting_confirms_and_writes_one_row_at_a_time(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    _, made = review_all(failures, proposals, path, ["c", "c", "q"])
    assert made == {"confirmed": 2}
    written = attribute.read_rows(path)
    assert [row["id"] for row in written] == ["v1-0002", "v1-0003"]
    assert {row["action"] for row in written} == {"confirmed"}
    assert [row["final"] for row in written] == ["intent_misjudged", "retrieval_failure"]
    assert json.loads(path.with_suffix(".meta.json").read_text())["complete"] is False


def test_the_pass_resumes_where_it_stopped(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    review_all(failures, proposals, path, ["c", "c", "q"])
    decided = attribute.current(path, failures, proposals)
    out = StringIO()
    attribute.review(
        failures, proposals, decided, "tester", path, {}, out, asker(["c", "q"]),
    )
    assert "v1-0004" in out.getvalue() and "v1-0002" not in out.getvalue()
    assert [row["id"] for row in attribute.read_rows(path)] == ["v1-0002", "v1-0003", "v1-0004"]


def test_an_override_is_not_taken_without_a_reason(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    out = StringIO()
    decided: dict[str, dict] = {}
    attribute.review(
        failures[:1], proposals, decided, "tester", path, {}, out,
        asker(["3", "  ", "the router was right, the answer dropped a requirement"]),
    )
    assert "an override needs a reason" in out.getvalue()
    row = attribute.read_rows(path)[0]
    assert row["final"] == "evidence_misread" and row["proposed"] == "intent_misjudged"
    assert row["action"] == "overrode_rule"
    assert row["reason"] == "the router was right, the answer dropped a requirement"


def test_overriding_a_proposal_and_a_rule_are_counted_apart(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    # v1-0002 is rule 1 (mechanical); v1-0004 is a proposal.
    rows = [f for f in failures if f.id in {"v1-0002", "v1-0004"}]
    _, made = review_all(rows, proposals, path, ["4", "it answered anyway", "o", "neither fits"])
    assert made == {"overrode_rule": 1, "overrode_proposal": 1}
    assert [row["final"] for row in attribute.read_rows(path)] == ["overconfident", "other"]


def test_a_file_with_a_reasonless_override_is_refused(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    review_all(failures, proposals, path, ["c", "q"])
    rows = attribute.read_rows(path)
    rows[0]["final"] = "overconfident"
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gives no reason"):
        attribute.read_rows(path)


def test_a_line_without_a_reviewer_is_not_a_decision(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    review_all(failures, proposals, path, ["c", "q"])
    rows = attribute.read_rows(path)
    path.write_text(json.dumps({**rows[0], "reviewer": ""}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no reviewer"):
        attribute.read_rows(path)


def test_a_decision_made_against_another_proposal_is_refused(loaded, cfg, runs):
    """The run moved under the reviewer: the stored class was about a different reading."""
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    review_all(failures, proposals, path, ["c", "q"])
    moved = {**proposals, "v1-0002": attribute.Proposal("overconfident", 4, "changed")}
    with pytest.raises(ValueError, match="reviewed against a proposal of intent_misjudged"):
        attribute.current(path, failures, moved)


def test_a_row_that_is_not_a_failure_of_this_run_is_refused(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    review_all(failures, proposals, path, ["c", "q"])
    rows = attribute.read_rows(path)
    path.write_text(json.dumps({**rows[0], "id": "v1-9999"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a failure of this run"):
        attribute.current(path, failures, proposals)


# --- the table ------------------------------------------------------------------------------------


def test_every_failure_appears_exactly_once_and_the_counts_add_up(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    decided, _ = review_all(failures, proposals, path, ["c"] * len(failures))
    result = attribute.table(failures, decided)
    counted = [i for block in result["classes"].values() for i in block["ids"]]
    assert sorted(counted) == sorted(f.id for f in failures)
    assert len(counted) == len(set(counted)) == result["n_failures"] == len(failures)
    assert sum(block["n"] for block in result["classes"].values()) == len(failures)
    assert sum(block["share"] for block in result["classes"].values()) == pytest.approx(1.0)


def test_the_table_splits_every_class_per_bucket(loaded, cfg, runs):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    decided, _ = review_all(failures, proposals, path, ["c"] * len(failures))
    result = attribute.table(failures, decided)
    for cls, block in result["classes"].items():
        assert sum(block["by_bucket"].values()) == block["n"], cls
    assert result["classes"]["evidence_misread"]["by_bucket"] == {"fact": 1, "multihop": 1,
                                                                 "unanswerable": 0}
    assert result["classes"]["overconfident"]["by_bucket"] == {"fact": 0, "multihop": 0,
                                                              "unanswerable": 1}
    assert result["buckets"] == ["fact", "multihop", "unanswerable"]


def test_the_sidecar_names_the_classes_and_the_reviewers(loaded, cfg, runs, dataset, sidecar):
    _, failures, proposals = loaded
    path = attribute.attribution_path(cfg, runs)
    meta = attribute.sources(cfg, runs, dataset, sidecar)
    decided: dict[str, dict] = {}
    attribute.review(
        failures, proposals, decided, "tester", path, meta, StringIO(),
        asker(["c"] * len(failures)),
    )
    written = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert written["complete"] is True
    assert written["n_attributed"] == len(failures)
    assert written["reviewers"] == {"tester": len(failures)}
    assert written["by_class"]["evidence_misread"] == 2
    assert written["actions"] == {"confirmed": 6, "overrode_proposal": 0, "overrode_rule": 0}
    assert written["rules"] == [rule.text for rule in attribute.RULES]
    assert written["reading"] == attribute.READING
