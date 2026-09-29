"""Tests for eval/scripts/e2e_metrics.py: the four numerators, their denominators, and the ids.

Everything is a fixture in ``tmp_path`` -- a nine-row test half, a run file over it, a verdict
CSV and a gold-intent sidecar -- never ``v1.jsonl`` or the committed run. The nine rows are
chosen so that each metric has at least one row on both sides of it, and so that the one case
the two readings disagree about (an abstaining row that answered and whose text a person judged
a decline) is in there exactly once.

The Intent guidelines are the real ones: :func:`eval.intents.stamp` digests the section the
labels were made under, and a fake section would test the digest rather than the metrics.
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
from eval.scripts import e2e, e2e_labels, e2e_metrics
from eval.scripts.conformal_eval import clopper_pearson
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path

#: (id, bucket, split, intent the router chose, gold intents, exit, the person's verdict).
#: Read down the columns: three fact rows, one multihop, three unanswerable, two ambiguous.
ROWS: list[tuple[str, str, str, str, list[str], str, bool | None]] = [
    # Answered and right: the plain case, and no false block.
    ("v1-0001", "fact", "test", "course_fact", ["course_fact"], "answer", True),
    # Answered and wrong, and its gold intent is a set of two -- the router picked the second.
    ("v1-0002", "fact", "test", "prereq_chain", ["course_fact", "prereq_chain"], "answer", False),
    # Refused a question it should have answered: a false block, and an intent miss with it.
    ("v1-0003", "fact", "test", "compare_recommend", ["course_fact"], "refuse_handoff", None),
    ("v1-1001", "multihop", "test", "prereq_chain", ["prereq_chain"], "answer", False),
    # Declined through an exit its bucket accepts.
    ("v1-2001", "unanswerable", "test", "out_of_scope", ["out_of_scope"], "out_of_scope", None),
    # The one row the two readings disagree about: it left through `answer`, and the person who
    # read the text said it declined.
    ("v1-2002", "unanswerable", "test", "course_fact", ["course_fact"], "answer", True),
    # Answered a question with no answer in the catalog: a false release under both readings.
    ("v1-2003", "unanswerable", "test", "course_fact", ["course_fact"], "answer", False),
    # The caveat row: ambiguous, handed off, and nobody read the fixed text.
    ("v1-3001", "ambiguous", "test", "prereq_chain", ["prereq_chain"], "refuse_handoff", None),
    # Picked one reading and answered it: a false release, and an intent miss.
    ("v1-3002", "ambiguous", "test", "course_fact", ["prereq_chain"], "answer", False),
    # The other half. It is in the dataset, in no other file, and in no number below.
    ("v1-0004", "fact", "calib", "course_fact", [], "answer", True),
]

#: Enough gold ids for the strictest bucket; :data:`eval.schema.RULES` says how many each takes.
GOLD_IDS = ["2026-sp:CS-374", "2026-sp:MATH-213"]

TEST_ROWS = [row for row in ROWS if row[2] == "test"]
ANSWERED = [row for row in TEST_ROWS if row[5] == "answer"]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A frozen nine-plus-one row dataset, with the sidecar ``load_split`` checks."""
    path = tmp_path / "v1.jsonl"
    items = [
        EvalItem(
            id=id_,
            bucket=bucket,
            question=f"question {id_}",
            gold_answer="an answer",
            gold_doc_ids=GOLD_IDS[: RULES[bucket].min_gold_ids],
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, *_ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


@pytest.fixture
def sidecar(tmp_path: Path, dataset: Path) -> Path:
    """#102's gold intents over the test half, stamped under the real Intent guidelines."""
    path = tmp_path / "v1_intents.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": id_,
                    "gold_intents": gold,
                    "note": "both readings stand" if len(gold) > 1 else "",
                }
            )
            + "\n"
            for id_, _, _, _, gold, _, _ in TEST_ROWS
        ),
        encoding="utf-8",
    )
    written, violations = gold_intents.stamp("tester", path, dataset, gold_intents.GUIDELINES_PATH)
    assert written is not None, violations
    return path


def run_line(row: tuple[Any, ...]) -> dict[str, Any]:
    """One #103 run line: every key it declares, and nothing this file needs invented."""
    id_, bucket, split, intent, _, exit_, _ = row
    answered = exit_ == "answer"
    return {
        **dict.fromkeys(e2e.KEYS),
        "id": id_,
        "bucket": bucket,
        "split": split,
        "question": f"question {id_}",
        "intent": intent,
        "exit": exit_,
        "answer": "what the user saw",
        "citations": ["2026-sp:CS-374"] if answered else None,
        "evidence_doc_ids": ["2026-sp:CS-374"] if answered else [],
        "evidence_source": "bm25" if answered else None,
        "top_score": 20.0 if answered else None,
        "gate_ok": answered or None,
        "error": None,
    }


def write_run(cfg, runs_dir: Path, dataset: Path, rows=TEST_ROWS, **meta: Any) -> Path:
    """The run file and the sidecar :func:`eval.scripts.e2e_metrics.read_run` checks."""
    path = e2e.run_path("test", cfg, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [run_line(row) for row in rows]
    path.write_text(
        "".join(json.dumps({key: line[key] for key in e2e.KEYS}) + "\n" for line in lines),
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


def write_csv(cfg, runs_dir: Path, run: Path, rows=ANSWERED, **meta: Any) -> Path:
    """The verdict CSV #103 writes for the answered rows, in its committed column order."""
    path = e2e_labels.csv_path("test", cfg, runs_dir)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(e2e_labels.CSV_COLUMNS)
        for id_, bucket, split, intent, _, exit_, verdict in rows:
            writer.writerow(
                [id_, bucket, split, intent, exit_, 20.0, 1, int(verdict), int(verdict), "tester"]
            )
    write_json(
        path.with_name(path.name + ".meta.json"),
        {
            "csv": str(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "e2e_run": str(run),
            "e2e_run_sha256": file_sha256(run),
            **meta,
        },
    )
    return path


@pytest.fixture
def measured(cfg, tmp_path, dataset, sidecar):
    """Everything in place, measured once: the result JSON and what the script printed."""
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run)
    out = StringIO()
    path = e2e_metrics.run(cfg, out, runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)
    return json.loads(path.read_text(encoding="utf-8")), out.getvalue(), path


# --- the four numbers -----------------------------------------------------------------------------


def test_each_numerator_and_denominator_is_the_one_the_issue_defines(measured):
    result, _, _ = measured
    m = result["metrics"]
    counts = {
        "intent": (m["intent_accuracy"]["k"], m["intent_accuracy"]["n"]),
        "correct_verdict": (
            m["answer_correctness"]["verdict_backed"]["k"],
            m["answer_correctness"]["verdict_backed"]["n"],
        ),
        "correct_exit": (
            m["answer_correctness"]["exit_only"]["k"],
            m["answer_correctness"]["exit_only"]["n"],
        ),
        "release_verdict": (
            m["false_release"]["verdict_backed"]["k"],
            m["false_release"]["verdict_backed"]["n"],
        ),
        "release_exit": (
            m["false_release"]["exit_only"]["k"],
            m["false_release"]["exit_only"]["n"],
        ),
        "block": (m["false_block"]["k"], m["false_block"]["n"]),
    }
    assert counts == {
        # Two rows miss: v1-0003 (compare_recommend for course_fact) and v1-3002.
        "intent": (7, 9),
        # v1-0001, v1-2001, v1-2002 (on its verdict) and v1-3001 (on its exit).
        "correct_verdict": (4, 9),
        # The same, without v1-2002: an abstaining row that answered released, whatever it said.
        "correct_exit": (3, 9),
        # 3 unanswerable + 2 ambiguous rows should have declined.
        "release_verdict": (2, 5),
        "release_exit": (3, 5),
        # 3 fact + 1 multihop should have answered; v1-0003 did not.
        "block": (1, 4),
    }
    assert result["n"] == 9
    assert result["buckets"] == {"fact": 3, "multihop": 1, "unanswerable": 3, "ambiguous": 2}


def test_the_two_readings_differ_only_on_an_abstaining_row_that_answered(measured):
    result, _, _ = measured
    correctness = result["metrics"]["answer_correctness"]
    verdict = set(correctness["verdict_backed"]["numerator_ids"])
    exit_only = set(correctness["exit_only"]["numerator_ids"])
    assert verdict - exit_only == {"v1-2002"} and exit_only - verdict == set()
    release = result["metrics"]["false_release"]
    assert set(release["exit_only"]["numerator_ids"]) - set(
        release["verdict_backed"]["numerator_ids"]
    ) == {"v1-2002"}


def test_a_false_block_is_an_answerable_row_that_did_not_answer(measured):
    result, _, _ = measured
    block = result["metrics"]["false_block"]
    assert block["numerator_ids"] == ["v1-0003"]
    assert block["denominator_ids"] == ["v1-0001", "v1-0002", "v1-0003", "v1-1001"]
    assert block["by_bucket"]["fact"]["k"] == 1
    assert block["by_bucket"]["multihop"] == {"k": 0, "n": 1, "rate": 0.0, "numerator_ids": []}


def test_a_false_release_is_an_abstaining_row_that_answered(measured):
    result, _, _ = measured
    release = result["metrics"]["false_release"]
    assert release["verdict_backed"]["numerator_ids"] == ["v1-2003", "v1-3002"]
    assert release["exit_only"]["numerator_ids"] == ["v1-2002", "v1-2003", "v1-3002"]
    assert release["verdict_backed"]["denominator_ids"] == [
        "v1-2001", "v1-2002", "v1-2003", "v1-3001", "v1-3002",
    ]  # fmt: skip


def test_intent_accuracy_reads_a_set_of_acceptable_labels_not_one(measured):
    result, _, _ = measured
    intent = result["metrics"]["intent_accuracy"]
    # v1-0002 was routed to prereq_chain and its labeller accepted both readings.
    assert "v1-0002" in intent["numerator_ids"]
    assert [miss["id"] for miss in intent["misses"]] == ["v1-0003", "v1-3002"]
    assert intent["misses"][0] == {
        "id": "v1-0003",
        "bucket": "fact",
        "predicted": "compare_recommend",
        "gold_intents": ["course_fact"],
        "exit": "refuse_handoff",
        "outcome_correct_verdict_backed": False,
    }


def test_an_intent_the_half_has_no_row_for_is_reported_as_no_sample(measured):
    result, printed, _ = measured
    labels = result["metrics"]["intent_accuracy"]["rows_with_gold_label"]
    assert labels["degree_requirement"] == 0 and labels["compare_recommend"] == 0
    assert labels["course_fact"] == 5
    assert "no test row is labelled degree_requirement, compare_recommend" in printed


# --- every number can be traced back to rows ------------------------------------------------------


def test_every_number_carries_the_ids_it_was_counted_from(measured):
    result, _, _ = measured
    for name, block in walk(result["metrics"]):
        assert block["k"] == len(block["numerator_ids"]), name
        assert block["rate"] == pytest.approx(block["k"] / block["n"]), name
        if "denominator_ids" in block:
            assert block["n"] == len(block["denominator_ids"]), name
            assert set(block["numerator_ids"]) <= set(block["denominator_ids"]), name


def test_the_buckets_of_a_metric_add_up_to_it(measured):
    result, _, _ = measured
    for name, block in walk(result["metrics"]):
        if "by_bucket" not in block:
            continue
        assert sum(b["k"] for b in block["by_bucket"].values()) == block["k"], name
        assert sum(b["n"] for b in block["by_bucket"].values()) == block["n"], name


def test_a_per_bucket_denominator_is_the_metrics_own_cut_by_rows_by_bucket(measured):
    result, _, _ = measured
    by_bucket = result["rows_by_bucket"]
    assert sorted(i for ids in by_bucket.values() for i in ids) == sorted(r[0] for r in TEST_ROWS)
    for name, block in walk(result["metrics"]):
        if "by_bucket" not in block:
            continue
        for bucket, cut in block["by_bucket"].items():
            expected = set(block["denominator_ids"]) & set(by_bucket[bucket])
            assert cut["n"] == len(expected), f"{name}.{bucket}"


def walk(metrics: dict[str, Any]):
    """Every ``score`` block in the result, by a dotted name."""
    for name, block in metrics.items():
        if "k" in block:
            yield name, block
        else:
            for reading, inner in block.items():
                yield f"{name}.{reading}", inner


# --- the interval ---------------------------------------------------------------------------------


def test_the_interval_is_clopper_pearson_at_the_configs_level(measured, cfg):
    result, _, _ = measured
    level = float(cfg["eval"]["bootstrap"]["ci"])
    assert result["ci_level"] == level
    assert result["ci_method"] == "Clopper-Pearson, two-sided"
    block = result["metrics"]["false_block"]
    assert block["ci"] == list(clopper_pearson(block["k"], block["n"], level))


def test_a_bucket_gets_a_point_estimate_and_no_interval(measured):
    result, printed, _ = measured
    for _, block in walk(result["metrics"]):
        for cut in block.get("by_bucket", {}).values():
            assert "ci" not in cut
    assert "point estimates, no interval" in printed


# --- the ambiguous caveat -------------------------------------------------------------------------


def test_an_ambiguous_handoff_is_counted_correct_and_flagged_as_unread(measured):
    result, printed, _ = measured
    caveat = result["caveats"]["ambiguous_handoff"]
    assert caveat["ids"] == ["v1-3001"] and caveat["n"] == 1
    # It is in the headline numerator, and the bound is the headline without it.
    correctness = result["metrics"]["answer_correctness"]["verdict_backed"]
    assert "v1-3001" in correctness["numerator_ids"]
    assert caveat["correctness_if_none_counted"]["k"] == correctness["k"] - 1
    assert caveat["correctness_if_none_counted"]["n"] == correctness["n"]
    assert "nobody read the text (v1-3001)" in printed
    assert "3 / 9" in printed


def test_an_unanswerable_handoff_is_not_in_the_caveat(measured):
    # v1-2001 declined through an exit, which is the whole of what its bucket asks for.
    result, _, _ = measured
    assert "v1-2001" not in result["caveats"]["ambiguous_handoff"]["ids"]


# --- provenance -----------------------------------------------------------------------------------


def test_the_result_names_every_file_it_read_and_who_labelled_the_intents(measured, cfg, dataset):
    result, _, path = measured
    assert path.name == f"e2e_metrics_{cfg.hash}.json"
    assert result["split"] == "test" and result["config_hash"] == cfg.hash
    assert result["intents_labeller"] == "tester"
    assert result["dataset_sha256"] == file_sha256(dataset)
    assert result["guidelines_intent_sha256"] == gold_intents.section_sha256()
    assert result["run"].endswith(f"e2e_test_{cfg.hash}.jsonl")
    assert result["csv"].endswith(f"e2e_test_{cfg.hash}.csv")


def test_reading_it_twice_gives_the_same_file(cfg, tmp_path, dataset, sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run)
    args = (runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)
    first = e2e_metrics.run(cfg, StringIO(), *args).read_bytes()
    assert e2e_metrics.run(cfg, StringIO(), *args).read_bytes() == first


def test_the_calib_row_is_not_measured(measured):
    result, _, _ = measured
    for _, block in walk(result["metrics"]):
        assert "v1-0004" not in block.get("denominator_ids", [])


# --- what it refuses ------------------------------------------------------------------------------


def test_a_run_with_an_errored_row_is_refused(cfg, tmp_path, dataset, sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset, errors=1)
    write_csv(cfg, runs_dir, run)
    with pytest.raises(ValueError, match="failed to run, so it is not a result"):
        e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)


def test_a_run_that_is_not_the_file_its_sidecar_describes_is_refused(cfg, tmp_path, dataset,
                                                                     sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run)
    run.write_text(run.read_text(encoding="utf-8").replace("v1-0001", "v1-9999"), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 is not the one its sidecar recorded"):
        e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)


def test_a_csv_that_labels_another_copy_of_the_run_is_refused(cfg, tmp_path, dataset, sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run, e2e_run_sha256="0" * 64)
    with pytest.raises(ValueError, match="labels another copy of"):
        e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)


def test_an_answered_row_with_no_verdict_is_refused(cfg, tmp_path, dataset, sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run, rows=ANSWERED[1:])
    with pytest.raises(ValueError, match="v1-0001 left through answer and the CSV has none"):
        e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)


def test_a_half_the_gold_intents_do_not_cover_is_refused(cfg, tmp_path, dataset, sidecar):
    runs_dir = tmp_path / "runs"
    run = write_run(cfg, runs_dir, dataset)
    write_csv(cfg, runs_dir, run)
    lines = sidecar.read_text(encoding="utf-8").splitlines(keepends=True)
    sidecar.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="does not pass"):
        e2e_metrics.run(cfg, StringIO(), runs_dir, dataset, sidecar, gold_intents.GUIDELINES_PATH)


def test_a_missing_run_says_which_command_makes_it(cfg, tmp_path, dataset, sidecar, capsys):
    code = e2e_metrics.main(
        [
            "--runs-dir", str(tmp_path / "runs"),
            "--dataset", str(dataset),
            "--intents", str(sidecar),
            "--guidelines", str(gold_intents.GUIDELINES_PATH),
        ]
    )  # fmt: skip
    assert code == 1
    assert "eval.scripts.e2e --split test" in capsys.readouterr().err


# --- the predicates, on their own -----------------------------------------------------------------


def test_correct_takes_one_of_the_two_readings():
    row = e2e_metrics.Row("v1-0001", "fact", "answer", "course_fact",
                          frozenset({"course_fact"}), False, True)  # fmt: skip
    with pytest.raises(ValueError, match="reading must be one of"):
        e2e_metrics.correct(row, "whatever")


def test_a_numerator_larger_than_its_denominator_is_a_bug_not_a_rate():
    row = e2e_metrics.Row("v1-0001", "fact", "answer", "course_fact",
                          frozenset({"course_fact"}), False, True)  # fmt: skip
    with pytest.raises(ValueError, match="are not nested"):
        e2e_metrics.score([row, row], [row])
