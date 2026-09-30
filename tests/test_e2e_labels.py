"""Tests for eval/scripts/e2e_labels.py: #88's judge on the answered rows, then a person on each.

The run file is written by hand into ``tmp_path``, so these tests are about the labelling pass and
not about the agent: what is judged, what carries over, what the CSV refuses to say. The chat
client is a fake that answers a judge prompt by the question in it, and the reviewer is a list of
keystrokes. Both logs live in ``tmp_path`` -- nothing here touches the committed ones.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from config import file_sha256, load_config, signals_key
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import e2e
from eval.scripts import e2e_labels as labels
from eval.scripts import review_signals as rs
from eval.scripts.draft_items import write_json
from eval.scripts.signals import sha256_text
from eval.scripts.split import meta_path

#: (id, bucket, exit, answer, the judge's reply). Two answered rows, two that never answered.
ROWS = [
    ("v1-0001", "fact", "answer", "CS 374 is 3 hours. [2026-sp:CS-374]", "correct. it matches"),
    ("v1-0002", "multihop", "answer", "Take MATH 213. [2026-sp:MATH-213]", "correct, sort of"),
    ("v1-0003", "unanswerable", "refuse_handoff", "I could not answer that.", None),
    ("v1-0004", "ambiguous", "out_of_scope", "That is outside the catalog.", None),
]  # fmt: skip

BY_ANSWER = {row[3]: row for row in ROWS}
ANSWERED = [row for row in ROWS if row[2] == "answer"]

KNOWN = frozenset(f"2026-sp:{code}" for code in ("CS-374", "CS-446", "MATH-213"))

#: What each bucket's rows must carry, per eval/schema.py's RULES.
GOLD = {
    "fact": ["2026-sp:CS-374"],
    "multihop": ["2026-sp:MATH-213", "2026-sp:CS-374"],
    "unanswerable": [],
    "ambiguous": [],
}


class FakeChat:
    """One judge reply per prompt, chosen by the answer quoted in it."""

    model = "fake-judge"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, temperature=None):
        self.prompts.append(prompt)
        row = next(row for answer, row in BY_ANSWER.items() if answer in prompt)
        return row[4]


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
            gold_answer=f"gold for {id_}",
            gold_doc_ids=GOLD[bucket],
            should_abstain=RULES[bucket].should_abstain,
            split="calib",
        )
        for id_, bucket, *_ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


@pytest.fixture
def runs_dir(cfg, tmp_path: Path) -> Path:
    """The run file #103 would have written, with two answered rows and two refusals."""
    path = e2e.run_path("calib", cfg, tmp_path / "runs")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            **dict.fromkeys(e2e.KEYS),
            "id": id_,
            "bucket": bucket,
            "split": "calib",
            "question": f"question {id_}?",
            "intent": "course_fact",
            "exit": exit_,
            "answer": answer,
            "evidence_doc_ids": ["2026-sp:CS-374"],
            "top_score": 20.0,
            "gate_ok": True,
        }
        for id_, bucket, exit_, answer, _ in ROWS
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path.parent


@pytest.fixture
def logs(tmp_path: Path) -> labels.Logs:
    return labels.Logs(prior=tmp_path / "prior_log.jsonl", own=tmp_path / "e2e_log.jsonl")


def prior_verdict(logs: labels.Logs, id_: str, answer: str, correct: bool = True) -> None:
    """One verdict in #88's log, on an answer text -- the carry-over this pass must honour."""
    entry = {
        "id": id_,
        "split": "calib",
        "signals_key": "40070d8d",
        "answer_sha256": sha256_text(answer),
        "judge_correct": True,
        "correct": correct,
        "action": "accepted_judge",
        "reviewer": "eve",
        "note": None,
        "ts": "2026-09-28T12:00:00.000000Z",
    }
    logs.prior.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def judge(cfg, dataset, runs_dir, chat=None):
    out = StringIO()
    chat = chat or FakeChat()
    path = labels.judge_answers(cfg, "calib", out, dataset, runs_dir, KNOWN, chat)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return lines, meta, out.getvalue(), chat


def run_main(args, dataset, runs_dir, logs, keys=(), chat=None):
    out = StringIO()
    typed = iter(keys)
    code = labels.main(
        ["--split", "calib", *args, "--dataset", str(dataset)],
        out=out,
        input_fn=lambda _: next(typed),
        runs_dir=runs_dir,
        logs=logs,
        known=KNOWN,
        chat=chat,
    )
    return code, out.getvalue()


# --- the judge ------------------------------------------------------------------------------------


def test_only_the_answered_rows_are_judged(cfg, dataset, runs_dir):
    lines, meta, printed, chat = judge(cfg, dataset, runs_dir)
    assert [line["id"] for line in lines] == ["v1-0001", "v1-0002"]
    assert meta["n_lines"] == 2
    assert len(chat.prompts) == 2
    assert "I could not answer that." not in "".join(chat.prompts)  # a fixed exit message
    assert "judging 2 answered rows" in printed


def test_a_judge_reply_that_is_not_a_verdict_is_null_and_kept(cfg, dataset, runs_dir):
    lines, meta, _, _ = judge(cfg, dataset, runs_dir)
    assert lines[0]["judge_correct"] is True and lines[0]["rationale"] == "it matches"
    # "correct, sort of" is a hedge, and #88's parser does not guess at one.
    assert lines[1]["judge_correct"] is None and lines[1]["judge_raw"] == "correct, sort of"
    assert meta["by_bucket"] == {"fact": {"correct": 1}, "multihop": {"unparsed": 1}}


def test_the_judge_file_names_the_run_it_graded_and_says_it_is_not_the_label(
    cfg, dataset, runs_dir
):
    _, meta, _, _ = judge(cfg, dataset, runs_dir)
    run = e2e.run_path("calib", cfg, runs_dir)
    assert meta["e2e_run"].endswith(run.name)
    assert meta["e2e_run_sha256"] == file_sha256(run)
    assert meta["config_hash"] == cfg.hash
    assert meta["model"] == "fake-judge"
    assert "only from a person's verdict" in meta["note"]
    assert labels.judge_path("test", cfg, runs_dir).name == f"e2e_judge_test_{cfg.hash}.jsonl"


def test_the_judge_grades_the_answer_against_the_gold_of_its_own_bucket(cfg, dataset, runs_dir):
    _, _, _, chat = judge(cfg, dataset, runs_dir)
    prompt = chat.prompts[1]
    assert "gold for v1-0002" in prompt and "question v1-0002?" in prompt
    assert "needs more than one catalog record" in prompt  # the multihop prompt, not the fact one


def test_labelling_before_the_judge_has_run_is_refused(cfg, dataset, runs_dir, logs):
    code, _ = run_main(["--stats"], dataset, runs_dir, logs)
    assert code == 1


def test_a_run_nobody_has_made_is_refused_by_name(cfg, dataset, tmp_path, logs):
    code, _ = run_main(["--stats"], dataset, tmp_path / "empty", logs)
    assert code == 1


# --- the human pass -------------------------------------------------------------------------------


def test_a_verdict_carries_over_on_the_answer_text_and_is_not_asked_again(
    cfg, dataset, runs_dir, logs
):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", ANSWERED[0][3])
    code, printed = run_main(["--review", "--reviewer", "eve"], dataset, runs_dir, logs, keys=["y"])
    assert code == 0
    assert "1 of 2 answered rows reviewed (1 carried over from another run)" in printed
    assert "v1-0001" not in printed  # already labelled, on this very answer
    entry = json.loads(logs.own.read_text(encoding="utf-8").strip())
    assert entry["id"] == "v1-0002" and entry["correct"] is True
    assert entry["action"] == "no_judge_verdict"  # the judge could not be read on that row
    assert entry["run"].endswith(e2e.run_path("calib", cfg, runs_dir).name)
    assert entry["run_key"] == cfg.hash and entry["signals_key"] == signals_key(cfg)
    assert tuple(entry) == labels.LOG_KEYS


def test_a_verdict_for_an_answer_that_has_changed_counts_as_unreviewed(
    cfg, dataset, runs_dir, logs
):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", "an answer the system no longer gives")
    code, printed = run_main(["--stats"], dataset, runs_dir, logs)
    assert code == 0
    assert "1 verdict(s) are for an answer that has since changed" in printed
    assert "0 of 2 answered rows reviewed" in printed


def test_this_pass_appends_to_its_own_log_and_leaves_88s_alone(cfg, dataset, runs_dir, logs):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", ANSWERED[0][3])
    before = logs.prior.read_text(encoding="utf-8")
    run_main(["--review", "--reviewer", "eve"], dataset, runs_dir, logs, keys=["n"])
    assert logs.prior.read_text(encoding="utf-8") == before
    assert len(logs.own.read_text(encoding="utf-8").splitlines()) == 1


def test_exactly_one_mode_is_required(cfg, dataset, runs_dir, logs):
    code, _ = run_main([], dataset, runs_dir, logs)
    assert code == 1
    code, _ = run_main(["--stats", "--write-csv"], dataset, runs_dir, logs)
    assert code == 1


# --- the CSV --------------------------------------------------------------------------------------


def test_the_csv_is_refused_while_any_answered_row_has_no_human_verdict(
    cfg, dataset, runs_dir, logs, capsys
):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", ANSWERED[0][3])
    code, _ = run_main(["--write-csv"], dataset, runs_dir, logs)
    assert code == 1
    error = capsys.readouterr().err
    assert "1 of 2 answered rows have no human verdict (v1-0002)" in error
    assert "comes only from the review log" in error
    assert not labels.csv_path("calib", cfg, runs_dir).exists()


def test_the_csv_holds_what_the_gate_saw_and_the_verdict_on_it(cfg, dataset, runs_dir, logs):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", ANSWERED[0][3])
    run_main(["--review", "--reviewer", "ada"], dataset, runs_dir, logs, keys=["n"])
    code, printed = run_main(["--write-csv"], dataset, runs_dir, logs)
    assert code == 0
    path = labels.csv_path("calib", cfg, runs_dir)
    assert path.name == f"e2e_calib_{cfg.hash}.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == ",".join(labels.CSV_COLUMNS)
    assert rows[1] == "v1-0001,fact,calib,course_fact,answer,20.0,1,1,1,eve"
    assert rows[2] == "v1-0002,multihop,calib,course_fact,answer,20.0,1,,0,ada"  # judge unparsed
    assert "wrote" in printed and "2 answered rows" in printed


def test_the_csv_sidecar_counts_the_verdicts_that_came_from_another_run(
    cfg, dataset, runs_dir, logs
):
    judge(cfg, dataset, runs_dir)
    prior_verdict(logs, "v1-0001", ANSWERED[0][3])
    run_main(["--review", "--reviewer", "ada"], dataset, runs_dir, logs, keys=["n"])
    run_main(["--write-csv"], dataset, runs_dir, logs)
    path = labels.csv_path("calib", cfg, runs_dir)
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["n_answered_rows"] == 2
    assert meta["verdicts_carried_over"] == 1 and meta["carried_over_ids"] == ["v1-0001"]
    assert meta["reviewers"] == {"ada": 1, "eve": 1}
    assert meta["sha256"] == file_sha256(path)
    assert meta["e2e_run_sha256"] == file_sha256(e2e.run_path("calib", cfg, runs_dir))
    assert set(meta["review_logs"]) == {rs.rel_to_repo(logs.prior), rs.rel_to_repo(logs.own)}
    assert meta["by_bucket"]["fact"] == {
        "rows": 1,
        "reviewed": 1,
        "accepted_judge": 1,
        "overturned": 0,
        "no_judge_verdict": 0,
    }
    assert meta["judge_x_human"]["unparsed"] == {"correct": 0, "incorrect": 1}
