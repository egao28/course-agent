"""Tests for eval/scripts/judge.py and eval/scripts/review_signals.py: #88's labels.

The signals run is written straight into ``tmp_path`` in #87's line shape, so nothing here runs the
agent. The judge is a fake chat that answers by bucket. The reviewer is a list of typed answers fed
through ``input_fn``, so a sitting is a transcript. Nothing touches ``eval/runs`` or ``v1.jsonl``.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from config import Config, correctness_key, file_sha256, load_config, signals_key
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import judge as jg
from eval.scripts import review_signals as rs
from eval.scripts.draft_items import write_json
from eval.scripts.signals import run_path, sha256_text
from eval.scripts.split import meta_path

KNOWN = frozenset(f"2026-sp:{code}" for code in ("CS-374", "CS-446", "MATH-213"))

#: (id, bucket, split, question, gold, answer, exit, the fake judge's reply).
ROWS = [
    ("v1-0001", "fact", "calib", "how many hours is cs 374?", "CS 374 is 4 hours.",
     "CS 374 is 4 hours [2026-sp:CS-374].", "answer", "correct\nSame hours as the reference."),
    ("v1-0002", "multihop", "calib", "what comes before cs 446?", "MATH 213, then CS 374.",
     "CS 374 [2026-sp:CS-374].", "answer", "Incorrect.\nStops after the first hop."),
    ("v1-0003", "unanswerable", "calib", "how many credits is CS 999?",
     "This catalog has no CS 999.", "CS 999 is 3 hours.", "handoff", "I think it is fine"),
    ("v1-0004", "ambiguous", "calib", "prereqs for the capstone?", "Which capstone do you mean?",
     "Which capstone course do you mean?", "answer", "CORRECT\nIt asks which one."),
    # The other half: a calib judge run must not read it.
    ("v1-0005", "fact", "test", "what is math 213?", "Discrete math.", "Discrete math.",
     "answer", "correct\nok"),
]  # fmt: skip

GOLD_IDS = {"fact": ["2026-sp:CS-374"], "multihop": ["2026-sp:MATH-213", "2026-sp:CS-374"]}


class FakeJudge:
    """Replies by which question is in the prompt, and keeps every prompt it was sent."""

    model = "fake-judge"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, temperature=None):
        self.prompts.append(prompt)
        return next(row[7] for row in ROWS if row[3] in prompt)


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
            gold_answer=gold,
            gold_doc_ids=GOLD_IDS.get(bucket, []),
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, question, gold, *_ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


def signals_line(row: tuple, n: int) -> dict:
    id_, bucket, split, question, _, answer, exit_, _ = row
    tool = n % 2 == 0
    return {
        "id": id_,
        "bucket": bucket,
        "split": split,
        "question": question,
        "intent": "course_fact",
        "exit": exit_,
        "answer": answer,
        "citations_unknown": ["x"] if exit_ != "answer" else None,
        "gate_ok": True,
        "retrieval_score": None if tool else 12.5,
        "retrieval_source": "course_tool" if tool else "bm25",
        "dense_top1": None if tool else 0.5,
        "verbalized": 100 if n != 2 else None,
        "verbalized_raw": "100",
        "samples": ["a", "b", "c"],
        "pair_agree": [True, True, True],
        "judge_raw": ["yes"] * 3,
        "self_consistency": 1.0 if n != 3 else 1 / 3,
    }


@pytest.fixture
def runs(tmp_path: Path, cfg: Config) -> Path:
    """The calib signals run, four lines, where #87 would have put it."""
    directory = tmp_path / "runs"
    directory.mkdir()
    lines = [signals_line(row, n) for n, row in enumerate(ROWS) if row[2] == "calib"]
    path = run_path("calib", cfg, directory)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return directory


def judge(cfg, dataset, runs, chat=None):
    out = StringIO()
    chat = chat or FakeJudge()
    path = jg.judge_split(cfg, "calib", out, dataset, runs, known=KNOWN, chat=chat)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return path, lines, out.getvalue(), chat


def review(dataset, runs, log, answers, *args):
    out = StringIO()
    feed = iter(answers)

    def input_fn(_: str) -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    argv = ["--split", "calib", "--dataset", str(dataset), *args]
    code = rs.main(argv, out, input_fn, runs, log, KNOWN)
    return code, out.getvalue()


# --- the judge ------------------------------------------------------------------------------------


def test_the_judge_labels_every_row_of_the_run_and_only_that_half(cfg, dataset, runs):
    path, lines, printed, chat = judge(cfg, dataset, runs)
    assert path.name == f"judge_calib_{correctness_key(cfg)}.jsonl"
    assert [line["id"] for line in lines] == ["v1-0001", "v1-0002", "v1-0003", "v1-0004"]
    assert [line["judge_correct"] for line in lines] == [True, False, None, True]
    assert lines[1]["rationale"] == "Stops after the first hop."
    assert lines[2]["judge_raw"] == "I think it is fine"  # unparsed: null, and the reply kept
    assert lines[2]["exit"] == "handoff"
    assert len(chat.prompts) == 4
    assert not any("what is math 213?" in prompt for prompt in chat.prompts)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["signals_key"] == signals_key(cfg)
    assert meta["by_bucket"]["unanswerable"] == {"unparsed": 1}
    assert "not the label" in meta["note"]
    assert "Not labels" in printed


def test_each_bucket_gets_its_own_prompt_with_the_gold_filled_in(cfg, dataset, runs):
    _, _, _, chat = judge(cfg, dataset, runs)
    by_question = {row[3]: prompt for row in ROWS for prompt in chat.prompts if row[3] in prompt}
    assert "every hop" in by_question["what comes before cs 446?"]
    assert "declines" in by_question["how many credits is CS 999?"]
    assert "asks the student to clarify" in by_question["prereqs for the capstone?"]
    assert "MATH 213, then CS 374." in by_question["what comes before cs 446?"]


@pytest.mark.parametrize(
    ("reply", "verdict", "rationale"),
    [
        ("correct\nbecause", True, "because"),
        ("Incorrect.\nwrong hop", False, "wrong hop"),
        ("  CORRECT  ", True, None),
        ("Incorrect. Misses the second hop.", False, "Misses the second hop."),
        ("correct: same hours\nand cited", True, "same hours and cited"),
        ("correct, mostly\nhmm", None, "hmm"),
        ("correctly answered", None, None),
        ("yes", None, None),
        ("", None, None),
    ],
)
def test_parsing_takes_the_first_line_or_nothing(reply, verdict, rationale):
    assert jg.parse_reply(reply) == jg.Parsed(verdict, rationale)


def test_a_missing_half_is_refused_by_name(cfg, dataset, tmp_path):
    with pytest.raises(ValueError, match="does not exist: run `python -m eval.scripts.signals"):
        jg.judge_split(cfg, "test", StringIO(), dataset, tmp_path, known=KNOWN, chat=FakeJudge())


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda p: {k: v for k, v in p.items() if k != "ambiguous"}, "missing ambiguous"),
        (lambda p: {**p, "other": p["fact"]}, "unknown other"),
        (lambda p: {**p, "fact": "no placeholders"}, "judge_prompts.fact must use exactly"),
    ],
)
def test_a_prompt_table_that_cannot_grade_every_bucket_is_refused(cfg, edit, message):
    data = cfg.to_dict()
    data["correctness"] = {"judge_prompts": edit(data["correctness"]["judge_prompts"])}
    with pytest.raises(ValueError, match=message):
        jg.prompts(Config(data))


def test_the_correctness_section_moves_neither_cfg_hash_nor_the_signals_key(cfg):
    # The run it grades is found by signals_key, and cannot be redone: a random draw.
    data = cfg.to_dict()
    prompts = data["correctness"]["judge_prompts"]
    edited = {**data, "correctness": {"judge_prompts": {**prompts, "fact": prompts["fact"] + "!"}}}
    assert Config(edited).hash == cfg.hash == "c73b197d"
    assert signals_key(Config(edited)) == signals_key(cfg)
    assert correctness_key(Config(edited)) != correctness_key(cfg)
    assert Config(edited).full_hash != cfg.full_hash


# --- the human pass -------------------------------------------------------------------------------


def log_lines(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_nothing_is_reviewed_without_a_reviewer_name(cfg, dataset, runs, tmp_path, capsys):
    judge(cfg, dataset, runs)
    code, _ = review(dataset, runs, tmp_path / "log.jsonl", ["a"])
    assert code == 1
    assert "--reviewer NAME is required" in capsys.readouterr().err
    assert not (tmp_path / "log.jsonl").exists()


def test_the_review_needs_the_judge_run_first(cfg, dataset, runs, tmp_path, capsys):
    code, _ = review(dataset, runs, tmp_path / "log.jsonl", [], "--reviewer", "egao")
    assert code == 1
    assert "python -m eval.scripts.judge --split calib" in capsys.readouterr().err


def test_a_sitting_records_each_verdict_with_who_when_and_what_the_judge_said(
    cfg, dataset, runs, tmp_path, monkeypatch
):
    judge(cfg, dataset, runs)
    monkeypatch.setattr(rs, "now", lambda: "2026-09-28T00:00:00.000000Z")
    log = tmp_path / "log.jsonl"
    # Queue order interleaves buckets: fact, multihop, unanswerable, ambiguous.
    code, printed = review(dataset, runs, log, ["a", "y", "", "wrong hop is fine", "q"],
                           "--reviewer", "egao")  # fmt: skip
    assert code == 0
    entries = log_lines(log)
    assert [(e["id"], e["correct"], e["action"]) for e in entries] == [
        ("v1-0001", True, "accepted_judge"),
        ("v1-0002", True, "overturned"),
    ]
    assert entries[1]["note"] == "wrong hop is fine" and entries[1]["judge_correct"] is False
    assert all(e["reviewer"] == "egao" and e["signals_key"] == signals_key(cfg) for e in entries)
    assert entries[0]["answer_sha256"] == sha256_text("CS 374 is 4 hours [2026-sp:CS-374].")
    assert "an overturn needs a reason" in printed
    assert "judge: incorrect -- Stops after the first hop." in printed
    assert "2 of 4 reviewed" in printed


def test_the_review_resumes_where_it_stopped(cfg, dataset, runs, tmp_path):
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a"], "--reviewer", "egao")
    _, printed = review(dataset, runs, log, ["n", "q"], "--reviewer", "egao")
    assert "v1-0001" not in printed.split("rows reviewed")[1]
    assert "[1/3] v1-0002" in printed
    # A row the judge could not read offers no "agree", and takes y/n with no reason asked.
    _, printed = review(dataset, runs, log, ["a", "n"], "--reviewer", "egao")
    assert "[1/2] v1-0003" in printed and "a, " not in printed.split("v1-0003")[1][:400]
    assert "y, n or q." in printed
    assert log_lines(log)[-1]["action"] == "no_judge_verdict"
    assert "the citation check discarded this draft" in printed


def test_a_partial_log_refuses_the_csv(cfg, dataset, runs, tmp_path, capsys):
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a", "a"], "--reviewer", "egao")
    code, _ = review(dataset, runs, log, [], "--write-csv")
    assert code == 1
    err = capsys.readouterr().err
    assert "refusing to write" in err and "2 of 4 rows have no human verdict" in err
    assert "v1-0003, v1-0004" in err
    assert not rs.csv_path("calib", cfg, runs).exists()


def test_a_full_log_writes_one_csv_row_per_signals_line(cfg, dataset, runs, tmp_path):
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a", "a", "n", "a"], "--reviewer", "egao")
    code, printed = review(dataset, runs, log, [], "--write-csv")
    assert code == 0
    path = rs.csv_path("calib", cfg, runs)
    assert path.name == f"signals_calib_{signals_key(cfg)}.csv"
    with path.open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    signals = run_path("calib", cfg, runs).read_text(encoding="utf-8").splitlines()
    assert len(table) == len(signals) == 4
    assert list(table[0]) == list(rs.CSV_COLUMNS)
    assert [r["correct"] for r in table] == ["1", "0", "0", "1"]
    assert [r["judge_correct"] for r in table] == ["1", "0", "", "1"]
    assert table[0]["retrieval_score"] == "" and table[1]["retrieval_score"] == "12.5"
    assert {r["reviewer"] for r in table} == {"egao"}
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["n_rows"] == 4 and meta["review_log_sha256"] == file_sha256(log)
    assert meta["judge_x_human"]["unparsed"] == {"correct": 0, "incorrect": 1}
    assert meta["by_bucket"]["unanswerable"]["no_judge_verdict"] == 1
    assert "overturned" in printed


def test_every_correct_in_the_csv_comes_from_the_log(cfg, dataset, runs, tmp_path):
    # The judge said correct on v1-0004; the person said no. The CSV follows the person.
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a", "a", "y", "n", "judge was wrong"], "--reviewer", "egao")
    review(dataset, runs, log, [], "--write-csv")
    with rs.csv_path("calib", cfg, runs).open(encoding="utf-8", newline="") as f:
        table = {r["id"]: r for r in csv.DictReader(f)}
    assert table["v1-0004"]["judge_correct"] == "1" and table["v1-0004"]["correct"] == "0"
    _, printed = review(dataset, runs, log, [], "--stats")
    assert "ambiguous" in printed and "overturned" in printed


def test_the_latest_verdict_wins_and_a_changed_answer_makes_a_verdict_stale(
    cfg, dataset, runs, tmp_path
):
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a", "a", "n", "a"], "--reviewer", "egao")
    review(dataset, runs, log, ["n", "second look", "q"], "--reviewer", "egao", "--redo")
    rows = rs.load_rows("calib", cfg, dataset, runs, KNOWN)
    current = rs.status(rows, rs.read_log(log), "calib")
    assert current.verdicts["v1-0001"]["correct"] is False
    # Rewrite one answer in place: its verdict no longer applies, and the judge's neither.
    path = run_path("calib", cfg, runs)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[0]["answer"] = "something else"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    with pytest.raises(ValueError, match="the judge graded a different answer"):
        rs.load_rows("calib", cfg, dataset, runs, KNOWN)


def test_a_log_entry_without_a_person_is_refused(tmp_path):
    log = tmp_path / "log.jsonl"
    entry = dict.fromkeys(rs.LOG_KEYS) | {"correct": True, "reviewer": ""}
    log.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="every entry is a person's verdict"):
        rs.read_log(log)


def test_a_verdict_follows_the_answer_text_across_signals_runs(cfg, dataset, runs, tmp_path):
    # A rerun of #87 with a new signals key: one answer changed, three came back word for word.
    judge(cfg, dataset, runs)
    log = tmp_path / "log.jsonl"
    review(dataset, runs, log, ["a", "a", "n", "a"], "--reviewer", "egao")
    entries = log_lines(log)
    for entry in entries:
        entry["signals_key"] = "0ldrun00"
    log.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    path = run_path("calib", cfg, runs)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[1]["answer"] = "MATH 213 [2026-sp:MATH-213], then CS 374 [2026-sp:CS-374]."
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    judge(cfg, dataset, runs)
    rows = rs.load_rows("calib", cfg, dataset, runs, KNOWN)
    current = rs.status(rows, rs.read_log(log), "calib")
    assert sorted(current.verdicts) == ["v1-0001", "v1-0003", "v1-0004"]
    assert current.stale == ["v1-0002"]
    code, printed = review(dataset, runs, log, [], "--write-csv")
    assert code == 1
    _, printed = review(dataset, runs, log, ["y", "both hops now"], "--reviewer", "egao")
    assert "[1/1] v1-0002" in printed
    assert review(dataset, runs, log, [], "--write-csv")[0] == 0
    csv_file = rs.csv_path("calib", cfg, runs)
    meta = json.loads(csv_file.with_name(csv_file.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["verdicts_carried_over"] == 3
