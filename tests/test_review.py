"""Tests for eval/scripts/review.py: the queue, the log, the screen, the sitting, the promotion.

The thing under test is a tool a person drives, so most of these drive it: a scripted list of
keystrokes goes in, a transcript and a log file come out, and the assertions are about what the
reviewer saw and what the file recorded. A review CLI whose tests only called its helpers would
pass with a prompt that never appeared and a log that was never flushed.

Five groups. The first three are over a synthetic three-row corpus built in ``tmp_path``, because
what they check is behaviour -- an edit that is refused, a log entry that replaces an earlier one
-- and a fixture keeps the expected strings readable. The fourth is the promotion, which is the
only place where a wrong answer is silent: a dataset is written either way and only its contents
say whether the log was replayed correctly. The fifth runs over the **committed** drafts, since
what #59 has to survive is the 50 real rows and their real chunk text, not three made-up ones.

Nothing here writes ``eval/datasets/review_log.jsonl``. Every test passes its own ``--log``: the
committed log is the record of a person's judgements, and a test suite that appended to it would
be forging them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_config
from eval.schema import ANNOTATED_FIELDS, EvalItem
from eval.scripts import build_multihop, build_unanswerable, review
from eval.scripts.draft_items import PROVENANCE_KEYS as DRAFTED_KEYS
from eval.validate import corpus_doc_ids

# --- a corpus and some drafts ---------------------------------------------------------------------

#: Three courses, enough for a fact row, a two-hop row and a dangling id.
CHUNKS = [
    {
        "doc_id": "2026-sp:CS-225",
        "text": "CS 225 — Data Structures\nCredits: 4\nPrerequisites: CS 173 or MATH 213.",
        "metadata": {"code": "CS 225"},
    },
    {
        "doc_id": "2026-sp:CS-374",
        "text": "CS 374 — Algorithms\nCredits: 4\nPrerequisites: One of CS 173, MATH 213; CS 225.",
        "metadata": {"code": "CS 374"},
    },
    {
        "doc_id": "2026-sp:MATH-213",
        "text": "MATH 213 — Discrete Mathematics\nCredits: 3\nPrerequisites: None listed.",
        "metadata": {"code": "MATH 213"},
    },
]

KNOWN = frozenset(chunk["doc_id"] for chunk in CHUNKS)
TEXTS = {chunk["doc_id"]: chunk["text"] for chunk in CHUNKS}


def drafted_provenance(**overrides: object) -> dict[str, object]:
    """A drafted row's provenance with every key #56 writes, so the reader accepts it."""
    provenance: dict[str, object] = dict.fromkeys(DRAFTED_KEYS, None)
    provenance.update({"flagged": False, "credit_question": False})
    provenance.update(overrides)
    return provenance


def row(
    item_id: str,
    bucket: str = "fact",
    question: str = "how many credits is CS 225?",
    answer: str = "CS 225 is 4 credit hours.",
    gold: tuple[str, ...] = ("2026-sp:CS-225",),
    **provenance: object,
) -> dict[str, object]:
    return {
        "id": item_id,
        "bucket": bucket,
        "question": question,
        "gold_answer": answer,
        "gold_doc_ids": list(gold),
        "should_abstain": bucket in {"unanswerable", "ambiguous"},
        "provenance": drafted_provenance(**provenance),
    }


def item(*args: object, **kwargs: object) -> EvalItem:
    """The six dataset fields of :func:`row`, without the provenance the schema forbids."""
    fields = {k: v for k, v in row(*args, **kwargs).items() if k != "provenance"}
    return EvalItem.model_validate(fields)


def write_drafts(directory: Path, name: str, rows: list[dict[str, object]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


@pytest.fixture
def drafts(tmp_path: Path) -> Path:
    """A drafts directory with one fact row, one multihop row and one unanswerable row."""
    directory = tmp_path / "drafts"
    write_drafts(
        directory,
        "fact_abc12345.jsonl",
        [
            row("v1-0001"),
            row(
                "v1-1001",
                bucket="multihop",
                question="what does CS 374 need, and what does that need?",
                answer="CS 374 needs CS 225, and CS 225 needs CS 173 or MATH 213.",
                gold=("2026-sp:CS-374", "2026-sp:CS-225"),
            ),
        ],
    )
    write_drafts(
        directory,
        "unanswerable_abc12345.jsonl",
        [
            {
                "id": "v1-2001",
                "bucket": "unanswerable",
                "question": "who teaches CS 225?",
                "gold_answer": "The catalog does not carry instructors.",
                "gold_doc_ids": [],
                "should_abstain": True,
                "provenance": {
                    key: None
                    for key in review.PROVENANCE_SETS[1][1]
                },
            }
        ],
    )
    return directory


@pytest.fixture
def session(drafts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the CLI over the fixture drafts with a scripted stdin; return (exit code, transcript)."""
    log = tmp_path / "review_log.jsonl"

    def run(keys: list[str], *args: str) -> tuple[int, str]:
        typed = iter(keys)
        printed: list[str] = []

        def fake_input(_prompt: str) -> str:
            """Exhausted input is a closed stdin, which is what a piped sitting ends with."""
            try:
                return next(typed)
            except StopIteration:
                raise EOFError from None

        class Out:
            def write(self, text: str) -> int:
                printed.append(text)
                return len(text)

            def flush(self) -> None:
                return None

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr("sys.stdout", Out())
        monkeypatch.setattr("sys.stderr", Out())
        monkeypatch.setattr(review, "corpus_doc_ids", lambda *a, **k: KNOWN)
        monkeypatch.setattr(review, "corpus_text", lambda *a, **k: TEXTS)
        monkeypatch.setattr(review, "credit_ranges", lambda *a, **k: None)
        code = review.main(["--drafts-dir", str(drafts), "--log", str(log), *args])
        return code, "".join(printed)

    run.log = log  # type: ignore[attr-defined]
    return run


# --- the queue ------------------------------------------------------------------------------------


def test_the_standin_file_is_not_reviewed(drafts: Path):
    """#56 writes canned rows to .standin.jsonl so nothing downstream reads them."""
    write_drafts(drafts, "fact_abc12345.standin.jsonl", [row("v1-0009")])
    names = [path.name for path in review.find_drafts(drafts)]
    assert names == ["fact_abc12345.jsonl", "unanswerable_abc12345.jsonl"]


def test_a_missing_drafts_directory_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        review.find_drafts(tmp_path / "nowhere")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("multihop_graph_7980f15f.jsonl", "build_multihop"),
        ("unanswerable_7980f15f.jsonl", "build_unanswerable"),
        ("multihop_7980f15f.jsonl", "draft_items"),
        ("fact_7980f15f.jsonl", "draft_items"),
        ("ambiguous_7980f15f.jsonl", "draft_items"),
    ],
)
def test_each_draft_file_is_held_to_its_own_builders_provenance(name: str, expected: str):
    """Three builders record three different things; a reader that took any of them takes none."""
    from eval.scripts import build_multihop, build_unanswerable, draft_items

    sets = {
        "build_multihop": tuple(build_multihop.PROVENANCE_KEYS),
        "build_unanswerable": tuple(build_unanswerable.PROVENANCE_KEYS),
        "draft_items": tuple(draft_items.PROVENANCE_KEYS),
    }
    assert review.provenance_keys(Path(name)) == sets[expected]


def test_a_graph_multihop_file_is_not_read_as_a_drafted_one():
    """The longest prefix wins, or #58's rows would be checked against #56's keys and fail."""
    graph = review.provenance_keys(Path("multihop_graph_x.jsonl"))
    drafted = review.provenance_keys(Path("multihop_x.jsonl"))
    assert graph != drafted


def test_the_queue_deals_one_row_from_each_bucket_in_turn():
    candidates = [
        review.Candidate(Path("f"), n, item(f"v1-{n:04d}"), {})
        for n in (1, 2, 3)
    ] + [
        review.Candidate(
            Path("u"),
            n,
            EvalItem.model_validate(
                {
                    "id": f"v1-{2000 + n:04d}",
                    "bucket": "unanswerable",
                    "question": f"q{n}",
                    "gold_answer": "no",
                    "gold_doc_ids": [],
                    "should_abstain": True,
                }
            ),
            {},
        )
        for n in (1, 2)
    ]
    ids = [c.id for c in review.interleave(candidates)]
    assert ids == ["v1-0001", "v1-2001", "v1-0002", "v1-2002", "v1-0003"]


def test_the_queue_is_a_function_of_the_drafts_alone(drafts: Path):
    """Two people reviewing the same files see the same order, so #60's κ is over one queue."""
    candidates, _ = review.load_candidates(review.find_drafts(drafts))
    first = [c.id for c in review.interleave(candidates)]
    second = [c.id for c in review.interleave(list(reversed(candidates)))]
    assert first == second == ["v1-0001", "v1-1001", "v1-2001"]


def test_an_id_in_two_draft_files_is_caught(drafts: Path, session):
    write_drafts(drafts, "ambiguous_abc12345.jsonl", [row("v1-0001")])
    code, _ = session(["q"])
    assert code == 1


def test_a_hand_edited_draft_line_stops_the_pass(drafts: Path, session):
    (drafts / "fact_abc12345.jsonl").write_text("{not json}\n", encoding="utf-8")
    code, _ = session(["q"])
    assert code == 1


# --- the log --------------------------------------------------------------------------------------


def test_an_accept_logs_the_seven_keys_with_the_edit_fields_null(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    review.append_verdict(log, review.Verdict("v1-0001", "accept", None, "T"))
    entry = json.loads(log.read_text(encoding="utf-8").strip())
    assert list(entry) == list(review.LOG_KEYS)
    assert entry["field_changed"] is entry["before"] is entry["after"] is None


def test_an_edit_of_three_fields_is_three_entries_sharing_a_timestamp(tmp_path: Path):
    """#59 fixes the log at seven scalar keys; the shared ts is what reassembles one verdict."""
    log = tmp_path / "log.jsonl"
    edits = (
        review.Edit("question", "old q", "new q"),
        review.Edit("gold_answer", "old a", "new a"),
        review.Edit("bucket", "fact", "multihop"),
    )
    review.append_verdict(log, review.Verdict("v1-0001", "edit", "wording", "T", edits))
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["field_changed"] for e in entries] == ["question", "gold_answer", "bucket"]
    assert {e["ts"] for e in entries} == {"T"}
    assert {e["reason"] for e in entries} == {"wording"}
    assert review.replay(entries)["v1-0001"].edits == edits


def test_a_later_verdict_replaces_an_earlier_one_outright(tmp_path: Path):
    """One row, one verdict, however many sittings -- the log is append-only, not a patch set."""
    log = tmp_path / "log.jsonl"
    review.append_verdict(
        log,
        review.Verdict("v1-0001", "edit", "wording", "T1", (review.Edit("question", "a", "b"),)),
    )
    review.append_verdict(log, review.Verdict("v1-0001", "delete", "range course", "T2"))
    verdict = review.read_verdicts(log)["v1-0001"]
    assert (verdict.verdict, verdict.reason, verdict.edits) == ("delete", "range course", ())


def test_a_log_entry_missing_a_key_is_an_error_and_names_it(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"id": "v1-0001", "verdict": "accept"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is missing reason, field_changed, before, after, ts"):
        review.read_entries(log)


def test_a_missing_log_is_an_empty_pass_not_an_error(tmp_path: Path):
    assert review.read_entries(tmp_path / "nothing.jsonl") == []


# --- what an edit is ------------------------------------------------------------------------------


def test_edits_are_recorded_against_the_draft_in_field_order():
    draft = item("v1-0001")
    edited = draft.model_copy(update={"gold_answer": "CS 225 is 4 hours.", "question": "credits?"})
    assert review.diff_edits(draft, edited) == (
        review.Edit("question", draft.question, "credits?"),
        review.Edit("gold_answer", draft.gold_answer, "CS 225 is 4 hours."),
    )


def test_a_draft_edited_after_it_was_reviewed_stops_the_replay():
    """The reviewer approved the old text; the new text has never been read by anyone."""
    draft = item("v1-0001", answer="something else")
    edits = (review.Edit("gold_answer", "CS 225 is 4 credit hours.", "CS 225 is 4 hours."),)
    with pytest.raises(ValueError, match="has to be re-reviewed"):
        review.apply_edits(draft, edits)


def test_the_log_may_not_edit_a_field_outside_the_five():
    draft = item("v1-0001")
    with pytest.raises(ValueError, match="'id', which is not one of"):
        review.apply_edits(draft, (review.Edit("id", "v1-0001", "v1-0002"),))


def test_a_row_with_no_verdict_is_the_draft():
    draft = item("v1-0001")
    candidate = review.Candidate(Path("f"), 1, draft, {})
    assert review.current(candidate, None) is draft


# --- the screen -----------------------------------------------------------------------------------


def rendered(candidate: review.Candidate, **kwargs: object) -> str:
    marks = review.flags(
        candidate, candidate.item, KNOWN, kwargs.get("ranges"), lambda t: t.lower().split()
    )
    return review.render_row(candidate, candidate.item, TEXTS, marks, 1, 1)


def test_every_cited_chunk_is_printed_in_full():
    """Without the chunk text on screen, check 2 does not happen -- it is approved by vibe."""
    candidate = review.Candidate(
        Path("f"),
        1,
        item(
            "v1-1001",
            bucket="multihop",
            gold=("2026-sp:CS-374", "2026-sp:CS-225"),
        ),
        {},
    )
    screen = rendered(candidate)
    for doc_id in ("2026-sp:CS-374", "2026-sp:CS-225"):
        assert doc_id in screen
        for line in TEXTS[doc_id].splitlines():
            assert line in screen


def test_the_three_checks_are_on_the_screen():
    screen = rendered(review.Candidate(Path("f"), 1, item("v1-0001"), {}))
    assert "gold_answer right" in screen
    assert "gold_doc_ids support it" in screen
    assert "GUIDELINES.md" in screen


def test_a_row_with_no_citations_says_what_there_is_to_check_instead():
    candidate = review.Candidate(
        Path("u"),
        1,
        item("v1-2001", bucket="unanswerable", answer="No instructors here.", gold=()),
        {},
    )
    screen = rendered(candidate)
    assert "claim of absence" in screen
    assert "open the catalog, not the agent" in screen


def test_a_code_the_answer_names_that_no_cited_chunk_carries_is_marked():
    """Check 2, on screen: a citation that is about the right topic is still the wrong id."""
    candidate = review.Candidate(
        Path("f"), 1, item("v1-0001", answer="CS 225 requires STAT 400."), {}
    )
    screen = rendered(candidate)
    assert "STAT 400 (in no cited chunk)" in screen
    assert "CS 225 (in no cited chunk)" not in screen


def test_a_dangling_gold_id_is_flagged_where_the_reviewer_can_still_fix_it():
    candidate = review.Candidate(Path("f"), 1, item("v1-0001", gold=("2026-sp:CS 225",)), {})
    screen = rendered(candidate)
    assert "not in the corpus" in screen
    assert "did you mean '2026-sp:CS-225'" in screen


def test_more_than_four_cited_records_is_flagged_as_a_list_question():
    gold = ("2026-sp:CS-225", "2026-sp:CS-374", "2026-sp:MATH-213")
    candidate = review.Candidate(
        Path("f"), 1, item("v1-1001", bucket="multihop", gold=gold * 2), {}
    )
    marks = review.flags(candidate, candidate.item, KNOWN, None, lambda t: t.split())
    assert any("past the cap of 4" in mark and "Which bucket" in mark for mark in marks)


def test_a_credit_question_names_the_rule_and_the_command_when_the_export_cannot_say():
    """#17 is on main, not in this stack, so the 1,415 variable-credit courses are unknowable."""
    candidate = review.Candidate(Path("f"), 1, item("v1-0001"), {})
    marks = review.flags(candidate, candidate.item, KNOWN, None, lambda t: t.lower().split())
    assert any("GUIDELINES rule 2" in mark and "creditsMin" in mark for mark in marks)


def test_a_credit_question_about_a_variable_credit_course_is_named_outright():
    candidate = review.Candidate(Path("f"), 1, item("v1-0001"), {})
    marks = review.flags(
        candidate, candidate.item, KNOWN, frozenset({"CS 225"}), lambda t: t.lower().split()
    )
    assert any("CS 225 has a credit range" in mark and "delete this row" in mark for mark in marks)


def test_a_credit_question_about_a_fixed_credit_course_is_not_flagged():
    candidate = review.Candidate(Path("f"), 1, item("v1-0001"), {})
    marks = review.flags(
        candidate, candidate.item, KNOWN, frozenset({"AAS 465"}), lambda t: t.lower().split()
    )
    assert marks == []


def test_the_top_decile_overlap_mark_reaches_the_reviewer():
    candidate = review.Candidate(Path("f"), 1, item("v1-0001"), {"flagged": True})
    marks = review.flags(candidate, candidate.item, KNOWN, None, lambda t: t.split())
    assert any("top decile of overlap" in mark for mark in marks)


def test_an_empty_credit_range_set_is_not_the_same_as_an_unknown_one(tmp_path: Path):
    """None is "cannot say"; an empty set is "none affected". Conflating them passes 1,415."""
    courses = tmp_path / "courses.json"
    courses.write_text(json.dumps([{"code": "CS 225", "credits": 4.0}]), encoding="utf-8")
    assert review.credit_ranges(courses) is None
    courses.write_text(
        json.dumps([{"code": "CS 225", "creditsMin": 4.0, "creditsMax": 4.0}]), encoding="utf-8"
    )
    assert review.credit_ranges(courses) == frozenset()


# --- the sitting ----------------------------------------------------------------------------------


def logged(log: Path) -> list[dict]:
    """The log as entries. A sitting that judged nothing writes no file, which is no entries."""
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_an_accept_is_written_before_the_next_row_is_shown(session):
    """Losing an hour of judgements to a closed terminal is how this task gets abandoned."""
    code, transcript = session(["a", "q"])
    assert code == 0
    entries = logged(session.log)
    assert [(e["id"], e["verdict"]) for e in entries] == [("v1-0001", "accept")]
    assert transcript.count("verdict?") == 2


def test_the_pass_resumes_by_id(session):
    session(["a", "a", "q"])
    _, transcript = session(["q"])
    assert "v1-0001" not in transcript
    assert "v1-1001" not in transcript
    assert "v1-2001" in transcript


def test_a_finished_pass_says_so_rather_than_showing_an_empty_queue(session):
    session(["a", "a", "a"])
    code, transcript = session(["q"])
    assert code == 0
    assert "every one of the 3 drafted rows has a verdict" in transcript


def test_redo_shows_a_judged_row_again_with_its_edits_applied(session):
    session(["e", "2", "CS 225 is 4 hours.", "done", "wording", "q"])
    _, transcript = session(["q"], "--redo")
    assert "CS 225 is 4 hours." in transcript
    assert "already judged edit (wording)" in transcript


def test_a_second_verdict_on_a_row_records_the_whole_diff_against_the_draft(session):
    """A re-review replaces the earlier verdict, so its edits must stand on the draft alone."""
    session(["e", "2", "CS 225 is 4 hours.", "done", "wording", "q"])
    session(["e", "1", "how many credits is CS 225 worth?", "done", "wording", "q"], "--redo")
    verdict = review.read_verdicts(session.log)["v1-0001"]
    assert {edit.field for edit in verdict.edits} == {"question", "gold_answer"}
    assert [edit.before for edit in verdict.edits] == [
        "how many credits is CS 225?",
        "CS 225 is 4 credit hours.",
    ]


def test_a_delete_will_not_go_through_without_a_reason(session):
    code, transcript = session(["d", "  ", "d", "range course", "q"])
    assert code == 0
    assert "a delete needs a reason" in transcript
    assert logged(session.log)[0]["reason"] == "range course"


def test_an_edit_reason_has_to_be_one_word_so_that_the_reasons_can_be_counted(session):
    _, transcript = session(["e", "2", "CS 225 is 4 hours.", "done", "it was badly worded", "q"])
    assert "one word, so the reasons can be counted" in transcript
    assert logged(session.log) == []


def test_an_edit_that_breaks_the_row_is_refused_and_the_reviewer_stays_in_the_editor(session):
    """Every check_row violation is fixable here; the way out of an unfixable row is delete."""
    _, transcript = session(
        ["e", "4", "unanswerable", "done", "5", "true", "3", "", "done", "relabel", "q"]
    )
    assert "one of the two is the mislabel" in transcript
    assert "fix those, or cancel and delete the row instead" in transcript
    verdict = review.read_verdicts(session.log)["v1-0001"]
    assert verdict.verdict == "edit"
    assert {edit.field: edit.after for edit in verdict.edits} == {
        "bucket": "unanswerable",
        "should_abstain": True,
        "gold_doc_ids": [],
    }


def test_cancelling_an_edit_leaves_the_row_unjudged_and_asks_again(session):
    session(["e", "2", "something", "cancel", "a", "q"])
    assert logged(session.log)[0]["verdict"] == "accept"
    assert logged(session.log)[0]["field_changed"] is None


def test_an_edit_that_changes_nothing_is_not_an_edit(session):
    _, transcript = session(["e", "done", "a", "q"])
    assert "nothing changed" in transcript
    assert logged(session.log)[0]["verdict"] == "accept"


def test_a_field_can_be_named_or_numbered_and_a_typo_is_not_a_verdict(session):
    _, transcript = session(
        ["e", "gold_answr", "gold_answer", "CS 225 is 4 hours.", "done", "w", "q"]
    )
    assert "no field 'gold_answr'" in transcript
    assert logged(session.log)[0]["field_changed"] == "gold_answer"


def test_a_value_the_field_cannot_hold_is_refused_with_the_options(session):
    _, transcript = session(["e", "4", "facts", "cancel", "q"])
    assert "bucket is one of fact, multihop, unanswerable, ambiguous" in transcript


def test_an_unknown_key_reprompts_rather_than_counting_as_a_verdict(session):
    _, transcript = session(["x", "a", "q"])
    assert "a, e, d or q." in transcript
    assert len(logged(session.log)) == 1


def test_a_closed_stdin_ends_the_sitting(session):
    """EOF is a quit, not a crash: a sitting driven by a pipe stops where the pipe does."""
    code, _ = session(["a"])
    assert code == 0
    assert len(logged(session.log)) == 1


def test_limit_stops_the_sitting_where_it_was_asked_to(session):
    code, transcript = session(["a", "a", "a"], "--limit", "2")
    assert code == 0
    assert "stopping at 2, as asked." in transcript
    assert len(logged(session.log)) == 2


# --- promotion ------------------------------------------------------------------------------------


def promoted(session, tmp_path: Path, *keys: str) -> tuple[int, str, Path]:
    out = tmp_path / "v1.jsonl"
    code, transcript = session(list(keys), "--promote", "--out", str(out))
    return code, transcript, out


def test_nothing_is_promoted_while_a_drafted_row_has_no_verdict(session, tmp_path: Path):
    """#59's first acceptance line: a partial dataset that looks whole is invisible to a reader."""
    session(["a", "q"])
    code, transcript, out = promoted(session, tmp_path)
    assert code == 1
    assert "2 drafted rows have no verdict (v1-1001, v1-2001)" in transcript
    assert not out.exists()


def test_the_dataset_is_the_drafts_replayed_through_the_log(session, tmp_path: Path):
    session(["a", "e", "2", "CS 374 needs CS 225.", "done", "wording", "d", "vague"])
    code, _, out = promoted(session, tmp_path)
    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["v1-0001", "v1-1001"]
    assert rows[1]["gold_answer"] == "CS 374 needs CS 225."


def test_a_deleted_row_leaves_the_log_and_nothing_else(session, tmp_path: Path):
    session(["d", "range course", "a", "a"])
    _, _, out = promoted(session, tmp_path)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert "v1-0001" not in [r["id"] for r in rows]
    assert review.read_verdicts(session.log)["v1-0001"].reason == "range course"


def test_the_dataset_carries_the_six_fields_sorted_by_id_and_no_trailing_whitespace(
    session, tmp_path: Path
):
    session(["a", "a", "a"])
    _, _, out = promoted(session, tmp_path)
    body = out.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert body.endswith("\n") and not body.endswith("\n\n")
    assert all(line == line.rstrip() for line in lines)
    assert [json.loads(line)["id"] for line in lines] == sorted(
        json.loads(line)["id"] for line in lines
    )
    # The six a reviewer works in, and no `split` column: promotion writes an unfrozen dataset,
    # and #61's freeze is the only thing that adds the seventh.
    assert all(list(json.loads(line)) == list(ANNOTATED_FIELDS) for line in lines)


def test_promotion_is_a_function_and_so_re_running_it_changes_nothing(session, tmp_path: Path):
    session(["a", "a", "a"])
    _, _, out = promoted(session, tmp_path)
    first = out.read_text(encoding="utf-8")
    promoted(session, tmp_path)
    assert out.read_text(encoding="utf-8") == first


def test_a_draft_edited_after_review_stops_the_promotion(session, drafts: Path, tmp_path: Path):
    session(["a", "a", "a"])
    path = drafts / "fact_abc12345.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["gold_answer"] = "CS 225 is 3 credit hours."
    write_drafts(drafts, "fact_abc12345.jsonl", rows)
    session(["e", "2", "CS 225 is 4 hours.", "done", "wording"], "--redo")
    rows[0]["gold_answer"] = "CS 225 is 5 credit hours."
    write_drafts(drafts, "fact_abc12345.jsonl", rows)
    code, transcript, out = promoted(session, tmp_path)
    assert code == 1
    assert "has to be re-reviewed" in transcript
    assert not out.exists()


def test_a_verdict_on_a_row_no_draft_holds_stops_the_promotion(
    session, drafts: Path, tmp_path: Path
):
    session(["a", "a", "a"])
    write_drafts(drafts, "fact_abc12345.jsonl", [row("v1-0001")])
    code, transcript, out = promoted(session, tmp_path)
    assert code == 1
    assert "the log judges v1-1001" in transcript
    assert not out.exists()


def test_a_short_bucket_is_a_warning_and_the_dataset_is_still_written(session, tmp_path: Path):
    """Deleting a bad item is what verification is for; a check that fails on 49 keeps a bad row."""
    session(["a", "a", "a"])
    code, transcript, out = promoted(session, tmp_path)
    assert code == 0
    assert "short of the planned 50" in transcript
    assert out.exists()


# --- what the pass has done -----------------------------------------------------------------------


def line_for(transcript: str, label: str) -> list[int]:
    """The counts on the stats row for ``label`` -- the table as numbers, not as a string."""
    for line in transcript.splitlines():
        if line.startswith(label):
            return [int(part) for part in line.split()[1:]]
    raise AssertionError(f"no {label!r} row in\n{transcript}")


def test_the_stats_table_counts_every_bucket_and_what_became_of_it(session):
    session(["a", "e", "2", "CS 374 needs CS 225.", "done", "wording", "d", "vague"])
    _, transcript = session([], "--stats")
    # drafted, accept, edit, delete, unreviewed
    assert line_for(transcript, "fact") == [1, 1, 0, 0, 0]
    assert line_for(transcript, "multihop") == [1, 0, 1, 0, 0]
    assert line_for(transcript, "unanswerable") == [1, 0, 0, 1, 0]
    assert line_for(transcript, "total") == [3, 1, 1, 1, 0]


def test_the_edit_reasons_are_counted_because_that_is_what_one_word_buys(session):
    session(
        ["e", "2", "a b c d.", "done", "wording", "e", "2", "e f g h.", "done", "wording", "q"]
    )
    _, transcript = session([], "--stats")
    assert "top edit reasons" in transcript
    assert "2  wording" in transcript


def test_an_unreviewed_bucket_is_counted_as_unreviewed_not_as_absent(session):
    _, transcript = session([], "--stats")
    assert line_for(transcript, "total") == [3, 0, 0, 0, 3]


def test_the_fatigue_line_reports_the_two_halves_rather_than_judging_them():
    judgements = [(f"T{n}", f"v1-{n:04d}", "delete") for n in range(4)]
    judgements += [(f"T{n}", f"v1-{n:04d}", "accept") for n in range(4, 8)]
    assert review.fatigue(judgements) == (
        "accept rate: 0% over the first 4 judgements, 100% over the last 4  <- read the late rows"
    )


def test_a_handful_of_judgements_is_not_a_fatigue_measurement():
    assert "too few judgements" in review.fatigue([("T", "v1-0001", "accept")])


def test_the_entries_of_one_edit_are_one_judgement():
    entries = [
        {"id": "v1-0001", "verdict": "edit", "reason": "w", "ts": "T", "field_changed": "question",
         "before": "a", "after": "b"},
        {"id": "v1-0001", "verdict": "edit", "reason": "w", "ts": "T", "field_changed": "bucket",
         "before": "fact", "after": "multihop"},
    ]
    assert review.events(entries) == [("T", "v1-0001", "edit")]


# --- over the committed drafts --------------------------------------------------------------------

#: What #57 and #58 actually committed, and what #56 will add. The tool has to survive these and
#: not three made-up rows: the real ones carry em dashes, four-id chains and empty gold id lists.
COMMITTED = review.find_drafts()

#: The two artifacts the pass produces. Neither is in the tree until a person has run it, so the
#: tests over them skip -- and hold them the moment they land, which is the point of writing them
#: now rather than after the file exists.
LOG = review.REVIEW_LOG_PATH
DATASET = review.DATASET_PATH


@pytest.fixture(scope="module")
def committed() -> list[review.Candidate]:
    candidates, violations = review.load_candidates(COMMITTED)
    assert violations == []
    return candidates


def test_every_committed_draft_file_is_read_against_its_own_provenance_set(committed):
    # Named off the live config rather than spelled out. #61 adds `eval:` to config.yaml, which
    # moves the hash and so the name of every artifact tagged with it; a test holding the old
    # spelling would pass over a rebuilt file and fail over a missing one for the wrong reason.
    cfg = load_config()
    assert {path.name for path in COMMITTED} >= {
        build_multihop.draft_path(build_multihop.STEM, cfg).name,
        build_unanswerable.draft_path(build_unanswerable.BUCKET, cfg).name,
    }
    assert len(committed) >= 50


def test_the_committed_queue_alternates_buckets(committed):
    queue = review.interleave(committed)
    buckets = [candidate.item.bucket for candidate in queue]
    assert len(set(buckets[:4])) == min(len(set(buckets)), 4)
    assert sorted(buckets) == sorted(c.item.bucket for c in committed)


def test_every_cited_id_in_the_committed_drafts_has_text_to_show(committed):
    """The screen is the pass; a gold id with no text behind it is a row nobody can check."""
    texts = review.corpus_text()
    missing = [
        (candidate.id, gold)
        for candidate in committed
        for gold in candidate.item.gold_doc_ids
        if gold not in texts
    ]
    assert missing == []


def test_no_committed_row_reaches_the_reviewer_with_a_validator_error(committed):
    """#54 and #56-#58 already ran; a flag here would mean a draft was committed broken."""
    known = corpus_doc_ids()
    for candidate in committed:
        for violation in review.check_row(review.Row(candidate.line, candidate.item), known):
            raise AssertionError(f"{candidate.id}: {violation.field} {violation.message}")


def test_every_committed_row_renders_with_its_chunk_text(committed):
    texts = review.corpus_text()
    for candidate in committed:
        screen = review.render_row(candidate, candidate.item, texts, [], 1, len(committed))
        assert candidate.id in screen
        for gold in candidate.item.gold_doc_ids:
            assert texts[gold].splitlines()[0] in screen
        if not candidate.item.gold_doc_ids:
            assert "claim of absence" in screen


@pytest.mark.skipif(not LOG.exists(), reason="the review log lands when a person has run the pass")
def test_the_committed_log_judges_only_rows_that_were_drafted(committed):
    drafted = {candidate.id for candidate in committed}
    judged = review.read_verdicts(LOG)
    assert set(judged) <= drafted
    assert all(verdict.verdict in review.VERDICT_NAMES for verdict in judged.values())


@pytest.mark.skipif(not LOG.exists(), reason="the review log lands when a person has run the pass")
def test_every_delete_and_every_edit_in_the_committed_log_carries_its_reason():
    """The reasons are the honest half of #62's table; an unreasoned delete explains nothing."""
    for verdict in review.read_verdicts(LOG).values():
        if verdict.verdict in {"edit", "delete"}:
            assert verdict.reason, f"{verdict.id}: a {verdict.verdict} with no reason"
        if verdict.verdict == "edit":
            assert review.one_word(verdict.reason or "") is not None
            assert verdict.edits, f"{verdict.id}: an edit that changed nothing"


@pytest.mark.skipif(
    not (LOG.exists() and DATASET.exists()),
    reason="the dataset lands when the pass has been run to the end",
)
def test_the_committed_dataset_is_the_drafts_replayed_through_the_committed_log(committed):
    """The invariant the whole design rests on: v1.jsonl is a function, not a second copy.

    If this fails, either a row was edited into the dataset without going through the log -- in
    which case the log understates what the pass changed -- or a draft moved under a verdict. Both
    are the same repair: re-review the row.
    """
    promotion = review.promote(committed, review.read_verdicts(LOG), corpus_doc_ids())
    assert promotion.unreviewed == []
    assert promotion.orphans == []
    assert [v for v in promotion.violations if v.level == "error"] == []
    written = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    assert written == [item.model_dump() for item in promotion.rows]
