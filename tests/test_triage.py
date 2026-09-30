"""Tests for eval/scripts/triage.py: the mapping, the reply, the file, and the line it is not.

The suggestion a reviewer sees is derived from four answers by :func:`eval.scripts.triage.suggest`,
so the first group pins that mapping with no model anywhere near it -- it is the part a person
will disagree with, and it has to be a function they can read.

The second group is the reply reader: a model's JSON, a fenced one, and every way one can come
back unusable. Each fault names the field, because a run over 152 rows that failed on row 90 has
to say which cell it could not read.

The third is the file. The point of these tests is the thing that must never happen: a triage row
being taken for a review log entry, or the other way round. They are different keys in different
files and both readers refuse the other's rows on the first line.

The fourth drives a run over a stub chat: the rows it reads, the ones it skips, the limit, and a
failure that does not stop the run.

The fifth is the review pass with ``--triage``: where on the screen the suggestion appears, which
is the whole of what makes this advisory rather than leading. It is **last**, after the three
checks and after every cited record.

Nothing here calls a model and nothing writes into ``eval/datasets/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config import load_config
from eval.schema import BUCKETS, EvalItem
from eval.scripts import review
from eval.scripts import triage as tr
from ingest.tokenize import get_tokenizer

# --- rows and a stub model ------------------------------------------------------------------------

CHUNKS = {
    "2026-sp:CS-225": "CS 225 — Data Structures\nCredits: 4\nPrerequisites: CS 173 or MATH 213.",
    "2026-sp:CS-374": "CS 374 — Algorithms\nCredits: 4\nPrerequisites: CS 225.",
}


#: What the one edited row's answer becomes, so the assertion and the edit share a string.
FIXED = "CS 225 is 4 hours, per the chunk."


def item(bucket: str = "fact", **overrides: Any) -> EvalItem:
    """One row, valid against its bucket's rule."""
    fields: dict[str, Any] = {
        "id": "v1-0001",
        "bucket": bucket,
        "question": "how many credits is CS 225?",
        "gold_answer": "CS 225 is 4 credit hours.",
        "gold_doc_ids": ["2026-sp:CS-225"],
        "should_abstain": bucket in {"unanswerable", "ambiguous"},
    }
    if bucket in {"unanswerable", "ambiguous"}:
        fields["gold_doc_ids"] = []
        fields["gold_answer"] = "This catalog does not carry instructors."
    fields.update(overrides)
    return EvalItem.model_validate(fields)


def reply(**overrides: Any) -> str:
    """A clean model reply, as JSON text."""
    body = {
        "answer_correct": "yes",
        "citations_support": "yes",
        "bucket": "fact",
        "belongs": "yes",
        "reason": "the chunk prints Credits: 4.",
    }
    body.update(overrides)
    return json.dumps(body)


class StubChat:
    """A chat client that returns canned replies in order, and records what it was asked.

    A list rather than one string, because the run has to be checked over several rows: a stub
    that answered identically would pass a loop that called it once and stopped.
    """

    def __init__(self, replies: list[str], model: str = "stub-1") -> None:
        self.model = model
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if not self._replies:
            raise RuntimeError("the stub ran out of replies")
        return self._replies.pop(0)


def candidates(n: int = 3) -> list[review.Candidate]:
    """``n`` drafted rows, each its own id, with the provenance #56 records."""
    from eval.scripts.draft_items import PROVENANCE_KEYS

    out = []
    for i in range(1, n + 1):
        provenance = dict.fromkeys(PROVENANCE_KEYS, None)
        provenance.update({"flagged": False, "credit_question": True, "model": "gpt-4o-mini"})
        one = item(id=f"v1-000{i}", question=f"how many credits is CS 225, take {i}?")
        out.append(review.Candidate(Path("fact_abc.jsonl"), i, one, provenance))
    return out


@pytest.fixture
def cfg():
    return load_config()


# --- the mapping ----------------------------------------------------------------------------------


def test_a_clean_row_is_an_accept():
    checks = {
        "answer_correct": "yes",
        "citations_support": "yes",
        "bucket": "fact",
        "belongs": "yes",
    }
    assert tr.suggest(checks, "fact") == "accept"


def test_no_citations_is_clean_on_a_refusal_row():
    """``n/a`` is the right answer where a row cites nothing, and it must not read as a fault."""
    checks = {
        "answer_correct": "yes",
        "citations_support": "n/a",
        "bucket": "unanswerable",
        "belongs": "yes",
    }
    assert tr.suggest(checks, "unanswerable") == "accept"


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer_correct", "no"),
        ("answer_correct", "unsure"),
        ("citations_support", "no"),
        ("bucket", "multihop"),
    ],
)
def test_anything_the_guidelines_would_have_changed_is_an_edit(field: str, value: str):
    """A wrong answer is an edit and not a delete: repairing rows is what the pass is for.

    ``unsure`` too -- a row a machine cannot settle is exactly a row that wants a person, and
    marking it accept would hide it in the one place a reviewer is scanning for work.
    """
    checks = {
        "answer_correct": "yes",
        "citations_support": "yes",
        "bucket": "fact",
        "belongs": "yes",
        field: value,
    }
    assert tr.suggest(checks, "fact") == "edit"


def test_only_belongs_no_is_a_delete():
    """And it outranks the other three: a row deleted under rule 2 may be correct in every way."""
    checks = {
        "answer_correct": "yes",
        "citations_support": "yes",
        "bucket": "fact",
        "belongs": "no",
    }
    assert tr.suggest(checks, "fact") == "delete"


def test_the_four_questions_are_the_pass_and_the_annotators_own():
    """#59's three checks plus one, and the values are #60's, so the two are comparable."""
    from eval.scripts.agreement import CATEGORIES

    assert tuple(tr.CHECKS) == ("answer_correct", "citations_support", "bucket", "belongs")
    for field in ("answer_correct", "citations_support", "bucket"):
        assert set(tr.CHECKS[field]) >= set(CATEGORIES[field]) - {"unsure"}
    assert tr.CHECKS["bucket"] == BUCKETS


# --- the reply ------------------------------------------------------------------------------------


def test_a_clean_reply_reads_as_four_answers_and_a_sentence():
    parsed = tr.parse_reply(reply())
    assert parsed["answer_correct"] == "yes"
    assert parsed["bucket"] == "fact"
    assert parsed["reason"] == "the chunk prints Credits: 4."


def test_a_fenced_reply_is_unwrapped_and_not_refused():
    """Like #56's: a fence is a formatting habit, and throwing the row away costs a call."""
    assert tr.parse_reply("```json\n" + reply() + "\n```")["bucket"] == "fact"


def test_case_and_space_are_folded():
    assert tr.parse_reply(reply(answer_correct="YES", reason="  two   spaces "))["reason"] == (
        "two spaces"
    )


@pytest.mark.parametrize(
    "text,message",
    [
        ("not json at all", "not JSON"),
        (json.dumps([1, 2]), "not an object"),
        (reply(answer_correct="probably"), "answer_correct"),
        (reply(bucket="facts"), "bucket"),
        (reply(belongs="maybe"), "belongs"),
        (reply(reason="  "), "reason is missing"),
    ],
)
def test_every_unusable_reply_names_its_fault(text: str, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        tr.parse_reply(text)


# --- the file, and the file it is not -------------------------------------------------------------


def test_a_triage_row_round_trips(tmp_path: Path):
    path = tmp_path / "triage.jsonl"
    row = tr.Triage("v1-0001", "accept", "because", {"bucket": "fact"}, "stub-1")
    tr.append_triage(path, row)
    back = tr.read_triage(path)
    assert back["v1-0001"].suggested == "accept"
    assert back["v1-0001"].model == "stub-1"


def test_the_later_line_wins(tmp_path: Path):
    """Append-only, like the log: a re-read replaces a reading rather than editing one."""
    path = tmp_path / "triage.jsonl"
    tr.append_triage(path, tr.Triage("v1-0001", "accept", "first", {}, "stub-1"))
    tr.append_triage(path, tr.Triage("v1-0001", "delete", "second", {}, "stub-1"))
    assert tr.read_triage(path)["v1-0001"].reason == "second"


def test_a_missing_file_is_no_suggestions_and_not_an_error(tmp_path: Path):
    assert tr.read_triage(tmp_path / "nothing.jsonl") == {}


def test_the_triage_reader_refuses_a_review_log(tmp_path: Path):
    """The failure this whole file is shaped to prevent, in the direction that would be worst."""
    path = tmp_path / "review_log.jsonl"
    review.append_verdict(path, review.Verdict("v1-0001", "accept", None, "2026-09-23T00:00:00Z"))
    with pytest.raises(ValueError, match="not a triage row"):
        tr.read_triage(path)


def test_the_log_reader_refuses_a_triage_file(tmp_path: Path):
    """And the other direction: six of seven keys missing, named on the first line."""
    path = tmp_path / "triage.jsonl"
    tr.append_triage(path, tr.Triage("v1-0001", "accept", "because", {}, "stub-1"))
    with pytest.raises(ValueError, match="review log entry carries all seven keys"):
        review.read_entries(path)


def test_the_two_key_sets_overlap_only_where_they_mean_the_same_thing():
    """An id and a reason. Neither of the two keys that make a log line a judgement is here.

    ``verdict`` and ``ts`` are what the readers above key on, so their absence is the property
    that keeps the files apart -- and it is asserted rather than left to the two readers, because
    adding a ``ts`` here would make :func:`eval.scripts.triage.read_triage` reject its own file
    and the reason would not be obvious.
    """
    assert set(tr.TRIAGE_KEYS) & set(review.LOG_KEYS) == {"id", "reason"}
    assert not {"verdict", "ts", "before", "after"} & set(tr.TRIAGE_KEYS)


# --- the run --------------------------------------------------------------------------------------


def test_a_run_reads_every_unread_row_and_writes_as_it_goes(cfg, tmp_path: Path):
    path = tmp_path / "triage.jsonl"
    rows = candidates(3)
    chat = StubChat([reply(), reply(belongs="no"), reply(answer_correct="no")])
    from io import StringIO

    run = tr.triage(
        cfg, rows, chat, CHUNKS, path, {}, StringIO(), guidelines="G", verdicts={}
    )
    assert [one.suggested for one in run.read] == ["accept", "delete", "edit"]
    assert not run.failed
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert all(one.config_hash == cfg.hash for one in run.read)
    assert all(one.model == "stub-1" for one in run.read)
    assert all(one.prompt_version == tr.TRIAGE_PROMPT_VERSION for one in run.read)


def test_rows_already_in_the_file_are_skipped(cfg, tmp_path: Path):
    from io import StringIO

    path = tmp_path / "triage.jsonl"
    rows = candidates(3)
    done = {"v1-0001": tr.Triage("v1-0001", "accept", "read already", {}, "stub-1")}
    chat = StubChat([reply(), reply()])
    run = tr.triage(cfg, rows, chat, CHUNKS, path, done, StringIO(), guidelines="G", verdicts={})
    assert run.skipped == 1
    assert [one.id for one in run.read] == ["v1-0002", "v1-0003"]


def test_the_limit_is_a_sitting(cfg, tmp_path: Path):
    from io import StringIO

    chat = StubChat([reply(), reply(), reply()])
    run = tr.triage(
        cfg, candidates(3), chat, CHUNKS, tmp_path / "t.jsonl", {}, StringIO(),
        limit=2, guidelines="G", verdicts={},
    )
    assert len(run.read) == 2
    assert len(chat.prompts) == 2


def test_one_unreadable_reply_does_not_stop_the_run(cfg, tmp_path: Path):
    """152 rows and one bad completion: the other 151 are worth having."""
    from io import StringIO

    chat = StubChat(["nonsense", reply(), reply()])
    run = tr.triage(
        cfg, candidates(3), chat, CHUNKS, tmp_path / "t.jsonl", {}, StringIO(),
        guidelines="G", verdicts={},
    )
    assert [one for one, _ in run.failed] == ["v1-0001"]
    assert [one.id for one in run.read] == ["v1-0002", "v1-0003"]


def test_the_row_read_is_the_row_as_edited(cfg, tmp_path: Path):
    """#59's ``current()``: a row a person has already fixed is triaged as fixed."""
    from io import StringIO

    rows = candidates(1)
    verdict = review.Verdict(
        "v1-0001",
        "edit",
        "wording",
        "2026-09-23T00:00:00Z",
        (review.Edit("gold_answer", rows[0].item.gold_answer, FIXED),),
    )
    chat = StubChat([reply()])
    tr.triage(
        cfg, rows, chat, CHUNKS, tmp_path / "t.jsonl", {}, StringIO(),
        guidelines="G", verdicts={"v1-0001": verdict},
    )
    assert FIXED in chat.prompts[0]


# --- the prompt -----------------------------------------------------------------------------------


def test_the_prompt_carries_the_guidelines_and_the_chunk_text():
    """Not a paraphrase of the guidelines: the same file the two annotators are held to."""
    body = tr.render_prompt(item(), CHUNKS, "THE GUIDELINES GO HERE")
    assert "THE GUIDELINES GO HERE" in body
    assert "Credits: 4" in body
    assert "2026-sp:CS-225" in body
    assert "how many credits is CS 225?" in body


def test_a_refusal_row_is_told_there_are_no_records_and_why():
    """An empty section would read as a missing prompt; the absence is the thing being checked."""
    body = tr.render_prompt(item("unanswerable"), CHUNKS, "G")
    assert "claim of absence" in body
    assert "[1]" not in body


def test_the_system_message_forbids_outside_knowledge():
    assert "not in the records quoted" in tr.TRIAGE_SYSTEM


# --- the review pass with --triage ----------------------------------------------------------------


def test_the_suggestion_is_shown_last_and_labelled_advisory():
    """After the three checks and after every cited record. That is what makes it advisory."""
    rows = candidates(1)
    screen = review.render_row(
        rows[0], rows[0].item, CHUNKS, [], 1, 1, None, "edit -- reads as multihop. a join."
    )
    assert "machine triage -- advisory, and not a verdict" in screen
    assert screen.index("is the gold_answer right?") < screen.index("machine triage")
    assert screen.index("Credits: 4") < screen.index("machine triage")
    assert screen.rstrip().endswith("edit -- reads as multihop. a join.")


def test_no_flag_means_no_suggestion_on_screen():
    rows = candidates(1)
    screen = review.render_row(rows[0], rows[0].item, CHUNKS, [], 1, 1)
    assert "machine triage" not in screen
    assert review.suggestions(None, rows) == {}


def test_the_line_names_only_what_would_change(tmp_path: Path):
    """A reviewer does not need to be told a machine also thought the answer was right."""
    row = tr.Triage(
        "v1-0001",
        "edit",
        "the answer is a join over two records.",
        {
            "answer_correct": "yes",
            "citations_support": "yes",
            "bucket": "multihop",
            "belongs": "yes",
        },
        "stub-1",
    )
    line = row.line("fact")
    assert "reads as multihop" in line
    assert "answer" not in line.split(".")[0]
    assert row.line("multihop").startswith("edit -- no objection")


def test_a_rule_two_delete_says_where_it_came_from():
    """Not one of the four answers, so a line built from those alone says the opposite."""
    row = tr.Triage(
        "v1-0001",
        "delete",
        "GUIDELINES rule 2: ACCY 290 has a credit range.",
        {
            "answer_correct": "yes",
            "citations_support": "yes",
            "bucket": "fact",
            "belongs": "yes",
        },
        "stub-1",
    )
    assert "deleted by the rule 2 lookup" in row.line("fact")
    assert "no objection" not in row.line("fact")


def test_suggestions_are_keyed_to_the_rows_being_reviewed(tmp_path: Path):
    """A triage row for a draft that has since gone is dropped, not shown against another row."""
    path = tmp_path / "triage.jsonl"
    tr.append_triage(path, tr.Triage("v1-0001", "accept", "fine", {"bucket": "fact"}, "stub-1"))
    tr.append_triage(path, tr.Triage("v1-9999", "delete", "gone", {"bucket": "fact"}, "stub-1"))
    found = review.suggestions(str(path), candidates(1))
    assert set(found) == {"v1-0001"}


# --- rule 2, which is a lookup and not a judgement ------------------------------------------------


def test_the_lookup_finds_the_courses_the_guidelines_cite():
    """1,415 of 4,502, from :data:`CREDIT_RANGE_REF` -- the number ``GUIDELINES.md`` rule 2 quotes.

    Skipped rather than failed where git cannot answer: a shallow clone and a machine with no
    remote are both real, and the run says so on its own summary line instead of deleting
    nothing and looking clean.
    """
    codes = tr.variable_credit_codes()
    if codes is None:
        pytest.skip(f"git show {tr.CREDIT_RANGE_REF}:data/courses.json is unavailable here")
    assert len(codes) == 1415
    assert "AAS 465" in codes


def test_an_export_without_the_range_fields_is_unavailable_not_empty(monkeypatch):
    """The #128 failure, pinned: 0 and "cannot say" are different answers.

    ``origin/main`` still resolves after #126 -- it is the six-day stack's export now, with a
    single ``credits`` float. Every row then compared ``None != None``, so the lookup returned an
    empty set, the summary line read ``0 variable-credit courses`` as though it had checked, and
    rule 2 was off for all 1,415 courses without one line of output saying so.
    """
    stack_export = json.dumps([{"code": "AAS 465", "credits": 3.0}])
    monkeypatch.setattr(tr, "_VARIABLE", {})
    monkeypatch.setattr(
        tr.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=stack_export),
    )
    assert tr.variable_credit_codes("some-ref-without-the-fields") is None


def test_an_export_with_the_fields_and_no_range_is_empty_not_unavailable(monkeypatch):
    """The other side of it: the fields are there, nothing is a range, and that is an answer."""
    fixed = json.dumps([{"code": "CS 225", "creditsMin": 4.0, "creditsMax": 4.0}])
    monkeypatch.setattr(tr, "_VARIABLE", {})
    monkeypatch.setattr(
        tr.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=fixed)
    )
    assert tr.variable_credit_codes("some-ref-with-the-fields") == frozenset()


def test_rule_two_needs_both_halves(cfg):
    """A credit question **and** a variable-credit course. Either alone is a good row."""
    tokenizer = get_tokenizer(cfg)
    variable = frozenset({"AAS 465"})
    credit_of_variable = item(question="how many credits is AAS 465?", gold_answer="AAS 465 is 3.")
    assert tr.rule_two(credit_of_variable, variable, tokenizer) == ["AAS 465"]
    prereq_of_variable = item(
        question="what are the prerequisites for AAS 465?", gold_answer="AAS 465 needs consent."
    )
    assert tr.rule_two(prereq_of_variable, variable, tokenizer) == []
    credit_of_fixed = item()
    assert tr.rule_two(credit_of_fixed, variable, tokenizer) == []


def test_an_unavailable_lookup_deletes_nothing(cfg):
    """``None`` is not an empty set: it must not read as "no course has a credit range"."""
    tokenizer = get_tokenizer(cfg)
    row = item(question="how many credits is AAS 465?", gold_answer="AAS 465 is 3.")
    assert tr.rule_two(row, None, tokenizer) == []


def test_rule_two_outranks_a_clean_reply():
    """The model is told not to judge it, so the mapping has to."""
    checks = {
        "answer_correct": "yes",
        "citations_support": "yes",
        "bucket": "fact",
        "belongs": "yes",
    }
    assert tr.suggest(checks, "fact") == "accept"
    assert tr.suggest(checks, "fact", ["AAS 465"]) == "delete"


def test_the_run_applies_rule_two_and_says_why(cfg, tmp_path: Path):
    """The reason names the rule and the course, because that is what the reviewer checks."""
    from io import StringIO

    rows = candidates(1)
    rows[0] = rows[0]._replace(
        item=item(
            id="v1-0001", question="how many credits is AAS 465?", gold_answer="AAS 465 is 3."
        )
    )
    chat = StubChat([reply()])
    run = tr.triage(
        cfg, rows, chat, CHUNKS, tmp_path / "t.jsonl", {}, StringIO(),
        guidelines="G", verdicts={}, variable=frozenset({"AAS 465"}),
    )
    assert run.read[0].suggested == "delete"
    assert "GUIDELINES rule 2: AAS 465" in run.read[0].reason
    assert run.read[0].checks["belongs"] == "yes", "the model's answer is recorded as given"


def test_the_prompt_tells_the_model_not_to_judge_rule_two():
    """It cannot: the chunk prints one number whether or not the catalog gives a range."""
    body = tr.render_prompt(item(), CHUNKS, "G")
    assert "Do **not** consider rule 2" in body
    assert tr.TRIAGE_PROMPT_VERSION == "v2"


def test_rule_two_does_not_touch_an_abstaining_row(cfg):
    """An `ambiguous` row's answer is a clarifying question, and rule 2 is about credit answers.

    *"Which 199 do you mean, UKR 199 or SOC 590?"* is correct whether or not either course has a
    credit range: the row asserts no number, so there is nothing for the rule to be wrong about.
    Deleting it would throw away a row that tests ambiguity on the grounds of a claim it never
    made -- which the first run of this triage did, on six rows.
    """
    tokenizer = get_tokenizer(cfg)
    variable = frozenset({"UKR 199", "SOC 590"})
    clarifying = EvalItem.model_validate(
        {
            "id": "v1-3020",
            "bucket": "ambiguous",
            "question": "how many credits is 199?",
            "gold_answer": "Which 199 do you mean — UKR 199 or SOC 590?",
            "gold_doc_ids": [],
            "should_abstain": True,
        }
    )
    assert tr.rule_two(clarifying, variable, tokenizer) == []
    refusal = EvalItem.model_validate(
        {
            "id": "v1-2020",
            "bucket": "unanswerable",
            "question": "how many credits is UKR 199 in the fall?",
            "gold_answer": "This export carries only 2026 spring, so it has no fall listing.",
            "gold_doc_ids": [],
            "should_abstain": True,
        }
    )
    assert tr.rule_two(refusal, variable, tokenizer) == []
