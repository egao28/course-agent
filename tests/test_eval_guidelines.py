"""Tests for eval/GUIDELINES.md: the spec stays a page, and its examples stay true of the catalog.

A prose spec cannot be unit tested into being *right*. What it can be held to is everything it
asserts about the corpus, and this file holds it to all of it: the eight boundary cases are the
ones that earn the labels in ``eval.schema.BUCKETS``, and every course, id and count they rest on
is read back out of ``data/``. CS 428 is the sharpest of them -- the whole ``unanswerable`` rule
is "term scope is absence", and if a re-export ever brings CS 428 in, the guideline's example
becomes a lie that no reader would catch, because it reads exactly like the gap it is not.

The second annotator (#60) gets this document and nothing else, so a stale example is not a typo:
it is a rule applied to the wrong facts for however long it takes someone to re-check by hand.
"""

from __future__ import annotations

import json
import re

import pytest

from agent.state import INTENTS
from agent.tools.courses import CourseDetail
from agent.tools.prereqs import load_prereqs
from config import REPO_ROOT, load_config
from eval.schema import BUCKETS, normalize_question
from eval.validate import corpus_doc_ids
from ingest.chunk import load_courses

GUIDELINES = REPO_ROOT / "eval" / "GUIDELINES.md"

#: A page of dense spec. The limit is a guard against drift, not a target -- the acceptance
#: criterion in #55 is "one page", and a document that has grown a third again is not one.
MAX_WORDS = 1400

#: The Intent section (#102) is counted apart: it labels a different thing, a sidecar and not a
#: bucket, and folding it into the page cap would have meant cutting a Day 3 rule to make room.
MAX_INTENT_WORDS = 550

#: ``2026-sp:CS-374``: the form a gold id takes, and the only form this document should use.
DOC_ID_RE = re.compile(r"\b\d{4}-[a-z]{2}:[A-Z]+-\d+\b")

#: ``CS 374``: the form a course code takes in prose, with the space the catalog uses.
CODE_RE = re.compile(r"\b[A-Z]{2,4} \d{3}\b")

#: ``- **IA1.** *"question"* → `["course_fact", ...]`.``: an Intent boundary case, whose label
#: list may wrap onto the next line.
INTENT_CASE_RE = re.compile(r'^- \*\*I[AU]\d\.\*\* \*"(.+?)"\* →\s+`(\[[^\]]+\])`', re.MULTILINE)


@pytest.fixture(scope="module")
def text() -> str:
    return GUIDELINES.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cases(text: str) -> dict[str, list[str]]:
    """The boundary section, as ``{bucket heading: [case, ...]}``.

    Each case is one ``- **F1.** ...`` bullet with its continuation lines, so a test can ask what
    a case says and not only that it is there.
    """
    _, _, tail = text.partition("## Eight boundary cases")
    section, _, _ = tail.partition("\n## ")
    out: dict[str, list[str]] = {}
    for block in section.split("\n### ")[1:]:
        heading, _, body = block.partition("\n")
        bullets = re.split(r"^- ", body, flags=re.MULTILINE)[1:]
        out[heading.strip()] = [b.strip() for b in bullets]
    return out


@pytest.fixture(scope="module")
def intent_section(text: str) -> str:
    """``## Intent`` up to the next ``## `` heading: #102's addition, labelled on its own."""
    _, found, tail = text.partition("\n## Intent\n")
    assert found, "#102's labels need their rules committed before any row is labelled"
    section, _, _ = tail.partition("\n## ")
    return section


@pytest.fixture(scope="module")
def intent_cases(intent_section: str) -> dict[str, list[tuple[str, list[str]]]]:
    """The Intent boundary cases, as ``{bucket: [(question, gold_intents), ...]}``."""
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for block in intent_section.split("\n### ")[1:]:
        heading, _, body = block.partition("\n")
        found = INTENT_CASE_RE.findall(body)
        out[heading.strip()] = [(q, json.loads(labels)) for q, labels in found]
    return out


@pytest.fixture(scope="module")
def v1_by_question() -> dict[str, dict]:
    with (REPO_ROOT / "eval" / "datasets" / "v1.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {normalize_question(r["question"]): r for r in rows}


@pytest.fixture(scope="module")
def by_code() -> dict[str, dict]:
    return {c["code"]: c for c in load_courses()}


def test_the_guidelines_are_in_the_tree():
    assert GUIDELINES.exists(), "#59 cannot start against a spec that is not committed"


def test_the_document_is_still_one_page(text: str, intent_section: str):
    words = len(text.split()) - len(intent_section.split())
    assert words <= MAX_WORDS, (
        f"{words} words; a criterion that does not fit a page is not decidable"
    )


def test_the_intent_section_is_its_own_half_page(intent_section: str):
    words = len(intent_section.split())
    assert words <= MAX_INTENT_WORDS, f"{words} words in the Intent section"


def test_the_buckets_are_the_schema_s_four(cases: dict[str, list[str]]):
    assert tuple(cases) == BUCKETS


def test_two_boundary_cases_per_bucket_and_eight_in_all(cases: dict[str, list[str]]):
    assert {b: len(c) for b, c in cases.items()} == dict.fromkeys(BUCKETS, 2)


def test_every_boundary_case_states_a_verdict_and_a_reason(cases: dict[str, list[str]]):
    for bucket, bullets in cases.items():
        for case in bullets:
            verdict, _, reason = case.partition("\n")
            assert any(b in case for b in BUCKETS) or "deleted" in case, f"{bucket}: {verdict}"
            assert reason.strip(), f"{bucket}: {verdict} is a call with no reason under it"


def test_every_gold_id_it_names_is_in_the_corpus(text: str):
    known = corpus_doc_ids()
    assert DOC_ID_RE.findall(text), "the examples should name gold ids in the form they take"
    assert not [i for i in DOC_ID_RE.findall(text) if i not in known]


def test_every_course_it_names_is_in_the_catalog_except_the_absent_one(text: str, by_code):
    named = set(CODE_RE.findall(text))
    assert "CS 428" in named, "the term-scope rule is argued from CS 428 and nothing else"
    assert "CS 428" not in by_code, "CS 428 is in the export now; the unanswerable example is wrong"
    assert not (named - {"CS 428"} - set(by_code))


def test_the_neighbours_that_make_cs_428_look_like_a_gap_are_still_there(by_code):
    assert {"CS 421", "CS 423", "CS 425", "CS 427"} <= set(by_code)


def test_one_record_answers_the_fact_example(by_code):
    assert by_code["CS 374"]["prereqRaw"] == "One of CS 173, MATH 213; CS 225."


def test_the_cross_reference_the_multihop_example_turns_on(by_code):
    ece = by_code["ECE 374"]
    assert ece["desc"] == "Same as CS 374. See CS 374."
    assert ece["prereqCodes"] == [], "ECE 374 carries prereqs now; M1's 'none' argument is gone"


def test_the_cycle_the_depth_rule_rests_on_is_still_a_cycle():
    graph = {row["course"]: row["prereqs"] for row in load_prereqs()}
    assert "CS 374" in graph["MATH 412"]
    assert "CS 225" in graph["CS 374"]
    assert "MATH 412" in graph["CS 225"]


def test_the_data_structures_referent_is_still_unique(by_code):
    titled = [c["code"] for c in by_code.values() if c["name"].casefold() == "data structures"]
    qualified = [
        c["code"]
        for c in by_code.values()
        if "data structures" in c["name"].casefold() and c["code"] not in titled
    ]
    assert titled == ["CS 225"]
    assert sorted(qualified) == ["CI 487", "CS 277"]
    assert by_code["CS 225"]["credits"] == 4.0


def test_the_bare_number_in_the_ambiguous_example_still_has_four_readings(by_code):
    assert sorted(c["code"] for c in by_code.values() if c["number"] == "374") == [
        "BADM 374",
        "CS 374",
        "ECE 374",
        "HIST 374",
    ]


def test_the_gen_ed_example_is_a_one_record_question_and_the_other_is_not(by_code):
    assert by_code["RHET 105"]["attrs"] == "Composition I course."


def test_the_field_no_chunk_carries_is_reachable_by_a_tool_and_not_by_a_citation():
    assert "section_count" in CourseDetail.model_fields
    fields = load_config()["chunk"]["fields"]
    assert not {"sectionCount", "partOfTerm", "types"} & set(fields)


# --- the Intent section (#102) --------------------------------------------------------------------


def test_the_intent_section_comes_before_the_amendment_rule(text: str):
    assert text.index("\n## Intent\n") < text.index("\n## Amending this file\n")


def test_two_intent_boundary_cases_for_each_bucket_with_no_obvious_intent(intent_cases):
    assert {b: len(c) for b, c in intent_cases.items()} == {"ambiguous": 2, "unanswerable": 2}


def test_every_intent_it_names_is_one_of_the_router_s_five(intent_section: str, intent_cases):
    for bucket, cases in intent_cases.items():
        for question, labels in cases:
            assert labels and len(set(labels)) == len(labels), f"{bucket}: {question}"
            assert set(labels) <= set(INTENTS), f"{bucket}: {question} -> {labels}"
    named = set(re.findall(r"`([a-z]+_[a-z_]+)`", intent_section))
    assert named - {"gold_intents"} <= set(INTENTS)


def test_the_intent_boundary_cases_are_calib_questions_in_their_own_bucket(
    intent_cases, v1_by_question
):
    """Quoting a test question here would label it before the labeller sees it."""
    for bucket, cases in intent_cases.items():
        for question, _ in cases:
            row = v1_by_question.get(normalize_question(question))
            assert row is not None, f"{question!r} is not a v1 row"
            assert row["split"] == "calib", f"{question!r} is a {row['split']} row"
            assert row["bucket"] == bucket, f"{question!r} is {row['bucket']}, not {bucket}"


def test_no_test_question_is_quoted_anywhere_in_the_intent_section(intent_section, v1_by_question):
    quoted = {normalize_question(q) for q in re.findall(r'\*"(.+?)"\*', intent_section)}
    tests = {q for q, row in v1_by_question.items() if row["split"] == "test"}
    assert not quoted & tests
