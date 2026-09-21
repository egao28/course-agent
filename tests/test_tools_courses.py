"""Tests for agent/tools/courses.py: the normalizer, the index guards, and both exits of the tool.

Two corpora, for two different jobs. Most tests run against the 17 courses
``tests/fixtures/catalog_sample.csv`` produces through the real exporter, so the shapes under test
(a course with no attrs, one with no prereqs, a repeated code, #25's credit fields) can be
constructed instead of hunted for -- and ``COURSES_PATH`` is redirected at a tmp_path copy, so no
test in here parses the live 2.9 MB corpus by accident.

The last section does read the live corpus, on purpose: #29's acceptance is stated about real
courses (``cs 374`` whole, ``CS 999`` absent), the file is committed, and every lookup here is
read-only.

``fresh_cache`` is autouse because the index is module state: without it the first test to touch
the fixture corpus would answer every later test, live section included.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.tools import courses as tc
from ingest import export_almabot as ex

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits the index another one built."""
    tc.clear_cache()
    yield
    tc.clear_cache()


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV).courses


@pytest.fixture
def corpus(tmp_path, courses, monkeypatch) -> Path:
    """Write the fixture corpus where the module will look for ``data/courses.json``."""
    path = tmp_path / "courses.json"
    ex.write_json(path, courses)
    monkeypatch.setattr(tc, "COURSES_PATH", path)
    return path


# --- normalizing ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling", ["CS 374", "cs 374", "CS374", "cs374", "  cs\t374  ", "Cs 374"]
)
def test_every_spelling_of_a_code_normalizes_to_one(spelling):
    assert tc.normalize_code(spelling) == "CS 374"


@pytest.mark.parametrize(
    "text", ["", "   ", "hello", "C 374", "COMPSCI 374", "CS 37", "CS 3745", "CS-374", "374"]
)
def test_normalize_code_returns_none_for_what_is_not_a_code(text):
    assert tc.normalize_code(text) is None


def test_a_sentence_mentioning_a_code_is_not_a_code():
    # The whole string has to be one code: 'CS 374 or ECE 374' names two, and picking one of them
    # here would be a retrieval decision hiding in a lookup. #35's router extracts the code.
    assert tc.normalize_code("what is cs 374 about") is None
    assert tc.normalize_code("CS 374 or ECE 374") is None


def test_zero_padding_survives_normalizing():
    # MUS 090 is the only zero-padded course in the 2026-sp catalog, and int('090') would lose it.
    assert tc.normalize_code("mus090") == "MUS 090"


# --- the index ------------------------------------------------------------------------------------


def test_the_index_is_keyed_by_normalized_code(courses):
    index = tc.build_index(courses)
    assert len(index.by_code) == len(courses)
    assert index.terms == ("2026-sp",)
    assert index.by_code["CS 374"]["name"] == "Introduction to Algorithms & Models"


def test_a_repeated_code_is_refused(courses):
    with pytest.raises(ValueError, match="appears twice"):
        tc.build_index([*courses, courses[0]])


def test_a_code_the_shared_matcher_cannot_read_is_refused(courses):
    with pytest.raises(ValueError, match="not a course code"):
        tc.build_index([{**courses[0], "code": "COMPSCI 374"}])


def test_pr25s_credit_fields_are_refused_until_this_model_moves(courses):
    moved = {k: v for k, v in courses[0].items() if k != "credits"}
    moved |= {"creditsRaw": "3 TO 4 hours.", "creditsMin": 3.0, "creditsMax": 4.0}
    with pytest.raises(ValueError, match="creditsRaw"):
        tc.build_index([moved])


def test_load_index_reads_the_path_it_is_given_and_does_not_cache(corpus, tmp_path, courses):
    three = tmp_path / "three.json"
    ex.write_json(three, courses[:3])
    assert len(tc.load_index(three).by_code) == 3
    assert len(tc.course_index().by_code) == len(courses)


def test_the_corpus_is_read_once(corpus, monkeypatch):
    reads: list = []
    real = tc.load_courses
    monkeypatch.setattr(tc, "load_courses", lambda path=None: (reads.append(path), real(path))[1])

    for code in ("CS 374", "CS 225", "CS 999"):
        tc.get_course(code)
    assert len(reads) == 1

    tc.clear_cache()
    tc.get_course("CS 374")
    assert len(reads) == 2


def test_a_missing_corpus_points_at_the_export(tmp_path, monkeypatch):
    # Not a found=False: a course that is absent is an answer, a corpus that is absent is a setup
    # step the caller has not run.
    monkeypatch.setattr(tc, "COURSES_PATH", tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError, match="export_almabot"):
        tc.get_course("CS 374")


# --- found ----------------------------------------------------------------------------------------


def test_get_course_returns_the_whole_record(corpus):
    detail = tc.get_course("cs 374")
    assert detail.found is True
    assert (detail.code, detail.name) == ("CS 374", "Introduction to Algorithms & Models")
    assert (detail.credits, detail.term, detail.section_count) == (4.0, "2026-sp", 3)
    assert detail.prereq_raw == "One of CS 173, MATH 213; CS 225."
    assert detail.message is None


def test_a_found_record_leaves_only_message_unset(corpus):
    # The guard against a field being added to CourseDetail and never wired to a record key.
    assert [k for k, v in tc.get_course("CS 374").model_dump().items() if v is None] == ["message"]


@pytest.mark.parametrize("spelling", ["CS 374", "cs374", "  cs 374  "])
def test_every_spelling_reaches_the_same_record(corpus, spelling):
    assert tc.get_course(spelling) == tc.get_course("CS 374")


def test_an_empty_field_comes_back_empty_not_missing(corpus):
    # CWL 593 has no description, no prereqs and no attributes; #38 says "none listed" from these,
    # so they must arrive as the empty strings the catalog gives, not as None.
    detail = tc.get_course("CWL 593")
    assert detail.found is True
    assert (detail.desc, detail.prereq_raw, detail.attrs) == ("", "", "")


# --- not found ------------------------------------------------------------------------------------


def test_a_missing_code_is_a_value_not_an_exception(corpus):
    detail = tc.get_course("CS 999")
    assert detail.found is False
    assert detail.code == "CS 999"
    assert "2026-sp" in detail.message
    assert (detail.name, detail.credits, detail.desc, detail.section_count) == (None,) * 4


def test_a_query_that_is_not_a_code_is_a_value_too(corpus):
    detail = tc.get_course("machine learning")
    assert detail.found is False
    assert detail.code is None
    assert "not a course code" in detail.message


@pytest.mark.parametrize("junk", ["", "   ", "?", "CS", "999", "what is cs374 about", "CS 3745"])
def test_no_query_reaches_the_graph_as_an_exception(corpus, junk):
    assert tc.get_course(junk).found is False


# --- the schemas MCP will publish -----------------------------------------------------------------


def test_the_query_schema_names_the_spellings_it_accepts():
    schema = tc.CourseQuery.model_json_schema()
    assert schema["required"] == ["course_id"]
    assert "CS374" in schema["properties"]["course_id"]["description"]


def test_found_is_the_only_required_field_of_a_result():
    assert tc.CourseDetail.model_json_schema()["required"] == ["found"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: tc.CourseQuery(course_id="CS 374", limit=5),
        lambda: tc.CourseDetail(found=False, reason="no such course"),
    ],
)
def test_the_models_refuse_a_field_they_do_not_declare(call):
    # extra="forbid" on both: an MCP client that invents an argument, or a node that reads a field
    # this tool never promised, is a mistake worth a ValidationError rather than a silent no-op.
    with pytest.raises(ValidationError):
        call()


# --- the live corpus: #29's acceptance ------------------------------------------------------------


def test_live_cs374_comes_back_whole():
    detail = tc.get_course("cs 374")
    assert detail.found is True
    assert (detail.code, detail.credits, detail.term) == ("CS 374", 4.0, "2026-sp")
    # The AlmaBot frontend corpus cut desc at 240 characters and lost text for 3,255 of the 4,502
    # courses, CS 374 among them; removing that cut was what #3 was for.
    assert len(detail.desc) > 240
    assert detail.desc.endswith("Same as ECE 374. Prerequisite: One of CS 173, MATH 213; CS 225.")


def test_live_missing_course_is_a_value():
    detail = tc.get_course("CS 999")
    assert detail.found is False
    assert detail.code == "CS 999"
    assert "2026-sp" in detail.message


def test_live_corpus_indexes_every_course_under_one_term():
    index = tc.course_index()
    assert len(index.by_code) == 4502
    assert index.terms == ("2026-sp",)
