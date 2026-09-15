"""Tests for agent/tools/requirements.py: the Gen Ed audit, and the refusal that guards the rest.

Two corpora. The audit runs against the 17 courses ``tests/fixtures/catalog_sample.csv`` produces
through the real exporter -- it carries both Quantitative Reasoning spellings, a course in two
categories at once, and three courses in the same one, which is every shape the audit has to get
right -- with the catalog and the rules redirected into ``tmp_path``. The last section reads the
live corpus, because #31's acceptance is stated about real courses and one of them (``MATH 221``)
is the unknown-course case for real: it is not offered in 2026-sp.

Nothing here asserts that a particular category is *wrong*. Open #23 fixes ``socbeh``'s rule in
``requirements.json`` (142 courses to 24), and this branch has the pre-#23 data, so the one thing
worth pinning is that the tool agrees with the exporter's evaluator whichever rule is on disk --
``test_the_audit_agrees_with_the_exporters_evaluator`` does that, and survives the merge.

``fresh_cache`` is autouse and clears all three tool caches: the audit reads the catalog index for
every course in the transcript, so a test that left ``courses``'s cache behind would audit the next
one's transcript against the wrong corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.tools import courses as tc
from agent.tools import prereqs as tp
from agent.tools import requirements as tr
from ingest import export_almabot as ex

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"

#: A transcript that reaches eight of the fixture's categories, and one course it does not have.
TRANSCRIPT = ["STAT 100", "MATH 101", "HIST 200", "AAS 100", "ANTH 103"]


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits the rules -- or the catalog index -- another one read."""
    for module in (tr, tp, tc):
        module.clear_cache()
    yield
    for module in (tr, tp, tc):
        module.clear_cache()


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV).courses


@pytest.fixture
def corpus(tmp_path, courses, monkeypatch) -> dict[str, Path]:
    """Write the catalog and the rules where the two modules will look for them."""
    paths = {"courses": tmp_path / "courses.json", "requirements": tmp_path / "requirements.json"}
    ex.write_json(paths["courses"], courses)
    rules = ex.build_requirements(courses, {"data": {"term": "2026-sp"}})
    ex.write_json(paths["requirements"], rules)
    monkeypatch.setattr(tc, "COURSES_PATH", paths["courses"])
    monkeypatch.setattr(tr, "REQUIREMENTS_PATH", paths["requirements"])
    return paths


def credited(status: tr.RequirementStatus) -> dict[str, str | None]:
    """``{category id: the course credited}`` for the satisfied half, which is most assertions."""
    return {c.id: c.satisfied_by for c in status.satisfied}


def rewrite(path: Path, **changes) -> None:
    """Rewrite the requirements document with ``changes`` applied, as a stale export would be."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    ex.write_json(path, {**doc, **changes})


# --- which programs are auditable at all ------------------------------------------------------


@pytest.mark.parametrize(
    "major",
    [
        "gened",
        "GenEd",
        "  gened  ",
        "gen ed",
        "Gen-Ed",
        "gen eds",
        "general education",
        "General Education",
        "general education requirements",
        "general ed",
    ],
)
def test_every_spelling_of_general_education_is_audited(corpus, major):
    status = tr.check_requirement(major, ["STAT 100"])
    assert status.supported is True
    assert credited(status) == {"qr1": "STAT 100"}


@pytest.mark.parametrize(
    "major",
    [
        "Computer Science, BS",
        "computer science",
        "CS",
        "cs major",
        "Liberal Arts & Sciences",
        "my major",
        "gen",
        "education",
        "",
        "   ",
    ],
)
def test_anything_that_is_not_general_education_is_refused(corpus, major):
    assert tr.check_requirement(major, ["STAT 100"]).supported is False


def test_a_refusal_leaves_nothing_for_an_answer_node_to_paraphrase(corpus):
    # The whole point of the typed refusal: no categories, satisfied or otherwise, so a fluent
    # answer node has no material to assemble a degree plan out of.
    status = tr.check_requirement("Computer Science, BS", TRANSCRIPT)
    assert (status.satisfied, status.unsatisfied, status.unknown_courses) == ([], [], [])
    assert status.major == "Computer Science, BS"


def test_a_refusal_says_what_is_missing_and_what_is_available(corpus):
    message = tr.check_requirement("Computer Science, BS", []).message
    assert "Computer Science, BS" in message
    assert "no major or degree requirements" in message
    assert "gened" in message
    assert "advisor" in message


def test_a_refusal_never_reads_the_rules(tmp_path, monkeypatch):
    # Refused before any I/O, so a program this tool cannot audit is declined the same way whether
    # or not the corpus is even built -- and the refusal can never fail with a FileNotFoundError.
    monkeypatch.setattr(tr, "REQUIREMENTS_PATH", tmp_path / "gone.json")
    monkeypatch.setattr(tc, "COURSES_PATH", tmp_path / "gone.json")
    assert tr.check_requirement("Computer Science, BS", ["CS 374"]).supported is False


@pytest.mark.parametrize(
    ("major", "expected"),
    [("Gen-Ed", "gen ed"), ("  GENERAL  EDUCATION  ", "general education"), ("CS, BS", "cs bs")],
)
def test_normalize_program_collapses_case_and_punctuation(major, expected):
    assert tr.normalize_program(major) == expected


# --- the audit ----------------------------------------------------------------------------------


def test_every_category_is_reported_exactly_once_in_the_order_the_file_lists_them(corpus):
    status = tr.check_requirement("gened", TRANSCRIPT)
    listed = [c["id"] for c in tr.requirements_doc()["genEd"]]
    satisfied = [c.id for c in status.satisfied]
    unsatisfied = [c.id for c in status.unsatisfied]
    assert sorted(satisfied + unsatisfied) == sorted(listed)
    assert len(listed) == 13
    # Both halves keep the file's order, which is gened.ts's, so an answer can walk them by group.
    assert satisfied == [i for i in listed if i in satisfied]
    assert unsatisfied == [i for i in listed if i in unsatisfied]


def test_a_category_carries_the_label_and_group_the_file_gives_it(corpus):
    status = tr.check_requirement("gened", ["STAT 100"])
    qr1 = next(c for c in status.satisfied if c.id == "qr1")
    assert (qr1.label, qr1.group) == ("Quantitative Reasoning I", "Quantitative Reasoning")


def test_the_transcript_credits_each_category_to_a_course(corpus):
    status = tr.check_requirement("gened", TRANSCRIPT)
    assert credited(status) == {
        "qr1": "STAT 100",
        "qr2": "MATH 101",
        "humhist": "HIST 200",
        "socsoc": "AAS 100",
        "socbeh": "AAS 100",
        "cultwest": "HIST 200",
        "cultnonwest": "ANTH 103",
        "cultus": "AAS 100",
    }
    missing = ["comp1", "advcomp", "natphys", "natlife", "humlit"]
    assert [c.id for c in status.unsatisfied] == missing


def test_one_course_satisfies_every_category_it_is_designated_for(corpus):
    # AAS 100 is "Social & Beh Sci - Soc Sci, and Cultural Studies - US Minority". The campus rule
    # lets one course carry both, so the audit does too -- gened.ts resolved each category alone.
    status = tr.check_requirement("gened", ["AAS 100"])
    assert set(credited(status)) >= {"socsoc", "cultus"}
    assert set(credited(status).values()) == {"AAS 100"}


def test_the_first_course_listed_is_the_one_credited(corpus):
    # Three fixture courses carry Soc Sci. The transcript's own order decides, like genEdStatus's
    # `completed.find`; re-ranking it would be a recommendation wearing an audit's clothes.
    assert credited(tr.check_requirement("gened", ["ECON 102", "AAS 100"]))["socsoc"] == "ECON 102"
    assert credited(tr.check_requirement("gened", ["AAS 100", "ECON 102"]))["socsoc"] == "AAS 100"


def test_the_none_clause_of_a_rule_is_evaluated(corpus):
    # "Quantitative Reasoning II" contains "Quantitative Reasoning I" as a substring; qr1's rule
    # excludes it. Evaluating `any` alone would credit MATH 101 with both.
    assert credited(tr.check_requirement("gened", ["MATH 101"])) == {"qr2": "MATH 101"}


def test_an_empty_transcript_is_an_answer(corpus):
    status = tr.check_requirement("gened", [])
    assert (status.supported, status.satisfied, status.unknown_courses) == (True, [], [])
    assert len(status.unsatisfied) == 13
    assert status.message is None


def test_the_transcript_argument_is_optional(corpus):
    assert len(tr.check_requirement("gened").unsatisfied) == 13


def test_a_course_with_no_degree_attributes_satisfies_nothing(corpus):
    status = tr.check_requirement("gened", ["CS 374", "CS 225"])
    assert (status.satisfied, status.unknown_courses) == ([], [])
    assert len(status.unsatisfied) == 13


def test_a_taken_course_is_normalized_the_way_the_lookup_normalizes_it(corpus):
    for spelling in ("stat 100", "STAT100", "  stat100  "):
        assert credited(tr.check_requirement("gened", [spelling])) == {"qr1": "STAT 100"}


def test_a_course_listed_twice_is_credited_once(corpus):
    status = tr.check_requirement("gened", ["STAT 100", "stat100"])
    assert credited(status) == {"qr1": "STAT 100"}


# --- courses the catalog does not have ------------------------------------------------------------


def test_a_course_outside_the_catalog_is_reported_not_dropped(corpus):
    status = tr.check_requirement("gened", ["STAT 100", "MATH 221"])
    assert status.unknown_courses == ["MATH 221"]
    assert credited(status) == {"qr1": "STAT 100"}


def test_a_string_that_is_not_a_code_is_reported_too(corpus):
    # Pulling a code out of "intro psych" is #35's job; what this tool must not do is drop it.
    assert tr.check_requirement("gened", ["intro psych"]).unknown_courses == ["intro psych"]


def test_unknown_courses_keep_their_order_and_appear_once(corpus):
    status = tr.check_requirement("gened", ["MATH 221", "ZZZ 100", "math221", "MATH 221"])
    assert status.unknown_courses == ["MATH 221", "ZZZ 100"]


def test_a_blank_entry_names_nothing_and_is_dropped(corpus):
    status = tr.check_requirement("gened", ["", "   ", "STAT 100"])
    assert status.unknown_courses == []
    assert credited(status) == {"qr1": "STAT 100"}


def test_the_message_names_the_unrecognized_courses_and_the_term(corpus):
    message = tr.check_requirement("gened", ["MATH 221", "ZZZ 100"]).message
    assert "MATH 221, ZZZ 100" in message
    assert "2026-sp" in message
    assert "not counted" in message


def test_the_message_is_none_when_every_course_was_recognized(corpus):
    assert tr.check_requirement("gened", ["STAT 100"]).message is None


# --- the rules the audit reads --------------------------------------------------------------------


def test_the_rules_come_from_the_file_and_not_from_the_constant(corpus):
    # #3 turned the 13 rules into data so there would be one copy. A tool that transcribed them
    # again would keep answering from its own copy after an export changed them -- this fails if
    # anything here stops reading the file.
    rewrite(
        corpus["requirements"],
        genEd=[
            {
                "id": "onlyone",
                "label": "The Only Category",
                "group": "Invented",
                "match": {"field": "attrs", "any": ["Quantitative Reasoning I"], "none": []},
                "courseCount": 1,
            }
        ],
    )
    status = tr.check_requirement("gened", ["STAT 100", "MATH 101"])
    assert credited(status) == {"onlyone": "STAT 100"}
    assert status.unsatisfied == []


def test_the_audit_agrees_with_the_exporters_evaluator(corpus, courses):
    # The standing guard for the socbeh divergence. On this branch ECON 102 satisfies socbeh (the
    # catalog's "Social & Beh Sci - Soc Sci" prefix contains the "Beh Sci" substring the pre-#23
    # rule tests); after #23's anchored rule lands in requirements.json it will not. Either way the
    # tool must report exactly what matches_attrs says about the rule on disk.
    by_code = {c["code"]: c for c in courses}
    for code in by_code:
        expected = {
            category["id"]
            for category in tr.requirements_doc()["genEd"]
            if ex.matches_attrs(category, by_code[code]["attrs"])
        }
        assert set(credited(tr.check_requirement("gened", [code]))) == expected


def test_a_missing_requirements_file_points_at_the_export(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "REQUIREMENTS_PATH", tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError, match="export_almabot"):
        tr.load_requirements()


def test_a_document_with_no_categories_is_refused(corpus):
    # Would otherwise report an empty audit: nothing satisfied, nothing missing, nothing left to do.
    rewrite(corpus["requirements"], genEd=[])
    with pytest.raises(ValueError, match="no Gen Ed categories"):
        tr.check_requirement("gened", ["STAT 100"])


def test_a_category_missing_a_field_the_result_reports_is_refused(corpus):
    rewrite(corpus["requirements"], genEd=[{"id": "x", "label": "X", "group": "G"}])
    with pytest.raises(ValueError, match="missing"):
        tr.check_requirement("gened", [])


def test_a_match_rule_missing_a_clause_is_refused(corpus):
    # matches_attrs reads field / any / none. A rule short of one of them would raise a KeyError
    # halfway through an audit, which is a stack trace where a sentence belongs.
    doc = json.loads(corpus["requirements"].read_text(encoding="utf-8"))
    broken = {**doc["genEd"][0], "match": {"field": "attrs", "any": ["Composition I"]}}
    rewrite(corpus["requirements"], genEd=[broken])
    with pytest.raises(ValueError, match="match rule missing"):
        tr.check_requirement("gened", [])


def test_a_repeated_category_id_is_refused(corpus):
    doc = json.loads(corpus["requirements"].read_text(encoding="utf-8"))
    rewrite(corpus["requirements"], genEd=[*doc["genEd"], doc["genEd"][0]])
    with pytest.raises(ValueError, match="appears twice"):
        tr.check_requirement("gened", [])


def test_a_rule_on_a_field_the_lookup_does_not_return_is_refused(corpus):
    # A rule against a field CourseDetail has no attribute for would match nothing and report the
    # category unsatisfied for every student, which looks exactly like an honest answer.
    doc = json.loads(corpus["requirements"].read_text(encoding="utf-8"))
    broken = {**doc["genEd"][0], "match": {"field": "degreeAttributes", "any": ["x"], "none": []}}
    rewrite(corpus["requirements"], genEd=[broken, *doc["genEd"][1:]])
    with pytest.raises(ValueError, match="does not return"):
        tr.check_requirement("gened", [])


def test_major_requirements_in_the_file_are_refused_loudly(corpus):
    # #32's data arriving is the day this tool's refusal becomes a lie. Fail then, rather than keep
    # telling a student their program cannot be audited while the file on disk holds it.
    rewrite(corpus["requirements"], majors=[{"id": "cs-bs", "label": "Computer Science, BS"}])
    with pytest.raises(ValueError, match="#32"):
        tr.check_requirement("gened", [])


def test_rules_from_another_export_are_refused(corpus):
    rewrite(corpus["requirements"], term="2025-fa")
    with pytest.raises(ValueError, match="different runs"):
        tr.check_requirement("gened", [])


def test_load_requirements_reads_the_path_it_is_given_and_does_not_cache(corpus, tmp_path):
    other = tmp_path / "other.json"
    rule = {"field": "attrs", "any": ["x"], "none": []}
    doc = {"term": "2026-sp", "genEd": [{"id": "x", "label": "X", "group": "G", "match": rule}]}
    ex.write_json(other, {**doc, "majors": []})
    assert [c["id"] for c in tr.load_requirements(other)["genEd"]] == ["x"]
    assert len(tr.requirements_doc()["genEd"]) == 13


def test_the_rules_are_read_once(corpus, monkeypatch):
    reads: list = []
    real = tr.load_requirements
    monkeypatch.setattr(
        tr, "load_requirements", lambda path=None: (reads.append(path), real(path))[1]
    )

    for major in ("gened", "general education", "gened"):
        tr.check_requirement(major, ["STAT 100"])
    assert len(reads) == 1

    tr.clear_cache()
    tr.check_requirement("gened", ["STAT 100"])
    assert len(reads) == 2


# --- the schemas MCP will publish -----------------------------------------------------------------


def test_the_query_schema_tells_a_router_that_a_major_cannot_be_audited():
    schema = tr.RequirementQuery.model_json_schema()
    assert schema["required"] == ["major"]
    assert "supported=false" in schema["properties"]["major"]["description"]
    assert "unknown_courses" in schema["properties"]["taken_courses"]["description"]


def test_a_result_requires_only_whether_it_is_supported_and_what_was_asked():
    assert tr.RequirementStatus.model_json_schema()["required"] == ["supported", "major"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: tr.RequirementQuery(major="gened", taken=[]),
        lambda: tr.CategoryStatus(id="qr1", label="X", group="G", satisfiedBy=None),
        lambda: tr.RequirementStatus(supported=True, major="gened", missing=[]),
    ],
)
def test_the_models_refuse_a_field_they_do_not_declare(call):
    with pytest.raises(ValidationError):
        call()


def test_a_transcript_that_is_not_a_list_of_strings_is_refused(corpus):
    with pytest.raises(ValidationError):
        tr.check_requirement("gened", [374])


# --- the live corpus: #31's acceptance ------------------------------------------------------------


def test_live_the_acceptance_query_credits_composition_and_flags_the_rest():
    status = tr.check_requirement("gened", ["RHET 105", "MATH 221"])
    assert status.supported is True
    assert credited(status) == {"comp1": "RHET 105"}
    assert len(status.unsatisfied) == 12
    # MATH 221 is not a typo: the catalog is one term, and it is not offered in 2026-sp.
    assert status.unknown_courses == ["MATH 221"]
    assert "MATH 221" in status.message


def test_live_a_degree_program_is_refused_by_name():
    status = tr.check_requirement("Computer Science, BS", ["CS 173", "CS 225", "CS 374"])
    assert status.supported is False
    assert (status.satisfied, status.unsatisfied) == ([], [])
    assert "Computer Science, BS" in status.message


def test_live_every_category_is_reachable():
    # Each of the 13 has at least one course in this term that satisfies it -- so an unsatisfied
    # category is always something a student could still take, not a dead rule.
    by_code = tc.course_index().by_code
    for category in tr.requirements_doc()["genEd"]:
        code = next(c for c in by_code if ex.matches_attrs(category, by_code[c]["attrs"]))
        assert category["id"] in credited(tr.check_requirement("gened", [code]))


def test_live_one_course_can_carry_three_categories():
    # IS 145 is "Quantitative Reasoning II, Social & Beh Sci - Soc Sci, and Cultural Studies -
    # US Minority": three designations, three categories, one course.
    assert set(credited(tr.check_requirement("gened", ["IS 145"]))) >= {"qr2", "socsoc", "cultus"}


def test_live_a_first_year_transcript_leaves_the_right_categories_open():
    status = tr.check_requirement(
        "general education", ["RHET 105", "MATH 231", "CS 225", "PHIL 105", "MUS 133"]
    )
    # CS 225 is "Quantitative Reasoning II course." -- a major requirement that carries a Gen Ed
    # designation too, and the audit credits it. Naming what satisfied each category is the point:
    # a bare count would leave a student re-taking a requirement they already hold.
    assert credited(status) == {
        "comp1": "RHET 105",
        "qr1": "MATH 231",
        "qr2": "CS 225",
        "humhist": "PHIL 105",
        "humlit": "MUS 133",
        "cultnonwest": "MUS 133",
    }
    assert [c.id for c in status.unsatisfied] == [
        "advcomp",
        "natphys",
        "natlife",
        "socsoc",
        "socbeh",
        "cultwest",
        "cultus",
    ]
    assert status.unknown_courses == []
