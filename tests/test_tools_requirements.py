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

#32 added the second half of this tool: ``requirements.json`` now carries two hand-encoded degree
programs, so "Computer Science, BS" is audited where it used to be refused, and the refusal moved
to the programs the file does not hold. The major tests run against the same fixture corpus, which
offers almost none of the courses a CS degree requires -- and that is the point of the last major
section: a requirement is credited from the *codes* a student gives, not from the catalog, or a
transcript full of courses this one term does not offer would audit as an empty one. The rule
language itself (``all_of`` / ``n_of`` / ``n_of_matching`` / ``advising``) is pinned in
``tests/test_majors.py``; what is pinned here is what the tool does with it.
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
from ingest import majors as mj

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
        "Aerospace Engineering, BS",
        "Statistics & Computer Science, BSLAS",  # a real degree, and not one of the two encoded
        "computer engineering",
        "Liberal Arts & Sciences",
        "my major",
        "gen",
        "education",
        "",
        "   ",
    ],
)
def test_a_program_this_corpus_has_no_requirements_for_is_refused(corpus, major):
    assert tr.check_requirement(major, ["STAT 100"]).supported is False


@pytest.mark.parametrize(
    "major",
    [
        "Computer Science, BS",
        "computer science",
        "CS",
        "cs",
        "cs major",
        "cs-bs",
        "Bachelor of Science in Computer Science",
    ],
)
def test_every_spelling_of_an_encoded_degree_reaches_it(corpus, major):
    # An alias list is how a router's free text finds a program. Matching is exact on the
    # normalized form, which is why "Statistics & Computer Science" above does not land here.
    status = tr.check_requirement(major, [])
    assert (status.supported, status.program) == (True, "Computer Science, BS")


def test_a_refusal_leaves_nothing_for_an_answer_node_to_paraphrase(corpus):
    # The whole point of the typed refusal: no requirements, satisfied or otherwise, so a fluent
    # answer node has no material to assemble a degree plan out of.
    status = tr.check_requirement("Aerospace Engineering, BS", TRANSCRIPT)
    assert (status.satisfied, status.unsatisfied, status.not_checked) == ([], [], [])
    assert (status.unknown_courses, status.catalog_year, status.program) == ([], None, None)
    assert status.major == "Aerospace Engineering, BS"


def test_a_refusal_says_what_is_missing_and_what_is_available(corpus):
    # #32 changed what "available" means: the refusal used to say this corpus has no degree
    # requirements at all, and now it has two -- a refusal that does not name them sends a student
    # away from an answer that exists.
    message = tr.check_requirement("Aerospace Engineering, BS", []).message
    assert "Aerospace Engineering, BS" in message
    assert "Computer Science, BS" in message
    assert "Statistics, BSLAS" in message
    assert "gened" in message
    assert "advisor" in message


def test_a_refusal_now_reads_the_rules_because_the_programs_are_data(tmp_path, monkeypatch):
    # Before #32 this tool could refuse a degree without any I/O: the answer was always no. Which
    # programs exist is now a property of the file, so it has to be read before anything can be
    # declined -- and a missing corpus must say so rather than refuse on the grounds that a
    # program it cannot see does not exist.
    monkeypatch.setattr(tr, "REQUIREMENTS_PATH", tmp_path / "gone.json")
    monkeypatch.setattr(tc, "COURSES_PATH", tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError, match="export_almabot"):
        tr.check_requirement("Aerospace Engineering, BS", ["CS 374"])


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
    # Pulling a code out of "intro psych" is #36's job; what this tool must not do is drop it.
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


def test_a_major_the_shared_checker_refuses_is_refused_here_too(corpus):
    # Before #32 any major in this file was refused outright. Now the file is read, and what
    # guards it is ingest.majors.check_majors -- the same function that guards the hand-written
    # source, so a document edited on disk cannot get past the tool by skipping the export.
    rewrite(corpus["requirements"], majors=[{"id": "cs-bs", "label": "Computer Science, BS"}])
    with pytest.raises(ValueError, match="missing"):
        tr.check_requirement("gened", [])


def test_a_corpus_with_no_degrees_at_all_still_refuses_in_a_sentence(corpus):
    # The pre-#32 state of this file, and the state of any export whose majors were dropped: the
    # refusal has no programs to offer and must not say "0 programs -- " and trail off.
    rewrite(corpus["requirements"], majors=[])
    message = tr.check_requirement("Computer Science, BS", []).message
    assert "no major or degree requirements at all" in message
    assert "gened" in message and "advisor" in message


def test_a_program_answering_to_gened_is_refused(corpus):
    # GENED_ALIASES is checked first, so such a program could never be reached -- and a student
    # asking for "general education" would get the campus audit while believing they asked for
    # the degree. ingest/majors.json has no way to know which spellings this module reserves.
    doc = json.loads(corpus["requirements"].read_text(encoding="utf-8"))
    hijacker = {**doc["majors"][0], "id": "hijack", "label": "Gen Ed, BS", "aliases": ["gened"]}
    rewrite(corpus["requirements"], majors=[hijacker])
    with pytest.raises(ValueError, match="could never be reached"):
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


# --- the degree audit #32 added ------------------------------------------------------------------


#: A transcript of the first two years of the CS degree, in the fixture corpus's own terms: none of
#: these courses are in it, which is exactly the case a degree audit has to get right.
CS_TRANSCRIPT = ["CS 124", "CS 128", "CS 173", "CS 222", "CS 225", "MATH 221", "MATH 231"]


def group_ids(statuses) -> list[str]:
    return [status.id for status in statuses]


def test_a_degree_audit_says_which_edition_of_the_requirements_it_used(corpus):
    # A student is held to the catalog year they enrolled under, and this is a hand transcription
    # of one page of one edition: an audit that does not name it cannot be checked by anybody.
    status = tr.check_requirement("Computer Science, BS", CS_TRANSCRIPT)
    assert (status.supported, status.program) == (True, "Computer Science, BS")
    assert status.major == "Computer Science, BS"
    assert status.catalog_year == "2025-2026"
    assert status.source.startswith("https://catalog.illinois.edu/")


def test_every_requirement_group_is_reported_exactly_once_in_the_files_order(corpus):
    status = tr.check_requirement("Computer Science, BS", CS_TRANSCRIPT)
    program = next(m for m in tr.requirements_doc()["majors"] if m["id"] == "cs-bs")
    listed = [g["id"] for g in program["groups"]]
    reported = group_ids(status.satisfied) + group_ids(status.unsatisfied)
    reported += group_ids(status.not_checked)
    assert sorted(reported) == sorted(listed)
    for half in (status.satisfied, status.unsatisfied, status.not_checked):
        assert group_ids(half) == [i for i in listed if i in group_ids(half)]


def test_a_requirement_a_course_list_cannot_decide_is_never_counted(corpus):
    # The honest third list. 128 hours, a 2.0 technical GPA, a language placement: reporting them
    # as satisfied would be a guess, and dropping them would read as "you are done".
    status = tr.check_requirement("Computer Science, BS", CS_TRANSCRIPT)
    assert "total-hours" in group_ids(status.not_checked)
    assert "total-hours" not in group_ids(status.satisfied) + group_ids(status.unsatisfied)
    for group in status.not_checked:
        assert (group.credited, group.still_needed, group.options) == ([], 0, [])
        assert group.note and group.source  # the catalog's own words, and where to read them


def test_a_partly_finished_requirement_names_what_is_left(corpus):
    status = tr.check_requirement("Computer Science, BS", CS_TRANSCRIPT)
    core = next(g for g in status.unsatisfied if g.id == "cs-core")
    assert core.credited == ["CS 124", "CS 128", "CS 173", "CS 222", "CS 225"]
    assert core.still_needed == 6
    assert core.options == ["CS 233", "CS 341", "CS 357", "CS 361", "CS 374", "CS 421"]
    assert core.group == "Computer Science Technical Core"


def test_a_requirement_with_one_course_left_to_choose_offers_the_alternatives(corpus):
    status = tr.check_requirement("Computer Science, BS", CS_TRANSCRIPT)
    linear = next(g for g in status.unsatisfied if g.id == "linear-algebra")
    assert (linear.credited, linear.still_needed) == ([], 1)
    assert linear.options == ["MATH 257", "MATH 415", "MATH 416"]


def test_a_course_this_catalog_does_not_offer_still_credits_the_requirement_that_names_it(corpus):
    # The decision #32 turns on. MATH 221 is required by both degrees and is not offered in
    # 2026-sp, so an audit that resolved requirements through the catalog index would tell a
    # student who has taken Calculus I to take it again -- wrong in the direction they cannot
    # check. It is credited, and it is still reported as outside the catalog.
    status = tr.check_requirement("Computer Science, BS", ["MATH 221"])
    assert group_ids(status.satisfied) == ["calculus-i"]
    assert next(g for g in status.satisfied if g.id == "calculus-i").satisfied_by == "MATH 221"
    assert status.unknown_courses == ["MATH 221"]


def test_an_entry_that_is_not_a_course_code_credits_nothing(corpus):
    status = tr.check_requirement("Computer Science, BS", ["intro to calculus"])
    assert status.satisfied == []
    assert status.unknown_courses == ["intro to calculus"]


def test_a_retake_listed_twice_is_credited_once(corpus):
    status = tr.check_requirement("Computer Science, BS", ["cs124", "CS 124", " CS  124 "])
    assert next(g for g in status.unsatisfied if g.id == "cs-core").credited == ["CS 124"]


def test_the_technical_electives_are_counted_by_pattern_and_refuse_the_core(corpus):
    # Six CS 400-level courses. CS 421 is in the technical core, so it cannot also be an elective;
    # CS 400 the catalog excludes by name; CS 225 is not 400-level at all.
    taken = ["CS 411", "CS 412", "CS 425", "CS 446", "CS 450", "CS 421", "CS 400", "CS 225"]
    status = tr.check_requirement("Computer Science, BS", taken)
    electives = next(g for g in status.unsatisfied if g.id == "technical-electives")
    assert electives.credited == ["CS 411", "CS 412", "CS 425", "CS 446", "CS 450"]
    assert electives.still_needed == 1
    # The answer to "what else counts" is 44 courses in this term, so the rule does not list them.
    assert electives.options == []
    # CS 411 is on the team project list, and the catalog lets it count for both.
    assert "technical-elective-team-project" in group_ids(status.satisfied)


def test_a_degree_audit_always_carries_its_caveats(corpus):
    message = tr.check_requirement("Computer Science, BS", ["MATH 221", "ZZZ 100"]).message
    assert "2025-2026" in message and "2026-sp" in message
    assert "catalog.illinois.edu" in message
    assert "8 of the 17 requirement groups" in message  # what it could not check
    assert "MATH 221" in message  # required, not offered this term
    assert "ZZZ 100" in message  # given, not in this catalog


def test_the_second_program_is_audited_from_its_own_page(corpus):
    status = tr.check_requirement("statistics", ["STAT 400", "STAT 410", "STAT 425", "STAT 426"])
    assert status.program == "Statistics, BSLAS"
    assert "statistics-bslas" in status.source
    assert "stat-core" in group_ids(status.satisfied)
    electives = next(g for g in status.unsatisfied if g.id == "advanced-electives")
    assert electives.still_needed == 4


def test_the_requirements_come_from_the_file_and_not_from_the_tool(corpus):
    # The same guard the Gen Ed rules have. A tool carrying its own copy of a degree would keep
    # answering from it after the catalog year turned over and the file was re-transcribed.
    invented = {
        "id": "demo-bs",
        "label": "Demonstration, BS",
        "college": "A College",
        "source": "https://example.edu/demo-bs/",
        "catalogYear": "2025-2026",
        "aliases": ["demo"],
        "groups": [
            {
                "id": "only",
                "label": "The only requirement",
                "group": "Everything",
                "rule": {"kind": "all_of", "courses": ["STAT 100"]},
                "source": "https://example.edu/demo-bs/",
                "catalogYear": "2025-2026",
                "courseCount": 1,
                "notOffered": [],
            }
        ],
    }
    rewrite(corpus["requirements"], majors=[invented])
    assert tr.check_requirement("Computer Science, BS", []).supported is False
    status = tr.check_requirement("demo", ["STAT 100"])
    assert (status.program, group_ids(status.satisfied)) == ("Demonstration, BS", ["only"])


def test_the_audit_agrees_with_the_shared_evaluator(corpus):
    # ingest.majors owns the rule language; this tool owns the walk over the groups. If the two
    # ever disagreed, the file would say one thing and the answer another.
    taken = ["CS 124", "CS 128", "MATH 221", "CS 411", "CS 412"]
    status = tr.check_requirement("Computer Science, BS", taken)
    reported = {g.id: (g.credited, g.still_needed) for g in status.satisfied + status.unsatisfied}
    program = next(m for m in tr.requirements_doc()["majors"] if m["id"] == "cs-bs")
    for group in program["groups"]:
        if group["rule"]["kind"] == "advising":
            continue
        credit = mj.credit_rule(group["rule"], taken)
        assert reported[group["id"]] == (credit.credited, credit.still_needed)


# --- the schemas MCP will publish -----------------------------------------------------------------


def test_the_query_schema_tells_a_router_that_a_major_cannot_be_audited():
    schema = tr.RequirementQuery.model_json_schema()
    assert schema["required"] == ["major"]
    assert "supported=false" in schema["properties"]["major"]["description"]
    assert "unknown_courses" in schema["properties"]["taken_courses"]["description"]


def test_a_result_requires_only_whether_it_is_supported_and_what_was_asked():
    assert tr.RequirementStatus.model_json_schema()["required"] == ["supported", "major"]


def test_the_query_schema_names_the_programs_a_router_may_ask_for():
    # #36's dispatch fills this from free text and this description is all the instruction it gets.
    # Before #32 it said no degree could be audited; saying so now would cost two real answers.
    description = tr.RequirementQuery.model_json_schema()["properties"]["major"]["description"]
    assert "Computer Science, BS" in description
    assert "Statistics, BSLAS" in description
    assert "must not be guessed at" in description


def test_the_fields_the_degree_audit_added_default_to_the_gen_ed_shape():
    # A Gen Ed category is one course and no citation, so every field #32 added has to be optional
    # or the 13 categories would have to carry empty values they cannot fill.
    category = tr.CategoryStatus(id="qr1", label="X", group="G")
    assert (category.credited, category.still_needed, category.options) == ([], 0, [])
    assert (category.note, category.source) == (None, None)
    status = tr.RequirementStatus(supported=True, major="gened")
    assert (status.not_checked, status.catalog_year, status.program) == ([], None, None)


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


def test_live_a_degree_this_corpus_does_not_carry_is_refused_by_name():
    status = tr.check_requirement("Aerospace Engineering, BS", ["CS 173", "CS 225", "CS 374"])
    assert status.supported is False
    assert (status.satisfied, status.unsatisfied, status.not_checked) == ([], [], [])
    assert "Aerospace Engineering, BS" in status.message


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


def test_live_a_second_year_cs_transcript_reads_the_way_the_program_page_does():
    # Everything the sample sequence puts in the first two years, minus the ethics course. The
    # catalog is one spring term and MATH 221 is not in it; the audit credits it anyway.
    # fmt: off
    taken = ["CS 124", "CS 128", "CS 173", "CS 222", "CS 225", "CS 233", "CS 341",
             "MATH 221", "MATH 231", "MATH 241", "MATH 257", "PHYS 211", "PHYS 212", "ENG 100"]
    # fmt: on
    status = tr.check_requirement("Computer Science, BS", taken)
    assert group_ids(status.satisfied) == [
        "orientation",
        "calculus-i",
        "calculus-sequence",
        "linear-algebra",
        "physics",
    ]
    assert group_ids(status.unsatisfied) == [
        "ethics",
        "cs-core",
        "technical-electives",
        "technical-elective-team-project",
    ]
    core = next(g for g in status.unsatisfied if g.id == "cs-core")
    assert (core.still_needed, core.options) == (4, ["CS 357", "CS 361", "CS 374", "CS 421"])
    assert status.unknown_courses == ["MATH 221"]


def test_live_the_gen_ed_half_of_a_degree_is_handed_back_as_its_own_question():
    # A degree page says "follows the campus General Education requirements" and stops. Rather
    # than fold 13 categories into a degree audit, the group says where the other half lives.
    status = tr.check_requirement("Computer Science, BS", ["RHET 105"])
    gened = next(g for g in status.not_checked if g.id == "gened")
    assert "gened" in gened.note
    # And that request is a different answer about the same transcript.
    assert tr.check_requirement("gened", ["RHET 105"]).satisfied[0].id == "comp1"


def test_live_a_degree_audit_credits_a_course_that_carries_a_gen_ed_too():
    # CS 225 is a technical core course and a Quantitative Reasoning II course. Both audits count
    # it, which is what the campus rule says: the degree does not spend the Gen Ed.
    degree = tr.check_requirement("Computer Science, BS", ["CS 225"])
    assert "CS 225" in next(g for g in degree.unsatisfied if g.id == "cs-core").credited
    assert credited(tr.check_requirement("gened", ["CS 225"])) == {"qr2": "CS 225"}


def test_live_the_refusal_names_exactly_the_programs_the_corpus_carries():
    labels = [m["label"] for m in tr.requirements_doc()["majors"]]
    assert labels == ["Computer Science, BS", "Statistics, BSLAS"]
    message = tr.check_requirement("Aerospace Engineering, BS", []).message
    for label in labels:
        assert label in message
