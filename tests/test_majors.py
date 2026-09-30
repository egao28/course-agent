"""Tests for ingest/majors.py and the hand-written ingest/majors.json it reads.

Three things are being pinned here, and they fail for different reasons.

The **encoding** is a small rule language -- ``all_of``, ``n_of``, ``n_of_matching``, ``advising``
-- and :func:`ingest.majors.check_majors` is the reason a hand-written file can be trusted: every
check it makes is a wrong audit it prevents, so each one has a test that shows the wrong audit.

The **evaluator** is :func:`ingest.majors.credit_rule`, and it runs on course *codes* rather than
catalog records. That is the load-bearing decision of #32: ``MATH 221`` is required by both
programs and is not offered in 2026-sp, so an evaluator that went through the catalog index would
tell a student who has taken it that they still need it.

The **data** is two real degrees transcribed from two real catalog pages, and the tests at the
bottom run against the live ``data/courses.json`` rather than a fixture -- a transcription error
that renames a course is exactly what a fixture cannot see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest import export_almabot as ex
from ingest import majors as mj

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"

#: A minimal well-formed program, deep-copied per test by :func:`program`.
TEMPLATE = {
    "id": "demo-bs",
    "label": "Demonstration, BS",
    "college": "A College",
    "source": "https://example.edu/demo-bs/",
    "aliases": ["demo"],
    "groups": [
        {
            "id": "core",
            "label": "The core",
            "group": "Core",
            "rule": {"kind": "all_of", "courses": ["CS 124", "CS 128"]},
        }
    ],
}


@pytest.fixture
def program() -> dict:
    return json.loads(json.dumps(TEMPLATE))


def check(program: dict) -> None:
    mj.check_majors([program], "ingest/majors.json")


# --- the rule language ---------------------------------------------------------------------------


def test_all_of_wants_every_course_and_names_the_ones_still_missing():
    rule = {"kind": "all_of", "courses": ["CS 124", "CS 128", "CS 173"]}
    credit = mj.credit_rule(rule, ["CS 128"])
    assert credit.credited == ["CS 128"]
    assert credit.still_needed == 2
    # In the catalog's own order, not the transcript's: a student reads the program page.
    assert credit.options == ["CS 124", "CS 173"]


def test_a_finished_requirement_offers_no_options():
    # "You have finished this, and here are three more courses that would finish it" is noise at
    # best and reads as a warning at worst.
    credit = mj.credit_rule({"kind": "all_of", "courses": ["CS 124"]}, ["CS 124"])
    assert (credit.still_needed, credit.options) == (0, [])


def test_n_of_takes_any_one_of_the_alternatives():
    rule = {"kind": "n_of", "n": 1, "courses": ["MATH 257", "MATH 415", "MATH 416"]}
    for taken in ("MATH 257", "MATH 415", "MATH 416"):
        assert mj.credit_rule(rule, [taken]).still_needed == 0


def test_every_alternative_taken_is_credited_even_past_the_number_required():
    # A student who took both CS 210 and CS 211 took both. Reporting one would look like the audit
    # had lost a course, and a student who cannot see their course in an audit re-takes it.
    rule = {"kind": "n_of", "n": 1, "courses": ["CS 210", "CS 211"]}
    credit = mj.credit_rule(rule, ["CS 211", "CS 210"])
    assert credit.credited == ["CS 210", "CS 211"]
    assert credit.still_needed == 0


def test_n_of_matching_counts_courses_by_pattern_and_honours_both_lists():
    rule = {
        "kind": "n_of_matching",
        "n": 2,
        "match": {
            "subjects": ["CS"],
            "numberMin": 400,
            "numberMax": 499,
            "include": ["CS 397"],
            "exclude": ["CS 400", "CS 421"],
        },
    }
    credit = mj.credit_rule(rule, ["CS 397", "CS 411", "CS 421", "CS 400", "CS 225", "MATH 415"])
    # CS 421 and CS 400 are excluded (one by the catalog, one because the core already requires it),
    # CS 225 and MATH 415 fall outside the pattern, CS 397 is included by name.
    assert credit.credited == ["CS 397", "CS 411"]
    assert credit.still_needed == 0
    # The answer to "what else counts?" is 44 courses, so the rule does not pretend to list them.
    assert credit.options == []


def test_an_advising_rule_is_not_evaluated_at_all():
    assert mj.credit_rule({"kind": "advising"}, ["CS 124"]) == mj.Credit([], 0, [])


def test_all_of_is_n_of_over_its_whole_list():
    assert mj.required_count({"kind": "all_of", "courses": ["CS 124", "CS 128"]}) == 2
    assert mj.required_count({"kind": "n_of", "n": 1, "courses": ["CS 124", "CS 128"]}) == 1


def test_a_program_name_is_normalized_before_it_is_matched():
    assert mj.normalize_program("  Computer Science, BS ") == "computer science bs"
    assert mj.normalize_program("Gen-Ed") == "gen ed"


# --- what check_majors refuses, and the wrong audit each refusal prevents -------------------------


def test_a_rule_kind_with_no_evaluator_is_refused(program):
    # credit_rule would return nothing for it and the group would report as satisfied: a
    # requirement quietly met because nobody implemented it.
    program["groups"][0]["rule"] = {"kind": "most_of", "courses": ["CS 124"]}
    with pytest.raises(ValueError, match="no evaluator implements"):
        check(program)


def test_a_requirement_no_transcript_can_satisfy_is_refused(program):
    program["groups"][0]["rule"] = {"kind": "n_of", "n": 3, "courses": ["CS 124", "CS 128"]}
    with pytest.raises(ValueError, match="no transcript"):
        check(program)


def test_a_requirement_every_transcript_satisfies_is_refused(program):
    program["groups"][0]["rule"] = {"kind": "n_of", "n": 0, "courses": ["CS 124"]}
    with pytest.raises(ValueError, match="every transcript satisfies"):
        check(program)


def test_a_course_code_outside_the_catalogs_spelling_is_refused(program):
    # "CS124" matches nothing a student has taken, because the transcript is normalized to the
    # catalog's spelling before it gets here -- so the audit would report a course they hold as
    # still needed, which is the failure that makes a wrong audit worse than no audit.
    program["groups"][0]["rule"]["courses"] = ["CS124"]
    with pytest.raises(ValueError, match="not a course code"):
        check(program)


def test_a_course_named_twice_in_one_rule_is_refused(program):
    program["groups"][0]["rule"]["courses"] = ["CS 124", "CS 124"]
    with pytest.raises(ValueError, match="twice"):
        check(program)


def test_an_exclusion_the_pattern_never_matched_is_refused(program):
    # An exclusion that excludes nothing is a typo, and the course it meant to exclude is still
    # counting toward the requirement.
    program["groups"][0]["rule"] = {
        "kind": "n_of_matching",
        "n": 1,
        "match": {"subjects": ["CS"], "numberMin": 400, "numberMax": 499, "exclude": ["MATH 415"]},
    }
    with pytest.raises(ValueError, match="never matched"):
        check(program)


def test_an_empty_number_range_is_refused(program):
    program["groups"][0]["rule"] = {
        "kind": "n_of_matching",
        "n": 1,
        "match": {"subjects": ["CS"], "numberMin": 499, "numberMax": 400},
    }
    with pytest.raises(ValueError, match="empty range"):
        check(program)


def test_an_advising_requirement_with_nothing_to_say_is_refused(program):
    # It would be reported as unchecked without saying what it is, which is worse than silence.
    program["groups"][0]["rule"] = {"kind": "advising"}
    program["groups"][0].pop("note", None)
    with pytest.raises(ValueError, match="advising requirement with no note"):
        check(program)


def test_a_group_missing_a_field_the_audit_reports_is_refused(program):
    program["groups"][0].pop("label")
    with pytest.raises(ValueError, match="missing"):
        check(program)


def test_a_repeated_group_id_is_refused(program):
    program["groups"].append(dict(program["groups"][0]))
    with pytest.raises(ValueError, match="appears twice"):
        check(program)


def test_two_programs_answering_to_one_name_are_refused(program):
    other = json.loads(json.dumps(program))
    other["id"] = "demo-ba"
    other["label"] = "Demonstration, BA"
    with pytest.raises(ValueError, match="depend on the order"):
        mj.check_majors([program, other], "ingest/majors.json")


def test_two_programs_sharing_an_id_are_refused(program):
    twin = json.loads(json.dumps(program))
    twin["aliases"] = ["other"]
    twin["label"] = "Something Else, BS"
    with pytest.raises(ValueError, match="share the id"):
        mj.check_majors([program, twin], "ingest/majors.json")


def test_a_program_is_reachable_by_its_label_and_its_id_as_well_as_its_aliases(program):
    # check_majors treats all three as names, so uniqueness covers all three.
    program["aliases"] = ["demonstration bs"]  # already the normalized label
    check(program)  # the same program answering twice to one name is fine
    other = json.loads(json.dumps(program))
    other.update(id="demo-ba", label="Other, BA", aliases=["demo bs"])
    with pytest.raises(ValueError, match="depend on the order"):
        mj.check_majors([program, other], "ingest/majors.json")


# --- the exported copy ----------------------------------------------------------------------------


def test_every_group_carries_the_page_and_year_it_was_read_off():
    # The issue's own requirement: an audit that cannot be cited is one a student has to take on
    # faith, and this is a file a human transcribed.
    built = mj.build_majors(ex.read_courses(FIXTURE_CSV).courses)
    for major in built:
        assert major["catalogYear"]
        for group in major["groups"]:
            assert group["source"].startswith("https://catalog.illinois.edu/")
            assert group["catalogYear"] == major["catalogYear"]


def test_a_groups_course_count_and_missing_courses_are_measured_against_the_catalog():
    courses = [{"code": "CS 124"}, {"code": "CS 411"}]
    doc = {
        "catalogYear": "2025-2026",
        "source": "https://example.edu/",
        "majors": [
            {
                **TEMPLATE,
                "groups": [
                    {
                        "id": "core",
                        "label": "The core",
                        "group": "Core",
                        "rule": {"kind": "all_of", "courses": ["CS 124", "CS 128"]},
                    },
                    {
                        "id": "electives",
                        "label": "Electives",
                        "group": "Core",
                        "rule": {
                            "kind": "n_of_matching",
                            "n": 1,
                            "match": {
                                "subjects": ["CS"],
                                "numberMin": 400,
                                "numberMax": 499,
                                "exclude": ["CS 400"],
                            },
                        },
                    },
                ],
            }
        ],
    }
    core, electives = mj.build_majors(courses, doc)[0]["groups"]
    assert (core["courseCount"], core["notOffered"]) == (1, ["CS 128"])
    # A pattern's count is how many courses in the catalog match it, and its exclusions are not
    # courses the degree is missing -- CS 400 is one the requirement refuses.
    assert (electives["courseCount"], electives["notOffered"]) == (1, [])


def test_a_group_the_catalog_has_no_course_for_stops_the_export():
    doc = {
        "catalogYear": "2025-2026",
        "source": "https://example.edu/",
        "majors": [json.loads(json.dumps(TEMPLATE))],
    }
    built = mj.build_majors([{"code": "MATH 231"}], doc)
    with pytest.raises(ValueError, match="renumbered program"):
        mj.check_coverage(built, "data/requirements.json")


def test_a_missing_course_is_not_a_missing_group():
    # One course absent is the one-term corpus and must not stop an export; all of them absent is
    # an encoding that has rotted.
    doc = {
        "catalogYear": "2025-2026",
        "source": "https://example.edu/",
        "majors": [json.loads(json.dumps(TEMPLATE))],
    }
    built = mj.build_majors([{"code": "CS 124"}], doc)
    mj.check_coverage(built, "data/requirements.json")
    assert built[0]["groups"][0]["notOffered"] == ["CS 128"]


def test_an_advising_group_is_not_asked_for_courses():
    doc = {
        "catalogYear": "2025-2026",
        "source": "https://example.edu/",
        "majors": [
            {
                **TEMPLATE,
                "groups": [
                    {
                        "id": "hours",
                        "label": "128 hours",
                        "group": "Graduation Requirements",
                        "rule": {"kind": "advising"},
                        "note": "Hours are not in this data.",
                    }
                ],
            }
        ],
    }
    built = mj.build_majors([], doc)
    mj.check_coverage(built, "data/requirements.json")  # zero courses, and that is correct
    assert built[0]["groups"][0]["courseCount"] == 0


# --- the two programs actually in the file ------------------------------------------------------


def test_the_source_file_is_valid():
    doc = mj.load_majors()
    assert [m["id"] for m in doc["majors"]] == ["cs-bs", "statistics-bslas"]
    assert doc["catalogYear"] == "2025-2026"


def test_every_program_cites_a_catalog_page_for_the_year_it_claims():
    doc = mj.load_majors()
    for major in doc["majors"]:
        assert doc["catalogYear"] in major["source"], major["id"]


def test_every_requirement_is_either_checkable_or_explained():
    # The split this data lives or dies on: a group is a course list, or it says in words why it
    # is not one. There is no third state in which the audit quietly decides something.
    for major in mj.load_majors()["majors"]:
        for group in major["groups"]:
            if group["rule"]["kind"] == "advising":
                assert len(group.get("note", "")) > 40, (major["id"], group["id"])
            else:
                assert mj.named_codes(group["rule"]) or group["rule"]["kind"] == "n_of_matching"


def test_the_cs_degree_reads_the_way_the_catalog_prints_it():
    cs = next(m for m in mj.load_majors()["majors"] if m["id"] == "cs-bs")
    groups = {g["id"]: g for g in cs["groups"]}
    # fmt: off
    assert groups["cs-core"]["rule"]["courses"] == [
        "CS 124", "CS 128", "CS 173", "CS 222", "CS 225", "CS 233",
        "CS 341", "CS 357", "CS 361", "CS 374", "CS 421",
    ]
    # fmt: on
    assert groups["technical-electives"]["rule"]["n"] == 6
    # CS 421 is excluded from the electives although the catalog does not say so: it is in the core
    # above, and a course counts once.
    assert "CS 421" in groups["technical-electives"]["rule"]["match"]["exclude"]


def test_live_every_requirement_group_still_has_a_course_in_this_catalog():
    # The standing guard against a rotted transcription, run over the real corpus: a group whose
    # courses have all vanished is what a renumbered program looks like (CS 125 became CS 124 and
    # CS 128, and no field in the data announced it).
    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    mj.check_coverage(mj.build_majors(courses), "data/courses.json")


def test_live_the_courses_this_term_does_not_offer_are_the_ones_we_expect():
    # Not a bug and not a typo: the corpus is one spring term and these run in the fall. Pinned so
    # that a *new* absence has to be looked at rather than absorbed.
    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    absent = {
        code
        for major in mj.build_majors(courses)
        for group in major["groups"]
        for code in group["notOffered"]
    }
    # fmt: off
    assert absent == {
        "CS 428", "CS 429", "CS 465", "CS 467", "ENG 300", "MATH 221", "STAT 424", "STAT 480",
    }
    # fmt: on
