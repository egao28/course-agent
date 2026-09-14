"""Tests for ingest/export_almabot.py: CSV de-duplication, the three data files, the Gen Ed rules.

Everything runs against ``tests/fixtures/catalog_sample.csv`` -- a 20-row synthetic catalog with
the real column layout -- so the suite never needs the private course-catalog.csv, and tuning the
live ``config.yaml`` cannot break it. The fixture deliberately contains a description longer than
the 240 chars AlmaBot truncates at, an empty description, a cross-reference-only description,
dangling and self-referencing prerequisites, a requirements block whose sentence break falls inside
"Ph.D.", stated and hedged corequisites, both Quantitative Reasoning spellings, and rows with no
Subject/Number.

``redirect_outputs`` is autouse: no test can write into the repo's ``data/``, whatever order the
guards inside :func:`export` run in.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from config import Config, load_config
from ingest import export_almabot as ex

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"
BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"
OUTPUT_PATHS = ("COURSES_PATH", "PREREQS_PATH", "REQUIREMENTS_PATH")
#: The real, repo-relative output paths, captured before any fixture redirects them.
REPO_OUTPUTS = {name: getattr(ex, name) for name in OUTPUT_PATHS}


@pytest.fixture(autouse=True)
def redirect_outputs(tmp_path, monkeypatch) -> dict[str, Path]:
    """Point every output path at tmp_path, for every test in this module.

    ``data/courses.json`` is 3 MB of committed corpus. A test that reaches :func:`export` -- even
    one that only means to assert a guard raises -- would otherwise overwrite it with the 17-course
    fixture the moment that guard moved or stopped raising.
    """
    paths = {name: tmp_path / f"{name.removesuffix('_PATH').lower()}.json" for name in OUTPUT_PATHS}
    for name, path in paths.items():
        monkeypatch.setattr(ex, name, path)
    return paths


@pytest.fixture
def cfg() -> Config:
    """Config pointing at the fixture CSV, with the source pin matching its content."""
    raw = load_config(BASE_CONFIG).to_dict()
    raw["data"]["source"] = str(FIXTURE_CSV)
    raw["data"]["source_sha256"] = hashlib.sha256(FIXTURE_CSV.read_bytes()).hexdigest()
    raw["data"]["term"] = "2026-sp"
    return Config(raw)


@pytest.fixture
def catalog() -> ex.Catalog:
    return ex.read_courses(FIXTURE_CSV)


@pytest.fixture
def courses(catalog) -> list[dict]:
    return catalog.courses


@pytest.fixture
def by_code(courses) -> dict[str, dict]:
    return {c["code"]: c for c in courses}


@pytest.fixture
def exported(cfg, redirect_outputs) -> dict:
    """Run the full export into tmp_path and return the parsed files plus the result dict."""
    result = ex.export(cfg)
    return {
        "result": result,
        "courses": json.loads(redirect_outputs["COURSES_PATH"].read_text(encoding="utf-8")),
        "prereqs": json.loads(redirect_outputs["PREREQS_PATH"].read_text(encoding="utf-8")),
        "requirements": json.loads(
            redirect_outputs["REQUIREMENTS_PATH"].read_text(encoding="utf-8")
        ),
        "paths": redirect_outputs,
    }


# --- field parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 hours.", (3.0, 3.0, "")),
        ("4 hours.", (4.0, 4.0, "")),
        ("1.5 hours.", (1.5, 1.5, "")),
        ("0 hours.", (0.0, 0.0, "")),  # 69 real courses; not the same as "unknown"
        # Both ends survive, whichever end the range starts at. The old parse_credits kept the
        # first number unless it was 0 -- the low end of 1,211 courses, the high end of 204.
        ("3 OR 4 hours.", (3.0, 4.0, "OR")),
        ("1 TO 5 hours.", (1.0, 5.0, "TO")),
        ("0 TO 4 hours.", (0.0, 4.0, "TO")),
        ("0 TO 20 hours.", (0.0, 20.0, "TO")),
        ("0 OR 1 hours.", (0.0, 1.0, "OR")),
        # Tolerated spellings: the catalog writes the connector upper-case and "hours." with the
        # stop, but neither is load-bearing.
        ("2 or 3 hours", (2.0, 3.0, "OR")),
        ("1 hour.", (1.0, 1.0, "")),
    ],
)
def test_parse_credit_hours_keeps_both_ends_of_a_range(raw, expected):
    credits = ex.parse_credit_hours(raw)
    assert (credits.low, credits.high, credits.connector) == expected
    assert credits.raw == raw.strip()


@pytest.mark.parametrize("raw", ["", None, "   ", "hours to be arranged.", "3 to 4 credits"])
def test_parse_credit_hours_reports_an_unreadable_value_as_no_bounds(raw):
    # No invented number: the text is kept and the bounds stay empty, which check_credits then
    # refuses to export rather than shipping a course no credit filter can see.
    credits = ex.parse_credit_hours(raw)
    assert (credits.low, credits.high) == (None, None)
    assert credits.raw == (raw or "").strip()


def test_parse_credit_hours_orders_the_bounds(by_code):
    # The field names promise min <= max; the catalog never writes a range the other way round.
    assert all(c["creditsMin"] <= c["creditsMax"] for c in by_code.values())
    assert ex.parse_credit_hours("4 TO 2 hours.")[1:3] == (2.0, 4.0)


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ("Intro. Prerequisite: CS 225.", "CS 225."),
        # The whole block is kept: the catalog runs it to the end of the description.
        (
            "Intro. Prerequisites: CS 225 or CS 240. Restricted to majors.",
            "CS 225 or CS 240. Restricted to majors.",
        ),
        (
            "Intro. Prerequisite: One of CS 173, MATH 213; CS 225.",
            "One of CS 173, MATH 213; CS 225.",
        ),
        ("Seminar. Prerequisite: Ph.D. standing in BADM.", "Ph.D. standing in BADM."),
        ("No prerequisite sentence here.", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_parse_prereq_keeps_the_whole_requirements_block(desc, expected):
    assert ex.parse_prereq(desc) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One sentence.", ["One sentence."]),
        ("First. Second.", ["First.", "Second."]),
        # The stop belongs to the abbreviation, not to a sentence boundary.
        ("Ph.D. standing in BADM.", ["Ph.D. standing in BADM."]),
        (
            "Ph.D. standing in BADM. Consent required.",
            ["Ph.D. standing in BADM.", "Consent required."],
        ),
        ("IS 507 or equivalent (e.g. STAT 100).", ["IS 507 or equivalent (e.g. STAT 100)."]),
        ("For the M.Eng. in Bioengineering.", ["For the M.Eng. in Bioengineering."]),
        ("", []),
    ],
)
def test_split_sentences_does_not_break_inside_abbreviations(text, expected):
    assert ex.split_sentences(text) == expected


@pytest.mark.parametrize(
    ("raw", "code", "expected"),
    [
        ("One of CS 173, MATH 213; CS 225.", "CS 374", ["CS 173", "CS 225", "MATH 213"]),
        ("CS374 and one of CS 233, CS 240.", "CS 421", ["CS 233", "CS 240", "CS 374"]),
        ("CS 225 or CS 225.", "CS 421", ["CS 225"]),
        ("PHYS 211 placement test or MATH 221.", "PHYS 211", ["MATH 221"]),
        ("consent of instructor", "CS 591", []),
        ("", "CS 591", []),
        # Codes recovered because the sentence no longer stops inside "e.g.".
        (
            "IS 507 or equivalent (e.g. STAT 100 or CS 361).",
            "IS 517",
            ["CS 361", "IS 507", "STAT 100"],
        ),
        # Only the prerequisite statement counts: later sentences say something else.
        ("CS 173. May not be taken concurrently with CS 225.", "ANTH 390", ["CS 173"]),
        ("CS 173. Recommended: CS 225.", "CS 464", ["CS 173"]),
        ("CS 173. Credit is not given for both CS 225 and CS 233.", "CS 400", ["CS 173"]),
        ("CS 125. Credit or concurrent registration in CS 173.", "CS 233", ["CS 125"]),
    ],
)
def test_prereq_codes_come_from_the_prerequisite_statement_only(raw, code, expected):
    # Sorted, de-duplicated, self-excluded; "CS374" and "CS 374" are the same course.
    assert ex.prereq_codes(raw, code) == expected


@pytest.mark.parametrize(
    ("raw", "code", "expected"),
    [
        (
            "AE 321 and AE 352. Credit or concurrent registration in AE 323 and AE 353.",
            "AE 461",
            ["AE 323", "AE 353"],
        ),
        ("CS 125. Concurrent enrollment in EDPR 442 is required.", "CI 404", ["EDPR 442"]),
        ("PHYS 211. Corequisites: MATH 231 and PHYS 225.", "PHYS 246", ["MATH 231", "PHYS 225"]),
        ("CS 173. Must enroll concurrently in SE 495.", "SE 494", ["SE 495"]),
        # Hedged or negated: not a corequisite edge.
        ("STAT 100. Concurrent enrollment in CS 225 is preferred.", "STAT 425", []),
        ("CS 173. Concurrent enrollment with JOUR 317 is allowed.", "JOUR 340", []),
        ("CS 173. May not be taken concurrently with CS 225.", "ANTH 390", []),
        ("CS 173. Concurrent enrollment in CS 225 is recommended.", "CS 400", []),
        # The prerequisite statement itself never yields a corequisite.
        ("Credit or concurrent registration in CS 173.", "CS 233", []),
        # A code that is already a prerequisite is not repeated as a corequisite.
        ("CS 173. Credit or concurrent registration in CS 173.", "CS 233", []),
    ],
)
def test_coreq_codes_need_an_explicit_unhedged_statement(raw, code, expected):
    assert ex.coreq_codes(raw, code) == expected


def test_row_term_prefers_yearterm_then_falls_back():
    assert ex.row_term({"YearTerm": "2026-sp", "Year": "2027", "Term": "Fall"}) == "2026-sp"
    assert ex.row_term({"YearTerm": "", "Year": "2026", "Term": "Spring"}) == "2026-sp"
    assert ex.row_term({"Year": "2025", "Term": "fall"}) == "2025-fa"


def test_row_term_rejects_unusable_rows():
    with pytest.raises(ValueError, match="no usable term"):
        ex.row_term({"YearTerm": "", "Year": "2026", "Term": "Interterm"})
    with pytest.raises(ValueError, match="no usable term"):
        ex.row_term({})


def test_clean_decodes_entities_and_trims():
    assert ex.clean("  Algorithms &amp; Models  ") == "Algorithms & Models"
    assert ex.clean(None) == ""


# --- read_courses ----------------------------------------------------------------------------


def test_section_rows_collapse_into_courses(courses):
    codes = [c["code"] for c in courses]
    assert len(codes) == len(set(codes))
    assert "CS 374" in codes
    assert len(codes) == 17  # 20 fixture rows: 3 CS 374 sections collapse, 1 row has no subject


def test_rows_without_subject_or_number_are_skipped(courses):
    assert all(c["subject"] and c["number"] for c in courses)
    assert not any(c["name"] == "Orphan row" for c in courses)


def test_courses_are_sorted_by_subject_then_number(courses):
    assert courses == sorted(courses, key=lambda c: (c["subject"], c["number"]))


def test_section_metadata_is_aggregated_across_rows(by_code):
    cs374 = by_code["CS 374"]
    assert cs374["sectionCount"] == 3
    assert cs374["partOfTerm"] == ["1", "A"]
    assert cs374["types"] == ["Discussion/Recitation", "Laboratory", "Lecture"]
    assert by_code["CS 173"]["sectionCount"] == 1


def test_read_courses_reports_the_section_row_count(catalog):
    assert catalog.section_rows == 19  # every fixture row except the one with no Subject/Number
    assert catalog.terms == {"2026-sp": 19}  # tallied per row, not per course


def test_description_is_not_truncated(by_code):
    # The whole point of the re-export: AlmaBot cuts desc at 240 chars, we keep all of it.
    desc = by_code["CS 374"]["desc"]
    assert len(desc) > 240
    assert desc.endswith("Prerequisite: One of CS 173, MATH 213; CS 225.")


def test_every_course_carries_the_term(by_code):
    assert all(c["term"] == "2026-sp" for c in by_code.values())
    assert by_code["STAT 100"]["term"] == "2026-sp"  # this row's YearTerm is empty


def test_html_entities_are_decoded(by_code):
    assert by_code["CS 374"]["name"] == "Introduction to Algorithms & Models"
    assert "&amp;" not in by_code["ECON 102"]["attrs"]


def test_course_fields_cover_the_chunk_template(by_code):
    # config.yaml's chunk.fields must all exist on a course record.
    fields = load_config()["chunk"]["fields"]
    assert set(fields) <= set(by_code["CS 374"])


def test_credit_hours_are_exported_raw_and_as_bounds(by_code):
    # The lossless string plus the two numbers a filter compares against. A fixed-credit course
    # gets equal bounds, so one comparison covers both kinds.
    assert by_code["CS 173"]["creditsRaw"] == "3 OR 4 hours."
    assert (by_code["CS 173"]["creditsMin"], by_code["CS 173"]["creditsMax"]) == (3.0, 4.0)
    assert by_code["CWL 593"]["creditsRaw"] == "1 TO 5 hours."
    assert (by_code["CWL 593"]["creditsMin"], by_code["CWL 593"]["creditsMax"]) == (1.0, 5.0)
    assert by_code["CS 374"]["creditsRaw"] == "4 hours."
    assert (by_code["CS 374"]["creditsMin"], by_code["CS 374"]["creditsMax"]) == (4.0, 4.0)
    assert "credits" not in by_code["CS 374"]  # the flattened field is gone, not renamed


def test_a_credit_filter_finds_every_course_that_can_be_taken_for_four(by_code):
    # #17's acceptance question, asked the way a consumer asks it: one comparison against the
    # pair, covering fixed-credit and variable-credit courses alike.
    four = sorted(c["code"] for c in by_code.values() if c["creditsMin"] <= 4 <= c["creditsMax"])
    assert four == [
        "ANTH 390",  # 2 TO 4
        "BADM 537",
        "CS 173",  # 3 OR 4
        "CS 225",
        "CS 233",
        "CS 374",
        "CS 421",  # 3 OR 4
        "CWL 593",  # 1 TO 5
        "ECON 102",  # 0 TO 4
        "PHYS 211",
        "STAT 425",  # 3 OR 4
    ]

    def flattened(raw: str) -> float:
        """What the old single ``credits`` field held: the first number unless it was 0."""
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
        return nums[0] if nums[0] > 0 else max(nums)

    # Five of the eleven were invisible to `credits == 4`, and ECON 102 only made it in because
    # its range starts at zero -- the inconsistency this issue is about.
    missed = set(four) - {c["code"] for c in by_code.values() if flattened(c["creditsRaw"]) == 4}
    assert missed == {"ANTH 390", "CS 173", "CS 421", "CWL 593", "STAT 425"}


def test_prereq_codes_are_unfiltered_on_the_course_record(by_code):
    # MATH 213 is not in this catalog but stays here; prereqs.json is where it gets dropped.
    assert by_code["CS 374"]["prereqCodes"] == ["CS 173", "CS 225", "MATH 213"]
    assert by_code["CS 374"]["prereqRaw"] == "One of CS 173, MATH 213; CS 225."


def test_requirements_block_is_not_cut_at_the_first_sentence_break(by_code):
    # The frontend regex stopped at the first ". ", which reduced this whole block to "Ph.D".
    assert by_code["BADM 537"]["prereqRaw"] == "Ph.D. standing in BADM or consent of instructor."
    assert by_code["CS 233"]["prereqRaw"] == (
        "CS 125. Credit or concurrent registration in CS 173."
    )
    assert by_code["ANTH 390"]["prereqRaw"] == (
        "CS 173. May not be taken concurrently with CS 225."
    )


def test_corequisites_are_recorded_apart_from_prerequisites(by_code):
    assert by_code["CS 233"]["prereqCodes"] == ["CS 125"]
    assert by_code["CS 233"]["coreqCodes"] == ["CS 173"]
    # Hedged ("preferred") and negated ("May not be taken") sentences yield neither.
    assert by_code["STAT 425"]["coreqCodes"] == []
    assert by_code["ANTH 390"]["coreqCodes"] == []
    assert by_code["ANTH 390"]["prereqCodes"] == ["CS 173"]


# --- check_terms -----------------------------------------------------------------------------


def test_check_terms_accepts_the_configured_term(catalog, cfg):
    ex.check_terms(catalog, cfg)


def test_check_terms_rejects_a_term_the_config_does_not_claim(catalog, cfg):
    other = Config({**cfg.to_dict(), "data": {**cfg["data"], "term": "2026-fa"}})
    with pytest.raises(ValueError, match="data.term"):
        ex.check_terms(catalog, other)


def test_check_terms_rejects_a_missing_term(catalog, cfg):
    broken = catalog._replace(courses=[{**catalog.courses[0], "term": ""}, *catalog.courses[1:]])
    with pytest.raises(ValueError, match="no term"):
        ex.check_terms(broken, cfg)


def test_check_terms_catches_a_mixed_term_csv_that_deduplication_would_hide(cfg, tmp_path):
    # Extra sections for a course that already exists, in a different term. De-duplication keeps
    # only the first row of each course, so every *course* still reads 2026-sp -- the mismatch
    # survives only in the per-row tally, which is what check_terms reads.
    lines = FIXTURE_CSV.read_text(encoding="utf-8").splitlines(keepends=True)
    stale = [line.replace("2026,Spring,2026-sp", "2025,Fall,2025-fa") for line in lines[1:4]]
    mixed = tmp_path / "mixed.csv"
    mixed.write_text("".join([*lines, *stale]), encoding="utf-8")

    catalog = ex.read_courses(mixed)
    assert {c["term"] for c in catalog.courses} == {"2026-sp"}  # de-duplication hid it
    assert catalog.terms == {"2026-sp": 19, "2025-fa": 3}  # the tally did not

    with pytest.raises(ValueError, match="2025-fa"):
        ex.check_terms(catalog, cfg)


def test_check_terms_error_names_the_offending_row_counts(cfg, tmp_path):
    catalog = ex.Catalog(
        courses=[{"code": "CS 1", "term": "2025-fa"}],
        section_rows=1,
        terms=ex.Counter({"2025-fa": 7}),
    )
    with pytest.raises(ValueError, match=r"2025-fa \(7 rows\)"):
        ex.check_terms(catalog, cfg)


# --- check_credits ---------------------------------------------------------------------------


def test_check_credits_accepts_the_fixture_catalog(catalog):
    ex.check_credits(catalog)


def test_check_credits_rejects_a_value_it_cannot_read(catalog):
    broken = catalog._replace(
        courses=[
            {**catalog.courses[0], "creditsRaw": "hours TBD", "creditsMin": None},
            *catalog.courses[1:],
        ]
    )
    with pytest.raises(ValueError, match="Credit Hours value this export cannot read"):
        ex.check_credits(broken)


def test_check_credits_error_names_the_offending_course_and_value(catalog):
    broken = catalog._replace(
        courses=[{"code": "CS 999", "creditsRaw": "hours TBD", "creditsMin": None}]
    )
    with pytest.raises(ValueError, match=r"CS 999 \('hours TBD'\)"):
        ex.check_credits(broken)


def test_export_runs_the_credit_check(cfg, tmp_path, monkeypatch, redirect_outputs):
    # The guard has to run inside export(), not just be available: a corpus with null bounds
    # must never reach data/.
    monkeypatch.setattr(ex, "parse_credit_hours", lambda raw: ex.CreditHours("?", None, None, ""))
    with pytest.raises(ValueError, match="Credit Hours value this export cannot read"):
        ex.export(cfg)
    assert not redirect_outputs["COURSES_PATH"].exists()


# --- build_prereqs ---------------------------------------------------------------------------


def test_prereq_edges_keep_only_courses_in_the_catalog(courses):
    edges = {e["course"]: e for e in ex.build_prereqs(courses)}
    assert edges["CS 374"]["prereqs"] == ["CS 173", "CS 225"]  # MATH 213 is not offered
    assert edges["CS 374"]["raw"] == "One of CS 173, MATH 213; CS 225."
    assert edges["CS 374"]["coreqs"] == []


def test_corequisite_edges_ride_in_their_own_field(courses):
    edges = {e["course"]: e for e in ex.build_prereqs(courses)}
    # CS 233's only prerequisite (CS 125) dangles, but its corequisite resolves, so it still
    # gets an entry -- and the corequisite never lands in `prereqs`, where it would become a
    # DAG edge. Mutual corequisites are common enough (DANC 498 <-> DANC 499) to make cycles.
    assert edges["CS 233"] == {
        "course": "CS 233",
        "prereqs": [],
        "coreqs": ["CS 173"],
        "raw": "CS 125. Credit or concurrent registration in CS 173.",
    }
    assert all("coreqs" in e for e in ex.build_prereqs(courses))


def test_courses_whose_references_all_dangle_get_no_entry(courses):
    edges = {e["course"] for e in ex.build_prereqs(courses)}
    assert "CS 225" not in edges  # its only reference, CS 125, is not in the catalog


def test_partially_dangling_references_keep_the_resolvable_ones(courses):
    edges = {e["course"]: e["prereqs"] for e in ex.build_prereqs(courses)}
    # CS 421 lists "CS374 and one of CS 233, CS 240": CS 240 is not in the catalog, and the
    # spaceless CS374 resolves to CS 374 like any other spelling.
    assert edges["CS 421"] == ["CS 233", "CS 374"]


def test_courses_without_prerequisites_get_no_entry(courses):
    edges = {e["course"] for e in ex.build_prereqs(courses)}
    assert "CS 173" not in edges
    assert "PHYS 211" not in edges  # self-reference dropped, MATH 221 not in the catalog


def test_every_edge_points_at_a_known_course(courses):
    known = {c["code"] for c in courses}
    for edge in ex.build_prereqs(courses):
        assert edge["course"] in known
        assert set(edge["prereqs"]) <= known
        assert set(edge["coreqs"]) <= known


# --- Gen Ed rules ----------------------------------------------------------------------------

# The matchers exactly as almabot-web/src/lib/gened.ts writes them, to check the declarative
# GENED rules against the behaviour they replace.
TS_MATCHERS = {
    "comp1": lambda a: "Composition I" in a,
    "advcomp": lambda a: "Advanced Composition" in a,
    "qr1": lambda a: re.search(r"Quantitative Reasoning I(?!I)", a) is not None,
    "qr2": lambda a: "Quantitative Reasoning II" in a,
    "natphys": lambda a: "Phys Sciences" in a,
    "natlife": lambda a: "Life Sciences" in a,
    "humhist": lambda a: "Hist & Phil" in a,
    "humlit": lambda a: "Lit & Arts" in a,
    "socsoc": lambda a: "Soc Sci" in a,
    "socbeh": lambda a: "Beh Sci" in a,
    "cultwest": lambda a: "Cultural Studies - Western" in a or "Western course" in a,
    "cultnonwest": lambda a: "Non-West" in a,
    "cultus": lambda a: "US Minority" in a,
}

# Degree Attributes strings as they appear in the 2026-sp catalog.
ATTRS_SAMPLES = [
    "",
    "Composition I course.",
    "Advanced Composition course.",
    "Quantitative Reasoning I course.",
    "Quantitative Reasoning II course.",
    "Nat Sci & Tech - Phys Sciences course.",
    "Nat Sci & Tech - Life Sciences, and Cultural Studies - Western course.",
    "Humanities - Hist & Phil, and Cultural Studies - Non-West course.",
    "Humanities - Lit & Arts course.",
    "Social & Beh Sci - Soc Sci, and Cultural Studies - US Minority course.",
    "Social & Beh Sci - Beh Sci course.",
    "Cultural Studies - Western, and Humanities - Lit & Arts course.",
    "Advanced Composition, Humanities - Hist & Phil, and Cultural Studies - Western course.",
]


def test_gened_ids_and_labels_match_the_frontend():
    assert [c["id"] for c in ex.GENED] == list(TS_MATCHERS)
    assert len(ex.GENED) == 13
    assert all(c["label"] and c["group"] for c in ex.GENED)


def test_gened_entries_already_carry_the_shape_that_is_written_out():
    # matches_attrs() and requirements.json must agree on where the rule lives, or a consumer
    # reading the file hits a KeyError and re-implements the qr1 exclusion by hand.
    for category in ex.GENED:
        assert set(category) == {"id", "label", "group", "match"}
        assert set(category["match"]) == {"field", "any", "none"}
        assert category["match"]["field"] == "attrs"
        assert category["match"]["any"]


@pytest.mark.parametrize("attrs", ATTRS_SAMPLES, ids=range(len(ATTRS_SAMPLES)))
def test_rules_agree_with_the_frontend_matchers(attrs):
    for category in ex.GENED:
        assert ex.matches_attrs(category, attrs) is TS_MATCHERS[category["id"]](attrs), (
            f"{category['id']} disagrees on {attrs!r}"
        )


def test_quantitative_reasoning_i_does_not_match_the_ii_designation():
    qr1 = next(c for c in ex.GENED if c["id"] == "qr1")
    assert ex.matches_attrs(qr1, "Quantitative Reasoning I course.")
    assert not ex.matches_attrs(qr1, "Quantitative Reasoning II course.")


def test_qr1_diverges_from_the_ts_regex_only_on_a_combination_the_catalog_never_uses():
    # gened.ts uses a lookahead, /Quantitative Reasoning I(?!I)/, which still matches a string that
    # carries BOTH designations; the declarative `none` rule excludes it. No course in the
    # 2026-sp catalog is attributed both, so the two agree on the real data (see PR #13).
    both = "Quantitative Reasoning I course. Quantitative Reasoning II course."
    qr1 = next(c for c in ex.GENED if c["id"] == "qr1")
    assert TS_MATCHERS["qr1"](both) is True
    assert ex.matches_attrs(qr1, both) is False


def test_empty_attrs_never_matches():
    assert all(not ex.matches_attrs(c, "") for c in ex.GENED)


# --- build_requirements ----------------------------------------------------------------------


def test_requirements_shape(courses, cfg):
    req = ex.build_requirements(courses, cfg)
    assert req["term"] == "2026-sp"
    assert len(req["genEd"]) == 13
    assert req["majors"] == []
    assert "not present in course-catalog.csv" in req["notes"]["majors"]
    assert req["notes"]["culturalStudies"]
    first = req["genEd"][0]
    assert set(first) == {"id", "label", "group", "match", "courseCount"}
    assert set(first["match"]) == {"field", "any", "none"}
    assert first["match"]["field"] == "attrs"


def test_rules_in_the_written_file_can_be_evaluated_by_matches_attrs(exported):
    # The contract a Day-2 consumer depends on: load requirements.json, call matches_attrs on an
    # entry straight out of it, get the same answer the export recorded.
    for category in exported["requirements"]["genEd"]:
        recount = sum(1 for c in exported["courses"] if ex.matches_attrs(category, c["attrs"]))
        assert recount == category["courseCount"]
    qr1 = next(c for c in exported["requirements"]["genEd"] if c["id"] == "qr1")
    assert ex.matches_attrs(qr1, "Quantitative Reasoning I course.")
    assert not ex.matches_attrs(qr1, "Quantitative Reasoning II course.")


def test_requirement_counts_match_the_courses(courses, cfg):
    counts = {c["id"]: c["courseCount"] for c in ex.build_requirements(courses, cfg)["genEd"]}
    assert counts["socsoc"] == 3  # AAS 100, ANTH 103, ECON 102
    assert counts["cultus"] == 1  # AAS 100
    assert counts["cultwest"] == 1  # HIST 200
    assert counts["cultnonwest"] == 1  # ANTH 103
    assert counts["qr1"] == 1  # STAT 100, not MATH 101
    assert counts["qr2"] == 1  # MATH 101
    assert counts["comp1"] == 0
    for category in ex.GENED:
        expected = sum(1 for c in courses if ex.matches_attrs(category, c["attrs"]))
        assert counts[category["id"]] == expected


# --- export ------------------------------------------------------------------------------------


def test_export_writes_all_three_files(exported):
    for path in exported["paths"].values():
        assert path.is_file()
    assert exported["courses"] == exported["result"]["courses"]
    assert exported["prereqs"] == exported["result"]["prereqs"]
    assert exported["requirements"] == exported["result"]["requirements"]


def test_exported_courses_meet_the_acceptance_criteria(exported):
    by_code = {c["code"]: c for c in exported["courses"]}
    assert len(by_code["CS 374"]["desc"]) > 240
    assert all(c["term"] for c in exported["courses"])


def test_exported_json_is_utf8_and_newline_terminated(exported):
    text = exported["paths"]["COURSES_PATH"].read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\\u" not in text  # ensure_ascii=False: non-ASCII stays readable


def test_export_verifies_the_source_pin(cfg, tmp_path, redirect_outputs):
    # A CSV whose content does not match data.source_sha256 must not be exported at all:
    # the pin is in the config hash, so mismatched data would be indexed under the wrong name.
    copy = tmp_path / "catalog.csv"
    copy.write_bytes(FIXTURE_CSV.read_bytes() + b"\n")
    bad = Config({**cfg.to_dict(), "data": {**cfg["data"], "source": str(copy)}})
    with pytest.raises(ValueError, match="pins"):
        ex.export(bad)
    assert not any(path.exists() for path in redirect_outputs.values())


def test_export_rejects_a_csv_with_no_courses(cfg, tmp_path, redirect_outputs):
    empty = tmp_path / "empty.csv"
    empty.write_text("Year,Term,YearTerm,Subject,Number,Name,Description\n", encoding="utf-8")
    bad = Config(
        {
            **cfg.to_dict(),
            "data": {
                **cfg["data"],
                "source": str(empty),
                "source_sha256": hashlib.sha256(empty.read_bytes()).hexdigest(),
            },
        }
    )
    with pytest.raises(ValueError, match="no courses"):
        ex.export(bad)
    assert not any(path.exists() for path in redirect_outputs.values())


def test_default_output_paths_live_in_the_data_directory():
    # REPO_OUTPUTS is captured at import time; redirect_outputs has since pointed the module
    # constants at tmp_path for every test in this file.
    assert REPO_OUTPUTS["COURSES_PATH"] == ex.DATA_DIR / "courses.json"
    assert REPO_OUTPUTS["PREREQS_PATH"] == ex.DATA_DIR / "prereqs.json"
    assert REPO_OUTPUTS["REQUIREMENTS_PATH"] == ex.DATA_DIR / "requirements.json"
    assert ex.DATA_DIR.is_dir()


def test_no_test_in_this_module_can_write_into_the_repo(redirect_outputs, tmp_path):
    # The guard behind the autouse fixture: export() writes to tmp_path, never to data/.
    for name, path in redirect_outputs.items():
        assert getattr(ex, name) == path
        assert path.parent == tmp_path
        assert REPO_OUTPUTS[name].parent == ex.DATA_DIR


# --- stats and CLI -----------------------------------------------------------------------------


def test_summary_reports_description_quality(exported):
    lines = ex.summarize(
        exported["result"]["catalog"], exported["prereqs"], exported["requirements"]
    )
    text = "\n".join(lines)
    assert "courses                17" in text
    assert "section rows read      19" in text
    assert "term                   2026-sp (19 rows)" in text
    assert "empty                1" in text  # CWL 593
    assert "cross-reference only 1" in text  # XREF 101
    assert "prereq codes mentioned 11" in text
    assert "resolved in catalog  6" in text
    assert "dangling (dropped)   5" in text
    assert "corequisite edges      1" in text  # CS 233 -> CS 173
    assert "credit hours fixed     11" in text
    assert "variable             6  (3 'X TO Y', 3 'X OR Y')" in text  # 11 fixed of 17
    assert "Quantitative Reasoning I" in text


def test_cli_runs_the_export_and_prints_stats(cfg, tmp_path, capsys):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")  # JSON is valid YAML
    ex.main(["--config", str(cfg_file)])
    out = capsys.readouterr().out
    assert "catalog_sample.csv" in out
    assert "courses                17" in out
    assert out.count("wrote    ") == 3
    assert str(Path.home()) not in out  # paths go through rel_to_repo()
