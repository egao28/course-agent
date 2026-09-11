"""Tests for ingest/export_almabot.py: CSV de-duplication, the three data files, the Gen Ed rules.

Everything runs against ``tests/fixtures/catalog_sample.csv`` -- a 16-row synthetic catalog with
the real column layout -- so the suite never needs the private course-catalog.csv, and tuning the
live ``config.yaml`` cannot break it. The fixture deliberately contains a description longer than
the 240 chars AlmaBot truncates at, an empty description, a cross-reference-only description,
dangling and self-referencing prerequisites, both Quantitative Reasoning spellings, and rows with
no Subject/Number.
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


@pytest.fixture
def cfg() -> Config:
    """Config pointing at the fixture CSV, with the source pin matching its content."""
    raw = load_config(BASE_CONFIG).to_dict()
    raw["data"]["source"] = str(FIXTURE_CSV)
    raw["data"]["source_sha256"] = hashlib.sha256(FIXTURE_CSV.read_bytes()).hexdigest()
    raw["data"]["term"] = "2026-sp"
    return Config(raw)


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV)[0]


@pytest.fixture
def by_code(courses) -> dict[str, dict]:
    return {c["code"]: c for c in courses}


@pytest.fixture
def exported(cfg, tmp_path, monkeypatch) -> dict:
    """Run the full export into tmp_path and return the parsed files plus the result dict."""
    paths = {
        "COURSES_PATH": tmp_path / "courses.json",
        "PREREQS_PATH": tmp_path / "prereqs.json",
        "REQUIREMENTS_PATH": tmp_path / "requirements.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(ex, name, path)
    result = ex.export(cfg)
    return {
        "result": result,
        "courses": json.loads(paths["COURSES_PATH"].read_text(encoding="utf-8")),
        "prereqs": json.loads(paths["PREREQS_PATH"].read_text(encoding="utf-8")),
        "requirements": json.loads(paths["REQUIREMENTS_PATH"].read_text(encoding="utf-8")),
        "paths": paths,
    }


# --- field parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 hours.", 3.0),
        ("4 hours.", 4.0),
        ("3 OR 4 hours.", 3.0),
        ("1 TO 5 hours.", 1.0),
        ("0 TO 4 hours.", 4.0),  # a zero lower bound means "up to", so keep the maximum
        ("1.5 hours.", 1.5),
        ("", 0.0),
        (None, 0.0),
        ("hours to be arranged.", 0.0),
    ],
)
def test_parse_credits(raw, expected):
    assert ex.parse_credits(raw) == expected


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ("Intro. Prerequisite: CS 225.", "CS 225"),
        ("Intro. Prerequisites: CS 225 or CS 240. More text.", "CS 225 or CS 240"),
        (
            "Intro. Prerequisite: One of CS 173, MATH 213; CS 225.",
            "One of CS 173, MATH 213; CS 225",
        ),
        ("No prerequisite sentence here.", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_parse_prereq(desc, expected):
    assert ex.parse_prereq(desc) == expected


@pytest.mark.parametrize(
    ("raw", "code", "expected"),
    [
        ("One of CS 173, MATH 213; CS 225", "CS 374", ["CS 173", "CS 225", "MATH 213"]),
        ("CS374 and one of CS 233, CS 240", "CS 421", ["CS 233", "CS 240", "CS 374"]),
        ("CS 225 or CS 225", "CS 421", ["CS 225"]),
        ("PHYS 211 placement test or MATH 221", "PHYS 211", ["MATH 221"]),
        ("consent of instructor", "CS 591", []),
    ],
)
def test_prereq_codes(raw, code, expected):
    # Sorted, de-duplicated, self-excluded; "CS374" and "CS 374" are the same course.
    assert ex.prereq_codes(raw, code) == expected


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
    assert len(codes) == 13  # 16 fixture rows: 3 CS 374 sections collapse, 1 row has no subject


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


def test_read_courses_reports_the_section_row_count():
    _, rows = ex.read_courses(FIXTURE_CSV)
    assert rows == 15  # every fixture row except the one with no Subject/Number


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


def test_prereq_codes_are_unfiltered_on_the_course_record(by_code):
    # MATH 213 is not in this catalog but stays here; prereqs.json is where it gets dropped.
    assert by_code["CS 374"]["prereqCodes"] == ["CS 173", "CS 225", "MATH 213"]
    assert by_code["CS 374"]["prereqRaw"] == "One of CS 173, MATH 213; CS 225"


# --- check_terms -----------------------------------------------------------------------------


def test_check_terms_accepts_the_configured_term(courses, cfg):
    ex.check_terms(courses, cfg)


def test_check_terms_rejects_a_term_the_config_does_not_claim(courses, cfg):
    other = Config({**cfg.to_dict(), "data": {**cfg["data"], "term": "2026-fa"}})
    with pytest.raises(ValueError, match="data.term"):
        ex.check_terms(courses, other)


def test_check_terms_rejects_a_missing_term(courses, cfg):
    broken = [{**courses[0], "term": ""}, *courses[1:]]
    with pytest.raises(ValueError, match="no term"):
        ex.check_terms(broken, cfg)


# --- build_prereqs ---------------------------------------------------------------------------


def test_prereq_edges_keep_only_courses_in_the_catalog(courses):
    edges = {e["course"]: e for e in ex.build_prereqs(courses)}
    assert edges["CS 374"]["prereqs"] == ["CS 173", "CS 225"]  # MATH 213 is not offered
    assert edges["CS 374"]["raw"] == "One of CS 173, MATH 213; CS 225"


def test_courses_whose_references_all_dangle_get_no_entry(courses):
    edges = {e["course"] for e in ex.build_prereqs(courses)}
    assert "CS 225" not in edges  # its only reference, CS 125, is not in the catalog


def test_partially_dangling_references_keep_the_resolvable_ones(courses):
    edges = {e["course"]: e["prereqs"] for e in ex.build_prereqs(courses)}
    # CS 421 lists "CS374 and one of CS 233, CS 240": only the spaceless CS374 is in the catalog.
    assert edges["CS 421"] == ["CS 374"]


def test_courses_without_prerequisites_get_no_entry(courses):
    edges = {e["course"] for e in ex.build_prereqs(courses)}
    assert "CS 173" not in edges
    assert "PHYS 211" not in edges  # self-reference dropped, MATH 221 not in the catalog


def test_every_edge_points_at_a_known_course(courses):
    known = {c["code"] for c in courses}
    for edge in ex.build_prereqs(courses):
        assert edge["course"] in known
        assert set(edge["prereqs"]) <= known


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


def test_export_verifies_the_source_pin(cfg, tmp_path, monkeypatch):
    # A CSV whose content does not match data.source_sha256 must not be exported at all:
    # the pin is in the config hash, so mismatched data would be indexed under the wrong name.
    copy = tmp_path / "catalog.csv"
    copy.write_bytes(FIXTURE_CSV.read_bytes() + b"\n")
    bad = Config({**cfg.to_dict(), "data": {**cfg["data"], "source": str(copy)}})
    monkeypatch.setattr(ex, "COURSES_PATH", tmp_path / "courses.json")
    with pytest.raises(ValueError, match="pins"):
        ex.export(bad)
    assert not (tmp_path / "courses.json").exists()


def test_export_rejects_a_csv_with_no_courses(cfg, tmp_path, monkeypatch):
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


def test_default_output_paths_live_in_the_data_directory():
    assert ex.COURSES_PATH == ex.DATA_DIR / "courses.json"
    assert ex.PREREQS_PATH == ex.DATA_DIR / "prereqs.json"
    assert ex.REQUIREMENTS_PATH == ex.DATA_DIR / "requirements.json"
    assert ex.DATA_DIR.is_dir()


# --- stats and CLI -----------------------------------------------------------------------------


def test_summary_reports_description_quality(exported):
    lines = ex.summarize(
        exported["courses"],
        exported["prereqs"],
        exported["requirements"],
        exported["result"]["section_rows"],
    )
    text = "\n".join(lines)
    assert "courses                13" in text
    assert "section rows read      15" in text
    assert "empty                1" in text  # CWL 593
    assert "cross-reference only 1" in text  # XREF 101
    assert "prereq codes mentioned 8" in text
    assert "resolved in catalog  3" in text  # CS 374 -> CS 173, CS 225; CS 421 -> CS 374
    assert "dangling (dropped)   5" in text
    assert "Quantitative Reasoning I" in text


def test_cli_runs_the_export_and_prints_stats(cfg, tmp_path, monkeypatch, capsys):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")  # JSON is valid YAML
    for name in ("COURSES_PATH", "PREREQS_PATH", "REQUIREMENTS_PATH"):
        monkeypatch.setattr(ex, name, tmp_path / f"{name.split('_')[0].lower()}.json")
    ex.main(["--config", str(cfg_file)])
    out = capsys.readouterr().out
    assert "catalog_sample.csv" in out
    assert "courses                13" in out
    assert out.count("wrote    ") == 3
    assert str(Path.home()) not in out  # paths go through rel_to_repo()
