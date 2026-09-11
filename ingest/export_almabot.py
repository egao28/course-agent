"""Re-export the UIUC catalog CSV into ``data/`` as the retrieval corpus.

``almabot-web`` already turns ``course-catalog.csv`` into ``public/courses.json``, but that file is
built for a React UI: ``desc`` is truncated to 240 chars, ``attrs`` to 160, and there is no term
field. Dense and BM25 both index the full description, so this exporter re-reads the **CSV** and
reuses only the *logic* from ``almabot-web/scripts/build_data.py`` (section de-duplication,
``parse_credits``, ``parse_prereq``, ``CODE_RE``), keeping every field untruncated.

Run from the repo root::

    uv run python -m ingest.export_almabot   # -> data/{courses,prereqs,requirements}.json

The input is ``data.source`` in ``config.yaml``, checked against the ``data.source_sha256`` pin
by :func:`config.verify_source`: a re-exported CSV changes the config hash, so indexes built from
the old data can never be mistaken for the new ones.

Outputs
-------
``data/courses.json``
    One record per course, sorted by (subject, number). ``prereqCodes`` holds *every* course code
    mentioned in the prerequisite text (minus self), resolved or not -- same as the frontend field.
``data/prereqs.json``
    The prerequisite edge list, filtered to codes that exist in this catalog (the filter
    ``dag.ts::prereqs()`` applies at read time). Courses whose references all dangle are omitted.
``data/requirements.json``
    The 13 campus Gen Ed categories from ``almabot-web/src/lib/gened.ts``, as declarative
    ``Degree Attributes`` substring rules. Major requirements are not in the CSV: ``majors`` is an
    empty placeholder.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import REPO_ROOT, Config, load_config, rel_to_repo, verify_source

DATA_DIR = REPO_ROOT / "data"
COURSES_PATH = DATA_DIR / "courses.json"
PREREQS_PATH = DATA_DIR / "prereqs.json"
REQUIREMENTS_PATH = DATA_DIR / "requirements.json"

# A course code as written in prose: "CS 374", "CS374", "MATH 241".
CODE_RE = re.compile(r"\b([A-Z]{2,4})\s?(\d{3})\b")
# Prerequisite sentence inside the description, up to the first sentence break.
PREREQ_RE = re.compile(r"Prerequisite[s]?:\s*(.+?)(?:\.\s|\.$|$)")
# Descriptions that carry no content of their own: "Same as HIST 283. See HIST 283."
CROSS_REF_RE = re.compile(r"^Same as [^.]*\.\s*See [^.]*\.$")

TERM_ABBR = {"spring": "sp", "summer": "su", "fall": "fa", "winter": "wi"}

# Gen Ed categories, transcribed from GENED in almabot-web/src/lib/gened.ts. The TS matchers are
# closures over the "Degree Attributes" string; here each is declarative so the JSON can carry it:
# a course matches when attrs contains ANY of `any` and NONE of `none`.
#   - qr1 replaces the lookahead /Quantitative Reasoning I(?!I)/ with an explicit `none` exclusion.
#   - cultwest keeps both spellings the TS matcher ORs together.
# fmt: off -- one line per category, aligned for review against the TS source
GENED: tuple[dict[str, Any], ...] = (
    {
        "id": "comp1",
        "label": "Composition I",
        "group": "Composition",
        "any": ["Composition I"],
        "none": [],
    },
    {
        "id": "advcomp",
        "label": "Advanced Composition",
        "group": "Composition",
        "any": ["Advanced Composition"],
        "none": [],
    },
    {
        "id": "qr1",
        "label": "Quantitative Reasoning I",
        "group": "Quantitative Reasoning",
        "any": ["Quantitative Reasoning I"],
        "none": ["Quantitative Reasoning II"],
    },
    {
        "id": "qr2",
        "label": "Quantitative Reasoning II",
        "group": "Quantitative Reasoning",
        "any": ["Quantitative Reasoning II"],
        "none": [],
    },
    {
        "id": "natphys",
        "label": "Physical Sciences",
        "group": "Natural Sciences & Tech",
        "any": ["Phys Sciences"],
        "none": [],
    },
    {
        "id": "natlife",
        "label": "Life Sciences",
        "group": "Natural Sciences & Tech",
        "any": ["Life Sciences"],
        "none": [],
    },
    {
        "id": "humhist",
        "label": "Historical & Philosophical",
        "group": "Humanities & the Arts",
        "any": ["Hist & Phil"],
        "none": [],
    },
    {
        "id": "humlit",
        "label": "Literature & the Arts",
        "group": "Humanities & the Arts",
        "any": ["Lit & Arts"],
        "none": [],
    },
    {
        "id": "socsoc",
        "label": "Social Sciences",
        "group": "Social & Behavioral Sci",
        "any": ["Soc Sci"],
        "none": [],
    },
    {
        "id": "socbeh",
        "label": "Behavioral Sciences",
        "group": "Social & Behavioral Sci",
        "any": ["Beh Sci"],
        "none": [],
    },
    {
        "id": "cultwest",
        "label": "Western/Comparative Cultures",
        "group": "Cultural Studies",
        "any": ["Cultural Studies - Western", "Western course"],
        "none": [],
    },
    {
        "id": "cultnonwest",
        "label": "Non-Western Cultures",
        "group": "Cultural Studies",
        "any": ["Non-West"],
        "none": [],
    },
    {
        "id": "cultus",
        "label": "US Minority Cultures",
        "group": "Cultural Studies",
        "any": ["US Minority"],
        "none": [],
    },
)
# fmt: on

MAJORS_NOTE = (
    "Major and degree requirements are not present in course-catalog.csv, which lists sections "
    "only. A separate source (catalog.illinois.edu program pages, or a department-provided plan) "
    "has to be scraped before this can be filled in."
)
CULTURAL_STUDIES_NOTE = (
    "Each campus Gen Ed category needs at least one course; Cultural Studies needs all three of "
    "cultwest / cultnonwest / cultus. Satisfying all 13 categories listed here covers the campus "
    "requirement."
)


# --- field parsing (ported from almabot-web/scripts/build_data.py) --------------------------------


def parse_credits(raw: str | None) -> float:
    """Credit hours from strings like ``"3 hours."`` or ``"0 TO 4 hours."``.

    The first number, unless it is zero -- a "0 TO 4" course is worth up to 4.
    """
    nums = re.findall(r"\d+(?:\.\d+)?", raw or "")
    if not nums:
        return 0.0
    first = float(nums[0])
    return first if first > 0 else max(float(n) for n in nums)


def parse_prereq(desc: str | None) -> str:
    """The raw prerequisite sentence from a description, or ``""``.

    Stops at the first sentence break, exactly like the frontend: the text is a human-readable
    excerpt, while :func:`prereq_codes` extracts the machine-readable part.
    """
    m = PREREQ_RE.search(desc or "")
    return m.group(1).strip() if m else ""


def prereq_codes(prereq_raw: str, code: str) -> list[str]:
    """Course codes mentioned in the prerequisite text, normalized and self-excluded.

    Not filtered against the catalog -- ``prereqs.json`` is where dangling references are dropped.
    """
    found = {f"{m.group(1)} {m.group(2)}" for m in CODE_RE.finditer(prereq_raw)}
    return sorted(found - {code})


def row_term(row: Mapping[str, str]) -> str:
    """Term slug for a section row, e.g. ``"2026-sp"``.

    ``YearTerm`` is already in that form; ``Year`` + ``Term`` is the fallback.
    """
    year_term = (row.get("YearTerm") or "").strip()
    if year_term:
        return year_term
    year = (row.get("Year") or "").strip()
    abbr = TERM_ABBR.get((row.get("Term") or "").strip().lower())
    if not year or not abbr:
        raise ValueError(
            f"row has no usable term: YearTerm={row.get('YearTerm')!r} "
            f"Year={row.get('Year')!r} Term={row.get('Term')!r}"
        )
    return f"{year}-{abbr}"


def clean(value: str | None) -> str:
    """Trimmed text with HTML entities decoded (the CSV carries ``&amp;``, ``&#39;``)."""
    return html.unescape((value or "").strip())


# --- CSV -> courses ------------------------------------------------------------------------------


def read_courses(csv_path: Path) -> tuple[list[dict[str, Any]], int]:
    """De-duplicate section rows into courses. Returns ``(courses, section_row_count)``.

    Course-level columns (Name, Description, Credit Hours, Degree Attributes) are identical across
    the sections of a course, so the first row of each code wins -- the same rule build_data.py
    uses. Section rows still contribute the counts and the ``partOfTerm`` / ``types`` sets, which
    are the only section facts worth keeping as retrieval metadata.
    """
    courses: dict[str, dict[str, Any]] = {}
    sections: dict[str, int] = {}
    part_of_term: dict[str, set[str]] = {}
    types: dict[str, set[str]] = {}
    rows = 0

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            subject = (row.get("Subject") or "").strip()
            number = (row.get("Number") or "").strip()
            if not subject or not number:
                continue
            rows += 1
            code = f"{subject} {number}"
            course = courses.get(code)
            if course is None:
                desc = clean(row.get("Description"))
                prereq_raw = parse_prereq(desc)
                courses[code] = {
                    "code": code,
                    "subject": subject,
                    "number": number,
                    "name": clean(row.get("Name")),
                    "term": row_term(row),
                    "credits": parse_credits(row.get("Credit Hours")),
                    "desc": desc,
                    "attrs": clean(row.get("Degree Attributes")),
                    "prereqRaw": prereq_raw,
                    "prereqCodes": prereq_codes(prereq_raw, code),
                }
                sections[code] = 0
                part_of_term[code] = set()
                types[code] = set()
            sections[code] += 1
            if pot := (row.get("Part of Term") or "").strip():
                part_of_term[code].add(pot)
            if typ := (row.get("Type") or "").strip():
                types[code].add(typ)

    for code, course in courses.items():
        course["sectionCount"] = sections[code]
        course["partOfTerm"] = sorted(part_of_term[code])
        course["types"] = sorted(types[code])

    ordered = sorted(courses.values(), key=lambda c: (c["subject"], c["number"]))
    return ordered, rows


def check_terms(courses: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> None:
    """Fail if the CSV does not hold exactly the term ``config.yaml`` claims.

    ``data.term`` is part of the config hash and ends up in every chunk, so a silent mismatch would
    tag an index with the wrong term.
    """
    missing = [c["code"] for c in courses if not c["term"]]
    if missing:
        raise ValueError(f"{len(missing)} courses have no term, e.g. {missing[:5]}")
    found = sorted({c["term"] for c in courses})
    expected = cfg["data"]["term"]
    if found != [expected]:
        raise ValueError(
            f"config data.term is {expected!r} but the CSV holds {found}; "
            "update config.yaml (the hash, and therefore the index names, must move with the data)"
        )


def build_prereqs(courses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prerequisite edge list, keeping only references to courses in this catalog.

    Mirrors ``dag.ts::prereqs()``, which filters ``prereqCodes`` through ``byCode()``. A course
    whose references all dangle (a prerequisite not offered this term) gets no entry at all.
    """
    known = {c["code"] for c in courses}
    edges = []
    for course in courses:
        resolved = [p for p in course["prereqCodes"] if p in known]
        if resolved:
            edges.append(
                {"course": course["code"], "prereqs": resolved, "raw": course["prereqRaw"]}
            )
    return edges


def matches_attrs(category: Mapping[str, Any], attrs: str) -> bool:
    """Whether a ``Degree Attributes`` string satisfies a Gen Ed category rule."""
    if not attrs:
        return False
    return any(s in attrs for s in category["any"]) and not any(
        s in attrs for s in category["none"]
    )


def build_requirements(courses: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict:
    """Gen Ed rules plus the placeholder for major requirements.

    ``courseCount`` is how many courses in *this* export satisfy each category -- derived from the
    same CSV in the same run, so it is a sanity check on the rules, not an independent source.
    """
    categories = []
    for category in GENED:
        categories.append(
            {
                "id": category["id"],
                "label": category["label"],
                "group": category["group"],
                "match": {"field": "attrs", "any": category["any"], "none": category["none"]},
                "courseCount": sum(1 for c in courses if matches_attrs(category, c["attrs"])),
            }
        )
    return {
        "term": cfg["data"]["term"],
        "source": (
            "almabot-web/src/lib/gened.ts (GENED), matched against the catalog's "
            "'Degree Attributes' column"
        ),
        "genEd": categories,
        "majors": [],
        "notes": {"majors": MAJORS_NOTE, "culturalStudies": CULTURAL_STUDIES_NOTE},
    }


# --- output ---------------------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> Path:
    """Write compact UTF-8 JSON (the corpus is a few MB; no indentation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    return path


def summarize(
    courses: Sequence[Mapping[str, Any]],
    prereqs: Sequence[Mapping[str, Any]],
    requirements: Mapping[str, Any],
    section_rows: int,
) -> list[str]:
    """Human-readable export stats: sizes, description quality, prerequisite coverage."""
    lengths = sorted(len(c["desc"]) for c in courses)
    empty = [c for c in courses if not c["desc"]]
    cross_ref = [c for c in courses if CROSS_REF_RE.match(c["desc"])]
    mentioned = sum(len(c["prereqCodes"]) for c in courses)
    resolved = sum(len(e["prereqs"]) for e in prereqs)
    with_attrs = sum(1 for c in courses if c["attrs"])
    truncated = sum(1 for n in lengths if n > 240)

    lines = [
        f"section rows read      {section_rows}",
        f"courses                {len(courses)}",
        f"term                   {courses[0]['term'] if courses else '-'}",
        (
            f"desc length p50 / max  {int(statistics.median(lengths)) if lengths else 0} / "
            f"{lengths[-1] if lengths else 0}"
        ),
        f"  longer than 240      {truncated}  (what the AlmaBot export truncated)",
        f"  empty                {len(empty)}"
        + (f"  {[c['code'] for c in empty[:5]]}" if empty else ""),
        f'  cross-reference only {len(cross_ref)}  ("Same as X. See X.")',
        f"prereq codes mentioned {mentioned}",
        f"  resolved in catalog  {resolved}  over {len(prereqs)} courses",
        f"  dangling (dropped)   {mentioned - resolved}",
        f"courses with attrs     {with_attrs}",
    ]
    lines += [f"  {c['id']:<12} {c['courseCount']:>4}  {c['label']}" for c in requirements["genEd"]]
    return lines


def export(cfg: Config) -> dict[str, Any]:
    """Read the pinned CSV and write all three data files. Returns the exported objects."""
    csv_path = verify_source(cfg)
    courses, section_rows = read_courses(csv_path)
    if not courses:
        raise ValueError(f"{rel_to_repo(csv_path)} yielded no courses; is it the catalog export?")
    check_terms(courses, cfg)
    prereqs = build_prereqs(courses)
    requirements = build_requirements(courses, cfg)
    write_json(COURSES_PATH, courses)
    write_json(PREREQS_PATH, prereqs)
    write_json(REQUIREMENTS_PATH, requirements)
    return {
        "courses": courses,
        "prereqs": prereqs,
        "requirements": requirements,
        "section_rows": section_rows,
        "source": csv_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-export the catalog CSV into data/ (full desc + term), for retrieval."
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    result = export(cfg)
    print(f"source   {rel_to_repo(result['source'])}  (sha256 pinned, config hash {cfg.hash})")
    for line in summarize(
        result["courses"], result["prereqs"], result["requirements"], result["section_rows"]
    ):
        print(line)
    for path in (COURSES_PATH, PREREQS_PATH, REQUIREMENTS_PATH):
        print(f"wrote    {rel_to_repo(path)}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
