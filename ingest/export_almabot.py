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
    One record per course, sorted by (subject, number). ``prereqRaw`` is the *whole* requirements
    block -- the catalog runs it from the ``Prerequisite:`` label to the end of the description --
    and ``prereqCodes`` / ``coreqCodes`` hold the course codes it names, resolved or not.
``data/prereqs.json``
    The prerequisite edge list, filtered to codes that exist in this catalog (the filter
    ``dag.ts::prereqs()`` applies at read time). Courses with no surviving edge are omitted.
``data/requirements.json``
    The 13 campus Gen Ed categories from ``almabot-web/src/lib/gened.ts``, as declarative
    ``Degree Attributes`` substring rules, and the two degree programs hand-transcribed into
    ``ingest/majors.json`` -- major requirements are not in the CSV, which lists sections, so that
    file is their source and this export only checks and measures them (see :mod:`ingest.majors`).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from config import REPO_ROOT, Config, load_config, rel_to_repo, verify_source
from ingest.majors import build_majors, check_coverage

DATA_DIR = REPO_ROOT / "data"
COURSES_PATH = DATA_DIR / "courses.json"
PREREQS_PATH = DATA_DIR / "prereqs.json"
REQUIREMENTS_PATH = DATA_DIR / "requirements.json"

# A course code as written in prose: "CS 374", "CS374", "MATH 241".
CODE_RE = re.compile(r"\b([A-Z]{2,4})\s?(\d{3})\b")
# The label that opens the requirements block; everything after it belongs to that block.
PREREQ_HEAD_RE = re.compile(r"Prerequisite[s]?:\s*")
# Descriptions that carry no content of their own: "Same as HIST 283. See HIST 283."
CROSS_REF_RE = re.compile(r"^Same as [^.]*\.\s*See [^.]*\.$")

# Sentence boundary: a stop followed by something that starts a new sentence. Splitting on ". "
# alone would cut "Ph.D. standing in BADM" down to "Ph.D" -- the abbreviations below are never
# treated as a boundary, and a continuation starting lower-case is not one either.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
# fmt: off
ABBREVIATIONS = frozenset({
    "e.g", "i.e", "etc", "vs", "approx", "no", "jr", "sr", "dr", "st",
    "ph.d", "m.s", "b.s", "m.a", "b.a", "m.eng", "sc.d", "ed.d", "ed.m", "m.b.a", "m.d", "d.v.m",
    "u.s",
})
# fmt: on

# A continuation sentence that states a corequisite: "Credit or concurrent registration in AE 323".
COREQ_RE = re.compile(
    r"credit or concurrent (?:registration|enrollment)"
    r"|concurrent(?:ly)? (?:registration|enrollment)"
    r"|concurrent enrollment"
    r"|enroll(?:ed)? concurrently"
    r"|corequisite",
    re.IGNORECASE,
)
# ...unless it is optional or negated: "Concurrent enrollment with JOUR 317 is allowed."
SOFT_REQUIREMENT_RE = re.compile(
    r"\bis allowed\b|preferred|recommend|suggest|encouraged|not required|optional|cannot|may not"
    r"|not given",
    re.IGNORECASE,
)

TERM_ABBR = {"spring": "sp", "summer": "su", "fall": "fa", "winter": "wi"}

# Gen Ed categories, transcribed from GENED in almabot-web/src/lib/gened.ts. The TS matchers
# are closures over the "Degree Attributes" string; here each rule is declarative so that
# requirements.json can carry it verbatim and :func:`matches_attrs` can evaluate it straight
# from the file: a course matches when attrs contains ANY of `any` and NONE of `none`.
#   - qr1 replaces the lookahead /Quantitative Reasoning I(?!I)/ with an explicit `none`.
#   - cultwest keeps both spellings the TS matcher ORs together.
# fmt: off -- one line per category, aligned for review against the TS source
GENED: tuple[dict[str, Any], ...] = (
    {
        "id": "comp1",
        "label": "Composition I",
        "group": "Composition",
        "match": {"field": "attrs", "any": ["Composition I"], "none": []},
    },
    {
        "id": "advcomp",
        "label": "Advanced Composition",
        "group": "Composition",
        "match": {"field": "attrs", "any": ["Advanced Composition"], "none": []},
    },
    {
        "id": "qr1",
        "label": "Quantitative Reasoning I",
        "group": "Quantitative Reasoning",
        "match": {
            "field": "attrs",
            "any": ["Quantitative Reasoning I"],
            "none": ["Quantitative Reasoning II"],
        },
    },
    {
        "id": "qr2",
        "label": "Quantitative Reasoning II",
        "group": "Quantitative Reasoning",
        "match": {"field": "attrs", "any": ["Quantitative Reasoning II"], "none": []},
    },
    {
        "id": "natphys",
        "label": "Physical Sciences",
        "group": "Natural Sciences & Tech",
        "match": {"field": "attrs", "any": ["Phys Sciences"], "none": []},
    },
    {
        "id": "natlife",
        "label": "Life Sciences",
        "group": "Natural Sciences & Tech",
        "match": {"field": "attrs", "any": ["Life Sciences"], "none": []},
    },
    {
        "id": "humhist",
        "label": "Historical & Philosophical",
        "group": "Humanities & the Arts",
        "match": {"field": "attrs", "any": ["Hist & Phil"], "none": []},
    },
    {
        "id": "humlit",
        "label": "Literature & the Arts",
        "group": "Humanities & the Arts",
        "match": {"field": "attrs", "any": ["Lit & Arts"], "none": []},
    },
    {
        "id": "socsoc",
        "label": "Social Sciences",
        "group": "Social & Behavioral Sci",
        "match": {"field": "attrs", "any": ["Soc Sci"], "none": []},
    },
    {
        "id": "socbeh",
        "label": "Behavioral Sciences",
        "group": "Social & Behavioral Sci",
        "match": {"field": "attrs", "any": ["Beh Sci"], "none": []},
    },
    {
        "id": "cultwest",
        "label": "Western/Comparative Cultures",
        "group": "Cultural Studies",
        "match": {
            "field": "attrs",
            "any": ["Cultural Studies - Western", "Western course"],
            "none": [],
        },
    },
    {
        "id": "cultnonwest",
        "label": "Non-Western Cultures",
        "group": "Cultural Studies",
        "match": {"field": "attrs", "any": ["Non-West"], "none": []},
    },
    {
        "id": "cultus",
        "label": "US Minority Cultures",
        "group": "Cultural Studies",
        "match": {"field": "attrs", "any": ["US Minority"], "none": []},
    },
)
# fmt: on

MAJORS_NOTE = (
    "Major and degree requirements are not present in course-catalog.csv, which lists sections "
    "only. The programs in `majors` were transcribed by hand from the catalog.illinois.edu "
    "program pages each one names, for the catalog year in force for this term; they are two of "
    "the roughly 250 the campus offers. Requirements a course list cannot decide -- credit hours, "
    "GPA, placement, approved lists -- are marked kind 'advising' and are never counted as met."
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
    """The requirements block: everything from the ``Prerequisite:`` label to the end, or ``""``.

    The catalog puts the whole enrollment-requirements block last, so the label runs to the end of
    the description. The frontend instead stops at the first sentence break, which drops text for
    483 of 2,732 courses and cuts mid-abbreviation ("Prerequisite: Ph.D. standing in BADM" becomes
    ``"Ph.D"``) -- exactly the truncation this export exists to undo.
    """
    m = PREREQ_HEAD_RE.search(desc or "")
    return (desc or "")[m.end() :].strip() if m else ""


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, without breaking inside ``ABBREVIATIONS``.

    ``"Ph.D. standing in BADM. Consent required."`` -> two sentences, not three.
    """
    sentences: list[str] = []
    pending = ""
    for part in SENTENCE_SPLIT_RE.split(text.strip()):
        pending = f"{pending} {part}" if pending else part
        words = pending.rstrip(".").split()
        # An abbreviation can open a bracket or quote: "(e.g. STAT 100)".
        tail = words[-1].lower().lstrip("([\"'") if words else ""
        if tail in ABBREVIATIONS:
            continue  # the stop belonged to an abbreviation: keep reading
        sentences.append(pending)
        pending = ""
    if pending:
        sentences.append(pending)
    return [s for s in sentences if s]


def _codes_in(text: str, code: str) -> set[str]:
    """Normalized course codes in a span of text, minus the course itself."""
    return {f"{m.group(1)} {m.group(2)}" for m in CODE_RE.finditer(text)} - {code}


def prereq_codes(prereq_raw: str, code: str) -> list[str]:
    """Course codes required *before* the course, normalized and self-excluded.

    Only the prerequisite statement itself -- the first sentence of the block -- contributes.
    Later sentences state something else: a corequisite (:func:`coreq_codes`), a restriction, a
    recommendation ("Recommended: PHYS 213"), or an exclusion ("May not be taken concurrently with
    ANTH 391"). Reading codes out of those would put edges into the DAG that are not prerequisites
    at all, so they are left to the raw text.

    Not filtered against the catalog -- ``prereqs.json`` is where dangling references are dropped.
    """
    sentences = split_sentences(prereq_raw)
    return sorted(_codes_in(sentences[0], code)) if sentences else []


def coreq_codes(prereq_raw: str, code: str) -> list[str]:
    """Course codes to be taken *alongside* the course.

    A continuation sentence counts only when it states a corequisite outright ("Credit or
    concurrent registration in AE 323 and AE 353.", "Corequisites: MATH 231, PHYS 212.") and does
    not hedge it ("Concurrent enrollment with JOUR 317 is allowed."). Corequisites are kept apart
    from prerequisites deliberately: they are mutual often enough (DANC 498 <-> DANC 499) that
    folding them into the prerequisite DAG would create cycles out of correct data.
    """
    sentences = split_sentences(prereq_raw)
    found: set[str] = set()
    for sentence in sentences[1:]:
        if COREQ_RE.search(sentence) and not SOFT_REQUIREMENT_RE.search(sentence):
            found |= _codes_in(sentence, code)
    return sorted(found - set(prereq_codes(prereq_raw, code)))


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


class Catalog(NamedTuple):
    """What one CSV read yields, including facts that de-duplication would otherwise hide."""

    courses: list[dict[str, Any]]
    section_rows: int
    #: every term seen across section rows -> how many rows carried it. De-duplication keeps only
    #: the first row per course, so this is the only place a mixed-term CSV is still visible.
    terms: Counter[str]


def read_courses(csv_path: Path) -> Catalog:
    """De-duplicate section rows into courses.

    Course-level columns (Name, Description, Credit Hours, Degree Attributes) are identical across
    the sections of a course, so the first row of each code wins -- the same rule build_data.py
    uses. Section rows still contribute the counts and the ``partOfTerm`` / ``types`` sets, which
    are the only section facts worth keeping as retrieval metadata, and every row's term is tallied
    for :func:`check_terms`.
    """
    courses: dict[str, dict[str, Any]] = {}
    sections: dict[str, int] = {}
    part_of_term: dict[str, set[str]] = {}
    types: dict[str, set[str]] = {}
    terms: Counter[str] = Counter()
    rows = 0

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            subject = (row.get("Subject") or "").strip()
            number = (row.get("Number") or "").strip()
            if not subject or not number:
                continue
            rows += 1
            terms[row_term(row)] += 1
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
                    "coreqCodes": coreq_codes(prereq_raw, code),
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
    return Catalog(courses=ordered, section_rows=rows, terms=terms)


def check_terms(catalog: Catalog, cfg: Mapping[str, Any]) -> None:
    """Fail unless *every section row* carries the term ``config.yaml`` claims.

    ``data.term`` is part of the config hash and ends up in every chunk, so a silent mismatch would
    tag an index with the wrong term. The check runs on ``catalog.terms`` -- the tally over all
    rows -- rather than on the de-duplicated courses: de-duplication keeps only the first row of
    each course, so a CSV mixing 2026-sp and 2025-fa would pass a per-course check while
    ``sectionCount`` silently blended the two terms.
    """
    missing = [c["code"] for c in catalog.courses if not c["term"]]
    if missing:
        raise ValueError(f"{len(missing)} courses have no term, e.g. {missing[:5]}")
    expected = cfg["data"]["term"]
    if set(catalog.terms) != {expected}:
        found = ", ".join(f"{t} ({n} rows)" for t, n in sorted(catalog.terms.items()))
        raise ValueError(
            f"config data.term is {expected!r} but the CSV holds {found or 'no terms'}; "
            "every section row must be in the configured term -- update config.yaml "
            "(the hash, and therefore the index names, must move with the data)"
        )


def build_prereqs(courses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prerequisite edge list, keeping only references to courses in this catalog.

    Mirrors ``dag.ts::prereqs()``, which filters ``prereqCodes`` through ``byCode()``. ``coreqs``
    rides along in its own field -- a consumer building the prerequisite DAG reads ``prereqs`` and
    is unaffected. A course whose references all dangle (a prerequisite not offered this term) gets
    no entry at all.
    """
    known = {c["code"] for c in courses}
    edges = []
    for course in courses:
        prereqs = [p for p in course["prereqCodes"] if p in known]
        coreqs = [p for p in course["coreqCodes"] if p in known]
        if prereqs or coreqs:
            edges.append(
                {
                    "course": course["code"],
                    "prereqs": prereqs,
                    "coreqs": coreqs,
                    "raw": course["prereqRaw"],
                }
            )
    return edges


def matches_attrs(category: Mapping[str, Any], attrs: str) -> bool:
    """Whether a ``Degree Attributes`` string satisfies a Gen Ed category rule.

    Reads ``category["match"]`` -- the same nested shape :func:`build_requirements` writes into
    ``requirements.json`` -- so a consumer can load a category straight out of the file and
    evaluate it with this function instead of re-implementing the qr1 exclusion. ``match.field``
    names the course field the rule applies to, for a consumer that holds a whole course.
    """
    if not attrs:
        return False
    rule = category["match"]
    return any(s in attrs for s in rule["any"]) and not any(s in attrs for s in rule["none"])


def build_requirements(courses: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict:
    """Gen Ed rules and the hand-encoded degree programs, both measured against this export.

    ``courseCount`` is how many courses in *this* export satisfy each category -- derived from the
    same CSV in the same run, so it is a sanity check on the rules, not an independent source. The
    same field on a major's requirement group means the same thing, and
    :func:`~ingest.majors.build_majors` adds the program page and catalog year each group was read
    off, so an audit can cite the source of every line it reports.
    """
    categories = []
    for category in GENED:
        categories.append(
            {
                **category,
                "match": dict(category["match"]),
                "courseCount": sum(1 for c in courses if matches_attrs(category, c["attrs"])),
            }
        )
    return {
        "term": cfg["data"]["term"],
        "source": (
            "genEd: almabot-web/src/lib/gened.ts (GENED), matched against the catalog's "
            "'Degree Attributes' column. majors: ingest/majors.json, hand-transcribed from the "
            "catalog.illinois.edu program page each one names"
        ),
        "genEd": categories,
        "majors": build_majors(courses),
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
    catalog: Catalog,
    prereqs: Sequence[Mapping[str, Any]],
    requirements: Mapping[str, Any],
) -> list[str]:
    """Human-readable export stats: sizes, description quality, prerequisite coverage."""
    courses = catalog.courses
    lengths = sorted(len(c["desc"]) for c in courses)
    empty = [c for c in courses if not c["desc"]]
    cross_ref = [c for c in courses if CROSS_REF_RE.match(c["desc"])]
    mentioned = sum(len(c["prereqCodes"]) for c in courses)
    resolved = sum(len(e["prereqs"]) for e in prereqs)
    prereq_courses = sum(1 for e in prereqs if e["prereqs"])
    coreq_resolved = sum(len(e["coreqs"]) for e in prereqs)
    coreq_courses = sum(1 for e in prereqs if e["coreqs"])
    with_attrs = sum(1 for c in courses if c["attrs"])
    untruncated = sum(1 for n in lengths if n > 240)

    lines = [
        f"section rows read      {catalog.section_rows}",
        f"courses                {len(courses)}",
        "term                   "
        + ", ".join(f"{t} ({n} rows)" for t, n in sorted(catalog.terms.items())),
        (
            f"desc length p50 / max  {int(statistics.median(lengths)) if lengths else 0} / "
            f"{lengths[-1] if lengths else 0}"
        ),
        f"  longer than 240      {untruncated}  (what the AlmaBot export truncated)",
        f"  empty                {len(empty)}"
        + (f"  {[c['code'] for c in empty[:5]]}" if empty else ""),
        f'  cross-reference only {len(cross_ref)}  ("Same as X. See X.")',
        f"prereq codes mentioned {mentioned}",
        f"  resolved in catalog  {resolved}  over {prereq_courses} courses",
        f"  dangling (dropped)   {mentioned - resolved}",
        f"corequisite edges      {coreq_resolved}  over {coreq_courses} courses",
        f"courses with attrs     {with_attrs}",
    ]
    lines += [f"  {c['id']:<12} {c['courseCount']:>4}  {c['label']}" for c in requirements["genEd"]]
    for major in requirements["majors"]:
        checked = [g for g in major["groups"] if g["rule"]["kind"] != "advising"]
        absent = sorted({code for g in major["groups"] for code in g["notOffered"]})
        advising = len(major["groups"]) - len(checked)
        lines += [
            f"{major['label']}  ({major['catalogYear']} catalog)",
            (
                f"  requirement groups   {len(major['groups'])}  ({len(checked)} checked against "
                f"a course list, {advising} needing an advisor)"
            ),
            f"  named, not this term {len(absent)}" + (f"  {absent}" if absent else ""),
        ]
    return lines


def export(cfg: Config) -> dict[str, Any]:
    """Read the pinned CSV and write all three data files. Returns the exported objects."""
    csv_path = verify_source(cfg)
    catalog = read_courses(csv_path)
    if not catalog.courses:
        raise ValueError(f"{rel_to_repo(csv_path)} yielded no courses; is it the catalog export?")
    check_terms(catalog, cfg)
    prereqs = build_prereqs(catalog.courses)
    requirements = build_requirements(catalog.courses, cfg)
    check_coverage(requirements["majors"], rel_to_repo(REQUIREMENTS_PATH))
    write_json(COURSES_PATH, catalog.courses)
    write_json(PREREQS_PATH, prereqs)
    write_json(REQUIREMENTS_PATH, requirements)
    return {
        "catalog": catalog,
        "courses": catalog.courses,
        "prereqs": prereqs,
        "requirements": requirements,
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
    for line in summarize(result["catalog"], result["prereqs"], result["requirements"]):
        print(line)
    for path in (COURSES_PATH, PREREQS_PATH, REQUIREMENTS_PATH):
        print(f"wrote    {rel_to_repo(path)}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
