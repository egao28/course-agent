"""Tool 3 of three: what a program still requires, given what a student has already taken.

"What Gen Eds do I still need?" is a question about a *rule set*, not about a course: the answer is
13 campus categories checked against a transcript, and neither the retriever nor the catalog lookup
can produce it. This module is that check, over the rules ``data/requirements.json`` carries.

Plain functions over Pydantic models, like the other two tools: #33 registers them with FastMCP,
#36 imports :func:`check_requirement` and calls it in-process.

**What this data supports, and what it does not.** ``requirements.json`` holds two keys that
matter here. ``genEd`` is the 13 campus categories, each a declarative rule against a course's
``Degree Attributes`` string. ``majors`` is two degree programs -- Computer Science, BS and
Statistics, BSLAS -- hand-transcribed in #32 from the catalog program pages into
``ingest/majors.json``, because ``course-catalog.csv`` lists sections and has never contained a
degree requirement. Everything else is still refused, and the refusal is the load-bearing part: a
degree audit assembled from an LLM's parametric memory is the most harmful output this agent could
produce -- fluent, specific, checkable only by the advisor the student is trying not to bother, and
wrong in a way that costs a semester. A program with no entry in the file returns
``supported=False`` with empty category lists, and a message naming the programs that do have one
-- a typed refusal #37/#38 can route to the hand-off exit, not a generic "no requirements found"
that a fluent answer node could talk its way past.

**A major is audited by course code; Gen Ed is audited by catalog record.** The two halves read
different fields for a reason. A Gen Ed category is a rule about a course's ``Degree Attributes``,
so it can only be decided for a course this export holds. A major requirement is a list of codes,
and the codes it names outlive one term: ``MATH 221`` is required by both programs and is not
offered in 2026-sp, and neither is ``ENG 300``. Auditing a major through the catalog index would
report the student who has taken MATH 221 as still needing it -- wrong in the direction they cannot
detect. So :func:`~ingest.majors.credit_rule` matches normalized code strings, the catalog is used
to *describe* the gap (``notOffered``, recorded at export), and ``unknown_courses`` stays a caveat
rather than a filter.

**What a course list cannot decide is reported, not guessed.** 128 hours, a 2.0 technical GPA, 40
hours of upper-division work, a language placement, an approved science-elective list that is not
in this export: those requirement groups carry ``{"kind": "advising"}`` in the data and come back
in ``not_checked``, with the catalog's own sentence and the page it is on. They are 8 of the 17
groups of the CS degree, so an audit that dropped them would read as "you are done" to a student
who is several requirements from done. ``satisfied`` and ``unsatisfied`` carry only what a
transcript actually decides.

**The Gen Ed rules are evaluated from the file, by the exporter's own evaluator.** Ports
``gened.ts::genEdStatus`` -- for each category, the first completed course whose ``attrs`` match --
but the matching itself is :func:`~ingest.export_almabot.matches_attrs` applied to the rule as
written in ``requirements.json``. The rules stopped being code in #3 precisely so there would be
one copy: the TS matcher for ``socbeh`` is wrong (the catalog's ``Social & Beh Sci - Soc Sci``
prefix contains the ``Beh Sci`` substring it tests, so all 118 Soc Sci courses match it too), open
#23 fixes that *in the data*, and deliberately leaves the TS broken. A tool carrying its own
transcription of those 13 rules would have to be fixed a third time. Evaluating the file means
this tool's answer changes when #23 merges, with no code change here -- which is the point, and is
asserted in the tests rather than assumed.

**A course can satisfy more than one category**, because the campus rule says so: ``AAS 100`` is
``Social & Beh Sci - Soc Sci, and Cultural Studies - US Minority``, and it counts for both. Each
category is resolved independently, exactly as the frontend did it. Which course is credited when
several qualify is the first in the order the caller listed them -- a transcript is the student's
own ordering, and re-ranking it would be a recommendation dressed up as an audit.

**Codes the catalog has never heard of are reported, not dropped.** ``data/courses.json`` is one
term: a real transcript is mostly courses this corpus does not contain -- ``MATH 221``, in #31's own
example, is not offered in 2026-sp. Silently ignoring them produces an audit that is wrong in the
one direction a student cannot detect ("you still need Quantitative Reasoning I" when the course
they took satisfies it), so they come back in ``unknown_courses`` and in the message. Whether a code
is in the catalog is decided by :func:`~agent.tools.courses.get_course`, so the three tools cannot
disagree about what the catalog holds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.courses import CourseDetail, course_index, get_course
from config import rel_to_repo
from ingest.export_almabot import REQUIREMENTS_PATH, matches_attrs
from ingest.majors import check_majors, credit_rule, normalize_program

#: The campus General Education audit, in the spellings a router or a person might produce.
#: Compared after :func:`normalize_program`, so "Gen-Ed", "GenEd" and "general education." all land
#: on a member of this set.
GENED_ALIASES = frozenset(
    {
        "gened",
        "geneds",
        "gen ed",
        "gen eds",
        "gen ed requirements",
        "general ed",
        "general education",
        "general education requirements",
    }
)

#: What the refusal offers alongside the degree programs. Two spellings, not the whole alias set
#: -- the message is read by a student, and a list of eight is noise.
GENED_OFFER = "'gened' (or 'general education')"


# --- models ---------------------------------------------------------------------------------------


class RequirementQuery(BaseModel):
    """A program, and the courses already completed.

    Declared as a model for the same reason the other two tools' queries are: #33 hands this schema
    to an MCP client and #35's router fills it from free text, so the descriptions below are the
    only instruction either one gets -- including the one that names the programs that can be
    audited, which is what keeps a router from confidently asking for one of the other 248.
    """

    model_config = ConfigDict(extra="forbid")

    major: str = Field(
        description=(
            "The program to check: 'gened' (or 'general education') for the campus General "
            "Education requirements, or a degree program this corpus carries requirements for -- "
            "'Computer Science, BS' and 'Statistics, BSLAS' today. Any other program returns "
            "supported=false and names the ones that are available; degree requirements that are "
            "not in the data must not be guessed at."
        )
    )
    taken_courses: list[str] = Field(
        default_factory=list,
        description=(
            "Course codes the student has completed, in any spelling: 'RHET 105', 'rhet 105', "
            "'RHET105'. Codes this catalog does not offer come back in unknown_courses."
        ),
    )


class CategoryStatus(BaseModel):
    """One requirement, and what the transcript did to it.

    ``id`` / ``label`` / ``group`` are copied from ``requirements.json`` rather than restated here:
    ``group`` is what lets an answer say "Cultural Studies: two of three" instead of listing 13
    categories flat, and for a degree it is the heading the program page files the requirement
    under ("Computer Science Technical Core").

    A Gen Ed category is carried by one course, so ``satisfied_by`` names it. A degree requirement
    can take eleven, so ``credited`` lists every course counted -- including any past the number
    required, because a student who took both CS 210 and CS 211 took both and an audit that hid one
    would look like it had lost a course. ``still_needed`` is how many more that requirement wants
    and ``options`` names the courses that would supply them, empty once nothing is missing and for
    a rule whose answer is a range rather than a list ("six CS 400-level courses").

    ``note`` is the catalog's own sentence about this requirement, and ``source`` the page it is
    on, so an answer can cite the line it is reporting rather than ask to be believed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    group: str
    satisfied_by: str | None = None
    credited: list[str] = Field(default_factory=list)
    still_needed: int = 0
    options: list[str] = Field(default_factory=list)
    note: str | None = None
    source: str | None = None


class RequirementStatus(BaseModel):
    """A program audit, or a typed refusal to attempt one.

    ``supported`` is the field to branch on, and it is not the same question as "is anything
    unsatisfied": an unsupported program comes back with *every* list empty, so there is nothing
    for an answer node to paraphrase into a degree plan. ``message`` is the sentence #38 turns into
    the hand-off, and on a supported audit it is the caveat -- which courses the catalog does not
    hold, and how much of the degree a course list cannot decide.

    ``major`` echoes the caller's own spelling (whitespace collapsed) so the answer can name what it
    was asked about; ``program`` is what the file calls it, which is what a citation needs.

    ``not_checked`` is the third list and the honest one. A degree is more than its course lists --
    hours, GPA, placement, approved lists nobody exported -- and those requirements are reported
    here, unevaluated, with their catalog text, and are never counted as satisfied.
    ``catalog_year`` and ``source`` say which edition of the requirements was audited against,
    which matters because a student is held to the catalog year they enrolled under.
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool
    major: str
    program: str | None = None
    catalog_year: str | None = None
    source: str | None = None
    satisfied: list[CategoryStatus] = Field(default_factory=list)
    unsatisfied: list[CategoryStatus] = Field(default_factory=list)
    not_checked: list[CategoryStatus] = Field(default_factory=list)
    unknown_courses: list[str] = Field(default_factory=list)
    message: str | None = None


# --- which program ----------------------------------------------------------------------------


# ``normalize_program`` is imported rather than written here: it decides both which spellings reach
# GENED_ALIASES and which reach a program's ``aliases`` list in ingest/majors.json, and two copies
# of that rule would make a program reachable by a spelling the file's own uniqueness check never
# saw.
def program_labels(doc: Mapping[str, Any]) -> list[str]:
    """The programs this corpus can audit, as a student would name them. Used by the refusal."""
    return [major["label"] for major in doc.get("majors", ())]


def resolve_program(doc: Mapping[str, Any], major: str) -> dict[str, Any] | None:
    """The program ``major`` names, or ``None`` if the file holds no such program.

    Matched on the normalized form of the label, the id, or one of the hand-written aliases, and
    matched *exactly*: "statistics and computer science" is a different degree from "statistics",
    and a substring match would audit a student against a program they are not in.
    """
    asked = normalize_program(major)
    for program in doc.get("majors", ()):
        spellings = {normalize_program(a) for a in program["aliases"]}
        spellings |= {normalize_program(program["label"]), normalize_program(program["id"])}
        if asked in spellings:
            return dict(program)
    return None


# --- the rules --------------------------------------------------------------------------------


def load_requirements(path: Path | None = None) -> dict[str, Any]:
    """The whole requirements document, checked. Uncached -- :func:`requirements_doc` is cached.

    The *whole* document, not ``genEd`` alone (which is all ``ingest.chunk.load_gened_categories``
    needs), because ``majors`` is where the degrees are: which programs can be audited and which
    are declined is a property of this file, read rather than assumed -- before #32 the answer was
    always "declined", and a tool that had kept that hard-coded would go on refusing a program the
    data now holds.
    """
    path = REQUIREMENTS_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{rel_to_repo(path)} is missing; run `uv run python -m ingest.export_almabot` first"
        )
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    check_requirements(doc, path)
    return doc


def check_requirements(doc: Mapping[str, Any], path: Path) -> None:
    """Refuse a document this tool would answer *wrongly* from.

    Each check is one such answer. No categories at all is a vacuously complete audit ("nothing
    left to take"). A repeated id double-counts one category. A rule naming a course field the
    lookup does not return would silently match nothing, i.e. report every category unsatisfied.

    The majors are checked by :func:`ingest.majors.check_majors`, the same function that guards
    the hand-written source they are exported from: a rule kind with no evaluator, an ``n`` no
    transcript can reach, a code spelled outside the catalog's own form. One check belongs here
    rather than there -- a program must not answer to a spelling Gen Ed already answers to, since
    ``ingest/majors.json`` has no way to know which spellings this module reserves.
    """
    where = rel_to_repo(path)
    categories = doc.get("genEd")
    if not categories:
        raise ValueError(
            f"{where} carries no Gen Ed categories, so every audit would report nothing left to "
            "take; re-run `uv run python -m ingest.export_almabot`"
        )
    seen: set[str] = set()
    for category in categories:
        missing = [key for key in ("id", "label", "group", "match") if key not in category]
        if missing:
            raise ValueError(f"{where}: Gen Ed category {category} is missing {missing}")
        if category["id"] in seen:
            raise ValueError(
                f"{where}: Gen Ed category {category['id']!r} appears twice, so it would be "
                "counted twice; re-run `uv run python -m ingest.export_almabot`"
            )
        seen.add(category["id"])
        missing = [key for key in ("field", "any", "none") if key not in category["match"]]
        if missing:
            raise ValueError(
                f"{where}: Gen Ed category {category['id']!r} has a match rule missing {missing}; "
                "ingest.export_almabot.matches_attrs reads all three"
            )
        field = category["match"]["field"]
        if field not in CourseDetail.model_fields:
            raise ValueError(
                f"{where}: Gen Ed category {category['id']!r} matches on course field {field!r}, "
                "which agent.tools.courses.CourseDetail does not return, so the rule could never "
                "match and the category would always read unsatisfied"
            )
    majors = doc.get("majors", [])
    check_majors(majors, where)
    for major in majors:
        for alias in [*major["aliases"], major["label"], major["id"]]:
            if normalize_program(alias) in GENED_ALIASES:
                raise ValueError(
                    f"{where}: {major['id']} answers to {alias!r}, which this module reads as a "
                    "request for the campus General Education audit, so the degree could never be "
                    "reached; rename the alias in ingest/majors.json"
                )


def check_term(doc: Mapping[str, Any], terms: Sequence[str], path: Path) -> None:
    """Fail unless the rules describe the catalog the transcript is checked against.

    ``courseCount`` on every category is measured during the export, and the messages here name the
    catalog's term, so a ``requirements.json`` left over from another run would quietly describe a
    different corpus than the one deciding which of the student's courses exist.
    """
    if doc.get("term") not in terms:
        raise ValueError(
            f"{rel_to_repo(path)} describes term {doc.get('term')!r} but the catalog covers "
            f"{', '.join(terms)}; the two files are from different runs, so re-run "
            "`uv run python -m ingest.export_almabot`"
        )


#: The one document, read on first use, like the catalog index and the prerequisite graph.
_REQUIREMENTS: dict[str, Any] | None = None


def requirements_doc() -> dict[str, Any]:
    """The cached document, cross-checked against the catalog on the first call."""
    global _REQUIREMENTS
    if _REQUIREMENTS is None:
        doc = load_requirements()
        check_term(doc, course_index().terms, REQUIREMENTS_PATH)
        _REQUIREMENTS = doc
    return _REQUIREMENTS


def clear_cache() -> None:
    """Drop the cached document, so the next audit re-reads the rules."""
    global _REQUIREMENTS
    _REQUIREMENTS = None


# --- the transcript, and what to say about it -------------------------------------------------


class Transcript(NamedTuple):
    """What the student said they have taken, read three ways.

    ``found`` is the catalog records, which is all a Gen Ed rule can be evaluated against.
    ``codes`` is every entry that was a course code at all, normalized -- including the ones this
    one-term catalog does not offer, because a degree requirement names codes and MATH 221 is
    required by both programs and runs in the fall. ``unknown`` is what the catalog does not hold,
    for the caveat.
    """

    found: list[CourseDetail]
    codes: list[str]
    unknown: list[str]


def resolve_taken(taken_courses: Sequence[str]) -> Transcript:
    """Read the transcript into catalog records, course codes, and what is neither.

    Every list keeps the order it was given and holds each course once: a transcript that lists a
    retake twice should not credit a requirement twice over, and an unknown code repeated is one
    unknown code. A blank entry is dropped -- it names nothing that could be reported back.
    """
    found: list[CourseDetail] = []
    codes: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for entry in taken_courses:
        detail = get_course(entry)
        label = detail.code or " ".join(entry.split())
        if not label or label in seen:
            continue
        seen.add(label)
        if detail.code:
            codes.append(detail.code)
        if detail.found:
            found.append(detail)
        else:
            unknown.append(label)
    return Transcript(found=found, codes=codes, unknown=unknown)


def satisfied_by(category: Mapping[str, Any], taken: Sequence[CourseDetail]) -> str | None:
    """The code of the first completed course whose record satisfies ``category``, if any.

    ``gened.ts::genEdStatus`` takes the first match in the completed list and so does this; the rule
    itself is read from the category through the exporter's evaluator, and the field it applies to
    is the one the rule names (:func:`check_requirements` has already refused a field the lookup
    does not return).
    """
    field = category["match"]["field"]
    for course in taken:
        if matches_attrs(category, getattr(course, field) or ""):
            return course.code
    return None


def unknown_note(unknown: Sequence[str], terms: Sequence[str]) -> str:
    """The sentence that keeps unrecognized courses visible in the answer, not just in the field."""
    one = len(unknown) == 1
    it, was, satisfies = ("it", "was", "satisfies") if one else ("they", "were", "satisfy")
    return (
        f"{len(unknown)} of the courses given ({', '.join(unknown)}) "
        f"{'is' if one else 'are'} not in this catalog, which covers {' and '.join(terms)} only, "
        f"so {it} {was} not counted toward any category -- anything {it} {satisfies} is missing "
        "from this audit"
    )


def unsupported_note(major: str, programs: Sequence[str]) -> str:
    """Why a program cannot be audited, in a sentence a student can act on.

    Names the gap and what is available rather than the ticket: with #32's data the gap is no
    longer "this catalog has no degree requirements at all" but "it has two of them", and a
    refusal that does not say which two sends a student away from an answer that exists. The
    honest next step for every other program is still a human advisor. Issue numbers live in this
    module's docstring, where the reader is a developer.
    """
    available = "; ".join(programs)
    carries = (
        f"this corpus carries degree requirements for {len(programs)} hand-encoded "
        f"program{'s' if len(programs) != 1 else ''} only -- {available} -- plus {GENED_OFFER} "
        "for the campus General Education requirements"
        if programs
        else (
            f"this catalog export lists course sections and carries no major or degree "
            f"requirements at all, so {GENED_OFFER} is the one thing it can check"
        )
    )
    return (
        f"{major!r} cannot be checked: {carries}. Its requirements are not in this data, and "
        "assembling them from anywhere else would be a fluent guess at a document a student acts "
        "on; for a degree audit this corpus does not hold, the department's advisor or the "
        "official catalog program page is the source."
    )


def major_note(program: Mapping[str, Any], unknown: Sequence[str], terms: Sequence[str]) -> str:
    """The caveats a degree audit is only honest with, in the order they change what it means.

    Which edition was audited (a student is held to their own catalog year, and this corpus is one
    term of courses), how much of the degree a course list cannot decide, which required courses
    this term does not offer, and which of the student's own courses the catalog does not hold.
    Always a sentence: unlike the Gen Ed audit, there is no state of this data in which a degree
    audit needs no caveat.
    """
    groups = program["groups"]
    parts = [
        (
            f"Audited against the {program['catalogYear']} catalog entry for {program['label']} "
            f"({program['source']}), over the {' and '.join(terms)} course catalog."
        )
    ]
    unchecked = [g for g in groups if g["rule"]["kind"] == "advising"]
    if unchecked:
        named = ", ".join(g["label"] for g in unchecked[:3])
        rest = f", and {len(unchecked) - 3} more" if len(unchecked) > 3 else ""
        parts.append(
            f"{len(unchecked)} of the {len(groups)} requirement groups ({named}{rest}) are not "
            "decided by a list of course codes; they come back in not_checked, unevaluated, and a "
            "student is held to them all the same."
        )
    absent = sorted({code for g in groups for code in g["notOffered"]})
    if absent:
        one = len(absent) == 1
        parts.append(
            f"{len(absent)} course{'' if one else 's'} these requirements name "
            f"({', '.join(absent)}) {'is' if one else 'are'} not offered in "
            f"{' or '.join(terms)}, so {'it is' if one else 'they are'} matched by code alone -- a "
            "student who has taken one is credited for it."
        )
    if unknown:
        one = len(unknown) == 1
        parts.append(
            f"{len(unknown)} of the courses given ({', '.join(unknown)}) {'is' if one else 'are'} "
            "not in this catalog: a course code still counts toward a requirement that names it, "
            "and an entry that is not a course code counts toward nothing."
        )
    return " ".join(parts)


# --- the audit ------------------------------------------------------------------------------------


def audit_gened(doc: Mapping[str, Any], taken: Sequence[CourseDetail]) -> tuple[list, list]:
    """The 13 campus categories against the courses the catalog holds, in the file's order."""
    satisfied: list[CategoryStatus] = []
    unsatisfied: list[CategoryStatus] = []
    for category in doc["genEd"]:
        by = satisfied_by(category, taken)
        status = CategoryStatus(
            id=category["id"],
            label=category["label"],
            group=category["group"],
            satisfied_by=by,
            credited=[by] if by else [],
            still_needed=0 if by else 1,
        )
        (satisfied if by else unsatisfied).append(status)
    return satisfied, unsatisfied


def audit_major(program: Mapping[str, Any], codes: Sequence[str]) -> tuple[list, list, list]:
    """One degree program against a transcript of course codes, group by group.

    Three lists, and the third is the one that keeps this honest: an ``advising`` group is not
    evaluated at all, because nothing in a list of course codes decides a GPA or an hour total.
    Each group is resolved independently, like a Gen Ed category -- a course counted twice across
    two groups is prevented in the data, by the exclusions the rules carry, and not by this walk.
    """
    satisfied: list[CategoryStatus] = []
    unsatisfied: list[CategoryStatus] = []
    not_checked: list[CategoryStatus] = []
    for group in program["groups"]:
        common = {
            "id": group["id"],
            "label": group["label"],
            "group": group["group"],
            "note": group.get("note"),
            "source": group.get("source"),
        }
        if group["rule"]["kind"] == "advising":
            not_checked.append(CategoryStatus(**common))
            continue
        credit = credit_rule(group["rule"], codes)
        status = CategoryStatus(
            **common,
            satisfied_by=credit.credited[0] if len(credit.credited) == 1 else None,
            credited=credit.credited,
            still_needed=credit.still_needed,
            options=credit.options,
        )
        (unsatisfied if credit.still_needed else satisfied).append(status)
    return satisfied, unsatisfied, not_checked


def check_requirement(major: str, taken_courses: Sequence[str] | None = None) -> RequirementStatus:
    """What ``major`` still requires given ``taken_courses``, or a typed refusal to guess.

    Takes bare arguments rather than a :class:`RequirementQuery` so the graph calls it as an
    ordinary function; the model is what #33 declares to MCP, and its validation runs here either
    way.

    Three outcomes. ``gened`` audits the 13 campus categories against the catalog records of the
    courses given. A program ``requirements.json`` carries audits that degree's requirement groups
    against the course codes given, and reports every group a course list cannot decide as
    unchecked. Anything else comes back ``supported=False`` with every list empty and a message
    naming what *is* available, because the requirements are not in the data and inventing them is
    the failure mode this tool exists to prevent.
    """
    query = RequirementQuery(major=major, taken_courses=list(taken_courses or []))
    asked = " ".join(query.major.split())
    doc = requirements_doc()
    transcript = resolve_taken(query.taken_courses)

    if normalize_program(query.major) in GENED_ALIASES:
        satisfied, unsatisfied = audit_gened(doc, transcript.found)
        return RequirementStatus(
            supported=True,
            major=asked,
            program="General Education",
            source=doc.get("source"),
            satisfied=satisfied,
            unsatisfied=unsatisfied,
            unknown_courses=transcript.unknown,
            message=(
                unknown_note(transcript.unknown, course_index().terms)
                if transcript.unknown
                else None
            ),
        )

    program = resolve_program(doc, query.major)
    if program is None:
        return RequirementStatus(
            supported=False, major=asked, message=unsupported_note(asked, program_labels(doc))
        )

    satisfied, unsatisfied, not_checked = audit_major(program, transcript.codes)
    return RequirementStatus(
        supported=True,
        major=asked,
        program=program["label"],
        catalog_year=program["catalogYear"],
        source=program["source"],
        satisfied=satisfied,
        unsatisfied=unsatisfied,
        not_checked=not_checked,
        unknown_courses=transcript.unknown,
        message=major_note(program, transcript.unknown, course_index().terms),
    )
