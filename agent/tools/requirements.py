"""Tool 3 of three: what a program still requires, given what a student has already taken.

"What Gen Eds do I still need?" is a question about a *rule set*, not about a course: the answer is
13 campus categories checked against a transcript, and neither the retriever nor the catalog lookup
can produce it. This module is that check, over the rules ``data/requirements.json`` carries.

Plain functions over Pydantic models, like the other two tools: #33 registers them with FastMCP,
#36 imports :func:`check_requirement` and calls it in-process.

**The half this data supports, and the half it does not.** ``requirements.json`` holds two keys
that matter here. ``genEd`` is the 13 categories, each a declarative rule against a course's
``Degree Attributes`` string. ``majors`` is ``[]`` -- empty, because ``course-catalog.csv`` lists
sections and has never contained a degree requirement; sourcing that is #32. So
``check_requirement("Computer Science, BS", ...)`` has nothing to answer *from*, and the one thing
it must never do is answer anyway. A degree audit assembled from an LLM's parametric memory is the
most harmful output this agent could produce: it is fluent, specific, checkable only by the advisor
the student is trying not to bother, and wrong in a way that costs a semester. Anything that is not
Gen Ed therefore returns ``supported=False`` with empty category lists and a message that says what
is missing and what is available -- a typed refusal #37/#38 can route to the hand-off exit, not a
generic "no requirements found" that a fluent answer node could talk its way past.

**The rules are evaluated from the file, by the exporter's own evaluator.** Ports
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
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.courses import CourseDetail, course_index, get_course
from config import rel_to_repo
from ingest.export_almabot import REQUIREMENTS_PATH, matches_attrs

#: The one program this tool can audit, in the spellings a router or a person might produce.
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

#: What the refusal offers instead. Two spellings, not the whole alias set -- the message is read
#: by a student, and a list of eight is noise.
GENED_OFFER = "'gened' (or 'general education')"


# --- models ---------------------------------------------------------------------------------------


class RequirementQuery(BaseModel):
    """A program, and the courses already completed.

    Declared as a model for the same reason the other two tools' queries are: #33 hands this schema
    to an MCP client and #35's router fills it from free text, so the descriptions below are the
    only instruction either one gets -- including the one that says a major cannot be audited, which
    is the field description that keeps a router from confidently asking for one.
    """

    model_config = ConfigDict(extra="forbid")

    major: str = Field(
        description=(
            "The program to check. Only general education is supported: pass 'gened' or 'general "
            "education'. A degree program ('Computer Science, BS') returns supported=false -- this "
            "catalog contains no major requirements, and they must not be guessed at."
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
    """One requirement category, and the course that satisfied it if any.

    ``id`` / ``label`` / ``group`` are copied from ``requirements.json`` rather than restated here:
    ``group`` is what lets an answer say "Cultural Studies: two of three" instead of listing 13
    categories flat.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    group: str
    satisfied_by: str | None = None


class RequirementStatus(BaseModel):
    """A program audit, or a typed refusal to attempt one.

    ``supported`` is the field to branch on, and it is not the same question as "is anything
    unsatisfied": an unsupported program comes back with *both* category lists empty, so there is
    nothing for an answer node to paraphrase into a degree plan. ``message`` is the sentence #38
    turns into the hand-off, and on a supported audit it is the caveat about courses the catalog
    does not hold -- ``None`` only when there is nothing to warn about.

    ``major`` echoes the caller's own spelling (whitespace collapsed) so the answer can name what it
    was asked about.
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool
    major: str
    satisfied: list[CategoryStatus] = Field(default_factory=list)
    unsatisfied: list[CategoryStatus] = Field(default_factory=list)
    unknown_courses: list[str] = Field(default_factory=list)
    message: str | None = None


# --- normalizing ----------------------------------------------------------------------------------


def normalize_program(major: str) -> str:
    """``"  Gen-Ed  "`` -> ``"gen ed"``: lower-cased, punctuation collapsed to single spaces.

    Only ever used to test membership of :data:`GENED_ALIASES`. It is deliberately not a general
    program normalizer -- there is no program list to normalize against, and there will not be one
    until #32.
    """
    return re.sub(r"[^a-z0-9]+", " ", major.lower()).strip()


# --- the rules --------------------------------------------------------------------------------


def load_requirements(path: Path | None = None) -> dict[str, Any]:
    """The whole requirements document, checked. Uncached -- :func:`requirements_doc` is cached.

    The *whole* document, not ``genEd`` alone (which is all ``ingest.chunk.load_gened_categories``
    needs), because the empty ``majors`` list is what makes the refusal true: this tool declines a
    major on the grounds that the data has none, so it has to read that and not assume it.
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

    And a non-empty ``majors`` is the one that will actually happen: when #32 lands that data, the
    refusal below becomes a lie -- it would tell a student their program cannot be audited while the
    file on disk holds it. Better to fail on the day the data arrives than to keep declining it.
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
    if doc.get("majors"):
        raise ValueError(
            f"{where} now carries {len(doc['majors'])} major requirement(s), but "
            "agent/tools/requirements.py still refuses every program except Gen Ed -- see issue "
            "#32. Teach check_requirement() to evaluate them before shipping this corpus."
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


# --- the audit --------------------------------------------------------------------------------


def resolve_taken(taken_courses: Sequence[str]) -> tuple[list[CourseDetail], list[str]]:
    """Split the transcript into catalog records and codes the catalog does not have.

    Both lists keep the order they were given and hold each course once: a transcript that lists a
    retake twice should not credit a category twice over, and an unknown code repeated is one
    unknown code. A blank entry is dropped -- it names nothing that could be reported back.
    """
    found: list[CourseDetail] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for entry in taken_courses:
        detail = get_course(entry)
        label = detail.code or " ".join(entry.split())
        if not label or label in seen:
            continue
        seen.add(label)
        if detail.found:
            found.append(detail)
        else:
            unknown.append(label)
    return found, unknown


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


def unsupported_note(major: str) -> str:
    """Why a program cannot be audited, in a sentence a student can act on.

    Names the gap rather than the ticket: what is missing is a source of degree requirements, and
    the honest next step is a human advisor. The issue number lives in this module's docstring,
    where the reader is a developer.
    """
    return (
        f"{major!r} cannot be checked: this catalog export lists course sections and carries no "
        f"major or degree requirements at all, so there is nothing to audit a program against and "
        f"guessing at one would be worse than declining. {GENED_OFFER} is the one thing it can "
        "check; for a degree audit, the department's advisor or the official catalog program page "
        "is the source."
    )


def check_requirement(major: str, taken_courses: Sequence[str] | None = None) -> RequirementStatus:
    """What ``major`` still requires given ``taken_courses``, or a typed refusal to guess.

    Takes bare arguments rather than a :class:`RequirementQuery` so the graph calls it as an
    ordinary function; the model is what #33 declares to MCP, and its validation runs here either
    way.

    Only Gen Ed is audited. Every other program -- including one this catalog has courses for --
    comes back ``supported=False`` with both category lists empty, because the requirements are not
    in the data and inventing them is the failure mode this tool exists to prevent.
    """
    query = RequirementQuery(major=major, taken_courses=list(taken_courses or []))
    asked = " ".join(query.major.split())
    if normalize_program(query.major) not in GENED_ALIASES:
        return RequirementStatus(supported=False, major=asked, message=unsupported_note(asked))

    doc = requirements_doc()
    taken, unknown = resolve_taken(query.taken_courses)
    satisfied: list[CategoryStatus] = []
    unsatisfied: list[CategoryStatus] = []
    for category in doc["genEd"]:
        by = satisfied_by(category, taken)
        status = CategoryStatus(
            id=category["id"], label=category["label"], group=category["group"], satisfied_by=by
        )
        (satisfied if by else unsatisfied).append(status)
    return RequirementStatus(
        supported=True,
        major=asked,
        satisfied=satisfied,
        unsatisfied=unsatisfied,
        unknown_courses=unknown,
        message=unknown_note(unknown, course_index().terms) if unknown else None,
    )
