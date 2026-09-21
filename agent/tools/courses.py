"""Tool 1 of three: the catalog record behind a course code.

"How many credits is CS 374?" names the row it wants, so answering it is a lookup, not a search --
putting that question through the retriever would rank five plausible chunks where the catalog
holds one exact answer. This module is that lookup, and it is what the ``course_fact`` intent
(#35) dispatches to when the query spells a code outright.

Plain functions over Pydantic models. #33 registers them with FastMCP so an MCP client can call
them; #36 imports :func:`get_course` and calls it directly, in-process -- there is no stdio
subprocess between the graph and its own tools -- and the models live here rather than in the
server because the schema is the contract either way.

Three decisions, all of them about what a graph can do with the result:

**Not found is a value, not an exception.** ``CS 999`` comes back as ``found=False`` with a message
naming the catalog that was searched, and so does a string that is not a course code at all. A tool
that raises takes the decision away from the node that called it, and "I don't know" is one of the
three exits Day 2 is built to reach -- not an error path.

**The code is normalized by the tokenizer's matcher, not by a second regex.**
``ingest.tokenize.CODE_RE`` already decides what a course code looks like, because the BM25 side
has to collapse ``CS 374`` / ``cs 374`` / ``CS374`` into one term stream. A tool carrying its own
pattern would drift from it -- accepting a spelling the index cannot find, or refusing one it can
-- and nothing anywhere would report the divergence. One pattern, two readers.

**The corpus is read once.** 4,502 records, 2.9 MB of JSON. The graph makes several tool calls per
query, and a re-read per call is the kind of cost that surfaces a week later as "the agent feels
slow" with nothing pointing at the cause. The index is built lazily on the first lookup and kept in
a module-level cache; :func:`clear_cache` drops it, which is what a re-export inside one process
(and every test in this module's suite) needs.

``data/courses.json`` holds exactly one term, and both places that fact shows up are deliberate.
``term`` is in the output so an answer can say *"4 hours in 2026-sp"* instead of implying the course
runs every semester, and the not-found message names the term because a code the catalog has never
issued and a course simply not offered this spring are indistinguishable from inside the graph --
the honest answer covers both.

One field is provisional. A record on this branch carries ``credits: 4.0``, so that is what
:class:`CourseDetail` exposes. Open PR #25 replaces it with ``creditsRaw`` / ``creditsMin`` /
``creditsMax``, because 1,415 of the 4,502 courses have a credit *range* and one number is the wrong
answer for every one of them; when #25 merges, this model grows those three fields and drops
``credits``. :func:`build_index` refuses a corpus that has already made that move, rather than
quietly report ``credits=None`` for all 4,502 courses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from config import rel_to_repo
from ingest.chunk import load_courses
from ingest.export_almabot import COURSES_PATH
from ingest.tokenize import CODE_RE

# --- models ---------------------------------------------------------------------------------------


class CourseQuery(BaseModel):
    """One course code, in whichever of its spellings the query happened to use.

    Declared as a model rather than left as a bare parameter because #33 hands this schema to an
    MCP client and #35's router fills it from free text: the description below is the only
    instruction either one gets about what belongs in the field.
    """

    model_config = ConfigDict(extra="forbid")

    course_id: str = Field(
        description="A course code: 'CS 374', 'cs 374' and 'CS374' all name the same course."
    )


class CourseDetail(BaseModel):
    """One catalog record, or the reason there is none.

    ``found`` is the field to branch on; every other value is ``None`` when it is ``False``, except
    ``code`` (echoed back normalized whenever the input *was* a code, so the answer can name what
    it looked for) and ``message``, which is the sentence #38 turns into the "I don't know" answer.

    ``desc`` is the catalog description in full. AlmaBot's frontend corpus truncated it at 240
    characters -- 3,255 of the 4,502 courses lost text there, CS 374 among them -- and re-exporting
    without that cut was the point of #3, so nothing downstream should re-introduce one.
    """

    model_config = ConfigDict(extra="forbid")

    found: bool
    code: str | None = None
    name: str | None = None
    credits: float | None = None
    term: str | None = None
    desc: str | None = None
    attrs: str | None = None
    prereq_raw: str | None = None
    section_count: int | None = None
    message: str | None = None


class CourseIndex(NamedTuple):
    """The corpus keyed for lookup: courses by normalized code, and the terms it covers."""

    by_code: dict[str, dict[str, Any]]
    terms: tuple[str, ...]


# --- normalizing ----------------------------------------------------------------------------------


def normalize_code(course_id: str) -> str | None:
    """``"cs374"`` / ``" cs 374 "`` -> ``"CS 374"``; ``None`` when the string is not a code at all.

    The display form -- subject upper-cased, one space, the digits exactly as written. The digits
    are copied rather than parsed because the padding is part of the code: ``MUS 090`` is the one
    zero-padded course in the 2026-sp catalog, and ``int("090")`` would spell it ``MUS 90``.
    """
    match = CODE_RE.fullmatch(course_id.strip())
    if match is None:
        return None
    return f"{match.group('subject').upper()} {match.group('number')}"


# --- the index ------------------------------------------------------------------------------------


def build_index(courses: Sequence[Mapping[str, Any]]) -> CourseIndex:
    """Key the corpus by normalized code, refusing anything a lookup could not answer from.

    Every check here is about a corpus that would answer *wrongly* rather than not at all: a code
    the shared matcher cannot read is a course no query can reach, a repeated code silently drops
    one of the two records, and a corpus carrying #25's credit fields would report ``credits=None``
    for all of them. All three raise a sentence naming the course and the command to re-run.
    """
    if courses and "credits" not in courses[0]:
        raise ValueError(
            f"{rel_to_repo(COURSES_PATH)} carries no `credits` field; if this is PR #25's corpus "
            "(creditsRaw / creditsMin / creditsMax), CourseDetail has to expose those three "
            "instead of one number -- see the credits note at the top of agent/tools/courses.py"
        )
    by_code: dict[str, dict[str, Any]] = {}
    for course in courses:
        code = normalize_code(course["code"])
        if code is None:
            raise ValueError(
                f"course {course['code']!r} is not a course code under ingest.tokenize.CODE_RE "
                "(two to four letters and three digits), so nothing could ever look it up; "
                "re-run `uv run python -m ingest.export_almabot`"
            )
        if code in by_code:
            raise ValueError(
                f"{code} appears twice in the corpus; the export de-duplicates by code, so re-run "
                "`uv run python -m ingest.export_almabot`"
            )
        by_code[code] = dict(course)
    return CourseIndex(by_code, tuple(sorted({c["term"] for c in courses})))


def load_index(path: Path | None = None) -> CourseIndex:
    """Read the corpus and index it. Uncached -- :func:`course_index` is the cached one.

    Read through ``ingest.chunk.load_courses`` rather than a second ``json.load`` here: it is the
    function that knows a missing corpus means "run the export", and both index builds already go
    through it for the same reason.
    """
    return build_index(load_courses(COURSES_PATH if path is None else path))


#: The one index, built on first use. Module-level, so the graph's second tool call in a query --
#: and its first in the next one -- costs a dict lookup rather than 2.9 MB of JSON parsing.
_INDEX: CourseIndex | None = None


def course_index() -> CourseIndex:
    """The cached index, built on the first call."""
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index()
    return _INDEX


def clear_cache() -> None:
    """Drop the cached index, so the next lookup re-reads the corpus."""
    global _INDEX
    _INDEX = None


# --- the tool -------------------------------------------------------------------------------------


def get_course(course_id: str) -> CourseDetail:
    """The catalog record for ``course_id``, or ``found=False`` saying why there is none.

    Takes the bare string rather than a :class:`CourseQuery` so that the graph calls it as an
    ordinary function; the model is what #33 declares to MCP, and its validation runs here either
    way.

    The whole argument has to be one code. ``"CS 374 or ECE 374"`` names two and ``"what is cs 374
    about"`` is a question, and picking a code out of either would be a retrieval decision hiding
    inside a lookup -- both come back as ``found=False`` saying so, and pulling the code out of a
    sentence is #35's job.
    """
    query = CourseQuery(course_id=course_id)
    code = normalize_code(query.course_id)
    if code is None:
        return CourseDetail(
            found=False,
            message=(
                f"{query.course_id!r} is not a course code -- a code is two to four letters and "
                "three digits, like 'CS 374'"
            ),
        )
    index = course_index()
    course = index.by_code.get(code)
    if course is None:
        terms = " and ".join(index.terms)
        return CourseDetail(
            found=False,
            code=code,
            message=(
                f"{code} is not in the catalog, which covers {len(index.by_code):,} courses for "
                f"{terms} only -- either there is no such course, or it is not offered that term"
            ),
        )
    return CourseDetail(
        found=True,
        code=course["code"],
        name=course["name"],
        credits=course["credits"],
        term=course["term"],
        desc=course["desc"],
        attrs=course["attrs"],
        prereq_raw=course["prereqRaw"],
        section_count=course["sectionCount"],
    )
