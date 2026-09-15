"""Tool 2 of three: the recursive prerequisite chain behind a course code.

"What do I need before SE 495?" is not a lookup and not a search -- the answer is a walk. One
course's ``prereqRaw`` names its immediate requirements; the courses it names have requirements of
their own, and a student asking the question wants the whole closure, not the first layer. This
module walks it over ``data/prereqs.json``, which the export already filtered to courses that exist
in this catalog (the same filter ``dag.ts::prereqs()`` applies at read time).

Ports AlmaBot's ``dag.ts`` (``prereqs()``, ``allPrereqs()``, ``buildSubgraph()``), which drew the
same graph for a diagram. A tool that answers a question in prose has to be stricter than a
diagram in three places -- the depth cap, the layering, and the boolean structure of the
requirement text -- and each of those is a decision below.

Plain functions over Pydantic models, like ``agent/tools/courses.py``: ``agent/mcp_server.py``
registers them with FastMCP (#33), #36 imports :func:`get_prereq_chain` and calls it in-process.

**Every walk carries a visited set, because the graph is not acyclic.** ``data/README.md`` said to
build the DAG from ``prereqs`` only, since corequisites are mutual often enough that folding them
in would turn correct data into cycles. That is true and not sufficient: the prerequisite edges
alone contain seven strongly-connected components over 19 courses (``THEA 375-378``,
``THEA 475-478``, ``CS 225 -> CS 374 -> MATH 412 -> CS 225``, and the mutual pairs ``ACCY 301/302``,
``CHEM 204/205``, ``MCB 581/582``, ``SE 311/312``). Sixteen of those 19 are corequisites after all,
stated *inside* the prerequisite sentence -- "SE 310; concurrent registration in SE 312." -- where
``export_almabot.coreq_codes`` cannot see them, because it only reads a continuation sentence. The
other three are a disjunction: ``MATH 412`` is one alternative in ``CS 225``'s "One of CS 173,
CS 413, MATH 213, MATH 314, MATH 347, MATH 412", and ``MATH 412`` in turn lists ``CS 374``, which
requires ``CS 225``. Both are data problems worth their own issue; neither is something this tool
can wait for, so the walk is cycle-safe and :attr:`PrereqChain.cycle_detected` says when it
mattered. ``data/README.md`` is corrected in this branch.

**No depth cap by default.** ``allPrereqs(code, depth = 6)`` stops at six levels; run against the
live graph, that cap drops closure members for 113 of the 1,135 courses. A cut is cosmetic in a
diagram and a wrong answer in a sentence, so this walk goes to exhaustion. It can afford to: a
visited set shared across the whole walk visits each course once, which bounds any single call by
the graph (1,442 courses, 2,086 edges) and in practice by the largest closure in it (47 courses,
``IS 557``; the median is 4). ``max_depth`` survives as something a *caller* can ask for -- one
level, for "what do I take first" -- and :attr:`PrereqChain.truncated` reports when it bit.

**Level 0 is the immediate prerequisites.** Each course sits at its *shortest* distance from the
queried one, which is the contract #38's answer node needs: "before CS 374 you need CS 173,
MATH 213 and CS 225". ``dag.ts::buildSubgraph`` layers by longest chain instead, because a diagram
wants every arrow pointing one way -- under that rule ``CS 374``'s level 0 would hold ``CS 225``
alone, and ``CS 173`` and ``MATH 213`` would sit two levels back, since ``CS 225`` also requires
them. So the layers here are not a schedule: reading them right-to-left can put a course in the
same level as something it requires. ``edges`` carries the full structure, and a consumer that
wants ``dag.ts``'s drawing layers back can compute them from it.

**Every node carries its raw requirement sentence.** ``prereqCodes`` is a flat list and the export
deliberately does not parse the boolean structure, so the list is *not* a conjunction: 744 of the
1,135 requirement statements contain a disjunction, and 397 of those resolve to more than one code.
``ABE 228`` reads "One of MATH 220, MATH 221, MATH 234." and exports ``["MATH 220", "MATH 234"]`` --
a chain that reports both claims a course the student does not need. 299 statements also offer a
"consent of ..." escape hatch that no code list can express. So ``raw`` travels with every node,
and #38's prompt quotes it rather than asserting the conjunction. Parsing that text into an AND/OR
tree is a separate piece of work (Day 3+), not something to sneak in here.

**Not found is a value, and :func:`~agent.tools.courses.get_course` is what decides it.** A code
the catalog does not have comes back as ``found=False`` carrying that tool's message, rather than a
second sentence written here that could drift from it: one decision about what is in the catalog,
made in the tool that owns the catalog. A course that *is* in the catalog and has no prerequisites
returns ``found=True`` with empty levels -- that is an answer. 1,597 of the 4,502 courses have a
requirements block but no surviving edge (a dangling reference, or consent only), so ``raw`` is
carried on the chain itself too: the answer can quote the sentence even when the walk is empty.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.courses import course_index, get_course, normalize_code
from config import rel_to_repo
from ingest.export_almabot import COURSES_PATH, PREREQS_PATH

# --- models ---------------------------------------------------------------------------------------


class PrereqQuery(BaseModel):
    """One course code, and optionally how far back to walk.

    Declared as a model for the same reason :class:`~agent.tools.courses.CourseQuery` is: #33 hands
    this schema to an MCP client and #36's dispatch node fills it from free text, so the
    descriptions below are the only instruction either one gets.
    """

    model_config = ConfigDict(extra="forbid")

    course_id: str = Field(
        description="A course code: 'CS 374', 'cs 374' and 'CS374' all name the same course."
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many levels to walk back; omit for the whole chain. 1 returns the immediate "
            "prerequisites only. A chain cut short by this comes back with truncated=true."
        ),
    )


class PrereqNode(BaseModel):
    """One course in the chain, with the sentence that put it there.

    ``raw`` is the *node's own* requirements block, verbatim from the catalog -- the reason to carry
    it is that the code list it produced may be a disjunction, so an answer has to quote it rather
    than assert it. It is ``""`` for a course whose catalog record has no requirements block (a leaf
    of the chain) and ``None`` only if a corpus stops carrying the field at all.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    name: str
    raw: str | None = None


class PrereqChain(BaseModel):
    """The prerequisite closure of one course, layered, or the reason there is none.

    ``found`` is the field to branch on. ``levels[0]`` is the immediate prerequisites and each
    course appears exactly once, at its shortest distance from the queried one; ``edges`` holds
    every ``(prereq, dependent)`` pair among the courses reported, including the ones that close a
    cycle, and both endpoints of every edge are always either ``course_id`` or a node in ``levels``.

    Two fields are not in #30's sketch. ``raw`` is the queried course's own requirements block, so
    an answer can quote it without a second call, and ``message`` is the sentence #38 turns into the
    "I don't know" exit -- the same field, for the same reason, as
    :attr:`~agent.tools.courses.CourseDetail.message`.
    """

    model_config = ConfigDict(extra="forbid")

    course_id: str
    found: bool
    levels: list[list[PrereqNode]] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    cycle_detected: bool = False
    truncated: bool = False
    raw: str | None = None
    message: str | None = None


class Walk(NamedTuple):
    """The result of walking the graph, in codes: :func:`get_prereq_chain` adds names and text."""

    levels: list[list[str]]
    edges: list[tuple[str, str]]
    cycle_detected: bool
    truncated: bool


# --- the graph ------------------------------------------------------------------------------------


def load_prereqs(path: Path | None = None) -> list[dict[str, Any]]:
    """The exported edge list, or a pointer at the command that writes it.

    Mirrors ``ingest.chunk.load_courses``: a missing file is a setup step the caller has not run,
    not an empty graph in which every course would answer "no prerequisites".
    """
    path = PREREQS_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{rel_to_repo(path)} is missing; run `uv run python -m ingest.export_almabot` first"
        )
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"{rel_to_repo(path)} holds no prerequisite records; re-run the export")
    return rows


def build_graph(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """``{course: its prerequisites}``, normalized, refusing anything a walk could not answer from.

    ``coreqs`` is read by nobody here: a corequisite is taken *alongside*, so it is not a step in
    the chain, and the export keeps the two apart precisely because folding them together would add
    cycles to correct data.

    The guards match :func:`~agent.tools.courses.build_index`'s: a code the shared matcher cannot
    read is a course no walk could reach, and a repeated record would silently drop one of the two
    edge sets. Both name the course and the command to re-run.
    """
    graph: dict[str, tuple[str, ...]] = {}
    for row in rows:
        code = _checked_code(row["course"])
        if code in graph:
            raise ValueError(
                f"{code} appears twice in {rel_to_repo(PREREQS_PATH)}; the export writes one "
                "record per course, so re-run `uv run python -m ingest.export_almabot`"
            )
        graph[code] = tuple(sorted({_checked_code(ref) for ref in row["prereqs"]}))
    return graph


def _checked_code(course_id: str) -> str:
    """``course_id`` normalized, or a ValueError pointing at the export."""
    code = normalize_code(course_id)
    if code is None:
        raise ValueError(
            f"{course_id!r} in {rel_to_repo(PREREQS_PATH)} is not a course code under "
            "ingest.tokenize.CODE_RE (two to four letters and three digits), so no walk could "
            "reach it; re-run `uv run python -m ingest.export_almabot`"
        )
    return code


def check_catalog(graph: Mapping[str, Sequence[str]], by_code: Mapping[str, Any]) -> None:
    """Fail unless every course the graph names is in the catalog.

    The export filters prerequisite references through the catalog, so a code here that the catalog
    does not have means the two files came from different runs -- and a node with no record has no
    ``name`` to report. Checked once, when the graph is built, rather than discovered mid-answer.
    """
    for code, prereqs in graph.items():
        for ref in (code, *prereqs):
            if ref not in by_code:
                raise ValueError(
                    f"{ref} appears in {rel_to_repo(PREREQS_PATH)} but not in "
                    f"{rel_to_repo(COURSES_PATH)}, which the export filters it against; the two "
                    "files are from different runs, so re-run "
                    "`uv run python -m ingest.export_almabot`"
                )


def load_graph(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Read the edge list and index it. Uncached -- :func:`prereq_graph` is the cached one."""
    return build_graph(load_prereqs(path))


#: The one graph, built on first use -- 1,135 records against several tool calls per query.
_GRAPH: dict[str, tuple[str, ...]] | None = None


def prereq_graph() -> dict[str, tuple[str, ...]]:
    """The cached graph, built and cross-checked against the catalog on the first call."""
    global _GRAPH
    if _GRAPH is None:
        graph = load_graph()
        check_catalog(graph, course_index().by_code)
        _GRAPH = graph
    return _GRAPH


def clear_cache() -> None:
    """Drop the cached graph, so the next chain re-reads the edge list."""
    global _GRAPH
    _GRAPH = None


# --- the walk -------------------------------------------------------------------------------------


def walk_chain(root: str, graph: Mapping[str, Sequence[str]], max_depth: int | None = None) -> Walk:
    """Breadth-first over ``graph`` from ``root``, one visited set for the whole walk.

    The visited set is what makes this terminate on the seven cycles in the live graph, and it is
    also what makes exhaustion affordable: each course is expanded once, so a call costs the
    closure it is reporting and nothing more.

    It is also why ``levels`` is shortest-distance rather than longest-chain -- a course already
    seen is not moved deeper when a longer path reaches it again.

    When ``max_depth`` stops the walk, ``truncated`` is set only if something was actually lost: a
    frontier whose every prerequisite is already reported drops nothing. Edges to courses left
    outside are dropped with them, so no edge in the result points at a course that is not in it.
    """
    seen = {root}
    levels: list[list[str]] = []
    edges: set[tuple[str, str]] = set()
    truncated = False
    frontier = [root]
    while frontier:
        if max_depth is not None and len(levels) >= max_depth:
            for code in frontier:
                for ref in graph.get(code, ()):
                    if ref in seen:
                        edges.add((ref, code))
                    else:
                        truncated = True
            break
        nxt: list[str] = []
        for code in frontier:
            for ref in graph.get(code, ()):
                edges.add((ref, code))
                if ref in seen:
                    continue
                seen.add(ref)
                nxt.append(ref)
        if nxt:
            levels.append(sorted(nxt))
        frontier = nxt
    return Walk(levels, sorted(edges), has_cycle(root, graph, seen), truncated)


def has_cycle(root: str, graph: Mapping[str, Sequence[str]], nodes: set[str]) -> bool:
    """True when a course in the reported subgraph is reachable from itself.

    A *back edge*, not a second visit. ``CS 374`` and ``CS 225`` both require ``CS 173``, and
    ``MATH 241 -> MATH 231 -> MATH 220`` sits under ``CS 101`` twice over: a breadth-first walk
    meets those courses again and again, and none of it is a cycle. Counting a re-encounter instead
    would flag 559 of the 1,135 chains, of which 165 actually contain one. Restricted to ``nodes``,
    so a truncated walk is only asked about the graph it actually reported.
    """
    on_path: set[str] = set()
    done: set[str] = set()

    def pending(code: str) -> list[str]:
        return [p for p in graph.get(code, ()) if p in nodes]

    stack = [(root, pending(root))]
    on_path.add(root)
    while stack:
        code, todo = stack[-1]
        if not todo:
            stack.pop()
            on_path.discard(code)
            done.add(code)
            continue
        ref = todo.pop()
        if ref in on_path:
            return True
        if ref in done:
            continue
        on_path.add(ref)
        stack.append((ref, pending(ref)))
    return False


# --- the tool -------------------------------------------------------------------------------------


def get_prereq_chain(course_id: str, max_depth: int | None = None) -> PrereqChain:
    """Every course required before ``course_id``, layered, or ``found=False`` saying why not.

    Takes bare arguments rather than a :class:`PrereqQuery` so the graph calls it as an ordinary
    function; the model is what #33 declares to MCP, and its validation runs here either way.

    ``found=False`` is delegated whole to :func:`~agent.tools.courses.get_course` -- a string that
    is not a code, and a code the catalog does not have, come back with its message.
    """
    query = PrereqQuery(course_id=course_id, max_depth=max_depth)
    detail = get_course(query.course_id)
    code = detail.code
    if not detail.found or code is None:
        return PrereqChain(
            course_id=code or query.course_id.strip(), found=False, message=detail.message
        )
    by_code = course_index().by_code
    walk = walk_chain(code, prereq_graph(), query.max_depth)
    return PrereqChain(
        course_id=code,
        found=True,
        levels=[[_node(c, by_code) for c in level] for level in walk.levels],
        edges=walk.edges,
        cycle_detected=walk.cycle_detected,
        truncated=walk.truncated,
        raw=detail.prereq_raw,
    )


def _node(code: str, by_code: Mapping[str, Mapping[str, Any]]) -> PrereqNode:
    """One chain node from its catalog record, which :func:`check_catalog` guarantees exists."""
    course = by_code[code]
    return PrereqNode(code=code, name=course["name"], raw=course.get("prereqRaw"))
