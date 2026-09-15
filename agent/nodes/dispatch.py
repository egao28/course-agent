"""Node 2: where the evidence comes from. Four rows, one record shape out.

    route_intent ──► dispatch ──► check_evidence

| intent | action |
|---|---|
| ``course_fact`` | ``get_course(code)`` when the query names a code, else ``retrieve(query)`` |
| ``prereq_chain`` | ``get_prereq_chain(code)``; no code -> ``tool_error``, never a guessed course |
| ``degree_requirement`` | ``check_requirement(major, taken)``, both read out of the query |
| ``compare_recommend`` | ``retrieve(query)`` -- several candidates, ranked, is the answer shape |
| ``out_of_scope`` | never arrives: :func:`agent.graph.route_after_intent` routes it to its exit |

The intent is a label, and this is the only place that turns it into an action. The alternative --
four nodes behind four conditional edges -- would put the same table in the graph's wiring, where
the rows could not be read side by side.

**Everything becomes :class:`~agent.state.Evidence`.** A retrieval hit keeps its score and names
the mode that produced it; a tool result is rendered to text, carries ``score=None`` and names the
tool. That ``None`` is load-bearing for #37: an exact catalog lookup has no ranking to be unsure
about, and comparing it to a BM25 number would compare it to nothing. #38 then has one input
format, and its citation check works the same whichever row ran.

**A tool that came back empty produces no evidence, not evidence about its emptiness.** ``docs`` is
empty and ``signals["tool_error"]`` names what happened (:data:`TOOL_ERRORS`), which is what #37
turns into the refuse/hand-off exit and :data:`agent.nodes.exits.REASONS` turns into a sentence.
Putting "CS 999 is not in the catalog" into ``docs`` would hand #38 a sentence about the absence of
an answer as though it were a source, and a model given that will write a fluent refusal with a
guess folded into it -- which is the whole reason the two refusal exits are fixed text.

**No code found is an outcome, not a failure to work around.** ``prereq_chain`` with no course in
the question sets ``no_course_code`` and stops, and a question about requirements that names no
auditable program sets ``no_program``. Asking the model which course or which degree was meant is
how an agent ends up answering confidently about the wrong one; the hand-off message asks instead.

**The requirement row resolves the program from the same file the tool audits it against**
(``ingest/majors.json``, through :data:`~agent.tools.requirements.GENED_ALIASES` and each
program's own aliases), so ``check_requirement``'s own ``supported=False`` refusal is not reachable
from here -- a spelling this node passes on is one the resolver knows. That refusal is for the MCP
surface (#33), where a client names a program directly; from the graph, an unreadable program is
``no_program`` and the hand-off says which programs there are to ask about.

**Codes come out of free text through the matcher the BM25 index was built with**
(:data:`ingest.tokenize.CODE_RE`, then :func:`agent.tools.courses.normalize_code`). A second regex
here would drift from that one, and a query would then retrieve under one normalization and look up
under another. What this module adds is a vocabulary filter: the catalog's own 187 subjects, so
that "I have 100 credits" does not name a course called ``HAVE 100``. The cost is that a code whose
*subject* has nothing at all in this export reads as no code -- accepting "have 100" and "top 100"
as courses is the worse trade, and the hand-off message for either case asks for a course code.

**The mode is a config value** (``agent.retrieval_mode``), not a constant here: which retriever
answers is an experiment setting, it is hashed into the report name like every other one, and a
checkout with no dense index has to be able to run. The table above says ``retrieve(query)`` for
that reason -- the row decides *that* retrieval answers, not which index does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import retrieval
from agent.nodes import Node
from agent.state import INTENTS, AgentState, Evidence, Intent, check_intent
from agent.tools import courses, prereqs, requirements
from config import Config
from ingest.chunk import format_credits
from ingest.majors import normalize_program
from ingest.tokenize import CODE_RE

#: The intents that reach this node: the five minus ``out_of_scope``, which ends its run at the
#: router. #39's trace groups its rows by these.
DISPATCHED: tuple[Intent, ...] = tuple(intent for intent in INTENTS if intent != "out_of_scope")

#: Every ``signals["tool_error"]`` this node can set, and what each one means. A row that cannot
#: produce evidence names one of these and writes no ``docs``; :data:`agent.nodes.exits.REASONS`
#: has a sentence and a remedy for each, and a test asserts it still does -- an error with no
#: sentence behind it reaches the reader as its own label.
TOOL_ERRORS: dict[str, str] = {
    "no_course_code": "the question needs one course and names none",
    "course_not_found": "the course it names is not in this catalog export",
    "no_program": "no program this corpus can audit could be read out of the question",
}

#: What :func:`~agent.tools.requirements.check_requirement` is asked for when the question names the
#: campus Gen Eds. One of its own aliases, so the tool's resolver decides what it means.
GENED = "gened"

#: How many of a requirement's options are named before the rest are counted. The CS technical
#: elective rule matches 44 courses this term, and an answer does not need the list to say that
#: three more are wanted.
MAX_OPTIONS = 6


class Gathered(NamedTuple):
    """One row's result: the evidence, why there is none, and which mode produced it.

    Three fields because the node writes three things, and a row that returns them together cannot
    write ``docs`` without saying where they came from. ``mode`` is ``None`` on every tool row --
    no retrieval ran, and a ``retrieval_mode`` filled in anyway would put a mode in #39's table for
    a query that never touched an index.
    """

    docs: list[Evidence]
    tool_error: str | None = None
    mode: str | None = None


# --- reading a course code out of a question ------------------------------------------------------


def catalog_subjects() -> frozenset[str]:
    """Every subject the catalog holds -- ``{"AE", ..., "CS", ...}``, 187 of them in 2026-sp.

    Read off the index each call rather than cached: it is a set comprehension over codes already
    in memory, and a cache here would be a second thing to clear when a test swaps the corpus.
    """
    return frozenset(code.split(" ", 1)[0] for code in courses.course_index().by_code)


def code_spans(query: str) -> list[str]:
    """Every code-shaped span in ``query``, spelled by the one normalizer, in order of appearance.

    ``CODE_RE`` decides what a code looks like (2-4 letters, an optional space, three digits) and
    :func:`~agent.tools.courses.normalize_code` decides how it is spelled; this function owns
    neither, so that ``CS374`` in a question and ``CS 374`` in the index cannot disagree.
    """
    spelled = (courses.normalize_code(match.group(0)) for match in CODE_RE.finditer(query))
    return [code for code in spelled if code is not None]


def extract_codes(query: str) -> list[str]:
    """The course codes ``query`` names, without repeats and in the order it names them.

    Filtered by the catalog's subject vocabulary, which is what separates a course from an English
    sentence that happens to fit the pattern: "I have 100 credits" matches ``CODE_RE`` and there is
    no subject ``HAVE``. The filter is the corpus, not a stopword list -- a word list here would be
    a retrieval decision hiding in a regex, and would need editing every time a subject was added.
    """
    subjects = catalog_subjects()
    found: list[str] = []
    for code in code_spans(query):
        if code.split(" ", 1)[0] in subjects and code not in found:
            found.append(code)
    return found


def pick_code(codes: Sequence[str]) -> str | None:
    """The one code a single-course row looks up: the first the catalog has, else the first named.

    Preferring a code the catalog holds is what keeps a stray match from shadowing a real course
    ("I have taken 3 courses, what comes before CS 374"). Falling back to the first named rather
    than to ``None`` is what keeps a *missing* course honest: ``CS 999`` reaches
    :func:`~agent.tools.courses.get_course` and comes back as "not in this catalog", which is a
    different sentence from "you did not name a course", and the reader is owed the right one.

    The first, not all of them: a question naming two courses is a comparison, which the router
    labels ``compare_recommend`` and this table sends to retrieval. Only the transcript row reads
    every code in the question.
    """
    index = courses.course_index().by_code
    for code in codes:
        if code in index:
            return code
    return codes[0] if codes else None


# --- reading a program out of a question ----------------------------------------------------------


def program_spellings() -> dict[str, str]:
    """Every program this corpus can audit, by the spellings it answers to.

    ``{normalized spelling: what to pass as ``major``}`` -- built from
    :data:`~agent.tools.requirements.GENED_ALIASES` and from each program's own ``aliases`` in
    ``ingest/majors.json``, so a program added to that file becomes askable with no change here.
    The value is the file's own label (or :data:`GENED`), which is what the tool's resolver matches
    exactly; this function only has to find *which* program was named.
    """
    spellings = {normalize_program(alias): GENED for alias in requirements.GENED_ALIASES}
    for program in requirements.requirements_doc().get("majors", ()):
        for spelling in (*program["aliases"], program["label"], program["id"]):
            spellings[normalize_program(spelling)] = program["label"]
    return spellings


def extract_program(query: str) -> str | None:
    """Which program the question is about, or ``None`` when it names none.

    Course codes are removed first: ``stat`` is one of the Statistics degree's aliases and also the
    subject of ``STAT 200``, and "what is left if I have taken STAT 200" names a transcript, not a
    program.

    **Gen Ed wins when a question names both.** "Which Gen Eds are left for the Statistics major"
    is answered by the campus audit, because a degree's audit is the one that cannot answer it:
    every program in ``ingest/majors.json`` reports the campus General Education requirements in
    ``not_checked``, unevaluated, and defers to the campus list. Among degrees the longest spelling
    wins, so a program whose name contains another's ("statistics" inside "statistics and computer
    science") is not shadowed by it.

    ``None`` is a real outcome, like a missing course code: this node does not pick a degree for a
    student, and an audit against the wrong program is wrong in the direction that costs a semester.
    """
    text = normalize_program(CODE_RE.sub(" ", query))
    named = [
        (spelling, major)
        for spelling, major in program_spellings().items()
        if re.search(rf"\b{re.escape(spelling)}\b", text)
    ]
    if not named:
        return None
    if any(major == GENED for _, major in named):
        return GENED
    return max(named, key=lambda pair: len(pair[0]))[1]


# --- rendering a tool result as evidence ----------------------------------------------------------


def render_course(detail: courses.CourseDetail) -> str:
    """One catalog record as an evidence block, in the labels the chunk template already uses.

    Not :func:`ingest.chunk.render_chunk`: that text is an index artifact whose template version is
    hashed into ``chunk_key``, and this block is neither indexed nor hashed -- it carries
    ``Sections``, which the template omits. The labels are deliberately identical, so #38 reads one
    vocabulary whether a block came from a lookup or from a retrieved chunk of the same course.
    """
    credits = "not listed" if detail.credits is None else format_credits(detail.credits)
    lines = [
        f"{detail.code} — {detail.name}",
        f"Credits: {credits}",
        f"Term: {detail.term}",
        f"Description: {detail.desc}",
        f"Prerequisites: {detail.prereq_raw or 'None listed'}",
        f"Degree Attributes: {detail.attrs or 'None'}",
        f"Sections: {detail.section_count}",
    ]
    return "\n".join(line.rstrip() for line in lines)


def render_chain(chain: prereqs.PrereqChain) -> str:
    """A prerequisite chain as an evidence block: the catalog's own sentence, then the layers.

    The statement comes first because it is the only part that is exactly right. The layers flatten
    the booleans -- "One of CS 173, MATH 213; CS 225." puts all three courses on the first level,
    and 744 of the catalog's 1,135 prerequisite statements contain a disjunction -- so a block whose
    layers could be read as a list of required courses says so in itself. An answer built from the
    layers alone would tell a student to take three courses where the catalog asks for two.
    """
    lines = [
        f"{chain.course_id} — what the catalog requires before it",
        f"Prerequisite statement: {chain.raw or 'None listed'}",
    ]
    for depth, level in enumerate(chain.levels, start=1):
        label = "Immediately before it" if depth == 1 else f"{depth} steps back"
        lines.append(f"{label}: " + "; ".join(f"{node.code} — {node.name}" for node in level))
    if not chain.levels:
        lines.append("Nothing in this catalog is listed as a prerequisite for it.")
    else:
        courses_named = sum(len(level) for level in chain.levels)
        lines.append(
            f"{courses_named} courses over {len(chain.levels)} levels, "
            f"{len(chain.edges)} prerequisite edges."
        )
    if any(len(level) > 1 for level in chain.levels):
        lines.append(
            "A level names every course the statements at that depth mention; where a statement "
            'says "one of", only one of them is required.'
        )
    if chain.cycle_detected:
        lines.append(
            "The catalog's own prerequisites form a cycle inside this chain -- some of these "
            "courses list each other -- so the walk stopped rather than looping."
        )
    if chain.truncated:
        lines.append("Cut off at the depth asked for; there are more prerequisites below this.")
    return "\n".join(lines)


def category_line(category: requirements.CategoryStatus) -> str:
    """One requirement as a line: what it is, what it took, and what it still wants."""
    parts = [f"{category.label} ({category.group})"]
    if category.satisfied_by:
        parts.append(f"satisfied by {category.satisfied_by}")
    if category.credited:
        parts.append("credited: " + ", ".join(category.credited))
    if category.still_needed:
        options = list(category.options[:MAX_OPTIONS])
        rest = len(category.options) - len(options)
        named = ", ".join(options) + (f", and {rest} more" if rest > 0 else "")
        parts.append(f"{category.still_needed} more" + (f" from {named}" if named else ""))
    if category.note:
        parts.append(category.note)
    return " -- ".join(parts)


def render_requirements(status: requirements.RequirementStatus) -> str:
    """An audit as an evidence block: both lists, what a course list cannot decide, the caveats.

    ``not_checked`` and the tool's ``message`` are in the block rather than dropped because they
    are what stops the answer reading as "you are done": 6 of the Statistics degree's 12 groups are
    hours, GPA and placement requirements no transcript of codes decides, and the message names the
    courses these rules require that this term does not offer.
    """
    head = f"{status.program} — requirement audit"
    if status.catalog_year:
        head += f", {status.catalog_year} catalog"
    lines = [head]
    if status.source:
        lines.append(f"Source: {status.source}")
    groups = (
        ("Satisfied", status.satisfied, True),
        ("Still needed", status.unsatisfied, True),
        # Printed only when there are any: every degree has some, a Gen Ed audit has none, and
        # "Not decided (0)" under the 13 campus categories would read as a disclaimer about them.
        ("Not decided by a list of courses", status.not_checked, False),
    )
    for label, group, always in groups:
        if not group and not always:
            continue
        lines.append(f"{label} ({len(group)}):" if group else f"{label} (0): none")
        lines.extend(f"- {category_line(category)}" for category in group)
    if status.unknown_courses:
        lines.append(
            "Courses given that this catalog does not offer: " + ", ".join(status.unknown_courses)
        )
    if status.message:
        lines.append(f"Caveats: {status.message}")
    return "\n".join(lines)


# --- the node -------------------------------------------------------------------------------------


def check_retrieval_mode(cfg: Mapping[str, Any], override: str | None = None) -> str:
    """Which mode the retrieval rows ask for: ``override``, or ``agent.retrieval_mode``.

    Validated when the node is *built*, like the router's ``llm`` section: it reads no index and
    costs nothing, and a config naming a mode :func:`retrieval.retrieve` does not implement should
    not survive until the first query of a report run.
    """
    if override is not None:
        return retrieval.check_mode(override)
    if "agent" not in cfg:
        raise ValueError(
            "config has no `agent` section; it must set retrieval_mode -- which of retrieve()'s "
            f"modes ({', '.join(retrieval.MODES)}) the dispatch table's retrieval rows ask for "
            "(see config.yaml)"
        )
    section = cfg["agent"]
    if "retrieval_mode" not in section:
        raise ValueError(
            "config's `agent` section has no retrieval_mode; it names the mode the dispatch node "
            f"retrieves in, one of {', '.join(retrieval.MODES)} (see config.yaml)"
        )
    mode = section["retrieval_mode"]
    if mode not in retrieval.MODE_SOURCES:
        raise ValueError(
            f"config sets agent.retrieval_mode to {mode!r}; retrieve() implements "
            f"{', '.join(retrieval.MODES)} (see config.yaml)"
        )
    return mode


def dispatch_node(
    cfg: Config, mode: str | None = None, retrieve: Callable[..., Any] | None = None
) -> Node:
    """Build the dispatcher: the intent picks the row, and every row returns :class:`Evidence`.

    ``mode`` overrides ``agent.retrieval_mode`` for the lifetime of the node -- a Day 4 ablation,
    or a test that must not depend on the shipped config. ``retrieve`` replaces
    :func:`retrieval.retrieve`, which is how the two retrieval rows are exercised without a 6 MB
    BM25 pickle on disk; the three tool rows are tested over a fixture catalog through their own
    modules' paths instead, since a tool's evidence is the whole point of its row.

    Nothing is opened here. The retrievers are built on the first query that needs one (and cached
    in :mod:`retrieval` for the rest of the process), and the tools read ``data/courses.json`` the
    same way -- so ``build_graph()`` still compiles in a checkout with no index and no API key.
    """
    retrieval_mode = check_retrieval_mode(cfg, mode)
    search = retrieval.retrieve if retrieve is None else retrieve

    def from_retrieval(query: str) -> Gathered:
        result = search(query, retrieval_mode, cfg=cfg)
        return Gathered(
            [
                Evidence(doc_id=hit.doc_id, text=hit.text, score=hit.score, source=retrieval_mode)
                for hit in result.hits
            ],
            mode=retrieval_mode,
        )

    def from_course(query: str) -> Gathered:
        code = pick_code(extract_codes(query))
        if code is None:
            # The table's "otherwise": a fact about a course nobody named is a search.
            return from_retrieval(query)
        detail = courses.get_course(code)
        if not detail.found:
            return Gathered([], "course_not_found")
        block = Evidence(doc_id=code, text=render_course(detail), score=None, source="get_course")
        return Gathered([block])

    def from_prereqs(query: str) -> Gathered:
        code = pick_code(extract_codes(query))
        if code is None:
            return Gathered([], "no_course_code")
        chain = prereqs.get_prereq_chain(code)
        if not chain.found:
            return Gathered([], "course_not_found")
        block = Evidence(
            doc_id=code, text=render_chain(chain), score=None, source="get_prereq_chain"
        )
        return Gathered([block])

    def from_requirements(query: str) -> Gathered:
        major = extract_program(query)
        if major is None:
            return Gathered([], "no_program")
        status = requirements.check_requirement(major, extract_codes(query))
        if not status.supported:
            # Not expected: the spelling came out of the file the tool resolves against. It is
            # here so that the two drifting apart refuses the run rather than answering it from
            # an audit with every list empty.
            return Gathered([], "no_program")
        block = Evidence(
            doc_id=status.program or major,
            text=render_requirements(status),
            score=None,
            source="check_requirement",
        )
        return Gathered([block])

    table: dict[str, Callable[[str], Gathered]] = {
        "course_fact": from_course,
        "prereq_chain": from_prereqs,
        "degree_requirement": from_requirements,
        "compare_recommend": from_retrieval,
    }

    def node(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent")
        if intent is None:
            raise ValueError(
                "dispatch ran with no `intent` set; the router writes one for every query and the "
                "graph routes on it (#35), so this is a wiring bug rather than a query"
            )
        intent = check_intent(intent, "route_intent")
        if intent == "out_of_scope":
            raise ValueError(
                "dispatch was reached with intent 'out_of_scope'; that label ends its run at the "
                "out-of-scope exit before any evidence is gathered (see agent/graph.py)"
            )
        try:
            row = table[intent]
        except KeyError:
            raise KeyError(
                f"dispatch has no row for intent {intent!r}; its rows are "
                f"{', '.join(DISPATCHED)} -- a new label needs one here, and #37 needs to know "
                "what its evidence looks like"
            ) from None
        gathered = row(state["query"])
        return {
            "docs": gathered.docs,
            "signals": {"retrieval_mode": gathered.mode, "tool_error": gathered.tool_error},
        }

    node.__name__ = "dispatch"
    return node
