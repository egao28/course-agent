"""Twenty-five rows the system must refuse, constructed rather than drawn from documents.

    uv run python -m eval.scripts.build_unanswerable
    uv run python -m eval.scripts.build_unanswerable --no-retrieval   # no BM25 index needed

The other three buckets are drafted (:mod:`eval.scripts.draft_items`); this one cannot be. A
model shown ten course records writes questions those records answer, which is the opposite of
the requirement, so the bucket is a **table in this file** -- :data:`ITEMS`, twenty-five of them
-- and what the script does is *check* that table against the corpus, score every row through
``retrieve()``, and write the result out as drafts in the same shape #56 produces.

**Four failure shapes, and every row records which one it is.** :data:`SHAPES` names them and
:func:`check_codes` enforces a different claim for each, because "this is unanswerable" is four
different claims and only one of them is about the retriever:

1. ``invented`` -- the course does not exist. Every declared code must be absent from the corpus
   **and named by no description in it**: a code the catalog talks about is a real course, and
   belongs in shape 2 rather than here. This is the one shape whose rows may declare no code at
   all -- an invented *subject* ("quantum origami") has none -- and such a row is held instead to
   naming no code in its question.
2. ``out_of_term`` -- a real course the 2026-sp export does not contain. Every code absent, and
   ``named_by`` a course that *is* in the export whose description names it. That is what makes
   these rows checkable rather than remembered: the corpus itself is the evidence that the course
   is real, so nothing here depends on what anyone knows about UIUC.
3. ``field_absent`` -- a real, present course, asked for a fact the knowledge base does not carry.
   Every code must be **present**, and ``field`` must be outside both the chunk template and
   :class:`agent.tools.courses.CourseDetail`. The second half is `eval/GUIDELINES.md`'s boundary
   case **U2**: ``get_course`` returns ``section_count``, so *"how many sections of CS 374 are
   running?"* is a deleted row and not an unanswerable one, and a table entry that asked for it
   is refused here rather than discovered in verification.
4. ``not_course`` -- not about courses at all. No codes, no field. Cheap to write, and they should
   be: they test the ``out_of_scope`` exit rather than the retriever, and a bucket made only of
   these would prove nothing.

Six each, except ``out_of_term``, which gets seven. Those are the rows that decide whether the
system knows its knowledge base is *one term* rather than "everything about this university", and
they are the only shape whose realness the corpus can vouch for.

**Scoring does not label anything.** `eval/GUIDELINES.md` forbids running ``retrieve()`` to decide
a label -- what the system does with a question is the thing under test and cannot also be the
thing that labels it -- so every label above comes from the corpus checks, which read the export
and never the index. The retrieval pass runs *afterwards*, changes no row, and exists to answer a
different question: which of these rows the evidence gate would refuse on the score alone, before
the model is asked anything. A row above the gate is still a correct row; it is a row whose
refusal Day 4 must not credit to the gate.

**What that measured, and it is the finding of this file.** Over the 25 rows in ``bm25`` mode at
``evidence.thresholds.bm25: 15.0``, **17 score above the gate** -- every ``out_of_term`` row,
every ``field_absent`` row, three of six ``invented`` and one ``not_course``. The mechanism is
visible in ``code_scores``, which scores each declared code *alone* beside the question it sits
in. Every bare invented code is refused -- 6.32 to 14.49, all under 15 -- and three of the five
invented rows that name one clear the gate anyway once the code is inside an ordinary student
sentence (15.03 to 20.76), because the rest of the sentence matches real documents. So the gate
separates an invented course from a real one only when the query is the bare code, which no
student types. Two shapes cannot
be gated at all, by construction: a ``field_absent`` row names a course that *is* in the corpus,
so retrieval succeeds and the gate must pass it -- for those six the question scores exactly what
the bare code does, on the course's own chunk -- and a real out-of-term course is one the corpus
talks about, so all seven are above the gate as bare codes already (19.35 to 31.02; ``CS 428``
alone scores 27.16 on CS 527, whose prerequisite sentence names it). For those the refusal has
to come from the answer node's grounding, and Day 5's abstention machinery has to do something a
threshold on this score cannot.

**Why ``named_by`` works at all**, since #57 records that every prereq code in
``data/prereqs.json`` resolves inside the catalog and so "a prereq pointing at a missing course"
is not an available shape: the exporter **filters** the flattened list. CS 225's raw sentence
names MATH 347 and CS 126 and its ``prereqs`` array lists neither; CS 527, whose prerequisites are
*only* absent courses, has no row in ``prereqs.json`` at all. The raw text keeps what the list
drops, so scanning descriptions finds 636 codes the 2026-sp catalog names and does not offer --
and seven of them are this bucket's shape 2.

**Nothing here carries a clock, a model or a seed that matters.** The table is fixed, the checks
are a function of the committed ``data/courses.json``, and the only thing that moves between runs
is the retrieval score. A re-run under the same config rewrites the same file byte for byte.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from agent.nodes.check_evidence import check_gateable, threshold_for, thresholds
from agent.nodes.dispatch import check_retrieval_mode
from agent.tools.courses import CourseDetail
from config import Config, load_config, rel_to_repo
from eval.schema import RULES, EvalItem, item_id, row_json
from eval.scripts.draft_items import (
    ID_BLOCKS,
    check_drafts,
    draft_path,
    meta_path,
    write_json,
    write_rows,
)
from eval.validate import Violation, corpus_doc_ids, plural
from ingest.chunk import TEMPLATE_FIELDS, load_courses
from ingest.export_almabot import COURSES_PATH
from ingest.tokenize import CODE_RE

#: The bucket every row here belongs to. One bucket, four shapes -- :mod:`eval.schema` fixes the
#: first and this file fixes the second, because Day 4 conditions its metrics on the bucket and
#: #62's report breaks the refusals down by what kind of refusal they are.
BUCKET = "unanswerable"


# --- the four shapes-------------------------------------------------------------------------------


class Shape(NamedTuple):
    """One failure shape: how many rows it wants, and what :func:`check_item` demands of them."""

    #: The shape's name, repeated here so a rule read out of :data:`SHAPES` stands alone.
    name: str
    #: How many rows the table holds. A **gate**, unlike :data:`eval.schema.RULES`' targets: this
    #: table is not sampled or drafted, so a shape that is short is an edit that did not finish.
    n: int
    #: What every declared code must be. See :func:`check_codes`.
    codes: str
    #: Whether the row declares ``field`` -- the fact it asks for. Only ``field_absent`` does.
    wants_field: bool
    #: One sentence, for the summary and for #62's table.
    gloss: str


#: The four, in the order #57 lists them and the order the summary prints them.
SHAPES: Mapping[str, Shape] = {
    "invented": Shape(
        name="invented",
        n=6,
        codes="unknown",
        wants_field=False,
        gloss="the course does not exist (an invented code, or an invented subject and no code)",
    ),
    "out_of_term": Shape(
        name="out_of_term",
        n=7,
        codes="named",
        wants_field=False,
        gloss="real course, outside the export's one term",
    ),
    "field_absent": Shape(
        name="field_absent",
        n=6,
        codes="present",
        wants_field=True,
        gloss="real course, fact the knowledge base does not carry",
    ),
    "not_course": Shape(
        name="not_course",
        n=6,
        codes="none",
        wants_field=False,
        gloss="not about courses at all",
    ),
}

#: 25. Summed rather than written down, so the table above stays the one place a count lives.
TOTAL = sum(shape.n for shape in SHAPES.values())


def shape(name: str) -> Shape:
    """The rule for one shape, or a sentence naming the four that exist."""
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; the four are {', '.join(SHAPES)}")
    return SHAPES[name]


# --- the table-------------------------------------------------------------------------------------


class Item(NamedTuple):
    """One constructed row, before it is checked, numbered or scored.

    Not an :class:`eval.schema.EvalItem`: the id is minted in table order by :func:`build_rows`
    and ``bucket`` / ``should_abstain`` are fixed by the bucket, so a table that carried them
    would be a table with three columns nobody may edit.
    """

    #: Which of :data:`SHAPES` this row tests.
    shape: str
    #: What a student would type. Lower case, except a course code, which keeps the catalog's
    #: spelling (`eval/GUIDELINES.md` rule 1).
    question: str
    #: The refusal that would be correct, as a sentence -- not an empty string. Day 4 scores
    #: whether the refusal was right, and a reviewer needs to know what right looked like.
    gold_answer: str
    #: Every course code the question names, in the catalog's spelling. What :func:`check_codes`
    #: asserts about them depends on the shape.
    codes: tuple[str, ...] = ()
    #: ``out_of_term`` only: a course that IS in the export whose description names ``codes``.
    #: The corpus's own evidence that the missing course is a real one.
    named_by: str = ""
    #: ``field_absent`` only: the fact being asked for, as a field name. Checked against the chunk
    #: template and against :class:`~agent.tools.courses.CourseDetail` -- see U2 in the module
    #: docstring.
    field: str = ""
    #: Why this row is in the table, where that is not obvious from reading it. Carried into
    #: provenance, so the reason survives into #59's verification and #62's report.
    note: str = ""


#: The bucket. Twenty-five rows, six per shape and seven for ``out_of_term``.
#:
#: Three are not new: *"How many credits is CS 428?"* is `eval/GUIDELINES.md`'s boundary case
#: **U1**, *"What should I take next semester?"* is its **A2**, and *"Is there a course about
#: quantum origami?"* is the negative query PR #51 measured at BM25 11.06. Carrying the spec's own
#: worked cases into the set it specifies is deliberate: a boundary case that is argued in the
#: guidelines and absent from the data is a rule nothing tests.
ITEMS: tuple[Item, ...] = (
    # --- 1. the course does not exist ---------------------------------------------------------
    Item(
        shape="invented",
        question="how many credits is CS 999?",
        gold_answer="This catalog has no CS 999, so there is no credit count to give.",
        codes=("CS 999",),
        note="real subject, invented number -- the shape a mistyped code takes",
    ),
    Item(
        shape="invented",
        question="what are the prerequisites for MATH 8123?",
        gold_answer=(
            "This catalog has no MATH 8123 -- every course number in it is three digits -- so it "
            "lists no prerequisites for one."
        ),
        codes=("MATH 8123",),
        note=(
            "#57 expects the code boost to fire on the subject here; it does not. "
            "ingest.tokenize.CODE_RE is three digits and no more, so 'MATH 8123' is not a code to "
            "the tokenizer at all: it indexes as the words 'math' and '8123' and mints no "
            "identity term. The row is still the hardest of the six, for the ordinary reason "
            "that 'math' and 'prerequisites' both match 4,502 documents"
        ),
    ),
    Item(
        shape="invented",
        question="is there a course about quantum origami?",
        gold_answer="The 2026 spring catalog has no course about quantum origami.",
        note=(
            "the floor case: no code at all, and an invented subject. PR #51 measured it at BM25 "
            "11.06, the lowest of that report's twelve queries"
        ),
    ),
    Item(
        shape="invented",
        question="what does QUAN 300 cover?",
        gold_answer="This catalog has no QUAN 300, and QUAN is not one of its subject codes.",
        codes=("QUAN 300",),
        note="invented subject in a well-formed code -- 187 subjects and QUAN is not one",
    ),
    Item(
        shape="invented",
        question="does ZZZ 100 count toward a general education requirement?",
        gold_answer=(
            "This catalog has no ZZZ 100, so it cannot say what requirement the course would "
            "satisfy."
        ),
        codes=("ZZZ 100",),
        note=(
            "the obviously-invented code, kept as the control for the plausible ones: bare "
            "'ZZZ 100' scores 6.58 and the question scores 19.44, so what carries it over the "
            "gate is 'general education requirement' and not the code"
        ),
    ),
    Item(
        shape="invented",
        question="is BIOL 150 an intro biology course?",
        gold_answer="This catalog has no BIOL 150; BIOL is not one of its subjects.",
        codes=("BIOL 150",),
        note=(
            "invented for THIS catalog and real at many universities -- UIUC teaches introductory "
            "biology as IB and MCB. A better test than ZZZ 100 of whether the system knows which "
            "catalog it is reading"
        ),
    ),
    # --- 2. real course, outside the export's scope ---------------------------------------------
    Item(
        shape="out_of_term",
        question="how many credits is CS 428?",
        gold_answer="This catalog covers 2026 spring only, and CS 428 is not offered in it.",
        codes=("CS 428",),
        named_by="CS 527",
        note=(
            "GUIDELINES U1, and PR #51's example. CS 527's prerequisite sentence reads "
            "'CS 428, CSE 429 or CS 429' and none of the three is in the export, which is why "
            "CS 527 has no row in prereqs.json at all"
        ),
    ),
    Item(
        shape="out_of_term",
        question="what are the prerequisites for CS 241?",
        gold_answer="CS 241 is not in the 2026 spring catalog, so its prerequisites are not here.",
        codes=("CS 241",),
        named_by="CS 341",
        note=(
            "eight courses still list CS 241 as their prerequisite; CS 341 is the course that "
            "replaced it, which is the ordinary way a code leaves a catalog"
        ),
    ),
    Item(
        shape="out_of_term",
        question="what topics does MATH 221 cover?",
        gold_answer=(
            "The 2026 spring catalog does not list MATH 221, so it says nothing about what the "
            "course covers."
        ),
        codes=("MATH 221",),
        named_by="MATH 220",
        note=(
            "the most-named absent course in the corpus: 35 descriptions mention MATH 221. A "
            "system that answered from what other courses say about it would answer this one"
        ),
    ),
    Item(
        shape="out_of_term",
        question="what is GGIS 205 about?",
        gold_answer=(
            "GGIS 205 is not in the 2026 spring catalog; BADM 205 points at it, but the record it "
            "points to is not here."
        ),
        codes=("GGIS 205",),
        named_by="BADM 205",
        note=(
            "the sharpest row in the bucket. BADM 205's whole description is 'Same as GGIS 205. "
            "See GGIS 205.', so the retriever returns a document that names the course, is about "
            "the course, and carries none of its content -- it is the one cross-reference of 497 "
            "that does not resolve inside the export"
        ),
    ),
    Item(
        shape="out_of_term",
        question="does RHET 101 satisfy the Composition I requirement?",
        gold_answer=(
            "RHET 101 is not offered in 2026 spring, so this catalog cannot say which requirement "
            "it satisfies."
        ),
        codes=("RHET 101",),
        named_by="RHET 105",
        note=(
            "a Gen Ed question about an absent course: the requirement is in the knowledge base "
            "and the course is not, so the answerable half is the trap"
        ),
    ),
    Item(
        shape="out_of_term",
        question="how many credit hours is PHYS 100?",
        gold_answer=(
            "PHYS 100 is not in the 2026 spring catalog, so it carries no credit hours here."
        ),
        codes=("PHYS 100",),
        named_by="PHYS 221",
        note=(
            "PHYS 221's raw prerequisite sentence is 'PHYS 100; concurrent registration in "
            "PHYS 211' and its exported prereqs list holds only PHYS 211 -- the filtered list and "
            "the raw text disagreeing, in one row"
        ),
    ),
    Item(
        shape="out_of_term",
        question="what are the degree attributes of STAT 409?",
        gold_answer=(
            "STAT 409 is not offered in 2026 spring, so this catalog lists no degree attributes "
            "for it."
        ),
        codes=("STAT 409",),
        named_by="STAT 410",
        note=(
            "degree attributes ARE a chunk field, so this row tests that absence of the course "
            "beats presence of the field -- the mirror of the field_absent shape below"
        ),
    ),
    # --- 3. real course, fact that is not in the knowledge base ---------------------------------
    Item(
        shape="field_absent",
        question="who teaches CS 225?",
        gold_answer=(
            "This catalog does not record who teaches a course, so the instructor for CS 225 is "
            "not something it can give."
        ),
        codes=("CS 225",),
        field="instructor",
        note="the commonest question a course catalog cannot answer",
    ),
    Item(
        shape="field_absent",
        question="what time does MATH 241 meet?",
        gold_answer=(
            "Meeting times and rooms are not in this catalog, so when MATH 241 meets is not "
            "something it can say."
        ),
        codes=("MATH 241",),
        field="meeting_time",
        note="",
    ),
    Item(
        shape="field_absent",
        question="which textbook does CHEM 102 use?",
        gold_answer="This catalog does not list textbooks, so the book CHEM 102 uses is not in it.",
        codes=("CHEM 102",),
        field="textbook",
        note="",
    ),
    Item(
        shape="field_absent",
        question="what fraction of students get an A in ECON 102?",
        gold_answer=(
            "Grade distributions are not part of this catalog, so what fraction of ECON 102 "
            "students earn an A is not something it knows."
        ),
        codes=("ECON 102",),
        field="grade_distribution",
        note="",
    ),
    Item(
        shape="field_absent",
        question="how many students are enrolled in PSYC 100?",
        gold_answer=(
            "Enrolment counts are not in this catalog, so how many students are in PSYC 100 is "
            "not something it can say."
        ),
        codes=("PSYC 100",),
        field="enrollment",
        note=(
            "the row nearest GUIDELINES U2, and the reason check_field exists: 'how many sections "
            "of PSYC 100 are running?' is a DELETED row, not this one, because get_course returns "
            "section_count. Enrolment reaches no field at all"
        ),
    ),
    Item(
        shape="field_absent",
        question="is STAT 200 considered a hard class?",
        gold_answer=(
            "This catalog carries no difficulty or quality rating, so it cannot say whether "
            "STAT 200 is hard."
        ),
        codes=("STAT 200",),
        field="difficulty",
        note="an opinion about a real course -- no clarification makes it answerable, so it is "
        "unanswerable and not ambiguous",
    ),
    # --- 4. not about courses at all ------------------------------------------------------------
    Item(
        shape="not_course",
        question="where can I park near the engineering quad?",
        gold_answer="Parking is not something this course catalog covers.",
        note="",
    ),
    Item(
        shape="not_course",
        question="what time does the dining hall close on Sundays?",
        gold_answer="Dining hall hours are not in this course catalog.",
        note="",
    ),
    Item(
        shape="not_course",
        question="will it snow in Champaign this weekend?",
        gold_answer="This is a course catalog; it has no weather forecast.",
        note="the cheapest row in the bucket, and the lowest-scoring: 7.44",
    ),
    Item(
        shape="not_course",
        question="how much is tuition for an out-of-state undergraduate?",
        gold_answer="Tuition and fees are not in this course catalog.",
        note="named in GUIDELINES' list of what the knowledge base excludes by construction",
    ),
    Item(
        shape="not_course",
        question="who is the dean of the Grainger College of Engineering?",
        gold_answer="This catalog lists courses, not the people who administer the colleges.",
        note=(
            "the one not_course row that clears the gate (21.60, on ENG 101) -- 'engineering' and "
            "'college' are ordinary catalog words, so being off-topic is not the same as being "
            "lexically far away"
        ),
    ),
    Item(
        shape="not_course",
        question="what should I take next semester?",
        gold_answer=(
            "Answering this needs your transcript, your standing and a future term's schedule, "
            "none of which this catalog has."
        ),
        note=(
            "GUIDELINES A2, and the one row whose shape was a judgement call: it IS about courses, "
            "and it is here because what it actually asks for -- the student's own record -- is "
            "outside the knowledge base by construction, exactly like parking. It is the row that "
            "separates this bucket from `ambiguous`: no clarifying question closes it"
        ),
    ),
)


# --- what the corpus says--------------------------------------------------------------------------


class Corpus(NamedTuple):
    """The export, read for the three questions this file asks of it."""

    #: Every code the 2026-sp export offers: 4,502 of them, ``CS 374`` form.
    codes: frozenset[str]
    #: Absent code -> the courses whose description names it, sorted. 636 entries, and the
    #: evidence behind every ``out_of_term`` row.
    named: Mapping[str, tuple[str, ...]]

    def has(self, code: str) -> bool:
        return code in self.codes


def read_corpus(courses_path: Path | None = None) -> Corpus:
    """Read ``data/courses.json`` and index it by code and by the codes its descriptions name.

    ``courses.json`` and not ``chunks.jsonl``: the descriptions are the same text either way
    (the chunk renders this field), the JSON is committed where the chunk file is gitignored, and
    a bucket whose realness evidence depended on a rebuildable artifact would be a bucket that
    cannot be checked on a fresh clone.

    A code is "named" when :data:`ingest.tokenize.CODE_RE` finds it in a description, which is the
    same matcher the BM25 tokenizer uses -- deliberately, since a shape-2 row's whole premise is
    that the index has something to say about a course the catalog does not offer. The pattern is
    a shape and not a subject list, so the scan also yields ``OF 100`` out of "of 100 students";
    those are harmless here, because nothing reads :attr:`Corpus.named` except a lookup of a code
    the table already declared.
    """
    courses = list(load_courses(COURSES_PATH if courses_path is None else Path(courses_path)))
    codes = frozenset(course["code"] for course in courses)
    named: dict[str, set[str]] = {}
    for course in courses:
        for match in CODE_RE.finditer(course.get("desc") or ""):
            found = f"{match.group('subject').upper()} {match.group('number')}"
            if found not in codes:
                named.setdefault(found, set()).add(course["code"])
    return Corpus(codes, {code: tuple(sorted(by)) for code, by in named.items()})


def known_fields() -> frozenset[str]:
    """Every field name a ``field_absent`` row may **not** ask for.

    Two sources, and the second is the one that matters. The chunk templates' fields are what a
    retrieved document prints; :class:`~agent.tools.courses.CourseDetail`'s are what ``get_course``
    returns, which is strictly more -- ``section_count`` among them. `eval/GUIDELINES.md`'s U2 is
    exactly that gap: the agent can answer a question about section counts, no chunk supports the
    number, and the row is deleted rather than labelled unanswerable. Reading the model and the
    template table rather than a list written down here means a field added to ``get_course``
    later invalidates the row that asks for it, in a test, instead of leaving a mislabel in the
    frozen set.

    Every template's fields, not the configured one's. A row must not become answerable because
    ``chunk.template_version`` was switched, and the two spellings of one field (``prereqRaw`` in
    the template, ``prereq_raw`` in the model) are both refused for the same reason.
    """
    template = frozenset(name for fields in TEMPLATE_FIELDS.values() for name in fields)
    return template | frozenset(CourseDetail.model_fields)


# --- checking the table----------------------------------------------------------------------------


def normalize(code: str) -> str:
    """A declared code in the catalog's spelling, or itself when it is not code-shaped.

    ``MATH 8123`` is deliberately not repaired: it is a row, and a normalizer that rejected it
    would make the table unable to say "this is not even a course number".
    """
    match = CODE_RE.fullmatch(code.strip())
    if match is None:
        return code.strip()
    return f"{match.group('subject').upper()} {match.group('number')}"


def codes_in(question: str) -> tuple[str, ...]:
    """Every code :data:`ingest.tokenize.CODE_RE` finds in a question, normalized, in order."""
    return tuple(
        f"{m.group('subject').upper()} {m.group('number')}" for m in CODE_RE.finditer(question)
    )


def check_codes(n: int, item: Item, corpus: Corpus) -> Iterator[Violation]:
    """The claim the row's shape makes about every code it declares.

    The four rules are the four shapes, and each of them is falsifiable against the export:
    ``unknown`` (absent and never named), ``named`` (absent and named, with the namer checked),
    ``present`` (offered this term), ``none`` (no codes at all).
    """
    rule = shape(item.shape)
    if rule.codes == "none":
        if item.codes:
            yield violation(
                n,
                item,
                "codes",
                f"declares {', '.join(item.codes)}; {item.shape} rows are not about a course, "
                "and one naming a code is a row in another shape",
            )
        return
    if not item.codes:
        if rule.codes != "unknown":
            yield violation(
                n, item, "codes", f"declares none; every {item.shape} row names a course"
            )
        # An `invented` row may name no code at all: #57's second kind of invention is a
        # *subject* ("is there a course about quantum origami?"), which has no code to declare.
        # There is nothing to check it against either -- the corpus cannot be asked whether it
        # lacks a topic -- so what stands in for the check is `check_question`, which refuses a
        # codeless row whose question names any code, invented or real.
        return

    for code in item.codes:
        if rule.codes == "present":
            if not corpus.has(code):
                yield violation(
                    n,
                    item,
                    "codes",
                    f"{code} is not in the export; a {item.shape} row asks a question the catalog "
                    "cannot answer ABOUT A COURSE IT HAS, and one about a missing course is an "
                    "out_of_term row wearing the wrong shape",
                )
            continue

        if corpus.has(code):
            yield violation(
                n,
                item,
                "codes",
                f"{code} IS in the 2026-sp export, so the catalog answers this question and the "
                "row is not unanswerable at all",
            )
            continue
        namers = corpus.named.get(code, ())
        if rule.codes == "unknown" and namers:
            yield violation(
                n,
                item,
                "codes",
                f"{code} is named by {', '.join(namers[:3])}; a code the catalog talks about is a "
                "real course that is out of term, not an invented one -- move the row to "
                "out_of_term",
            )
        if rule.codes == "named" and not namers:
            yield violation(
                n,
                item,
                "codes",
                f"nothing in the export names {code}, so the corpus cannot vouch for it being a "
                "real course; an out_of_term row whose realness is only remembered is a row the "
                "verification pass will relabel",
            )


def check_named_by(n: int, item: Item, corpus: Corpus) -> Iterator[Violation]:
    """``named_by`` is present exactly for ``out_of_term``, and really does name the code."""
    wants = shape(item.shape).codes == "named"
    if not wants:
        if item.named_by:
            yield violation(
                n, item, "named_by", f"is {item.named_by!r}; only an out_of_term row carries one"
            )
        return
    if not item.named_by:
        yield violation(
            n,
            item,
            "named_by",
            "is empty; an out_of_term row records the course whose description names the missing "
            "one, because that is the corpus's own evidence that it is real",
        )
        return
    if not corpus.has(item.named_by):
        yield violation(
            n,
            item,
            "named_by",
            f"{item.named_by} is not in the export either, so it vouches for nothing",
        )
        return
    for code in item.codes:
        if item.named_by not in corpus.named.get(code, ()):
            yield violation(
                n,
                item,
                "named_by",
                f"{item.named_by}'s description does not name {code}",
            )


def check_field(n: int, item: Item, fields: frozenset[str]) -> Iterator[Violation]:
    """``field`` is present exactly for ``field_absent``, and is outside the knowledge base."""
    if not shape(item.shape).wants_field:
        if item.field:
            yield violation(
                n, item, "field", f"is {item.field!r}; only a field_absent row carries one"
            )
        return
    if not item.field:
        yield violation(
            n,
            item,
            "field",
            "is empty; a field_absent row records WHICH fact it asks for, because that is the "
            "claim being checked against the chunk template and get_course",
        )
        return
    if item.field in fields:
        yield violation(
            n,
            item,
            "field",
            f"{item.field!r} is a field the knowledge base carries, so the question is answerable "
            "-- if a chunk prints it the row is a fact row, and if only get_course returns it "
            "(section_count) the row is deleted, per GUIDELINES U2",
        )


def check_question(n: int, item: Item) -> Iterator[Violation]:
    """The question names the codes the row declares, and no others.

    A row that declares ``CS 241`` and asks about ``CS 249`` passes every other check in this
    file -- both are absent -- and tests nothing anyone meant. Codes the pattern cannot see
    (``MATH 8123``) are looked for as literal text instead.
    """
    declared = tuple(normalize(code) for code in item.codes)
    found = codes_in(item.question)
    for code in found:
        if code not in declared:
            yield violation(
                n,
                item,
                "question",
                f"names {code}, which the row does not declare in `codes`",
            )
    for code in declared:
        if CODE_RE.fullmatch(code) is None:
            if code not in item.question:
                yield violation(n, item, "question", f"does not contain {code!r}")
        elif code not in found:
            yield violation(n, item, "question", f"does not name {code}, which the row declares")


def check_answer(n: int, item: Item) -> Iterator[Violation]:
    """The refusal names every code it is about (`eval/GUIDELINES.md` rule 1) and is a sentence."""
    if not item.gold_answer.strip():
        yield violation(
            n,
            item,
            "gold_answer",
            "is empty; the row records the refusal that would "
            "be correct, because Day 4 scores whether the refusal was right",
        )
        return
    for code in item.codes:
        if code not in item.gold_answer:
            yield violation(
                n,
                item,
                "gold_answer",
                f"does not name {code}; a refusal whose subject is ambiguous cannot be scored",
            )


def violation(n: int, item: Item, field: str, message: str) -> Violation:
    """A finding about the table, addressed the way the validator addresses one about a file."""
    return Violation(n, row_id(n), f"{item.shape}.{field}", message)


def row_id(n: int) -> str:
    """The id the ``n``-th row of the table gets: ``v1-2001`` upwards.

    #56's :data:`~eval.scripts.draft_items.ID_BLOCKS` reserves the block and does not use it, so
    the four buckets can be built independently and still never collide when #59 merges them.
    """
    return item_id(ID_BLOCKS[BUCKET] + n - 1)


def check_table(items: Sequence[Item], corpus: Corpus) -> list[Violation]:
    """Every check over the table, in table order: the shapes' claims, then the counts.

    Every violation, never the first -- the same rule :mod:`eval.validate` follows, for the same
    reason: a table with eight problems should be one run with eight messages.
    """
    fields = known_fields()
    found: list[Violation] = []
    counts = {name: 0 for name in SHAPES}
    for n, item in enumerate(items, start=1):
        if item.shape not in SHAPES:
            found.append(
                Violation(
                    n,
                    row_id(n),
                    "shape",
                    f"is {item.shape!r}; the four are {', '.join(SHAPES)}",
                )
            )
            continue
        counts[item.shape] += 1
        found.extend(check_codes(n, item, corpus))
        found.extend(check_named_by(n, item, corpus))
        found.extend(check_field(n, item, fields))
        found.extend(check_question(n, item))
        found.extend(check_answer(n, item))
    for name, rule in SHAPES.items():
        if counts[name] != rule.n:
            found.append(
                Violation(
                    None,
                    None,
                    name,
                    f"{plural(counts[name], 'row')}, not the {rule.n} the shape is built to hold "
                    f"-- {rule.gloss}",
                )
            )
    if len(items) != TOTAL:
        found.append(
            Violation(None, None, None, f"{plural(len(items), 'row')} in total, not {TOTAL}")
        )
    return found


# --- scoring, which labels nothing-----------------------------------------------------------------


class Score(NamedTuple):
    """What ``retrieve()`` says about one query: the top hit, or nothing when it returned none."""

    score: float | None
    doc_id: str | None


def score_query(query: str, mode: str, cfg: Config) -> Score:
    """Top-1 for one query. Imported at call time, so ``--no-retrieval`` needs no index."""
    from retrieval import retrieve

    hits = retrieve(query, mode=mode, k=1, cfg=cfg).hits  # type: ignore[arg-type]
    if not hits:
        return Score(None, None)
    return Score(round(hits[0].score, 4), hits[0].doc_id)


def gate_for(mode: str, cfg: Config) -> float:
    """The threshold this mode's scores are judged against, read from the gate's own helpers.

    Through :func:`agent.nodes.check_evidence.check_gateable` first, so ``--mode hybrid`` is
    refused here with the measurement that says why rather than being compared against a number
    #52 established cannot separate anything.
    """
    return threshold_for(check_gateable(mode), thresholds(cfg))


# --- building the rows-----------------------------------------------------------------------------

#: Every key a constructed row's ``provenance`` carries. Deliberately **not** #56's set: those
#: rows record a batch, a model and a leakage overlap, and none of the three exists for a row a
#: person wrote. ``source`` is where a drafted row keeps ``model``; a ``model: null`` there would
#: read as a call that failed rather than a call that never happened. #59 merges two provenance
#: shapes, which is the honest thing for it to see.
PROVENANCE_KEYS = (
    "shape",
    "source",
    "codes",
    "named_by",
    "field",
    "note",
    "config_hash",
    "retrieval_mode",
    "top_score",
    "top_doc_id",
    "code_scores",
    "gate",
    "above_gate",
)

#: What ``source`` says. One value today; #59 may promote a row and record that it rewrote it.
SOURCE = "constructed"


def build_rows(
    items: Sequence[Item],
    cfg: Config,
    mode: str | None = None,
    scored: bool = True,
) -> list[dict[str, Any]]:
    """The table as draft rows: six dataset fields, then provenance, in table order.

    ``scored`` off writes ``top_score``, ``top_doc_id`` and ``code_scores`` as ``None`` rather
    than dropping the keys -- a row that was never scored and a row that scored nothing must not
    read the same in the file, and ``retrieval_mode`` records which it was.

    The mode is validated either way, ``--no-retrieval`` included. A mode ``top_score`` cannot
    gate would make this bucket's one number meaningless, and accepting the flag on the run that
    does not use it would only move the failure to the run that does.
    """
    used = check_gateable(check_retrieval_mode(cfg, mode))
    gate = gate_for(used, cfg) if scored else None
    rows: list[dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        top = score_query(item.question, used, cfg) if scored else Score(None, None)
        codes = (
            {code: score_query(code, used, cfg).score for code in item.codes} if scored else None
        )
        row = EvalItem(
            id=row_id(n),
            bucket=BUCKET,
            question=item.question,
            gold_answer=item.gold_answer,
            gold_doc_ids=[],
            should_abstain=RULES[BUCKET].should_abstain,
        )
        rows.append(
            {
                **row_json(row),
                "provenance": {
                    "shape": item.shape,
                    "source": SOURCE,
                    "codes": list(item.codes),
                    "named_by": item.named_by or None,
                    "field": item.field or None,
                    "note": item.note or None,
                    "config_hash": cfg.hash,
                    "retrieval_mode": used if scored else None,
                    "top_score": top.score,
                    "top_doc_id": top.doc_id,
                    "code_scores": codes,
                    "gate": gate,
                    "above_gate": (
                        None if top.score is None or gate is None else top.score >= gate
                    ),
                },
            }
        )
    return rows


def sidecar(rows: Sequence[Mapping[str, Any]], cfg: Config, scored: bool) -> dict[str, Any]:
    """What the run records beside the rows: the counts, the gate, and the rows it did not refuse.

    Deliberately not a copy of the rows. It answers the two questions #62 asks of this bucket --
    how the 25 split across the four shapes, and how many of them the gate would have refused on
    the score alone -- without anybody having to aggregate a JSONL file to find out.
    """
    provenance = [dict(row["provenance"]) for row in rows]
    scores = [p["top_score"] for p in provenance if p["top_score"] is not None]
    above = [row["id"] for row, p in zip(rows, provenance, strict=True) if p["above_gate"]]
    return {
        "bucket": BUCKET,
        "source": SOURCE,
        "config_hash": cfg.hash,
        "rows": len(rows),
        "shapes": {name: sum(1 for p in provenance if p["shape"] == name) for name in SHAPES},
        "retrieval_mode": provenance[0]["retrieval_mode"] if provenance else None,
        "gate": provenance[0]["gate"] if provenance else None,
        "scored": scored,
        "top_score": (
            None if not scores else {"min": min(scores), "max": max(scores), "n": len(scores)}
        ),
        "above_gate": above,
    }


# --- the run---------------------------------------------------------------------------------------


class Report(NamedTuple):
    """One run: where it wrote, what it wrote, and everything wrong with either."""

    path: Path
    meta: Path
    rows: list[dict[str, Any]]
    table: list[Violation]
    written: list[Violation]
    scored: bool
    mode: str | None


def build(
    cfg: Config,
    mode: str | None = None,
    scored: bool = True,
    drafts_dir: Path | None = None,
    courses_path: Path | None = None,
    items: Sequence[Item] | None = None,
) -> Report:
    """Check the table, score it, write it, and re-read what was written.

    Nothing is written when a table check fails. A drafting run keeps the rows it paid a model
    for and drops the rest; there is nothing to salvage here -- the table is in the repository
    either way -- and half a bucket on disk under a name #59 reads is worse than none.

    ``items`` defaults to :data:`ITEMS` and is resolved on the call rather than bound to the
    signature, so a caller that replaces the table -- a test, or #59 promoting a rewritten row --
    gets the table it set and not the one this module held at import.
    """
    items = ITEMS if items is None else items
    corpus = read_corpus(courses_path)
    table = check_table(items, corpus)
    path = draft_path(BUCKET, cfg, drafts_dir)
    if any(v.level == "error" for v in table):
        return Report(path, meta_path(path), [], table, [], scored, mode)

    rows = build_rows(items, cfg, mode=mode, scored=scored)
    write_rows(path, rows)
    write_json(meta_path(path), sidecar(rows, cfg, scored))
    written = check_drafts(path, corpus_doc_ids(), keys=PROVENANCE_KEYS)
    return Report(path, meta_path(path), rows, table, written, scored, mode)


# --- saying so-------------------------------------------------------------------------------------


def gate_lines(report: Report) -> list[str]:
    """The callout #57's acceptance asks for: which rows the gate would not have refused, and why.

    Not a warning. A row above the gate is a correct row -- it is a row whose refusal, if Day 4
    sees one, came from the model and not from the evidence check, and reporting it as a refusal
    the gate earned would credit the wrong component.
    """
    scored = [
        (row["id"], dict(row["provenance"]))
        for row in report.rows
        if row["provenance"]["top_score"] is not None
    ]
    if not scored:
        return ["retrieval            not run, so no row has a score (--no-retrieval)"]
    gate = scored[0][1]["gate"]
    above = [(rid, p) for rid, p in scored if p["above_gate"]]
    values = sorted(p["top_score"] for _, p in scored)
    mode = scored[0][1]["retrieval_mode"]
    lines = [
        f"retrieval            {mode} top-1, gate {gate} (evidence.thresholds)",
        (
            f"top_score            min {values[0]}  max {values[-1]}  "
            f"above the gate {len(above)} of {len(scored)}"
        ),
    ]
    if not above:
        return lines
    lines += [
        "",
        (
            "Above the gate -- the gate would not refuse these, so a Day 4 refusal here is the "
            "model's and not the evidence check's:"
        ),
    ]
    for rid, p in above:
        bare = p["code_scores"] or {}
        alone = "  ".join(f"{code} alone {score}" for code, score in bare.items())
        lines.append(
            f"  {rid}  {p['shape']:<12} {p['top_score']:>8}  {p['top_doc_id']}"
            + (f"   ({alone})" if alone else "")
        )
    return lines


def summary(cfg: Config, report: Report) -> str:
    """What the terminal gets: the shapes, the scores, the callout, and where it all went."""
    if any(v.level == "error" for v in report.table):
        return (
            f"config hash {cfg.hash}  bucket {BUCKET}\n"
            f"the table does not check out, so nothing was written to "
            f"{rel_to_repo(report.path)}"
        )
    counts = {name: 0 for name in SHAPES}
    for row in report.rows:
        counts[row["provenance"]["shape"]] += 1
    ids = [row["id"] for row in report.rows]
    lines = [
        f"config hash {cfg.hash}  bucket {BUCKET}",
        f"rows                 {len(report.rows)} ({ids[0]} .. {ids[-1]})" if ids else "rows 0",
    ]
    for name, rule in SHAPES.items():
        lines.append(f"  {name:<18} {counts[name]}  {rule.gloss}")
    lines.extend(gate_lines(report))
    lines.append(
        f"wrote                {rel_to_repo(report.path)}, sidecar {rel_to_repo(report.meta)}"
    )
    return "\n".join(lines)


def render(violations: Sequence[Violation], where: str) -> str:
    """The table's findings, one line each, addressed by row number rather than by file line."""
    out = []
    for v in violations:
        at = f"{where} row {v.line}" if v.line else where
        what = " ".join(part for part in (v.item_id, v.field) if part is not None)
        out.append(f"{at}  {what}  {v.message}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.build_unanswerable",
        description="Check, score and write the 25 constructed unanswerable rows into "
        "eval/datasets/drafts/.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="retrieval mode to score with (default: agent.retrieval_mode); hybrid is refused, "
        "since top_score cannot gate an RRF score",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="skip scoring: the rows are written with a null top_score, and no index is opened",
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument(
        "--courses", default=None, help=f"the export (default: {rel_to_repo(COURSES_PATH)})"
    )
    parser.add_argument(
        "--drafts-dir", default=None, help="where the rows go (default: eval/datasets/drafts)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    report = build(
        cfg,
        mode=args.mode,
        scored=not args.no_retrieval,
        drafts_dir=Path(args.drafts_dir) if args.drafts_dir else None,
        courses_path=Path(args.courses) if args.courses else None,
    )
    print(summary(cfg, report))

    status = 0
    if report.table:
        print(
            "\nThe table in eval/scripts/build_unanswerable.py does not check out against the "
            "corpus:\n" + render(report.table, "ITEMS"),
            file=sys.stderr,
        )
        status = 1
    errors = [v for v in report.written if v.level == "error"]
    if errors:
        print(
            f"\n{rel_to_repo(report.path)} does not validate after the run, which is a bug in "
            "this script -- every row was checked before it was written:\n"
            + "\n".join(f"  line {v.line}  {v.field}  {v.message}" for v in errors),
            file=sys.stderr,
        )
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
