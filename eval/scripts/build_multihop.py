"""Twenty-five rows whose answer is a walk, drawn from the prereq graph rather than from prose.

    uv run python -m eval.scripts.build_multihop

The `fact` and `ambiguous` buckets are drafted from documents (:mod:`eval.scripts.draft_items`)
and this one is not, for a reason #58 states plainly: a model shown ten course records writes
questions those records answer, and it will write a one-hop question and label it ``multihop``.
A bucket full of one-hop questions measures nothing. So the bucket is a **table in this file** --
:data:`ITEMS`, twenty-five of them -- built against ``data/prereqs.json``, and what the script
does is check that table against the committed graph and the catalog's own sentences.

**Depth is two measures, and they disagree.** ``get_prereq_chain`` layers a course's closure by
*shortest* distance, which is the contract #38's answer node needs; #58's numbers -- 738 courses
at depth >= 2, deepest chain 13 -- are *longest*-chain depths. Both are computed here
(:class:`Graph`) and both are recorded per row, because they are not the same census: by levels
the graph has **718** courses at depth >= 2 and its deepest is **9**. Twenty courses are depth 2
one way and depth 1 the other, all of them courses every one of whose prerequisites is also a
direct prerequisite. A row must clear 2 on **both**, so no row's multi-hop-ness depends on which
measure you picked; the stratification below uses the longest-chain depth, which is the one #58
sampled from.

**Four records is the ceiling, and it comes from the spec.** #58 says ``gold_doc_ids`` lists every
course in the chain; `eval/GUIDELINES.md` caps a join at four records and deletes an unbounded
chain outright ("Bounded chains only... The question states its depth, or the row is deleted").
Those meet where the *question* bounds the chain: not "everything before SE 495" -- 27 courses,
and a cycle -- but "is MATH 257 required for SE 495?", whose answer rests on three records and
whose root still sits 13 levels deep. So every row declares a :attr:`Item.chain` of two to four
courses, that chain is the row's ``gold_doc_ids``, and :func:`check_chain` refuses a fifth.

**The tool proposes, the raw sentence disposes.** The obvious way to write these gold answers is
to run ``get_prereq_chain()`` and copy what it returns, which would make the set a regression test
for the walker rather than a measure of correctness: if the walk is wrong the gold is wrong the
same way and Day 4 scores 100%. So every answer here was written from the ``prereqRaw`` sentences
of the courses in its chain, and :func:`disagreements` re-derives, per row, where the exported
``prereqs`` list and that sentence do not match. It is a **log, not a gate**: a disagreement is a
bug report against #30 or the ingest parser and a row that has one must carry a ``note`` saying
which it is, but the run still writes.

**What that log found, and it is the finding of this file.** Over the 46 courses in the whole
graph whose parsed list drops a code its own sentence names -- counting only codes the 2026-sp
catalog offers, so the export's term filter is not mistaken for a parser bug -- three kinds turn
up, and the rows here were chosen to carry one of each. ``AE 312`` reads *"AE 202 and MATH 285.
Credit or concurrent registration in ME 200."* and exports two codes: a **requirement dropped**,
and ME 200 is offered this term, so a chain that reports it is complete is wrong. ``ASTR 405``
drops ``PHYS 213`` and ``PHYS 214`` from *"Recommended: PHYS 213 and PHYS 214"*, which is the
parser being **right** -- a recommendation is not a prerequisite -- and is in the log all the
same, because "the tool and the text differ" is what the log records and "and the tool is
correct" is the reviewer's line, not the scanner's. ``JOUR 340`` drops ``JOUR 217`` out of *"May
not be taken concurrently with JOUR 217"*, a **negation** read as a mention. The fourth kind is
not a parser bug at all and is the commonest: ``MATH 221`` and ``MATH 347`` are named in sentences
all over the catalog and are not offered in 2026 spring, so the export filters them and the
surviving list reads as a conjunction where the sentence offered a choice. ``ATMS 202`` requires
*"MATH 220 or MATH 221."* and exports ``["MATH 220"]``; #58's own example, ``ABE 228``, is this.

One more, which no scan finds and a reviewer does: ``MATH 220``'s sentence asks for *an adequate
ALEKS placement score ... demonstrating knowledge of topics of MATH 115* and the export made that
an edge to MATH 115. The code is named, so the two "agree"; the requirement is a placement score
and not a course. Row ``v1-1003`` is written against the sentence and says so.

**Nothing here carries a clock, a model, a seed or a retrieval score.** #57 scored its rows
because its finding was about the evidence gate; this bucket's finding is the agreement log above,
and `eval/GUIDELINES.md` forbids asking ``retrieve()`` anything that could bear on a label. The
table is fixed and every check is a function of ``data/courses.json`` and ``data/prereqs.json``,
both committed, so a re-run rewrites the same file byte for byte and needs no index.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from agent.tools.courses import course_index
from agent.tools.prereqs import load_graph, walk_chain
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
from ingest.chunk import doc_id
from ingest.tokenize import CODE_RE

#: The bucket every row here belongs to. :mod:`eval.schema` fixes what that means for the six
#: dataset fields; this file fixes the five question shapes below, which Day 4 and #62 break the
#: bucket down by.
BUCKET = "multihop"

#: The file stem this bucket is written under: ``multihop_graph_{hash}.jsonl``, not
#: ``multihop_{hash}.jsonl``. #56 still drafts ``multihop`` from documents and says in its own
#: docstring that those are *candidates only* -- "the real multi-hop rows come off the prereq
#: graph in #58" -- so both files exist and both are wanted. What they must not do is share a
#: name: they carry different provenance keys, ``parse_drafts`` is told which set to hold a file
#: to, and whichever script ran second would silently clobber the other. The stem says which
#: file is which, and ``source`` in every row says it again.
STEM = f"{BUCKET}_graph"

#: Fewest and most records a row may join. The floor is :data:`eval.schema.RULES`' -- a one-record
#: multihop row is a ``fact`` row -- and the ceiling is `eval/GUIDELINES.md`'s: "Cap the join at
#: four records. More than that is a list question."
MIN_CHAIN, MAX_CHAIN = RULES[BUCKET].min_gold_ids, 4

#: The depth every root must clear, on **both** measures. See the module docstring.
MIN_DEPTH = 2

#: How many rows must sit at longest-chain depth >= 5. #58: "Include a few deep ones: they are
#: where a recursive walk that half-works still looks fine on depth 2."
MIN_DEEP, DEEP_AT = 5, 5

#: Most rows allowed at the shallowest depth the bucket accepts, so the distribution is a
#: stratification and not a sample of the easy end. A third of the bucket.
MAX_SHALLOW = 8

#: How many rows must name a course inside one of the graph's seven cycles. #58: "A cycle-safe
#: answer is a real requirement, and an eval set that avoids the hard part of the graph will not
#: catch it regressing."
MIN_CYCLIC = 2

#: What makes a requirement sentence a disjunction. `eval/GUIDELINES.md` rule 3: the flattened
#: ``prereqCodes`` list has the booleans taken out of it, so an answer that turns "One of CS 173,
#: MATH 213" into "CS 173 and MATH 213" is wrong and not loose. Checked in both directions by
#: :func:`check_disjunction`: a row whose sentences offer a choice must say so.
DISJUNCTIONS = (" or ", "one of", "either", "and/or")


# --- the five question shapes ---------------------------------------------------------------------


class Shape(NamedTuple):
    """One question shape: how many rows it wants, and what a row of it must declare."""

    #: The shape's name, repeated here so a rule read out of :data:`SHAPES` stands alone.
    name: str
    #: How many rows the table holds. A **gate**, unlike :data:`eval.schema.RULES`' targets: this
    #: table is not sampled or drafted, so a shape that is short is an edit that did not finish.
    n: int
    #: ``True`` when the chain runs *out* of the root -- the courses that list it -- rather than
    #: into it. Only ``unlocks``; see :func:`check_chain`.
    reverse: bool
    #: Whether the row names the course the question asks *about* (``reachable``).
    wants_target: bool
    #: Whether the row names the courses the student says they have (``eligibility``).
    wants_held: bool
    #: One sentence, for the summary and for #62's table.
    gloss: str


#: The five, in the order #58 lists them and the order the summary prints them. Five rather than
#: four so the bucket is not 25 rewordings of "what do I need before X": the shapes traverse the
#: same graph four different ways, and ``unlocks`` traverses it backwards.
SHAPES: Mapping[str, Shape] = {
    "chain": Shape(
        name="chain",
        n=6,
        reverse=False,
        wants_target=False,
        wants_held=False,
        gloss="what do I need before X, and what do those require",
    ),
    "reachable": Shape(
        name="reachable",
        n=6,
        reverse=False,
        wants_target=True,
        wants_held=False,
        gloss="is Y required for X, directly or indirectly (yes/no)",
    ),
    "eligibility": Shape(
        name="eligibility",
        n=5,
        reverse=False,
        wants_target=False,
        wants_held=True,
        gloss="can I take X if I have only taken Y",
    ),
    "unlocks": Shape(
        name="unlocks",
        n=4,
        reverse=True,
        wants_target=False,
        wants_held=False,
        gloss="if I take X, what does it open up (reverse edges)",
    ),
    "distance": Shape(
        name="distance",
        n=4,
        reverse=False,
        wants_target=False,
        wants_held=False,
        gloss="how many courses stand between me and X",
    ),
}

#: 25. Summed rather than written down, so the table above stays the one place a count lives.
TOTAL = sum(s.n for s in SHAPES.values())


def shape(name: str) -> Shape:
    """The rule for one shape, or a sentence naming the five that exist."""
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; the five are {', '.join(SHAPES)}")
    return SHAPES[name]


# --- the table ------------------------------------------------------------------------------------


class Item(NamedTuple):
    """One row, before it is checked, numbered or turned into gold ids.

    Not an :class:`eval.schema.EvalItem`: ``id`` is minted in table order by :func:`build_rows`,
    ``bucket`` and ``should_abstain`` are fixed by the bucket, and ``gold_doc_ids`` is
    :attr:`chain` rendered through :func:`ingest.chunk.doc_id` -- so a table that carried them
    would be a table with four columns nobody may edit.
    """

    #: Which of :data:`SHAPES` this row tests.
    shape: str
    #: The course the question is about, in the catalog's spelling. Its depth is what the row is
    #: stratified by, and :func:`check_depth` holds it to :data:`MIN_DEPTH` on both measures.
    root: str
    #: What a student would type. Lower case, except a course code (`eval/GUIDELINES.md` rule 1).
    question: str
    #: The correct answer, written from the ``prereqRaw`` sentences of the courses in
    #: :attr:`chain` and not from what ``get_prereq_chain`` returned. Disjunctions stay
    #: disjunctions.
    gold_answer: str
    #: Every course whose record the answer rests on, root first: the row's ``gold_doc_ids``.
    #: Two to four (:data:`MIN_CHAIN`, :data:`MAX_CHAIN`), connected in the direction the shape
    #: walks.
    chain: tuple[str, ...]
    #: ``reachable`` only: the course the question asks about. Need **not** be in the chain -- for
    #: half these rows the answer is "no", and what shows that is the root's sentence.
    target: str = ""
    #: ``eligibility`` only: the courses the question says the student has taken.
    held: tuple[str, ...] = ()
    #: Why this row is in the table, and the reviewer's verdict on any disagreement between the
    #: exported list and the raw sentence. **Required** when :func:`disagreements` finds one, so
    #: the judgement is in the file rather than in somebody's head.
    note: str = ""


#: The bucket. Twenty-five rows over five shapes, stratified by the root's longest-chain depth.
#:
#: Two of `eval/GUIDELINES.md`'s worked cases are answered here rather than restated: **M2**
#: deletes *"list every course you have to take before MATH 412"* because the transitive answer
#: does not terminate, and ``v1-1009`` asks the bounded question about the same course instead;
#: **M1**'s point -- that a cross-reference makes a second record load-bearing -- is the same
#: point ``v1-1012`` makes about a disjunction whose other branch is out of term.
ITEMS: tuple[Item, ...] = (
    # --- 1. the bounded chain ------------------------------------------------------------------
    Item(
        shape="chain",
        root="FR 104",
        question="what do I need before FR 104, and what do those courses require in turn?",
        gold_answer=(
            "FR 104 requires FR 103, FR 103 requires FR 102, and FR 102 requires FR 101 -- and "
            "each of the three sentences also accepts a placement score or the matching year of "
            "high school French instead of the course below it."
        ),
        chain=("FR 104", "FR 103", "FR 102", "FR 101"),
        note=(
            "the cleanest three-level chain in the catalog and the floor case for the shape: "
            "a single path, no branching, and the only thing a system can get wrong is stopping "
            "at FR 103"
        ),
    ),
    Item(
        shape="chain",
        root="ACCY 202",
        question="what do I need before ACCY 202, and what does that course require?",
        gold_answer=(
            "ACCY 202 requires ACCY 201 or equivalent, and ACCY 201 asks for either ECON 102 or "
            "ECON 103 completed, with the remaining one of the two completed or taken "
            "concurrently."
        ),
        chain=("ACCY 202", "ACCY 201", "ECON 102", "ECON 103"),
        note=(
            "the exported list gives ACCY 201 both ECON courses flat, which is nearly right and "
            "loses the part a student needs: one of them may be taken alongside"
        ),
    ),
    Item(
        shape="chain",
        root="ATMS 202",
        question="what do I have to take before ATMS 202, and what comes before that?",
        gold_answer=(
            "ATMS 202 requires MATH 220 or MATH 221. MATH 220 does not require a course: it asks "
            "for an adequate ALEKS placement score demonstrating knowledge of the topics of "
            "MATH 115. MATH 115 itself requires MATH 112."
        ),
        chain=("ATMS 202", "MATH 220", "MATH 115", "MATH 112"),
        note=(
            "two disagreements in one row, and the second is the one no scanner finds. MATH 221 "
            "is not offered in 2026 spring, so the export filters it and ATMS 202's disjunction "
            "reads as a single requirement -- #58's ABE 228 case exactly. And MATH 220's sentence "
            "names MATH 115 only to say what the placement score must demonstrate; the export "
            "made it an edge, so the codes agree and the requirement does not. Filed against the "
            "ingest parser; the answer is written from the sentence"
        ),
    ),
    Item(
        shape="chain",
        root="MATH 503",
        question="what do I need before MATH 503, and what does that course require?",
        gold_answer=(
            "MATH 503 requires MATH 500 or equivalent; MATH 500 requires MATH 417 and MATH 418; "
            "and MATH 418 requires MATH 417 or consent of instructor."
        ),
        chain=("MATH 503", "MATH 500", "MATH 417", "MATH 418"),
        note=(
            "the deepest root in the bucket that still bounds to four records (longest chain 12, "
            "closure 20). Its walk reports cycle_detected, because MATH 417 reaches CS 374 and "
            "the CS 225 / CS 374 / MATH 412 cycle sits under it -- three levels below anything "
            "this question asks about, which is the point: a walker that loops here loops on a "
            "question whose answer is four courses"
        ),
    ),
    Item(
        shape="chain",
        root="CEE 462",
        question="what do I need before CEE 462, and what does that course require?",
        gold_answer="CEE 462 requires CEE 460, and CEE 460 requires CEE 360.",
        chain=("CEE 462", "CEE 460", "CEE 360"),
        note=(
            "the control for the rest of the bucket: a deep root (longest chain 11) whose three "
            "sentences are each a single code with no alternative, no concurrency and no consent "
            "clause. An answer that hedges here is hedging out of habit"
        ),
    ),
    Item(
        shape="chain",
        root="ANSC 306",
        question="what do I need before ANSC 306, and what does that course require?",
        gold_answer=(
            "ANSC 306 asks for ANSC 222 or equivalent and credit or concurrent enrollment in "
            "ANSC 224 or equivalent; ANSC 224 requires ANSC 100 and ANSC 221. Of the four "
            "courses named, only ANSC 224 and ANSC 100 are in the 2026 spring catalog."
        ),
        chain=("ANSC 306", "ANSC 224", "ANSC 100"),
        note=(
            "half of what these two sentences require is out of term: ANSC 222 and ANSC 221 are "
            "named and not offered, so the exported chain is a two-course path and the catalog "
            "text is a four-course requirement. The answer states the gap rather than reporting "
            "the path as complete -- the failure mode being tested is a system that does"
        ),
    ),
    # --- 2. is Y required for X, directly or indirectly -----------------------------------------
    Item(
        shape="reachable",
        root="SE 495",
        target="MATH 257",
        question="is MATH 257 required for SE 495, directly or indirectly?",
        gold_answer=(
            "Not strictly. SE 495 requires IE 310, and IE 310 asks for credit or concurrent "
            "registration in MATH 257 or MATH 415 -- so MATH 415 satisfies that clause instead."
        ),
        chain=("SE 495", "IE 310", "MATH 257"),
        note=(
            "the deepest root in the graph (longest chain 13, closure 27) asked a question that "
            "rests on three records. MATH 257 IS in the closure, so a system that answers "
            "reachability by dumping get_prereq_chain says yes; the sentence says one of two"
        ),
    ),
    Item(
        shape="reachable",
        root="CS 374",
        target="MATH 213",
        question="is MATH 213 required for CS 374, directly or indirectly?",
        gold_answer=(
            'No. CS 374\'s prerequisite sentence is "One of CS 173, MATH 213; CS 225.", so '
            "MATH 213 is one of two ways to satisfy the first clause and CS 173 is the other. "
            "CS 225 lists MATH 213 too, as one of seven alternatives."
        ),
        chain=("CS 374", "CS 225"),
        note=(
            "GUIDELINES F1 is the one-record version of this question and is a fact row; adding "
            "'indirectly' is what makes CS 225's own sentence load-bearing, because MATH 213 "
            "appears in it as well and a yes/no answer has to account for both mentions"
        ),
    ),
    Item(
        shape="reachable",
        root="MATH 412",
        target="MATH 231",
        question="is MATH 231 required for MATH 412, directly or indirectly?",
        gold_answer=(
            "Only on one route. MATH 412 accepts MATH 314 or MATH 347 or equivalent experience "
            "or CS 374; the MATH 314 route runs MATH 314, then MATH 241, then MATH 231, so "
            "MATH 231 is required if you come that way and not otherwise."
        ),
        chain=("MATH 412", "MATH 314", "MATH 241", "MATH 231"),
        note=(
            "GUIDELINES M2 deletes the unbounded version of this row -- MATH 412 sits in the "
            "CS 225 / CS 374 / MATH 412 cycle and 'everything before it' does not terminate. "
            "Asking whether one named course is required bounds it to four records and keeps the "
            "cycle in the row. MATH 347 is not offered this term, so the export shows two "
            "options where the sentence offers four"
        ),
    ),
    Item(
        shape="reachable",
        root="AE 434",
        target="MATH 285",
        question="is MATH 285 required for AE 434, directly or indirectly?",
        gold_answer=(
            "Yes. AE 434 requires AE 312, and AE 312 requires AE 202 and MATH 285, plus credit "
            "or concurrent registration in ME 200."
        ),
        chain=("AE 434", "AE 312", "MATH 285"),
        note=(
            "the yes case, and the row carrying the clearest parser bug in the graph: AE 312's "
            "sentence reads 'AE 202 and MATH 285. Credit or concurrent registration in ME 200.' "
            "and its exported prereqs are AE 202 and MATH 285. ME 200 IS offered in 2026 spring, "
            "so this is not the term filter -- a continuation sentence naming a requirement was "
            "dropped. Filed against the ingest parser (export_almabot); one of 46 such courses"
        ),
    ),
    Item(
        shape="reachable",
        root="ACCY 202",
        target="ECON 103",
        question="is ECON 103 required for ACCY 202, directly or indirectly?",
        gold_answer=(
            "Yes, though not necessarily beforehand. ACCY 202 requires ACCY 201 or equivalent, "
            "and ACCY 201 asks for either ECON 102 or ECON 103 completed and the remaining one "
            "completed or taken concurrently -- so ECON 103 is needed for ACCY 201 either before "
            "it or alongside it."
        ),
        chain=("ACCY 202", "ACCY 201"),
        note=(
            "the shallowest root in the bucket (longest chain 2), kept because the answer is the "
            "one the flattened list cannot give: ECON 102 and ECON 103 are both edges, and the "
            "sentence says one of the two may be concurrent"
        ),
    ),
    Item(
        shape="reachable",
        root="MSE 501",
        target="PHYS 427",
        question="is PHYS 427 required for MSE 501, directly or indirectly?",
        gold_answer=(
            "Only on one route. MSE 501 accepts MSE 500 or PHYS 560, and PHYS 560 requires "
            "PHYS 427 and PHYS 580 -- so PHYS 427 is required if you come through PHYS 560. "
            "MSE 500 is not in the 2026 spring catalog, so what the other route needs cannot be "
            "read here."
        ),
        chain=("MSE 501", "PHYS 560", "PHYS 427"),
        note=(
            "the disjunction whose other branch is out of term, which is what makes the exported "
            "graph read as a conjunction: MSE 500 and PHYS 580 are both named and neither is "
            "offered, so MSE 501 exports one prerequisite and PHYS 560 exports one. The answer "
            "says which claims the catalog can and cannot support"
        ),
    ),
    # --- 3. can I take X if I have only taken Y -------------------------------------------------
    Item(
        shape="eligibility",
        root="SE 312",
        held=("SE 310",),
        question="can I take SE 312 if I have only taken SE 310?",
        gold_answer=(
            "Yes, provided you register for SE 311 at the same time: SE 312's sentence is "
            '"SE 310; concurrent registration in SE 311." SE 311 asks for the mirror of it, '
            "SE 310 with concurrent registration in SE 312, so the two are taken together."
        ),
        chain=("SE 312", "SE 310", "SE 311"),
        note=(
            "one of the seven cycles, and the one the export cannot avoid: SE 311 and SE 312 are "
            "each other's prerequisite because the corequisite is stated inside the prerequisite "
            "sentence, where export_almabot.coreq_codes cannot see it. A walk without a visited "
            "set does not return on this row"
        ),
    ),
    Item(
        shape="eligibility",
        root="CS 374",
        held=("CS 173", "MATH 213"),
        question="can I take CS 374 if I have only taken CS 173 and MATH 213?",
        gold_answer=(
            'No. CS 374 reads "One of CS 173, MATH 213; CS 225." -- CS 173 already satisfies '
            "the first clause, but CS 225 is still required, and CS 225 itself asks for CS 126 "
            "or CS 128 or ECE 220 and one of seven discrete-mathematics courses."
        ),
        chain=("CS 374", "CS 225"),
        note=(
            "the row that punishes reading the code list as a conjunction in both directions at "
            "once: having two of the three listed codes is not two thirds of the way there, and "
            "the one that is missing is the one with a chain behind it"
        ),
    ),
    Item(
        shape="eligibility",
        root="MATH 412",
        held=("MATH 241",),
        question="I have taken MATH 241 -- can I take MATH 412?",
        gold_answer=(
            "Not yet. MATH 412 accepts MATH 314 or MATH 347 or equivalent experience or CS 374, "
            "and MATH 241 is none of those. MATH 241 is the prerequisite for MATH 314, so "
            "MATH 314 is the one course standing between them."
        ),
        chain=("MATH 412", "MATH 314"),
        note=(
            "MATH 241 is in MATH 412's closure, so a system that answers eligibility by testing "
            "membership in the closure answers yes. Membership is not eligibility -- MATH 241 is "
            "in there because something else requires it"
        ),
    ),
    Item(
        shape="eligibility",
        root="ACCY 302",
        held=("ACCY 201",),
        question="can I take ACCY 302 if I have only taken ACCY 201?",
        gold_answer=(
            "No. ACCY 302 asks for ACCY 202 or equivalent and for BADM 210 or concurrent "
            "enrollment or equivalent. ACCY 201 is the prerequisite for ACCY 202, not a "
            "substitute for it, and BADM 210 has no prerequisites of its own."
        ),
        chain=("ACCY 302", "ACCY 202", "BADM 210"),
        note=(
            "ACCY 301 and ACCY 302 are one of the seven cycles -- each sentence recommends "
            "concurrent enrollment in the other and the export made both an edge. The cycle is "
            "one level above this question and does not enter the answer, which is the useful "
            "case: the walk has to survive it, not report it"
        ),
    ),
    Item(
        shape="eligibility",
        root="ASTR 405",
        held=("PHYS 212", "ASTR 210"),
        question="can I take ASTR 405 if I have only taken PHYS 212 and ASTR 210?",
        gold_answer=(
            "No. ASTR 405 also requires ASTR 310, or equivalent programming experience, and "
            "ASTR 310 itself asks for PHYS 211, MATH 220, and credit or concurrent registration "
            "in ASTR 210."
        ),
        chain=("ASTR 405", "ASTR 310"),
        note=(
            "the disagreement the parser gets RIGHT, kept so the log is not read as a bug list: "
            "ASTR 405's sentence ends 'Recommended: PHYS 213 and PHYS 214' and neither is "
            "exported, which is correct -- a recommendation is not a prerequisite. The scanner "
            "cannot tell that from a dropped requirement, and this is the reviewer's line"
        ),
    ),
    # --- 4. the reverse edges -------------------------------------------------------------------
    Item(
        shape="unlocks",
        root="ASTR 310",
        question="if I take ASTR 310 this term, what does it open up?",
        gold_answer=(
            "ASTR 405, ASTR 406 and ASTR 414. Each of the three asks for PHYS 212, ASTR 210 and "
            "ASTR 310, or equivalent programming experience in place of ASTR 310."
        ),
        chain=("ASTR 310", "ASTR 405", "ASTR 406", "ASTR 414"),
        note=(
            "the same graph, walked the other way, and the shape where a system that only "
            "implements get_prereq_chain has nothing to answer with: the tool takes a course and "
            "returns what is behind it"
        ),
    ),
    Item(
        shape="unlocks",
        root="CEE 460",
        question="if I take CEE 460, which courses does that open up?",
        gold_answer=(
            "Three list it: CEE 462, which requires CEE 460 alone; CEE 465, which asks for "
            "credit in either CEE 460 or CEE 461 with concurrent registration in the other; and "
            "CEE 562, which requires CEE 460, CEE 461 and CEE 472 together."
        ),
        chain=("CEE 460", "CEE 462", "CEE 465", "CEE 562"),
        note=(
            '"unlocks" is three different relations here and only the first is unconditional. '
            "CEE 472 is not offered this term, so CEE 562's exported prerequisites are a pair "
            "and its sentence is a triple"
        ),
    ),
    Item(
        shape="unlocks",
        root="ADV 281",
        question="if I take ADV 281, what does it unlock?",
        gold_answer=(
            "ADV 497, which requires ADV 281 and consent of instructor, and ADV 396, whose "
            'sentence is "ADV 150, ADV 281, or written consent of instructor." -- so ADV 281 '
            "is one of three ways into ADV 396 rather than the only one."
        ),
        chain=("ADV 281", "ADV 396", "ADV 497"),
        note=(
            "a shallow root (longest chain 2) on the reverse shape, so depth is not confounded "
            "with direction. Both dependents also require consent, which no code list can carry"
        ),
    ),
    Item(
        shape="unlocks",
        root="AE 323",
        question="what does AE 323 open up if I take it?",
        gold_answer=(
            "AE 370, which asks for credit or concurrent registration in AE 311 or AE 312 and in "
            "AE 321 or AE 323, and AE 442, which asks for credit or concurrent registration in "
            "AE 311, AE 323 and AE 352. AE 323 is one of two options for AE 370 and required "
            "outright for AE 442."
        ),
        chain=("AE 323", "AE 370", "AE 442"),
        note=(
            "both dependents are 'credit or concurrent registration', so neither is unlocked in "
            "the sense the question implies -- they can be taken alongside. The answer says what "
            "the sentences say and leaves the word alone"
        ),
    ),
    # --- 5. how many courses stand between me and X ---------------------------------------------
    Item(
        shape="distance",
        root="ITAL 104",
        question="how many Italian courses stand between me and ITAL 104 if I have taken none?",
        gold_answer=(
            "Three stand before ITAL 104: ITAL 103, ITAL 102 and ITAL 101 -- unless high "
            "school Italian places you further along, since each of the three sentences accepts "
            "one, two or three years of it in place of the course below."
        ),
        chain=("ITAL 104", "ITAL 103", "ITAL 102", "ITAL 101"),
        note=(
            "the count is right and conditional, which is the whole shape: a system that reads "
            "the graph gets three, and a system that reads the sentences gets three with the "
            "condition that makes it useful"
        ),
    ),
    Item(
        shape="distance",
        root="JOUR 340",
        question="how many courses do I have to get through before JOUR 340?",
        gold_answer=(
            "Three stand before JOUR 340, in order: JOUR 200, then JOUR 210, then JOUR 215 "
            "-- though JOUR 215 accepts JOUR 210 or consent of the Journalism Department."
        ),
        chain=("JOUR 340", "JOUR 215", "JOUR 210", "JOUR 200"),
        note=(
            "JOUR 340's sentence names two more courses and neither is a prerequisite: "
            "concurrent enrollment with JOUR 317 is allowed, and it may NOT be taken "
            "concurrently with JOUR 217. The export drops both, correctly, and the log flags "
            "them -- a negation read as a mention is the third kind of disagreement in the log "
            "and the one most likely to become a wrong edge later. Both are out of term anyway"
        ),
    ),
    Item(
        shape="distance",
        root="DANC 460",
        question="how many courses stand between me and DANC 460?",
        gold_answer=(
            "Three courses and six semesters of them: DANC 460 asks for a passing grade in two "
            "semesters of DANC 360, DANC 360 for two semesters of DANC 260, and DANC 260 for two "
            "semesters of DANC 160 -- or graduate standing in Dance, which DANC 460 accepts "
            "instead."
        ),
        chain=("DANC 460", "DANC 360", "DANC 260", "DANC 160"),
        note=(
            "the row where the graph's answer and the catalog's answer are different numbers. "
            "Three edges, six semesters, and nothing in prereqs.json carries the multiplier -- "
            "it is in the sentence and only in the sentence"
        ),
    ),
    Item(
        shape="distance",
        root="MATH 500",
        question="I have taken MATH 417 -- how many more courses before MATH 500?",
        gold_answer=(
            "One: MATH 418. MATH 500 requires MATH 417 and MATH 418, and MATH 418's own "
            "prerequisite is MATH 417 or consent of instructor."
        ),
        chain=("MATH 500", "MATH 417", "MATH 418"),
        note=(
            "a deep root (longest chain 11) with the question bounded by what the student "
            "already has, which is how a distance question about a deep course stays inside four "
            "records. The prerequisites of MATH 500 are not independent -- one of them requires "
            "the other -- so the count is one and not two"
        ),
    ),
)


# --- the graph, read twice ------------------------------------------------------------------------


class Graph(NamedTuple):
    """The committed prereq graph and catalog, indexed for the questions this file asks.

    ``data/prereqs.json`` and ``data/courses.json``, both committed, and never
    ``data/chunks.jsonl``: a bucket whose evidence depended on a gitignored rebuildable artifact
    could not be checked on a fresh clone.
    """

    #: ``{course: its exported prerequisites}`` -- 1,135 records, the same index
    #: :func:`agent.tools.prereqs.load_graph` builds.
    edges: Mapping[str, tuple[str, ...]]
    #: ``{course: the courses that list it}``. The ``unlocks`` shape, and nothing else.
    reverse: Mapping[str, tuple[str, ...]]
    #: Longest-chain depth: #58's measure, the one that counts 738 courses at >= 2 and 13 as the
    #: deepest. Cycle-safe -- a path may not revisit a course.
    depth: Mapping[str, int]
    #: Shortest-distance depth, i.e. ``len(get_prereq_chain(c).levels)``: what the tool reports,
    #: which counts 718 and 9. Recorded beside the other so the gap is visible per row.
    levels: Mapping[str, int]
    #: How many courses are in a root's closure, which is what the question would have to enumerate
    #: if it were not bounded.
    closure: Mapping[str, int]
    #: The 19 courses in the graph's 7 cycles: ``ACCY 301/302``, ``CHEM 204/205``, ``MCB 581/582``,
    #: ``SE 311/312``, ``THEA 375-378``, ``THEA 475-478`` and ``CS 225 -> CS 374 -> MATH 412``.
    cycles: frozenset[str]
    #: The catalog record per code, for ``prereqRaw``, ``name`` and :func:`ingest.chunk.doc_id`.
    by_code: Mapping[str, Mapping[str, Any]]

    def has(self, code: str) -> bool:
        return code in self.by_code

    def raw(self, code: str) -> str:
        return self.by_code.get(code, {}).get("prereqRaw") or ""


def read_graph(prereqs_path: Path | None = None) -> Graph:
    """Read the two committed files and index them six ways."""
    edges = load_graph(prereqs_path)
    by_code = course_index().by_code
    reverse: dict[str, set[str]] = {}
    for code, prereqs in edges.items():
        for ref in prereqs:
            reverse.setdefault(ref, set()).add(code)
    levels: dict[str, int] = {}
    closure: dict[str, int] = {}
    for code in edges:
        walk = walk_chain(code, edges)
        levels[code] = len(walk.levels)
        closure[code] = sum(len(level) for level in walk.levels)
    return Graph(
        edges=edges,
        reverse={code: tuple(sorted(deps)) for code, deps in reverse.items()},
        depth=longest_depths(edges),
        levels=levels,
        closure=closure,
        cycles=cycle_members(edges),
        by_code=by_code,
    )


def longest_depths(edges: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Longest-chain depth for every course with prerequisites: #58's 738-and-13 measure.

    Iterative, and a path may not revisit a course, which is what makes it terminate on the seven
    cycles. Not memoized: the longest simple path out of a course depends on how you got there
    when a cycle is in reach, and a cache keyed on the course alone would be wrong for exactly
    the 165 chains that touch one. It costs about a second over the live graph, once.
    """
    depths: dict[str, int] = {}
    for root in edges:
        best = 0
        stack: list[tuple[str, int, set[str]]] = [(root, 0, {root})]
        while stack:
            code, so_far, on_path = stack.pop()
            best = max(best, so_far)
            for ref in edges.get(code, ()):
                if ref not in on_path:
                    stack.append((ref, so_far + 1, on_path | {ref}))
        depths[root] = best
    return depths


def cycle_members(edges: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """Every course in a strongly connected component of more than one course.

    Tarjan, iteratively -- the recursive form overflows on a 1,135-node graph under the default
    limit, and a script that needs ``setrecursionlimit`` to read its own data is a script that
    will fail on somebody else's machine. A component of one is not a cycle: no course in this
    graph is its own prerequisite.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    found: set[str] = set()
    counter = 0

    for start in edges:
        if start in index:
            continue
        work: list[tuple[str, int]] = [(start, 0)]
        while work:
            node, position = work[-1]
            if position == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            recursed = False
            successors = edges.get(node, ())
            for i in range(position, len(successors)):
                ref = successors[i]
                if ref not in index:
                    work[-1] = (node, i + 1)
                    work.append((ref, 0))
                    recursed = True
                    break
                if ref in on_stack:
                    low[node] = min(low[node], index[ref])
            if recursed:
                continue
            if low[node] == index[node]:
                component = []
                while True:
                    popped = stack.pop()
                    on_stack.discard(popped)
                    component.append(popped)
                    if popped == node:
                        break
                if len(component) > 1:
                    found.update(component)
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
    return frozenset(found)


# --- the tool proposes, the raw sentence disposes -------------------------------------------------


class Disagreement(NamedTuple):
    """One course whose exported prerequisites and own sentence do not say the same thing."""

    #: The course whose ``prereqRaw`` was read.
    course: str
    #: Codes the sentence names that the exported list drops, **and that the catalog offers**: a
    #: requirement the chain would not report. The parser-bug kind (AE 312 and ME 200).
    dropped_offered: tuple[str, ...]
    #: Codes the sentence names that the export drops because they are not in this term. Not a
    #: bug -- but it turns a disjunction into a bare requirement, which is #58's ABE 228 case.
    dropped_absent: tuple[str, ...]
    #: Codes in the exported list that the sentence does not name at all. None in the live graph;
    #: checked because an invented edge is worse than a dropped one.
    added: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "course": self.course,
            "dropped_offered": list(self.dropped_offered),
            "dropped_absent": list(self.dropped_absent),
            "added": list(self.added),
        }


def named_in_raw(code: str, graph: Graph) -> tuple[str, ...]:
    """Every course code the sentence names, in order, minus the course itself.

    :data:`ingest.tokenize.CODE_RE`, the same matcher the BM25 tokenizer and the export use, so a
    code this finds is a code the pipeline would have found.
    """
    seen: list[str] = []
    for match in CODE_RE.finditer(graph.raw(code)):
        found = f"{match.group('subject').upper()} {match.group('number')}"
        if found != code and found not in seen:
            seen.append(found)
    return tuple(seen)


def disagreement(code: str, graph: Graph) -> Disagreement | None:
    """Where ``code``'s exported prerequisites and its own sentence differ, or ``None``.

    Names only -- whether a dropped code was a requirement, a recommendation or a negation is a
    judgement, and it goes in the row's ``note``. See the module docstring for the four kinds
    this turns up and which of them are bugs.
    """
    named = named_in_raw(code, graph)
    exported = set(graph.edges.get(code, ()))
    dropped = [ref for ref in named if ref not in exported]
    return (
        Disagreement(
            course=code,
            dropped_offered=tuple(ref for ref in dropped if graph.has(ref)),
            dropped_absent=tuple(ref for ref in dropped if not graph.has(ref)),
            added=tuple(sorted(exported - set(named))),
        )
        if dropped or exported - set(named)
        else None
    )


def asserted(item: Item, graph: Graph) -> tuple[str, ...]:
    """The courses in the chain whose requirement sentence the answer characterizes.

    A course with at least one prerequisite *inside the chain*: the row says something about what
    that course requires, so its sentence is one the reviewer had to read. The others are there
    as the thing required. For ``unlocks`` the induced edges run the other way and this picks out
    the dependents, which is right for the same reason.

    A floor and not a proof. An answer may also quote the sentence of a chain course with no
    prerequisite inside the chain -- ``v1-1017`` quotes ASTR 310's -- and the checks below simply
    do not demand anything of those. What they are for is the case that would otherwise pass
    silently: a row asserting a conjunction the sentence never stated.
    """
    inside = set(item.chain)
    return tuple(code for code in item.chain if inside & set(graph.edges.get(code, ())) - {code})


def disagreements(item: Item, graph: Graph) -> tuple[Disagreement, ...]:
    """Every disagreement among the sentences this row's answer characterizes."""
    found = (disagreement(code, graph) for code in asserted(item, graph))
    return tuple(one for one in found if one is not None)


# --- checking the table ---------------------------------------------------------------------------


def codes_in(text: str) -> tuple[str, ...]:
    """Every code :data:`ingest.tokenize.CODE_RE` finds in a string, normalized, in order."""
    return tuple(
        f"{m.group('subject').upper()} {m.group('number')}" for m in CODE_RE.finditer(text)
    )


def check_chain(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """The chain is the root, two to four real courses, connected the way the shape walks.

    Connectivity is the check that makes a ``multihop`` row multi-hop: a second id that the graph
    does not join to the first is a second document, not a second hop.
    """
    rule = shape(item.shape)
    if not item.chain or item.chain[0] != item.root:
        yield violation(
            n, item, "chain", f"does not start with {item.root}; the root is the row's first id"
        )
        return
    if not MIN_CHAIN <= len(item.chain) <= MAX_CHAIN:
        yield violation(
            n,
            item,
            "chain",
            f"joins {plural(len(item.chain), 'record')}; a multihop row joins {MIN_CHAIN} to "
            f"{MAX_CHAIN} -- one record is a fact row, and more than four is a list question "
            "GUIDELINES says to narrow or drop",
        )
    if len(set(item.chain)) != len(item.chain):
        yield violation(n, item, "chain", "names a course twice; one document is one hop")
    for code in item.chain:
        if not graph.has(code):
            yield violation(
                n, item, "chain", f"{code} is not in the 2026-sp export, so it has no chunk to cite"
            )
            return

    for code in item.chain[1:]:
        if rule.reverse:
            if item.root not in graph.edges.get(code, ()):
                yield violation(
                    n,
                    item,
                    "chain",
                    f"{code} does not list {item.root} as a prerequisite, so {item.root} unlocks "
                    "nothing for it",
                )
        elif code not in reachable_from(item.root, graph):
            yield violation(
                n,
                item,
                "chain",
                f"{code} is not in {item.root}'s prerequisite closure, so joining the two is not "
                "a hop the graph supports",
            )


def reachable_from(root: str, graph: Graph) -> frozenset[str]:
    """Every course in ``root``'s closure, cycle-safe.

    Not ``root`` itself, unless a cycle brings the walk back around to it.
    """
    walk = walk_chain(root, graph.edges)
    return frozenset(code for level in walk.levels for code in level)


def check_depth(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """The root clears :data:`MIN_DEPTH` on both measures -- see the module docstring."""
    if item.root not in graph.edges:
        yield violation(
            n,
            item,
            "root",
            f"{item.root} has no record in data/prereqs.json, so it has no chain at all",
        )
        return
    depth, levels = graph.depth[item.root], graph.levels[item.root]
    if depth < MIN_DEPTH or levels < MIN_DEPTH:
        yield violation(
            n,
            item,
            "root",
            f"{item.root} is depth {depth} by longest chain and {levels} by the tool's levels; a "
            f"multihop root clears {MIN_DEPTH} on both, or the row is one hop wearing this bucket",
        )


def check_target(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """``target`` is present exactly for ``reachable``, and is a course.

    Deliberately **not** checked for membership in the root's closure: half these rows answer
    "no", and requiring the target to be reachable would delete them. Whether it is reachable is
    recorded in provenance instead, because the gap between "in the closure" and "required" is
    what several of these rows test.
    """
    if not shape(item.shape).wants_target:
        if item.target:
            yield violation(
                n, item, "target", f"is {item.target!r}; only a reachable row names one"
            )
        return
    if not item.target:
        yield violation(
            n, item, "target", "is empty; a reachable row names the course it asks about"
        )
    elif not graph.has(item.target):
        yield violation(n, item, "target", f"{item.target} is not in the 2026-sp export")


def check_held(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """``held`` is present exactly for ``eligibility``, and every course in it is real."""
    if not shape(item.shape).wants_held:
        if item.held:
            yield violation(
                n,
                item,
                "held",
                f"is {', '.join(item.held)}; only an eligibility row says what the student has",
            )
        return
    if not item.held:
        yield violation(
            n,
            item,
            "held",
            "is empty; an eligibility row records which courses the question grants, because "
            "that is half of what makes the answer no",
        )
    for code in item.held:
        if not graph.has(code):
            yield violation(n, item, "held", f"{code} is not in the 2026-sp export")


def check_question(n: int, item: Item) -> Iterator[Violation]:
    """The question names the root, and names no course the row has not declared.

    A row about ``CS 374`` whose question asks about ``CS 473`` passes every graph check in this
    file and tests nothing anyone meant.
    """
    declared = {item.root, *item.chain, *item.held} | ({item.target} if item.target else set())
    found = codes_in(item.question)
    if item.root not in found:
        yield violation(n, item, "question", f"does not name {item.root}, which the row is about")
    for code in found:
        if code not in declared:
            yield violation(
                n,
                item,
                "question",
                f"names {code}, which is neither the root nor in chain, target or held",
            )
    if item.target and item.target not in found:
        yield violation(n, item, "question", f"does not name its target {item.target}")
    for code in item.held:
        if code not in found:
            yield violation(n, item, "question", f"does not name {code}, which it grants in held")


def check_answer(n: int, item: Item) -> Iterator[Violation]:
    """The answer is written, and it names every course it cites.

    `eval/GUIDELINES.md` rule 4: a gold id supports the claim, not the topic. An id whose course
    the answer never mentions is an id that rewards retrieving a document the answer did not use.
    Rule 1 is the other half -- *"the data structures course is 4 hours"* is not scoreable -- and
    both come to the same check here.
    """
    if not item.gold_answer.strip():
        yield violation(n, item, "gold_answer", "is empty")
        return
    for code in item.chain:
        if code not in item.gold_answer:
            yield violation(
                n,
                item,
                "gold_answer",
                f"does not name {code}, which the row cites as a gold id; either the answer uses "
                "that record or the id does not belong in the chain",
            )


def check_disjunction(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """A row whose sentences offer a choice must say so.

    `eval/GUIDELINES.md` rule 3: 744 of the 1,135 requirement statements say "or" or "one of",
    and turning *"One of CS 173, MATH 213; CS 225."* into a conjunction is wrong rather than
    loose. This does not check that the answer got the disjunction *right* -- that is the
    reviewer's job, and #59's -- only that a row built on a sentence offering alternatives does
    not read as a list of requirements.
    """
    sentences = [graph.raw(code) for code in asserted(item, graph)]
    if not any(marker in one.casefold() for one in sentences for marker in DISJUNCTIONS):
        return
    if not any(marker in item.gold_answer.casefold() for marker in DISJUNCTIONS):
        yield violation(
            n,
            item,
            "gold_answer",
            "states no alternative, and a requirement sentence this row is built on offers one; "
            "GUIDELINES rule 3 keeps a disjunction a disjunction",
        )


def check_note(n: int, item: Item, graph: Graph) -> Iterator[Violation]:
    """A row whose exported chain disagrees with the catalog text carries the verdict on it."""
    found = disagreements(item, graph)
    if found and not item.note.strip():
        which = ", ".join(one.course for one in found)
        yield violation(
            n,
            item,
            "note",
            f"is empty, and the exported prerequisites of {which} do not match the sentence; #58 "
            "asks for the disagreement to be judged and filed, not resolved in the tool's favour",
        )


def violation(n: int, item: Item, field: str, message: str) -> Violation:
    """A finding about the table, addressed the way the validator addresses one about a file."""
    return Violation(n, row_id(n), f"{item.shape}.{field}", message)


def row_id(n: int) -> str:
    """The id the ``n``-th row of the table gets: ``v1-1001`` upwards.

    #56's :data:`~eval.scripts.draft_items.ID_BLOCKS` reserves the block, so the buckets can be
    built independently and still never collide when #59 merges them.
    """
    return item_id(ID_BLOCKS[BUCKET] + n - 1)


def check_table(items: Sequence[Item], graph: Graph) -> list[Violation]:
    """Every check over the table, in table order, then the four the whole bucket must satisfy.

    Every violation, never the first -- the same rule :mod:`eval.validate` follows, for the same
    reason: a table with eight problems should be one run with eight messages.
    """
    found: list[Violation] = []
    counts = {name: 0 for name in SHAPES}
    depths: list[int] = []
    cyclic = 0
    for n, item in enumerate(items, start=1):
        if item.shape not in SHAPES:
            found.append(
                Violation(
                    n, row_id(n), "shape", f"is {item.shape!r}; the five are {', '.join(SHAPES)}"
                )
            )
            continue
        counts[item.shape] += 1
        depth_checks = list(check_depth(n, item, graph))
        found.extend(depth_checks)
        if not depth_checks:
            depths.append(graph.depth[item.root])
        chain_checks = list(check_chain(n, item, graph))
        found.extend(chain_checks)
        if not chain_checks and set(item.chain) & graph.cycles:
            cyclic += 1
        found.extend(check_target(n, item, graph))
        found.extend(check_held(n, item, graph))
        found.extend(check_question(n, item))
        found.extend(check_answer(n, item))
        found.extend(check_disjunction(n, item, graph))
        found.extend(check_note(n, item, graph))

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
    found.extend(check_distribution(depths, cyclic))
    return found


def check_distribution(depths: Sequence[int], cyclic: int) -> Iterator[Violation]:
    """#58's two acceptance criteria about the bucket as a whole, rather than about a row.

    Both are about what a bucket that passed every per-row check could still fail to be: 25
    depth-2 rows that avoid the cycles are all individually valid and measure the easy half of
    the graph.
    """
    deep = sum(1 for d in depths if d >= DEEP_AT)
    if deep < MIN_DEEP:
        yield Violation(
            None,
            None,
            "depth",
            f"{plural(deep, 'row')} at depth {DEEP_AT} or more, fewer than {MIN_DEEP}; a walk "
            "that half-works still looks fine on depth 2",
        )
    shallow = sum(1 for d in depths if d == MIN_DEPTH)
    if shallow > MAX_SHALLOW:
        yield Violation(
            None,
            None,
            "depth",
            f"{plural(shallow, 'row')} at depth {MIN_DEPTH}, more than {MAX_SHALLOW}; the bucket "
            "is stratified by depth, not sampled from the shallow end",
        )
    if cyclic < MIN_CYCLIC:
        yield Violation(
            None,
            None,
            "cycles",
            f"{plural(cyclic, 'row')} names a course inside one of the 7 cycles, fewer than "
            f"{MIN_CYCLIC}; get_prereq_chain is cycle-safe and nothing here would catch it "
            "regressing",
        )


# --- building the rows ----------------------------------------------------------------------------

#: Every key a graph-built row's ``provenance`` carries. Not #56's set and not #57's: a drafted
#: row records a batch, a model and a leakage overlap, a constructed one records a failure shape
#: and a gate reading, and this one records where in the graph the row sits and whether the tool
#: and the catalog text agreed about it. ``parse_drafts`` takes the key set to hold a file to, so
#: three shapes of provenance go through one reader; #59 sees all three, which is honest.
PROVENANCE_KEYS = (
    "shape",
    "source",
    "root",
    "target",
    "held",
    "depth",
    "levels",
    "closure",
    "cycle_members",
    "target_in_closure",
    "raw_agrees",
    "disagreements",
    "note",
    "config_hash",
)

#: What ``source`` says. One value today; #59 may promote a row and record that it rewrote it.
SOURCE = "prereq-graph"


def build_rows(items: Sequence[Item], graph: Graph, cfg: Config) -> list[dict[str, Any]]:
    """The table as draft rows: six dataset fields, then provenance, in table order."""
    rows: list[dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        found = disagreements(item, graph)
        closure = reachable_from(item.root, graph) if item.root in graph.edges else frozenset()
        row = EvalItem(
            id=row_id(n),
            bucket=BUCKET,
            question=item.question,
            gold_answer=item.gold_answer,
            gold_doc_ids=[doc_id(graph.by_code[code]) for code in item.chain],
            should_abstain=RULES[BUCKET].should_abstain,
        )
        rows.append(
            {
                **row_json(row),
                "provenance": {
                    "shape": item.shape,
                    "source": SOURCE,
                    "root": item.root,
                    "target": item.target or None,
                    "held": list(item.held),
                    "depth": graph.depth.get(item.root),
                    "levels": graph.levels.get(item.root),
                    "closure": graph.closure.get(item.root),
                    "cycle_members": sorted(set(item.chain) & graph.cycles),
                    # The tool's proposal, recorded and not acted on: a `reachable` row whose
                    # target IS in the closure and whose answer is "no" is the row that separates
                    # membership from requirement, and it should be visible as that in #59.
                    "target_in_closure": (item.target in closure) if item.target else None,
                    "raw_agrees": not found,
                    "disagreements": [one.as_json() for one in found],
                    "note": item.note or None,
                    "config_hash": cfg.hash,
                },
            }
        )
    return rows


def sidecar(rows: Sequence[Mapping[str, Any]], graph: Graph, cfg: Config) -> dict[str, Any]:
    """What the run records beside the rows: the two distributions and the agreement log.

    Deliberately not a copy of the rows. It answers the three questions #62 asks of this bucket --
    how the 25 split across the shapes, how they are spread over depth, and which of them sit on
    a sentence the export does not reproduce -- without anybody aggregating a JSONL file.
    """
    provenance = [dict(row["provenance"]) for row in rows]
    depths = sorted(p["depth"] for p in provenance if p["depth"] is not None)
    return {
        "bucket": BUCKET,
        "source": SOURCE,
        "config_hash": cfg.hash,
        "rows": len(rows),
        "shapes": {name: sum(1 for p in provenance if p["shape"] == name) for name in SHAPES},
        "depth": {
            "measure": "longest chain, cycle-safe",
            "histogram": {str(d): depths.count(d) for d in sorted(set(depths))},
            "min": depths[0] if depths else None,
            "max": depths[-1] if depths else None,
            "at_least_5": sum(1 for d in depths if d >= DEEP_AT),
        },
        "graph": {
            "courses_with_prerequisites": len(graph.edges),
            "depth_2_or_more_by_longest_chain": sum(1 for d in graph.depth.values() if d >= 2),
            "depth_2_or_more_by_levels": sum(1 for d in graph.levels.values() if d >= 2),
            "deepest_longest_chain": max(graph.depth.values(), default=0),
            "deepest_by_levels": max(graph.levels.values(), default=0),
            "courses_in_cycles": len(graph.cycles),
        },
        "cyclic_rows": [
            row["id"] for row, p in zip(rows, provenance, strict=True) if p["cycle_members"]
        ],
        "disagreements": {
            row["id"]: p["disagreements"]
            for row, p in zip(rows, provenance, strict=True)
            if p["disagreements"]
        },
    }


# --- the run ------------------------------------------------------------------------------------


class Report(NamedTuple):
    """One run: where it wrote, what it wrote, and everything wrong with either."""

    path: Path
    meta: Path
    rows: list[dict[str, Any]]
    table: list[Violation]
    written: list[Violation]
    graph: Graph


def build(
    cfg: Config,
    drafts_dir: Path | None = None,
    prereqs_path: Path | None = None,
    items: Sequence[Item] | None = None,
) -> Report:
    """Check the table, write it, and re-read what was written.

    Nothing is written when a table check fails. A drafting run keeps the rows it paid a model
    for; there is nothing to salvage here -- the table is in the repository either way -- and half
    a bucket on disk under a name #59 reads is worse than none.

    ``items`` defaults to :data:`ITEMS` and is resolved on the call rather than bound to the
    signature, so a caller that replaces the table -- a test, or #59 promoting a rewritten row --
    gets the table it set and not the one this module held at import.
    """
    items = ITEMS if items is None else items
    graph = read_graph(prereqs_path)
    table = check_table(items, graph)
    path = draft_path(STEM, cfg, drafts_dir)
    if any(v.level == "error" for v in table):
        return Report(path, meta_path(path), [], table, [], graph)

    rows = build_rows(items, graph, cfg)
    write_rows(path, rows)
    write_json(meta_path(path), sidecar(rows, graph, cfg))
    written = check_drafts(path, corpus_doc_ids(), keys=PROVENANCE_KEYS)
    return Report(path, meta_path(path), rows, table, written, graph)


# --- saying so ------------------------------------------------------------------------------------


def depth_lines(report: Report) -> list[str]:
    """The distribution #58's acceptance asks to see recorded, and the census behind it."""
    graph = report.graph
    depths = [p["depth"] for p in (row["provenance"] for row in report.rows)]
    if not depths:
        return []
    histogram = "  ".join(f"{d}:{depths.count(d)}" for d in sorted(set(depths)))
    by_longest = sum(1 for d in graph.depth.values() if d >= 2)
    by_levels = sum(1 for d in graph.levels.values() if d >= 2)
    return [
        (
            f"graph                {len(graph.edges)} courses with prerequisites, "
            f"{by_longest} at depth >= 2 by longest chain ({by_levels} by the tool's levels), "
            f"{len(graph.cycles)} in cycles"
        ),
        (
            f"depth                {histogram}   "
            f"(longest chain; {sum(1 for d in depths if d >= DEEP_AT)} at {DEEP_AT}+)"
        ),
    ]


def agreement_lines(report: Report) -> list[str]:
    """The log #58 asks for: every row where the exported chain and the catalog text differ.

    Not a warning. A disagreement is a bug report filed against #30 or the ingest parser, and
    some of them are the parser being right -- the row's note is where that is settled.
    """
    rows = [
        (row["id"], dict(row["provenance"]))
        for row in report.rows
        if row["provenance"]["disagreements"]
    ]
    lines = [
        f"tool vs raw          {len(report.rows) - len(rows)} of {len(report.rows)} rows agree"
    ]
    if not rows:
        return lines
    lines += [
        "",
        (
            "Where the exported prerequisites and the course's own sentence differ -- the gold "
            "answers below were written from the sentence:"
        ),
    ]
    for rid, p in rows:
        for one in p["disagreements"]:
            parts = []
            if one["dropped_offered"]:
                parts.append(f"drops {', '.join(one['dropped_offered'])} (offered this term)")
            if one["dropped_absent"]:
                parts.append(f"drops {', '.join(one['dropped_absent'])} (not in this term)")
            if one["added"]:
                parts.append(f"adds {', '.join(one['added'])} (not in the sentence)")
            lines.append(f"  {rid}  {one['course']:<10} {'; '.join(parts)}")
    return lines


def summary(cfg: Config, report: Report) -> str:
    """What the terminal gets: the shapes, the depths, the agreement log, and where it went."""
    if any(v.level == "error" for v in report.table):
        return (
            f"config hash {cfg.hash}  bucket {BUCKET}\n"
            f"the table does not check out, so nothing was written to "
            f"{rel_to_repo(report.path)}"
        )
    counts = {name: 0 for name in SHAPES}
    cyclic = 0
    for row in report.rows:
        counts[row["provenance"]["shape"]] += 1
        cyclic += bool(row["provenance"]["cycle_members"])
    ids = [row["id"] for row in report.rows]
    lines = [
        f"config hash {cfg.hash}  bucket {BUCKET}",
        f"rows                 {len(report.rows)} ({ids[0]} .. {ids[-1]})" if ids else "rows 0",
    ]
    for name, rule in SHAPES.items():
        lines.append(f"  {name:<18} {counts[name]}  {rule.gloss}")
    lines.extend(depth_lines(report))
    lines.append(f"cycles               {cyclic} rows name a course inside one of the 7")
    lines.extend(agreement_lines(report))
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
        prog="python -m eval.scripts.build_multihop",
        description="Check and write the 25 multi-hop rows, built from the prereq graph, into "
        "eval/datasets/drafts/.",
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument(
        "--prereqs", default=None, help="the exported graph (default: data/prereqs.json)"
    )
    parser.add_argument(
        "--drafts-dir", default=None, help="where the rows go (default: eval/datasets/drafts)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    report = build(
        cfg,
        drafts_dir=Path(args.drafts_dir) if args.drafts_dir else None,
        prereqs_path=Path(args.prereqs) if args.prereqs else None,
    )
    print(summary(cfg, report))

    status = 0
    if report.table:
        print(
            "\nThe table in eval/scripts/build_multihop.py does not check out against the "
            "graph:\n" + render(report.table, "ITEMS"),
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
