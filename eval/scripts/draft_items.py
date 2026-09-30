"""Ten real course records at a time: the machine half of Day 3's eval set.

    uv run python -m eval.scripts.draft_items --bucket fact --batches 5 --fake-model
    uv run --env-file .env python -m eval.scripts.draft_items --bucket fact --batches 20

This script produces **drafts, never dataset rows**. Everything it writes lands in
``eval/datasets/drafts/`` and only the verification pass (#59) promotes anything into
``eval/datasets/v1.jsonl``. What it does is feed a model ten real chunks out of
``data/chunks.jsonl`` and ask for questions in one bucket, with an answer and the ids that
support it; what it owes the rest of Day 3 is that every row it keeps is re-generatable, checked
against :mod:`eval.schema` before a person ever reads it, and carries enough provenance for Day 4
to slice on.

**Sampling is seeded, stratified and a prefix.** The order is a deterministic function of the
corpus and ``seed`` alone -- :func:`stratified_order` shuffles inside each subject with
``random.Random(cfg["seed"])`` (a private stream, so a second script in the same process cannot
shift it) and then deals one course per subject, round-robin. ``--batches N`` takes the first
``N`` batches of that one order, so a run of 3 and a later run of 5 agree on batches 1-3 and the
resume never re-rolls anything. Round-robin means ten *different* subjects per batch and, over a
whole run, roughly the same number of questions from a 3-course subject as from MUSC's 123 -- the
point of stratifying at all, since proportional sampling over 187 subjects would spend a fifth of
Day 3 on the five largest departments.

**The gold ids are constrained to the batch, and checked anyway.** The prompt hands the model the
ten ``doc_id``s and says the supporting ids must come from that list; it will still occasionally
invent one. Every drafted row is checked here -- against the ten, then against
:class:`eval.schema.EvalItem` and :func:`eval.validate.check_row` -- and a row that fails is
**dropped with a counted reason, never repaired**. A model that cited an id from a list it was
looking at got the question wrong too. The counts are printed and stored in the meta sidecar, so
"the model invented 4 ids in 20 batches" is a number #62 can report rather than an impression.

**The real hazard is that a question drafted from a chunk is trivially retrievable.** A question
that reuses its own chunk's rare vocabulary is found by BM25 at rank 1, and an eval set full of
those reports retrieval recall the system does not have -- the single biggest threat to Day 4's
numbers. Three things here, in increasing order of how much they are worth: the prompt forbids
copying distinctive phrases and asks for the question a student would actually type; every
``fact``/``multihop`` row records the Jaccard overlap between its question and its gold chunk,
over ``ingest.tokenize``'s own tokenizer, so the claim is measured and not assumed; and the top
decile of that distribution is marked ``flagged`` for rewriting during verification (#59). The
absolute numbers are small -- a 6-token question against a 110-token chunk cannot have a large
Jaccard -- which is why the *distribution* and its top decile are what is acted on, not a
threshold anyone could quote.

**The drafter and the agent under test are the same model family, and that cannot be fixed
here.** ``gpt-4o-mini`` drafting questions for ``gpt-4o-mini`` to answer shares blind spots with
it, and no amount of human verification detects that a question is one this family finds easy --
a person checks that an answer is *right*, not that it was unfairly reachable. So ``--model``
drafts with a different model where one is available, every row records the model that wrote it,
and Day 4 can slice on it. Where nothing else is available this stays a limitation of the set and
belongs in #62's report.

**Buckets.** ``fact``, ``multihop`` (candidates only -- the real multi-hop rows come off the
prereq graph in #58) and ``ambiguous``. ``unanswerable`` is *not* drafted from documents at all
and is constructed instead (#57): a model shown ten course records will not reliably invent
questions those records cannot answer, and the four failure shapes that bucket wants are a design,
not a sample.

**Nothing here carries a clock.** Like #39's report, a draft file moves only when the run moved:
the rows, the meta sidecar and the summary hold ids, counts and hashes, and a re-run under the
same seed and the same model rewrites the same rows. What a run *was* is in the file name (the
config hash) and in every row's provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from config import REPO_ROOT, Config, config_hash, load_config, rel_to_repo
from eval.schema import RULES, Bucket, EvalItem, item_id, normalize_question, row_json
from eval.validate import Row, Violation, check_across_rows, check_row, pydantic_violations
from ingest.chunk import CHUNKS_PATH, chunk_key, load_chunks
from ingest.tokenize import get_tokenizer
from llm import Chat, OpenAIChat, check_llm_config

#: Where drafts live. Not ``eval/datasets/`` itself: that directory holds artifacts a later day
#: reads (``day2_queries.jsonl``, and #62's ``v1.jsonl``), and a draft is neither.
DRAFTS_DIR = REPO_ROOT / "eval" / "datasets" / "drafts"

#: Records per model call. The issue's number, and the one the prompt's wording assumes.
BATCH_SIZE = 10

#: Questions asked of one batch unless ``--per-batch`` says otherwise. Three rather than ten: a
#: model told to produce one question per record produces ten questions about credit hours, and
#: the duplicate check then throws most of them away after paying for them.
DEFAULT_PER_BATCH = 3

#: The three buckets drafted from documents. ``unanswerable`` is #57's, constructed and not drawn.
#: ``multihop`` stays here as a *candidate* generator only -- the bucket itself is built from the
#: prereq graph by :mod:`eval.scripts.build_multihop`, which writes ``multihop_graph_{hash}.jsonl``
#: so that the two never share a file.
DRAFTABLE: tuple[Bucket, ...] = ("fact", "multihop", "ambiguous")

#: Where each bucket's ids start, so three draft files can be written independently and still
#: never collide when #59 merges them. Renumbering at promotion is allowed -- what
#: :mod:`eval.schema` calls a corruption is renumbering *after* #60's sample and #61's split name
#: rows by id, and both of those run on the promoted file. ``unanswerable`` has a block it does
#: not use here so that #57 can mint into the same space.
ID_BLOCKS: Mapping[Bucket, int] = {
    "fact": 1,
    "multihop": 1001,
    "unanswerable": 2001,
    "ambiguous": 3001,
}

#: The quantile above which a question is marked as reusing too much of its chunk's wording.
FLAG_QUANTILE = 0.9

#: Bumped whenever any of the prompt tables below changes in a way a model could notice. Pinned by
#: a digest assertion in ``tests/test_draft_items.py`` per bucket, so editing the prompt without
#: bumping the version fails a test rather than quietly producing a second "v1" -- and provenance
#: records it per row, since a set drafted under two prompts is two sets.
DRAFT_PROMPT_VERSION = "v1"


# --- the prompt -----------------------------------------------------------------------------------

#: One record as the model sees it. The ``doc_id`` sits on the *heading* line rather than on one
#: of its own, which is the one deliberate difference from :data:`agent.prompts.EVIDENCE_TEMPLATE`:
#: :class:`llm.StandInChat` matches a ``doc_id:`` line at the start of a line, so with that shape
#: the answer node's stand-in would happily "answer" a drafting prompt with the first line of
#: record one. It exists so a Day 2 report can be produced with no credits, and a drafting run
#: that quietly used it would write canned questions into a file the whole of Day 3 trusts.
#: ``tests/test_draft_items.py`` asserts that it refuses.
DOCUMENT_TEMPLATE = "Document {n} of {total} (doc_id: {doc_id})\n---\n{text}"

#: Who the model is and what it is writing. No persona: the completion is a JSON array.
DRAFT_HEADER = (
    "You write evaluation questions for a university course-planning assistant. Ten catalog "
    "records follow in the message. Everything you write is checked against those ten records "
    "and nothing else."
)

#: What each bucket asks for, in the words the model reads. Derived from `eval/GUIDELINES.md`
#: (#55) -- the spec a person will judge these rows against, so the drafting instruction and the
#: verification instruction say the same thing about where the bucket boundaries run.
BUCKET_TASKS: Mapping[Bucket, str] = {
    "fact": (
        "Write questions that one of the ten records answers on its own. The record you cite must "
        "carry every fact your answer states -- a record that is merely about the right course is "
        "the wrong record. Count records, not reasoning steps: a prerequisite sentence naming "
        "three courses is still one record."
    ),
    "multihop": (
        "Write questions whose answer needs two or more of the ten records joined, where the "
        "second record is load-bearing: remove it and the answer changes or disappears. Ten "
        "records drawn from ten different subjects usually support none at all, and an empty "
        "array is the right answer far more often than a question that only looks like a join."
    ),
    "ambiguous": (
        "Write questions that do not say enough to be answered, where exactly one clarifying "
        "question would pick out which record is meant. Put that clarifying question in `answer` "
        "-- it is what the assistant should have asked -- and list in `doc_ids` the two or more "
        "records the question could be about. A question with one plausible referent is not "
        "ambiguous; it is a fact question, and does not belong in this batch."
    ),
}

#: The instructions, in the order the model reads them. Rules 1 and 2 are the whole defence
#: against the leakage this file measures; 3 keeps the draft inside the knowledge base
#: `eval/GUIDELINES.md` defines (a question about instructors or seats is #57's bucket, not a
#: badly-written row of this one); 4 is the constraint the harness enforces afterwards anyway; 5
#: and 6 are the spec's two correctness rules, which a draft that ignores them cannot be verified
#: into satisfying.
DRAFT_RULES: tuple[str, ...] = (
    (
        "Write the question a student would type: short, lower case, and usually without the "
        "course code."
    ),
    (
        "Do not reuse a record's distinctive wording. A question assembled out of the rare "
        "phrases in its own record is found by keyword matching alone, and a set of those "
        "measures a retrieval quality the system does not have."
    ),
    (
        "Ask only about what a record prints: its title, credit hours, term, description, "
        "prerequisites and degree attributes. Instructors, meeting times, seats, enrolment, "
        "difficulty, textbooks and any other term are not in this catalog at all -- they are not "
        "hard questions, they are a different bucket, and writing one here wastes the row."
    ),
    (
        "Copy every id in `doc_ids` from a `doc_id:` heading above, exactly as it is written. A "
        "row citing anything else is discarded whole."
    ),
    (
        "Name courses by code in the answer -- `CS 225 is 4 credit hours`, never `the data "
        "structures course` -- because an answer whose subject is ambiguous cannot be scored."
    ),
    (
        "Quote the catalog's own sentence for a prerequisite or a requirement rather than "
        'paraphrasing it: "one of" and "concurrent registration" decide what a student actually '
        "has to take, and a flattened list of codes states a requirement the catalog does not."
    ),
    (
        "Write at most the number of questions the message asks for, and write fewer -- or none "
        "at all, as an empty array -- rather than one the records do not support."
    ),
    (
        "Reply with a JSON array and nothing else: no prose before it, no explanation after it, "
        "no code fence around it."
    ),
)

#: The three keys of one drafted question, glossed for the model. ``id``, ``bucket`` and
#: ``should_abstain`` are deliberately not among them: the bucket is what was asked for,
#: ``should_abstain`` follows from it (:data:`eval.schema.RULES`), and the id is minted by
#: :func:`eval.schema.item_id` in draft order. A model asked for a field it cannot know is a model
#: inventing one.
DRAFT_FIELDS: tuple[tuple[str, str], ...] = (
    ("question", "what a student would type, in one sentence"),
    (
        "answer",
        (
            "the correct answer, short and checkable -- for an ambiguous question, the "
            "clarifying question to ask instead"
        ),
    ),
    ("doc_ids", "the ids of the records the answer comes from, copied from the headings above"),
)

#: The ids the worked examples cite. Neither ``SUBJ`` nor ``OTHR`` is one of this catalog's 187
#: subjects, so an example id echoed into a real draft is dropped by the batch check instead of
#: landing in the set as a citation of a course nobody asked about. A test asserts that both are
#: absent from the corpus.
EXAMPLE_DOC_ID = "2026-sp:SUBJ-100"

#: The example records, in the shape :mod:`ingest.chunk` renders a real one. Three, because a
#: bucket whose rows cite two records needs an example batch that holds two: an example reply
#: citing an id the example batch does not contain would break rule 4 in the act of teaching it.
EXAMPLE_RECORDS: Mapping[str, tuple[str, str]] = {
    EXAMPLE_DOC_ID: (
        "SUBJ 100 — Example Course\n"
        "Credits: 3\n"
        "Term: 2026-sp\n"
        "Description: An example record, in the shape the real ones arrive in.\n"
        "Prerequisites: None listed"
    ),
    "2026-sp:SUBJ-200": (
        "SUBJ 200 — Second Example Course\n"
        "Credits: 3\n"
        "Term: 2026-sp\n"
        "Description: A second example record, so that a join has something to join to.\n"
        "Prerequisites: One of SUBJ 100, SUBJ 150."
    ),
    "2026-sp:OTHR-100": (
        "OTHR 100 — Example Course In Another Subject\n"
        "Credits: 4\n"
        "Term: 2026-sp\n"
        "Description: A record that shares a catalog number with SUBJ 100 and nothing else.\n"
        "Prerequisites: None listed"
    ),
}

#: Which of those records each bucket's example batch shows.
EXAMPLE_BATCHES: Mapping[Bucket, tuple[str, ...]] = {
    "fact": (EXAMPLE_DOC_ID,),
    "multihop": (EXAMPLE_DOC_ID, "2026-sp:SUBJ-200"),
    "ambiguous": (EXAMPLE_DOC_ID, "2026-sp:OTHR-100"),
}

#: One worked reply per bucket, over that bucket's example batch. Each demonstrates the rule its
#: bucket is most likely to break: a question that does not quote its own record, a join whose
#: second record is load-bearing, and an ``answer`` that is a question.
EXAMPLE_REPLIES: Mapping[Bucket, str] = {
    "fact": (
        '[{"question": "how many hours is subj 100?", "answer": "SUBJ 100 is 3 credit hours.", '
        '"doc_ids": ["2026-sp:SUBJ-100"]}]'
    ),
    "multihop": (
        '[{"question": "can i take subj 200 after just the intro one?", "answer": '
        '"Yes. SUBJ 200 lists \\"One of SUBJ 100, SUBJ 150.\\", so SUBJ 100 satisfies it on its '
        'own.", "doc_ids": ["2026-sp:SUBJ-100", "2026-sp:SUBJ-200"]}]'
    ),
    "ambiguous": (
        '[{"question": "what are the prerequisites for 100?", "answer": "Which 100 do you mean — '
        'SUBJ 100 or OTHR 100?", "doc_ids": ["2026-sp:SUBJ-100", "2026-sp:OTHR-100"]}]'
    ),
}


def render_documents(chunks: Sequence[Mapping[str, Any]]) -> str:
    """The batch as the model sees it: numbered records, each headed by its own ``doc_id``."""
    if not chunks:
        raise ValueError("a batch with no records in it has nothing to draft from")
    total = len(chunks)
    return "\n\n".join(
        DOCUMENT_TEMPLATE.format(n=n, total=total, doc_id=chunk["doc_id"], text=chunk["text"])
        for n, chunk in enumerate(chunks, start=1)
    )


def render_draft_prompt(bucket: str) -> str:
    """The system prompt for one bucket: what to write, the rules, the shape, and one example.

    Assembled from the tables above rather than written out three times, so a rule added for
    ``fact`` cannot go missing for ``multihop``. Rendered per bucket because the *task* differs
    and nothing else does -- and each of the three is pinned by its own digest in the tests.
    """
    if bucket not in BUCKET_TASKS:
        raise ValueError(
            f"{bucket!r} is not drafted from documents; the three that are: "
            f"{', '.join(DRAFTABLE)} (unanswerable rows are constructed in #57)"
        )
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(DRAFT_RULES, start=1))
    fields = "\n".join(f"- {name}: {gloss}" for name, gloss in DRAFT_FIELDS)
    shown = EXAMPLE_BATCHES[bucket]
    example_batch = render_documents(
        [{"doc_id": one, "text": EXAMPLE_RECORDS[one]} for one in shown]
    )
    records = "record" if len(shown) == 1 else "records"
    return (
        f"{DRAFT_HEADER}\n\n{BUCKET_TASKS[bucket]}\n\nRules:\n\n{rules}\n\n"
        f"Reply with a JSON array. Each element has exactly three keys:\n\n{fields}\n\n"
        f"Example — for a batch holding these {records} (a real batch holds {BATCH_SIZE}):"
        f"\n\n{example_batch}\n\n{EXAMPLE_REPLIES[bucket]}"
    )


def draft_user_prompt(chunks: Sequence[Mapping[str, Any]], bucket: str, per_batch: int) -> str:
    """The user half: the ten records, then how many questions, then the cue for the array."""
    if per_batch < 1:
        raise ValueError(f"a batch asks for at least one question, got {per_batch!r}")
    return (
        f"{render_documents(chunks)}\n\n"
        f"Write at most {per_batch} {bucket} question(s) from these {len(chunks)} records, as a "
        f"JSON array.\n\nJSON:"
    )


# --- the stand-in ---------------------------------------------------------------------------------

#: How :class:`StandInDrafter` reads a batch back out of the user prompt: the heading's ``doc_id``
#: and the record's text, up to the next heading or the instruction that follows the last one.
STANDIN_DOC_RE = re.compile(
    r"^Document \d+ of \d+ \(doc_id: (?P<doc_id>[^)]+)\)\n-{3}\n(?P<text>.*?)"
    r"(?=\n\nDocument \d+ of |\n\nWrite at most |\Z)",
    re.MULTILINE | re.DOTALL,
)

#: A record's first line: ``CS 374 — Algorithms & Models of Computation``.
STANDIN_CODE_RE = re.compile(r"^(?P<code>[A-Z]{2,4} \d{3})")

#: The ``Credits:`` line :mod:`ingest.chunk` renders.
STANDIN_CREDITS_RE = re.compile(r"^Credits: (?P<credits>.+)$", re.MULTILINE)


class StandInDrafter:
    """Not a model: the canned drafter that lets this harness run end to end with no credits.

    The OpenAI account behind this repo returns 429 ``insufficient_quota``, and a drafting run
    that cannot be executed is a drafting run that cannot be tested. This reads the batch back out
    of the prompt it was handed and emits the same JSON array a model is asked for, so every part
    of the machinery around the words is exercised for real -- the batch the sampler dealt, the
    parse, the id check against the ten, the schema check, the duplicate check, the overlap
    measurement and the file it all lands in. Only the questions are canned, and they are canned
    *badly*: they name the course code and quote its credit line, which is exactly the leakage
    rule 2 forbids, so a stand-in run's overlap distribution is a smoke test of the measurement
    and not a measurement.

    **Nothing it writes may be reported as a model's work, or promoted into the set.** Its
    ``model`` is ``stand-in``, every row it produces records that in provenance, its answers say
    so in their own text, and :func:`draft_path` sends it to a ``.standin.jsonl`` file that #59
    does not read -- a canned row in ``fact_{hash}.jsonl`` would be indistinguishable from a
    drafted one by the time anybody looked.
    """

    #: What provenance, the summary and the file name record. Deliberately not the configured
    #: model: a stand-in run must never read as a run of ``gpt-4o-mini``.
    model = "stand-in"

    def __init__(self, bucket: str, per_batch: int = DEFAULT_PER_BATCH) -> None:
        """Take the bucket whose canned shape to produce, and how many rows per batch."""
        if bucket not in DRAFTABLE:
            raise ValueError(
                f"unknown bucket {bucket!r}; the three drafted: {', '.join(DRAFTABLE)}"
            )
        self.bucket = bucket
        self.per_batch = per_batch

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """The JSON array a model would have replied with, read off the records in ``prompt``."""
        records = [
            (match["doc_id"].strip(), match["text"].strip())
            for match in STANDIN_DOC_RE.finditer(prompt)
        ]
        if not records:
            raise ValueError(
                "a stand-in has nothing to draft from: it reads the records out of the prompt, "
                "and this prompt carries none"
            )
        drafts = (
            self._fact(records) if self.bucket == "fact" else self._paired(records)
        )
        return json.dumps(drafts[: self.per_batch])

    @staticmethod
    def _code(text: str) -> str:
        match = STANDIN_CODE_RE.match(text)
        return match["code"] if match else text.split("\n", 1)[0][:12]

    @staticmethod
    def _credits(text: str) -> str:
        match = STANDIN_CREDITS_RE.search(text)
        return match["credits"].strip() if match else "an unstated number of"

    def _fact(self, records: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """One credit-hours question per record: the shape a real drafter is warned against."""
        return [
            {
                "question": f"how many hours is {self._code(text).lower()}?",
                "answer": (
                    f"{self._code(text)} is {self._credits(text)} credit hours. (stand-in draft, "
                    "copied off the record -- no model wrote this.)"
                ),
                "doc_ids": [doc_id],
            }
            for doc_id, text in records
        ]

    def _paired(self, records: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """Consecutive pairs, for the two buckets whose rows cite more than one record."""
        drafts = []
        for (first_id, first), (second_id, second) in zip(records[::2], records[1::2]):
            one, two = self._code(first), self._code(second)
            if self.bucket == "multihop":
                question = f"do i have to take {one.lower()} before {two.lower()}?"
                answer = (
                    f"{one} and {two} are two separate records. (stand-in draft: no join was "
                    "reasoned about here.)"
                )
            else:
                question = f"what are the prerequisites for {one.split()[-1]}?"
                answer = f"Which course do you mean — {one} or {two}? (stand-in draft.)"
            drafts.append(
                {"question": question, "answer": answer, "doc_ids": [first_id, second_id]}
            )
        return drafts


def make_drafter(
    cfg: Mapping[str, Any],
    bucket: str,
    *,
    model: str | None = None,
    per_batch: int = DEFAULT_PER_BATCH,
    client: Any | None = None,
    fake: bool = False,
) -> Chat:
    """The client this run drafts with: the configured chat model, another one, or the stand-in.

    ``model`` overrides ``llm.model`` **for this run only**, and deliberately not through
    ``config.yaml``: drafting with a model other than the one under test is the one mitigation
    there is for the shared-blind-spot problem in this module's docstring, and it must not cost
    every artifact in the repo a new config hash to try. What was used is recorded per row
    instead, which is where Day 4 needs it anyway.

    ``client`` replaces the SDK object inside :class:`llm.OpenAIChat`, exactly as it does for the
    agent's two calls -- so the prompt assembly, the cut-off check and the empty-completion check
    are all exercised over a stub transport, with no key and no network.
    """
    section = check_llm_config(cfg)
    if fake:
        return StandInDrafter(bucket, per_batch)
    return OpenAIChat(
        model or section["model"],
        temperature=float(section["temperature"]),
        max_tokens=int(section["max_tokens"]),
        client=client,
    )


# --- sampling -------------------------------------------------------------------------------------


class Batch(NamedTuple):
    """Ten records, and which call of the plan they are."""

    #: 1-based, and stable: batch 3 is the same ten records however many batches were asked for.
    n: int
    chunks: tuple[Mapping[str, Any], ...]

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """The batch's doc ids in the order the records were drawn."""
        return tuple(chunk["doc_id"] for chunk in self.chunks)

    @property
    def ids(self) -> frozenset[str]:
        """The same ids as a set, for checking a drafted gold id came out of this batch."""
        return frozenset(self.doc_ids)


def stratified_order(
    chunks: Sequence[Mapping[str, Any]], seed: int
) -> list[Mapping[str, Any]]:
    """The whole corpus in one seeded order: shuffled inside each subject, then dealt round-robin.

    Two decisions, both visible in the output. Subjects, and the courses inside each one, are
    sorted before they are shuffled, so the order depends on the seed and not on the order
    ``data/chunks.jsonl`` happens to be written in -- a re-export that emits the same 4,502
    records in another order is not a new eval set. And courses are dealt one subject at a time
    rather than in proportion to how many each subject has, so ten consecutive records are ten
    different subjects and MUSC's 123 courses get no more of the eval set than a department with
    three. Proportional sampling over this corpus would put a fifth of Day 3 in the five largest
    departments.

    The cost is worth naming: a batch of ten unrelated subjects is poor ground for a ``multihop``
    question, which is why that bucket is candidates-only here and #58 builds the real rows from
    the prerequisite graph.
    """
    rng = random.Random(seed)
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_subject[chunk["metadata"]["subject"]].append(chunk)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    for subject in subjects:
        by_subject[subject].sort(key=lambda chunk: chunk["doc_id"])
        rng.shuffle(by_subject[subject])
    order: list[Mapping[str, Any]] = []
    while len(order) < len(chunks):
        for subject in subjects:
            queue = by_subject[subject]
            if queue:
                order.append(queue.pop())
    return order


def plan_batches(
    chunks: Sequence[Mapping[str, Any]], seed: int, size: int = BATCH_SIZE
) -> list[Batch]:
    """:func:`stratified_order` cut into batches of ``size``; the short tail is dropped.

    A prefix of this list is what ``--batches N`` runs, which is what makes a resume free: the
    plan is a function of the corpus and the seed only, so it cannot change because a later run
    asked for more. 4,502 records make 450 batches of ten and leave two over -- and a batch of
    two is a batch the prompt's wording ("these ten records") does not describe.
    """
    order = stratified_order(chunks, seed)
    return [
        Batch(n, tuple(order[start : start + size]))
        for n, start in enumerate(range(0, len(order) - size + 1, size), start=1)
    ]


# --- reading a completion -------------------------------------------------------------------------

#: A fenced reply, which rule 8 forbids and models produce anyway. Unwrapped rather than refused:
#: the fence is not a drafting error, and a batch thrown away for one costs ten records and a call.
FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL)


def parse_completion(text: str) -> list[Any]:
    """The JSON array a reply carries, or a sentence saying what arrived instead.

    Only the outer shape is decided here -- it is an array, and that is all. A malformed *element*
    is one row's problem and is dropped with a reason further down, because a batch of three
    questions where one is missing its ``answer`` should cost one question and not three.
    """
    stripped = text.strip()
    fenced = FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced["body"].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as error:
        head = stripped[:120] + ("..." if len(stripped) > 120 else "")
        raise ValueError(f"reply is not JSON ({error.msg}); it starts {head!r}") from None
    if not isinstance(parsed, list):
        raise TypeError(f"reply is a JSON {type(parsed).__name__}, not the array rule 8 asks for")
    return parsed


# --- measuring the leak ---------------------------------------------------------------------------

#: Words that make a question a credit-hours question. See :func:`credit_question`.
CREDIT_WORDS = frozenset({"credit", "credits", "hour", "hours"})


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Jaccard over two term sets: ``|a ∩ b| / |a ∪ b|``, and 0.0 when either side is empty."""
    a, b = frozenset(left), frozenset(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_overlap(
    question: str, chunks: Sequence[Mapping[str, Any]], tokenizer: Callable[[str], list[str]]
) -> float | None:
    """How much of its own gold record a question re-uses: the **largest** Jaccard over them.

    Largest rather than mean or union: the hazard is one document that a keyword search finds at
    rank 1, and a row whose second gold record shares nothing is not less leaky for it. ``None``
    when there is no gold record to compare against, which is every ``ambiguous`` row -- the
    issue asks for this measure on ``fact`` and ``multihop``, where a gold chunk is what the
    question was written from.

    The numbers are small in absolute terms -- a six-term question against a 110-term record
    cannot be otherwise -- so what is acted on is the shape of the distribution and its top
    decile, never a threshold quoted out of this function.
    """
    if not chunks:
        return None
    terms = tokenizer(question)
    return max(jaccard(terms, tokenizer(chunk["text"])) for chunk in chunks)


def credit_question(question: str, tokenizer: Callable[[str], list[str]]) -> bool:
    """Whether a question asks about credit hours -- marked in provenance, never dropped here.

    `eval/GUIDELINES.md` deletes a credit-count question about a variable-credit course: the chunk
    prints ``Credits: 3`` where the catalog says "3 OR 4 hours.", so the answer is wrong about one
    or the other. Which 1,415 courses those are cannot be decided on this branch -- #17's
    ``creditsMin``/``creditsMax`` fix is on ``main`` and the ``data/courses.json`` under this
    stack carries a single ``credits`` field -- so the harness marks the rows it would apply to
    and #59 checks them against the one-liner the spec carries. Marking beats guessing: a drop
    rule that cannot see the data it depends on would delete good rows and keep bad ones.
    """
    return bool(CREDIT_WORDS & set(tokenizer(question)))


def quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile of ``values`` (which need not be sorted).

    Written here rather than reused from :func:`ingest.chunk.percentile`, which takes and returns
    ``int`` because it measures token counts. Nearest-rank, so the returned number is one that a
    row actually has -- a flag threshold that no row sits on would be harder to explain than one
    that several do.
    """
    if not values:
        raise ValueError("no values to take a quantile of")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


# --- turning a reply into rows --------------------------------------------------------------------

#: Why a drafted question did not become a row. Every drop is counted under one of these and
#: printed; a run that quietly kept fewer rows than it drafted would hide the model's error rate,
#: which is a number #62 reports.
DROP_REASONS = ("shape", "dangling", "outside-batch", "bucket", "duplicate")

#: Every key a draft row's ``provenance`` object carries. A draft is a dataset row plus this, and
#: a file of rows without it is not re-generatable, so its absence is an error and not a warning.
PROVENANCE_KEYS = (
    "batch",
    "batch_doc_ids",
    "cited_doc_ids",
    "subjects",
    "levels",
    "model",
    "prompt_version",
    "config_hash",
    "seed",
    "overlap",
    "flagged",
    "credit_question",
)


class Settings(NamedTuple):
    """What every row of one run records about how it was made."""

    bucket: Bucket
    per_batch: int
    model: str
    prompt_version: str = DRAFT_PROMPT_VERSION
    config_hash: str = ""
    seed: int = 0


class Kept(NamedTuple):
    """A drafted question that passed every check, and what it records about itself."""

    item: EvalItem
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """The row as it is written: the six dataset fields, then ``provenance``."""
        return {**row_json(self.item), "provenance": self.provenance}


class Dropped(NamedTuple):
    """A drafted question that did not, and why -- in the words that will be printed."""

    reason: str
    detail: str


def build_row(
    raw: Any,
    batch: Batch,
    settings: Settings,
    n: int,
    known: frozenset[str],
    seen: Mapping[str, str],
    tokenizer: Callable[[str], list[str]],
) -> Kept | Dropped:
    """One element of a reply, checked into a row or dropped with a reason.

    The order of the checks is the order of how much they say about the model. The ids come
    first: they were in front of it in the prompt, and citing one that is not is the failure this
    harness exists to catch before a person does. Shape and the bucket's own rules follow, over
    the same :class:`eval.schema.EvalItem` and :func:`eval.validate.check_row` that #59's output
    will be held to -- one definition of a valid row, not a drafting-time approximation of one.
    Duplicates come last because "the same question again" only means anything once the row is
    otherwise good.
    """
    if not isinstance(raw, Mapping):
        return Dropped("shape", f"element is a {type(raw).__name__}, not an object")
    missing = [key for key, _ in DRAFT_FIELDS if key not in raw]
    if missing:
        return Dropped("shape", f"element is missing {', '.join(missing)}")
    cited = raw["doc_ids"]
    if not isinstance(cited, list) or not all(isinstance(one, str) for one in cited):
        return Dropped("shape", f"doc_ids is {cited!r}, not a list of ids")

    for one in cited:
        if one not in known:
            return Dropped("dangling", f"cited {one!r}, which is not in the corpus at all")
        if one not in batch.ids:
            return Dropped(
                "outside-batch",
                f"cited {one!r}, a real id that was not among the ten it was shown",
            )

    if settings.bucket == "ambiguous" and len(set(cited)) < 2:
        return Dropped(
            "bucket",
            f"cited {len(set(cited))} record(s); a question ambiguous between fewer than two of "
            "them has one referent, which makes it a fact question and not this bucket",
        )

    # An ambiguous row carries no gold ids -- `eval/GUIDELINES.md`'s two worked cases carry none,
    # and the row's answer is a clarifying question rather than a claim traceable to a record. The
    # candidates are not thrown away, though: they go to `cited_doc_ids` below, which is what #59
    # needs to check that the ambiguity is real.
    gold = [] if settings.bucket == "ambiguous" else list(cited)
    try:
        item = EvalItem(
            id=item_id(ID_BLOCKS[settings.bucket] + n - 1),
            bucket=settings.bucket,
            question=str(raw["question"]),
            gold_answer=str(raw["answer"]),
            gold_doc_ids=gold,
            should_abstain=RULES[settings.bucket].should_abstain,
        )
    except ValidationError as error:
        return Dropped("shape", "; ".join(e["msg"] for e in error.errors()))

    violations = [v for v in check_row(Row(0, item), known) if v.level == "error"]
    if violations:
        return Dropped("bucket", "; ".join(v.message for v in violations))

    key = normalize_question(item.question)
    if key in seen:
        return Dropped("duplicate", f"repeats {seen[key]}: {item.question!r}")

    by_id = {chunk["doc_id"]: chunk for chunk in batch.chunks}
    cited_chunks = [by_id[one] for one in cited]
    scored = [] if settings.bucket == "ambiguous" else cited_chunks
    measured = question_overlap(item.question, scored, tokenizer)
    return Kept(
        item,
        {
            "batch": batch.n,
            "batch_doc_ids": list(batch.doc_ids),
            "cited_doc_ids": list(cited),
            "subjects": [chunk["metadata"]["subject"] for chunk in cited_chunks],
            "levels": [chunk["metadata"]["level"] for chunk in cited_chunks],
            "model": settings.model,
            "prompt_version": settings.prompt_version,
            "config_hash": settings.config_hash,
            "seed": settings.seed,
            "overlap": None if measured is None else round(measured, 4),
            "flagged": False,
            "credit_question": credit_question(item.question, tokenizer),
        },
    )


# --- the files ------------------------------------------------------------------------------------


def draft_path(
    bucket: str, cfg: Mapping[str, Any], drafts_dir: Path | None = None, fake: bool = False
) -> Path:
    """Where this run's drafts go: ``eval/datasets/drafts/{bucket}_{hash}.jsonl``.

    A stand-in run gets ``{bucket}_{hash}.standin.jsonl`` instead. The two must not share a file:
    once a canned row is in ``fact_{hash}.jsonl`` nothing downstream can tell it from a drafted
    one except by reading its provenance, and a set is not an artifact you have to audit row by
    row to trust.
    """
    directory = DRAFTS_DIR if drafts_dir is None else Path(drafts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".standin.jsonl" if fake else ".jsonl"
    return directory / f"{bucket}_{config_hash(cfg)}{suffix}"


def meta_path(path: Path) -> Path:
    """The sidecar beside a draft file: ``fact_{hash}.meta.json``.

    A file of rows cannot say which batches were *attempted*, only which produced something -- and
    a batch whose three questions were all dropped is exactly the batch a resume must not pay for
    twice. So the record of what has run lives here, next to the rows it produced, the way
    ``data/chunks.meta.json`` sits beside the corpus.
    """
    return path.with_name(path.name.removesuffix(".jsonl") + ".meta.json")


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a temporary file, so a crash cannot leave half a sidecar behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Add a batch's rows to the draft file, flushed before the sidecar records the batch.

    Rows first, sidecar second, deliberately. A crash between them leaves the rows written and
    the batch unmarked, so the resume re-drafts it -- and the duplicate check then drops what it
    re-drafts, because those questions are already in the file. The other order would lose rows
    that were paid for.
    """
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rewrite the whole draft file atomically -- how the flag pass lands."""
    tmp = path.with_name(path.name + ".tmp")
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


class Draft(NamedTuple):
    """A draft row read back off disk: a dataset row, its line, and its provenance."""

    line: int
    item: EvalItem
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """The row as it is written back: the six schema fields, then ``provenance``."""
        return {**row_json(self.item), "provenance": self.provenance}


def parse_drafts(
    path: Path, keys: Sequence[str] = PROVENANCE_KEYS
) -> tuple[list[Draft], list[Violation]]:
    """Read a draft file into rows, reporting every line that is not one.

    A draft is a dataset row **plus** ``provenance``, so it is checked as both: the six fields
    against :class:`eval.schema.EvalItem` (which forbids extras, hence the split) and the
    provenance object against ``keys``. A row whose provenance is missing or short is an error
    and not a warning -- it is the half that makes the row re-generatable, and a file of
    un-re-generatable drafts is a file of anonymous questions.

    ``keys`` defaults to :data:`PROVENANCE_KEYS`, which is what a *drafted* row records. #57's
    bucket is constructed rather than drafted, so its rows record a failure shape and a gate
    reading where these record a batch, a model and a leakage overlap; it passes its own set. The
    reader is shared because everything else about the two files -- the six fields, the id rules,
    the duplicate check -- is the same, and a second copy of it would be a second set of messages
    for one fault.
    """
    rows: list[Draft] = []
    violations: list[Violation] = []
    with Path(path).open(encoding="utf-8") as f:
        for line, text in enumerate(f, start=1):
            if not text.strip():
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                violations.append(Violation(line, None, None, f"not JSON: {exc.msg}"))
                continue
            if not isinstance(raw, dict):
                kind = type(raw).__name__
                violations.append(Violation(line, None, None, f"not an object, but a {kind}"))
                continue
            found = raw.get("id") if isinstance(raw.get("id"), str) else None
            provenance = raw.pop("provenance", None)
            if not isinstance(provenance, dict):
                violations.append(
                    Violation(
                        line,
                        found,
                        "provenance",
                        "is missing; a draft row is a dataset row plus the record of how it was "
                        "made, and one without it cannot be re-generated or sliced on",
                    )
                )
                provenance = {}
            else:
                missing = [key for key in keys if key not in provenance]
                unknown = [key for key in provenance if key not in keys]
                if missing:
                    violations.append(
                        Violation(line, found, "provenance", f"is missing {', '.join(missing)}")
                    )
                if unknown:
                    violations.append(
                        Violation(
                            line,
                            found,
                            "provenance",
                            f"carries {', '.join(map(repr, unknown))}, which nothing reads",
                        )
                    )
            try:
                rows.append(Draft(line, EvalItem.model_validate(raw), provenance))
            except ValidationError as exc:
                violations.extend(pydantic_violations(line, found, exc))
    return rows, violations


def check_drafts(
    path: Path, known: frozenset[str], keys: Sequence[str] = PROVENANCE_KEYS
) -> list[Violation]:
    """Every check a draft file gets: the row's, the file's, and the provenance's.

    :func:`eval.validate.check_counts` is deliberately **not** among them. It compares a file
    against the 120-row plan, and a drafts file is one bucket, mid-run, by construction short --
    a warning saying so on every run is a warning nobody reads.

    ``keys`` is the provenance set to hold the file to; see :func:`parse_drafts`.
    """
    rows, violations = parse_drafts(path, keys)
    violations.extend(v for row in rows for v in check_row(Row(row.line, row.item), known))
    violations.extend(check_across_rows([Row(row.line, row.item) for row in rows]))
    violations.sort(key=lambda v: (v.line or 0, v.field or ""))
    return violations


# --- the run --------------------------------------------------------------------------------------


class BatchResult(NamedTuple):
    """What one model call produced: rows, drops, and whatever went wrong."""

    batch: Batch
    kept: list[Kept]
    dropped: list[Dropped]
    #: Elements past ``per_batch``. A model that answers with eight questions when asked for three
    #: is not making an error worth a drop reason, but the extras are not free rows either: they
    #: were drafted under an instruction the rest of the run did not follow.
    ignored: int = 0
    #: The call or the parse failing. Recorded rather than raised, so a long run keeps the
    #: batches it already paid for -- and the batch stays unmarked, so a resume retries it.
    error: str | None = None


class RunReport(NamedTuple):
    """One invocation, whole: what it wrote, what it dropped, and what the file now holds."""

    path: Path
    sidecar: Path
    results: list[BatchResult]
    #: Batches the sidecar already had, in plan order.
    skipped: list[int]
    #: Rows in the file after the run, drafted by this invocation or an earlier one.
    rows: int
    overlaps: list[float]
    threshold: float | None
    flagged: int
    credit_questions: int
    #: :func:`check_drafts` over the file as written. Non-empty is a bug in this script: every
    #: row was checked before it was written.
    violations: list[Violation]
    fake: bool

    @property
    def kept(self) -> int:
        """How many rows survived every check across all batches."""
        return sum(len(result.kept) for result in self.results)

    @property
    def failed(self) -> list[BatchResult]:
        """The batches that raised, so the summary can name them instead of hiding a short run."""
        return [result for result in self.results if result.error]

    @property
    def drops(self) -> dict[str, int]:
        """Dropped rows per reason, reasons with no drops left out."""
        counts = {reason: 0 for reason in DROP_REASONS}
        for result in self.results:
            for drop in result.dropped:
                counts[drop.reason] += 1
        return {reason: n for reason, n in counts.items() if n}


def run_batch(
    chat: Chat,
    batch: Batch,
    settings: Settings,
    known: frozenset[str],
    seen: dict[str, str],
    next_n: int,
    tokenizer: Callable[[str], list[str]],
) -> BatchResult:
    """One model call over ten records, and the rows that survive it.

    ``seen`` is updated as rows are kept, so two identical questions inside one reply cost one
    row rather than two -- drafting ten records at a time re-asks "how many credits is X?"
    forever, which is what :func:`eval.schema.normalize_question` exists for.
    """
    system = render_draft_prompt(settings.bucket)
    user = draft_user_prompt(batch.chunks, settings.bucket, settings.per_batch)
    try:
        drafts = parse_completion(chat.complete(user, system=system))
    except Exception as error:  # noqa: BLE001 -- recorded, so the batches already written survive
        return BatchResult(batch, [], [], error=f"{type(error).__name__}: {error}")

    kept: list[Kept] = []
    dropped: list[Dropped] = []
    for raw in drafts[: settings.per_batch]:
        row = build_row(raw, batch, settings, next_n + len(kept), known, seen, tokenizer)
        if isinstance(row, Dropped):
            dropped.append(row)
            continue
        seen[normalize_question(row.item.question)] = row.item.id
        kept.append(row)
    return BatchResult(batch, kept, dropped, ignored=max(0, len(drafts) - settings.per_batch))


def flag_top_decile(path: Path, quantile_q: float = FLAG_QUANTILE) -> tuple[float | None, int]:
    """Mark the rows whose question re-uses most of its gold record, and say how many.

    Computed over the **whole file** at the end of every run and written back, so the flag is a
    property of the distribution and not of the row: a later batch can move the threshold, and a
    row that was in the top decile of 12 rows may not be in the top decile of 60. That is the
    honest behaviour -- the decile is what #59 is asked to rewrite, and it only exists relative to
    a set. Ties at the threshold are all flagged: they have the same overlap, and flagging one of
    them would be arbitrary.
    """
    rows, _ = parse_drafts(path)
    measured = [
        row.provenance["overlap"] for row in rows if row.provenance.get("overlap") is not None
    ]
    if not measured:
        return None, 0
    threshold = quantile(measured, quantile_q)
    flagged = 0
    for row in rows:
        overlap = row.provenance.get("overlap")
        hot = overlap is not None and overlap >= threshold
        row.provenance["flagged"] = hot
        flagged += hot
    write_rows(path, [row.to_json() for row in rows])
    return threshold, flagged


def draft(
    cfg: Config,
    bucket: str,
    *,
    chat: Chat,
    batches: int,
    per_batch: int = DEFAULT_PER_BATCH,
    chunks: Sequence[Mapping[str, Any]] | None = None,
    chunks_path: Path | None = None,
    drafts_dir: Path | None = None,
    fake: bool = False,
) -> RunReport:
    """Draft ``batches`` batches of one bucket into ``eval/datasets/drafts/``, and resume cleanly.

    The corpus is read from ``data/chunks.jsonl``: unlike :mod:`eval.validate`, which only needs
    the *ids* and can derive them from the committed JSON, this needs the record text a question
    is drafted from, so a fresh clone runs ``uv run python -m ingest.chunk`` first and
    :func:`ingest.chunk.load_chunks` says so.
    """
    if bucket not in DRAFTABLE:
        raise ValueError(
            f"{bucket!r} is not drafted from documents; the three that are: "
            f"{', '.join(DRAFTABLE)}. The unanswerable bucket is constructed instead (#57): a "
            "model shown ten course records will not reliably invent questions they cannot answer"
        )
    corpus = list(load_chunks(chunks_path) if chunks is None else chunks)
    known = frozenset(chunk["doc_id"] for chunk in corpus)
    plan = plan_batches(corpus, cfg["seed"])
    if batches > len(plan):
        raise ValueError(
            f"asked for {batches} batches of {BATCH_SIZE}; {len(corpus)} records make {len(plan)}"
        )

    path = draft_path(bucket, cfg, drafts_dir, fake)
    sidecar = meta_path(path)
    meta = check_resumable(path, sidecar, bucket, cfg, corpus)

    existing, broken = parse_drafts(path) if path.exists() else ([], [])
    if broken:
        raise ValueError(
            f"{rel_to_repo(path)} has {len(broken)} row(s) this harness did not write "
            f"(first: {broken[0].message}); it is not safe to append to, so move it aside"
        )
    seen = {normalize_question(row.item.question): row.item.id for row in existing}
    next_n = len(existing) + 1
    done = {record["batch"] for record in meta["batches"]}

    settings = Settings(
        bucket=bucket,  # type: ignore[arg-type]
        per_batch=per_batch,
        model=getattr(chat, "model", "unknown"),
        prompt_version=DRAFT_PROMPT_VERSION,
        config_hash=cfg.hash,
        seed=cfg["seed"],
    )
    tokenizer = get_tokenizer(cfg)

    results: list[BatchResult] = []
    skipped: list[int] = []
    for batch in plan[:batches]:
        if batch.n in done:
            skipped.append(batch.n)
            continue
        result = run_batch(chat, batch, settings, known, seen, next_n, tokenizer)
        results.append(result)
        if result.error:
            continue
        append_rows(path, [row.to_json() for row in result.kept])
        next_n += len(result.kept)
        meta["batches"].append(
            {
                "batch": batch.n,
                "doc_ids": list(batch.doc_ids),
                "model": settings.model,
                "per_batch": per_batch,
                "prompt_version": settings.prompt_version,
                "kept": [row.item.id for row in result.kept],
                "dropped": count_drops(result.dropped),
                "ignored": result.ignored,
            }
        )
        meta["rows"] = next_n - 1
        write_json(sidecar, meta)

    threshold, flagged = flag_top_decile(path) if path.exists() else (None, 0)
    rows, _ = parse_drafts(path) if path.exists() else ([], [])
    overlaps = [
        row.provenance["overlap"] for row in rows if row.provenance.get("overlap") is not None
    ]
    return RunReport(
        path=path,
        sidecar=sidecar,
        results=results,
        skipped=skipped,
        rows=len(rows),
        overlaps=overlaps,
        threshold=threshold,
        flagged=flagged,
        credit_questions=sum(1 for row in rows if row.provenance.get("credit_question")),
        violations=check_drafts(path, known) if path.exists() else [],
        fake=fake,
    )


def count_drops(dropped: Sequence[Dropped]) -> dict[str, int]:
    """Drops by reason, zeroes omitted -- what the sidecar stores and the summary prints."""
    counts: dict[str, int] = {}
    for drop in dropped:
        counts[drop.reason] = counts.get(drop.reason, 0) + 1
    return counts


def check_resumable(
    path: Path,
    sidecar: Path,
    bucket: str,
    cfg: Config,
    corpus: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The sidecar to continue, or a fresh one -- and a refusal if the two cannot be the same run.

    A draft file and its sidecar are one artifact. One without the other is a run whose state
    cannot be known: rows with no record of which batches produced them would be re-drafted, and
    a sidecar with no rows would skip batches that wrote nothing. Both are refused with the fix
    in the sentence.

    What may not change under an existing file is what decides the *plan*: the bucket, the
    corpus and the seed. The model, the prompt version and ``--per-batch`` may -- every row
    records its own, which is what makes a file drafted by two models legible rather than
    corrupt, and drafting the second half with a different model is a mitigation this module's
    docstring actively recommends.
    """
    fresh: dict[str, Any] = {
        "bucket": bucket,
        "config_hash": cfg.hash,
        "chunk_key": chunk_key(cfg),
        "seed": cfg["seed"],
        "batch_size": BATCH_SIZE,
        "corpus": {
            "chunks": len(corpus),
            "subjects": len({chunk["metadata"]["subject"] for chunk in corpus}),
        },
        "batches": [],
        "rows": 0,
    }
    if not sidecar.exists():
        if path.exists():
            raise FileNotFoundError(
                f"{rel_to_repo(path)} exists but {rel_to_repo(sidecar)} does not, so which "
                "batches have run cannot be known; move the draft file aside to start over"
            )
        return fresh
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if not path.exists() and meta.get("rows"):
        raise FileNotFoundError(
            f"{rel_to_repo(sidecar)} records {meta['rows']} row(s) but {rel_to_repo(path)} is "
            "gone; restore it or delete the sidecar to draft those batches again"
        )
    for key in ("bucket", "config_hash", "chunk_key", "seed", "batch_size"):
        if meta.get(key) != fresh[key]:
            raise ValueError(
                f"{rel_to_repo(sidecar)} was written with {key}={meta.get(key)!r} and this run "
                f"has {fresh[key]!r}; the batch plan is a function of the corpus and the seed, so "
                "the two runs are not the same set -- start a new file"
            )
    meta.setdefault("batches", [])
    return meta


# --- saying so ------------------------------------------------------------------------------------


def overlap_line(report: RunReport) -> str:
    """The distribution the issue asks to be computed and printed, on one line.

    Min, median, the flag threshold and max -- four numbers, because the useful reading is how
    far the top sits above the middle, and a nine-decile table on every run is a table nobody
    compares. The full distribution is in the file: every row carries its own ``overlap``.
    """
    if not report.overlaps:
        return "no fact/multihop rows yet, so nothing to measure"
    values = report.overlaps
    threshold = "—" if report.threshold is None else f"{report.threshold:.3f}"
    return (
        f"n {len(values)}  min {min(values):.3f}  p50 {quantile(values, 0.5):.3f}  "
        f"p90 {threshold}  max {max(values):.3f}   flagged {report.flagged} (>= {threshold})"
    )


def summary(cfg: Config, report: RunReport, bucket: str, batches: int) -> str:
    """What the terminal gets: what ran, what was kept, what was dropped, and where it went."""
    models = sorted(
        {result.kept[0].provenance["model"] for result in report.results if result.kept}
    )
    drops = report.drops
    ids = [row.item.id for row in parse_drafts(report.path)[0]] if report.path.exists() else []
    span = f"{ids[0]} .. {ids[-1]}" if ids else "none"
    drafters = ", ".join(models) or "nothing -- no batch produced a row"
    detail = f"  ({', '.join(f'{k} {n}' for k, n in drops.items())})" if drops else ""
    lines = [
        f"config hash {cfg.hash}  bucket {bucket}",
        (
            f"batches                {len(report.results)} run, {len(report.skipped)} already "
            f"done, {len(report.failed)} failed  (plan 1-{batches}, {BATCH_SIZE} records each)"
        ),
        f"drafted by             {drafters}, prompt {DRAFT_PROMPT_VERSION}",
        f"kept                   {report.kept} this run, {report.rows} in the file ({span})",
        f"dropped                {sum(drops.values())}{detail}",
        f"overlap                {overlap_line(report)}",
        (
            f"credit questions       {report.credit_questions}  (GUIDELINES rule 2 deletes the "
            "ones about a variable-credit course; #59 checks them)"
        ),
        (
            f"wrote                  {rel_to_repo(report.path)}, sidecar "
            f"{rel_to_repo(report.sidecar)}"
        ),
    ]
    details = [
        f"  {drop.reason:<14} batch {result.batch.n}: {drop.detail}"
        for result in report.results
        for drop in result.dropped
    ]
    if details:
        lines += ["", "dropped rows, one line each -- none of them was repaired:", *details]
    if report.fake:
        lines += [
            "",
            (
                "NO MODEL WAS CALLED. Every row above is canned by `StandInDrafter`: the "
                "questions quote their own records, so the overlap numbers are a test of the "
                "measurement and not a measurement. The file is `.standin.jsonl`, which #59 "
                "does not read."
            ),
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Draft one bucket into ``eval/datasets/drafts/`` and print what was kept and dropped.

    Exits 1 if a batch failed or if the file does not validate afterwards. The second case is a
    bug in this script rather than a bad draft: every row is checked before it is written.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.draft_items",
        description="Draft eval questions from ten real course records at a time, seeded and "
        "resumable, into eval/datasets/drafts/.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        choices=DRAFTABLE,
        help="which bucket to draft; unanswerable rows are constructed in #57, not drafted",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=5,
        help="how many batches of ten records to run, counting from the first (default: 5); "
        "batches already recorded in the sidecar are skipped, not re-drafted",
    )
    parser.add_argument(
        "--per-batch",
        type=int,
        default=DEFAULT_PER_BATCH,
        help=f"questions to ask of each batch (default: {DEFAULT_PER_BATCH})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="draft with this model instead of llm.model, without touching config.yaml -- the "
        "one mitigation there is for drafter and agent under test being the same model",
    )
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="draft with the canned stand-in: no model is called, and the rows go to a separate "
        ".standin.jsonl file that nothing downstream reads",
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument(
        "--chunks", default=None, help=f"the corpus (default: {rel_to_repo(CHUNKS_PATH)})"
    )
    parser.add_argument(
        "--drafts-dir", default=None, help=f"where drafts go (default: {rel_to_repo(DRAFTS_DIR)})"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    chat = make_drafter(
        cfg,
        args.bucket,
        model=args.model,
        per_batch=args.per_batch,
        fake=args.fake_model,
    )
    report = draft(
        cfg,
        args.bucket,
        chat=chat,
        batches=args.batches,
        per_batch=args.per_batch,
        chunks_path=Path(args.chunks) if args.chunks else None,
        drafts_dir=Path(args.drafts_dir) if args.drafts_dir else None,
        fake=args.fake_model,
    )
    print(summary(cfg, report, args.bucket, args.batches))

    status = 0
    if report.failed:
        print(
            "\n"
            + "\n".join(
                f"batch {result.batch.n} failed: {result.error}" for result in report.failed
            )
            + "\nThose batches are not marked done; re-running the same command retries them.",
            file=sys.stderr,
        )
        status = 1
    errors = [v for v in report.violations if v.level == "error"]
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
