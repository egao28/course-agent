"""One eval row, and what each of its four buckets commits to.

Day 3's deliverable is a dataset, and a dataset without a schema is a text file. This module is
the contract -- six fields, four buckets -- and :mod:`eval.validate` is the runnable check over a
file of them. Both land *before* the first question is drafted (#56-#58): the verification pass
(#59) is the only part of Day 3 that needs a person, and it should be spent judging answers
rather than finding typos and dangling ids.

**Six fields, no extras.** ``extra="forbid"``, so a drafting run that invents a ``difficulty``
field fails loudly instead of writing a column nothing reads. The field descriptions are not
decoration either: #56 hands this model to the drafting model as a JSON schema, so they are the
only instruction it gets about what belongs in each one.

**Six annotated fields and one assigned one.** :attr:`EvalItem.split` is the seventh, and it is
not a seventh thing anyone judges: no annotator fills it, no drafting model sees it (#56 hands
the model the six through a schema that predates it), and it is ``None`` everywhere until #61
freezes the dataset. It is on the row rather than in a sidecar because ``v1.jsonl`` is the
artifact Day 4 and Day 5 read, and an id-keyed file beside it is a second thing to keep in step
-- a row deleted from one leaves a line in the other, and nothing would say so. Being a field
means :func:`eval.validate.validate_file` sees it, :func:`eval.validate.parse_items` round-trips
it, and a misspelt ``spilt`` is still the extra-field error it always was. Use :func:`row_json`
rather than ``model_dump()`` when writing rows to a file: it drops ``split`` while it is unset,
so a draft file and a promoted-but-unfrozen dataset keep the six columns they have always had.

**``id`` is minted here, never chosen by a model.** :func:`item_id` assigns ``v1-NNNN`` in
pipeline order. Ids are how #60's agreement sample and #61's frozen split refer to rows, so a
re-numbering after either of those lands is a silent corruption -- the split would still load, and
point at different questions.

**``should_abstain`` is redundant with ``bucket``, on purpose.** The bucket fixes it (see
:data:`RULES`), so the two can disagree, and a row where they do is a mislabel that one field
alone would have hidden. It stays a single bit -- "must not emit a confident answer" -- because
the behaviour the two abstaining buckets want is not the same, and that distinction is carried by
``bucket`` rather than by a fifth state in a field the plan fixes at six: an ``unanswerable`` row
should end at *"this catalog does not have that"*, an ``ambiguous`` one at *"which course do you
mean?"*. So for ``ambiguous`` rows ``gold_answer`` holds **the clarifying question that should
have been asked**, not a factual answer. Every Day 4 metric conditions on ``bucket``, never on
``should_abstain`` alone.

**The counts in :data:`RULES` are targets, not gates.** :mod:`eval.validate` warns when a bucket
is short and never fails on it. Deleting a bad item during verification is the whole point of
verification, and a schema that refuses 49 facts is a schema that pressures the reviewer into
keeping a bad row.

``should_abstain`` is the one field declared strict. Pydantic would otherwise read the string
``"true"`` as ``True``, and the row would validate while the *file* -- which is the artifact, and
which Day 4 and Day 5 will also read with tools that do no coercion at all -- still held a
string in a column everything else treats as a bit.

One asymmetry worth naming: ``question`` and ``gold_answer`` are stripped of surrounding
whitespace on load, because they are prose and a stray space is noise -- and because #56 drafts in
batches and the duplicate check downstream should not be defeated by one. ``gold_doc_ids`` are
**not** stripped or repaired in any way: a doc id is a key, whitespace inside it is the error this
dataset is most likely to contain (``2026-sp:MATH 241`` for ``2026-sp:MATH-241``), and repairing
it here would hide exactly what the validator exists to catch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from agent.state import Exit

# --- the labels -----------------------------------------------------------------------------------

#: What a row is testing. ``Literal`` so a typo in a drafting script is a validation error rather
#: than a fifth bucket that no metric knows how to read.
Bucket = Literal["fact", "multihop", "unanswerable", "ambiguous"]

#: The same four, as data -- for the validator's messages and #62's per-bucket table.
BUCKETS: tuple[Bucket, ...] = ("fact", "multihop", "unanswerable", "ambiguous")

#: Which half of the frozen dataset a row is in (#61). ``calib`` is what Day 5's conformal
#: prediction calibrates on; ``test`` is what every final number is reported on, and it is read
#: once per milestone rather than once per experiment. A ``Literal`` for the same reason
#: :data:`Bucket` is one: ``"cal"`` should be a validation error, not a third half.
Split = Literal["calib", "test"]

#: The two, as data. Order is ``calib`` then ``test`` wherever both are printed.
SPLITS: tuple[Split, ...] = ("calib", "test")

#: The six fields a person fills in, in the order the issue writes them -- and exactly the keys
#: :func:`row_json` writes for a row that has not been frozen. ``split`` is deliberately not one
#: of them: it is assigned, not annotated, and the tests that pin a file's columns are about what
#: was written by hand. Named here so those tests and #61's writers share one list.
ANNOTATED_FIELDS: tuple[str, ...] = (
    "id",
    "bucket",
    "question",
    "gold_answer",
    "gold_doc_ids",
    "should_abstain",
)

#: The dataset version these ids belong to. Bumping it means a new file, not a renumbered one.
VERSION = "v1"

#: ``v1-0001``: the version, a dash, and four zero-padded digits. Anchored, so a trailing space
#: or a five-digit id is a failure and not a near-match.
ID_RE = re.compile(r"^v[1-9][0-9]*-[0-9]{4}$")

#: The largest id :func:`item_id` can mint without changing width -- and so without changing what
#: an id sorts next to. 120 rows is the plan; 9,999 is where the format runs out.
MAX_N = 9999


def item_id(n: int, version: str = VERSION) -> str:
    """The ``n``-th row's id, zero-padded: ``item_id(7) == 'v1-0007'``.

    The pipeline (#56) calls this in draft order and the number is never revisited. Refusing
    ``n = 0`` is not pedantry: a zero-th row would sort before every real one and read as a
    header in every table Day 4 prints.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N:
        raise ValueError(f"eval item numbers run from 1 to {MAX_N}, got {n!r}")
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError(f"version must look like 'v1', got {version!r}")
    return f"{version}-{n:04d}"


def normalize_question(question: str) -> str:
    """Whitespace collapsed, case folded -- what "the same question twice" means here.

    Only used for the duplicate check. Drafting ten documents at a time (#56) will re-ask *"How
    many credits is X?"* forever, and two rows that differ by a capital letter are one row.
    """
    return " ".join(question.split()).casefold()


# --- the bucket table -----------------------------------------------------------------------------


class BucketRule(NamedTuple):
    """What a bucket fixes about every row in it, and what Day 4 expects such a row to do."""

    #: The bucket this rule is for, repeated here so a rule read out of :data:`RULES` stands alone.
    bucket: Bucket
    #: How many rows the plan wants. A **target**: the validator warns, it does not fail.
    target_n: int
    #: The one value ``should_abstain`` may take in this bucket.
    should_abstain: bool
    #: Fewest gold ids a row may carry.
    min_gold_ids: int
    #: Most it may carry, or ``None`` for no ceiling.
    max_gold_ids: int | None
    #: The Day 2 exits (:data:`agent.state.EXITS`) that count as right for this bucket. Read by
    #: Day 4, not by the validator -- a row does not record what it actually did.
    exits: tuple[Exit, ...]
    #: One sentence for a validator message and for #62's table.
    gloss: str


#: The four buckets, their counts, and the shape each one forces on a row. 120 rows in total.
RULES: Mapping[Bucket, BucketRule] = {
    "fact": BucketRule(
        bucket="fact",
        target_n=50,
        should_abstain=False,
        min_gold_ids=1,
        max_gold_ids=None,
        exits=("answer",),
        gloss="one catalog record answers it",
    ),
    "multihop": BucketRule(
        bucket="multihop",
        target_n=25,
        should_abstain=False,
        min_gold_ids=2,
        max_gold_ids=None,
        exits=("answer",),
        gloss="two or more records must be joined",
    ),
    "unanswerable": BucketRule(
        bucket="unanswerable",
        target_n=25,
        should_abstain=True,
        min_gold_ids=0,
        max_gold_ids=0,
        exits=("out_of_scope", "refuse_handoff"),
        gloss="this catalog cannot answer it, so there is nothing to cite",
    ),
    "ambiguous": BucketRule(
        bucket="ambiguous",
        target_n=20,
        should_abstain=True,
        min_gold_ids=0,
        max_gold_ids=None,
        exits=("refuse_handoff",),
        gloss="under-specified; gold_answer holds the clarifying question",
    ),
}

#: 120. Summed rather than written down, so the table stays the one place a count lives.
TARGET_TOTAL = sum(r.target_n for r in RULES.values())


def rule(bucket: str) -> BucketRule:
    """The rule for one bucket, or a sentence naming the four that exist."""
    if bucket not in RULES:
        raise ValueError(f"unknown bucket {bucket!r}; the four are {', '.join(BUCKETS)}")
    return RULES[bucket]


# --- the row --------------------------------------------------------------------------------------


class EvalItem(BaseModel):
    """One question, its gold answer, and the documents that answer it.

    Shape only. Whether ``should_abstain`` agrees with ``bucket``, whether a ``multihop`` row
    carries two ids, and whether those ids exist are :mod:`eval.validate`'s job -- not because
    they are less important, but because a model that raises on the first cross-field problem
    reports one problem per run, and a 120-row file wants every one of them at once.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        pattern=ID_RE.pattern,
        description="Assigned by the pipeline, never by you: 'v1-0001'.",
    )
    bucket: Bucket = Field(
        description="fact = one catalog record answers it; multihop = two or more must be "
        "joined; unanswerable = this catalog cannot answer it at all; ambiguous = it does not "
        "say enough to be answered, and the right reply asks which course is meant.",
    )
    question: str = Field(
        description="What a student would type, in one sentence.",
    )
    gold_answer: str = Field(
        description="The correct answer, short and checkable. For an ambiguous row this is the "
        "clarifying question that should have been asked instead.",
    )
    gold_doc_ids: list[str] = Field(
        description="The chunk ids an answer must be traceable to, '2026-sp:MATH-241' form -- a "
        "dash, where a course code has a space. Empty for an unanswerable row.",
    )
    should_abstain: StrictBool = Field(
        description="True when no confident answer may be given: every unanswerable and "
        "ambiguous row, and no other.",
    )
    split: Split | None = Field(
        default=None,
        description="Assigned by the freeze (#61), never by you: 'calib' or 'test'. None on a "
        "draft and on a promoted set that has not been split yet.",
    )

    @field_validator("question", "gold_answer")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        """Strip the prose fields, and refuse one that was only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


def row_json(item: EvalItem) -> dict[str, Any]:
    """The row as it is written to a file: the six fields, and ``split`` once it has one.

    Everything that writes a row goes through this rather than ``model_dump()``. A drafted row
    has no half yet, and ``"split": null`` in a draft file would be a column that means "not
    frozen" sitting in files that can never be frozen -- three builders' output, the review
    pass's promoted set, #60's sample -- where it would read as a missing value instead of an
    inapplicable one. Dropping it also keeps every file already in the tree byte-identical to
    what a re-run produces, which is the property that makes a regenerated artifact reviewable.
    """
    return item.model_dump(exclude_none=True)
