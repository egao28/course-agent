"""The second annotator, on fifteen items: a blind sheet, two numbers, and every disagreement.

    uv run python -m eval.scripts.agreement           # export the packet, for a second person
    uv run python -m eval.scripts.agreement --score          # the filled sheet -> the report
    uv run python -m eval.scripts.agreement --score --dry-run

A single-annotator dataset has no measurable reliability: the first reviewer who asks *"how do you
know your labels are any good?"* has no answer without this, and "I was careful" is not one. So
fifteen rows of ``v1.jsonl`` go to one other person, with ``eval/GUIDELINES.md`` and nothing else,
and whatever number comes back is the number the Day 3 report prints.

**This module is the machinery, and it is all of what this file can honestly contain.** It samples,
exports, reads a returned sheet and computes the statistics. It cannot invent the sheet. Nothing
here writes a judgement, and :func:`score` refuses to run against anything but a file a person
actually filled in -- for the same reason #59's log is not seeded with verdicts.

**Stratified 6 / 3 / 3 / 3, not uniform.** Fifteen rows drawn uniformly from 120 miss ``ambiguous``
entirely 5.4% of the time and bring back fewer than three of it 52.6% of the time (hypergeometric,
asserted in the tests), and ``ambiguous`` is where disagreement lives -- it is the bucket whose
boundary (**A2** against **U1**) is a judgement call rather than a lookup. The allocation is
roughly proportional to :data:`eval.schema.RULES` (6.25 / 3.1 / 3.1 / 2.5 at the targets) with
``ambiguous`` rounded up, and it is a table here rather than arithmetic over the file so that the
draw does not quietly change shape when verification deletes rows.

**Seeded per bucket, and the sheet's row order shuffled.** Each bucket draws from
``random.Random(f"{seed}:{bucket}")`` over its **sorted** ids, so the draw depends on the seed and
the id set alone -- not on line order, and not on how many rows the other three buckets hold. One
shared stream would make the ``ambiguous`` draw move whenever a ``fact`` row was deleted, which is
reproducibility that only holds while nothing changes. The fifteen are then shuffled once more
before they are written, because a sheet whose first six rows are all ``fact`` hands the annotator
the stratification and a good guess at every bucket boundary in it.

**Blind means the labels are stripped -- and two of them leak anyway.** ``bucket`` and
``should_abstain`` are not in the sheet: showing them turns the exercise into a review of the first
annotator's work and pushes agreement toward 100%. But the sheet has to show the candidate answer,
because field 2 is *"is this answer right?"*, and two things follow that no amount of column
stripping fixes:

1. **The answer's wording carries ``should_abstain``.** *"This catalog has no CS 999 …"* is visibly
   a refusal. So the abstain/non-abstain split is given away, and what the bucket field
   actually measures is agreement on the two *within-pair* calls -- ``fact`` against
   ``multihop``, and ``unanswerable`` against ``ambiguous``. Those are the two hard ones
   (**F1**/**M1**, **A2**/**U1**), so the measurement is not worthless; it is narrower than
   "four-way bucket agreement" sounds, and the report says so beside the confusion matrix
   rather than leaving a reader to infer it.
2. **The citation count nearly determines the bucket.** :data:`eval.schema.RULES` gives
   ``unanswerable`` exactly zero gold ids and ``multihop`` at least two, so a sheet listing the
   cited ids would label three of the four buckets by arithmetic. But field 3 cannot be judged
   without them. Hence **two passes**: the sheet is filled for ``bucket`` and ``answer_correct``
   first, and only then is ``agreement_citations.md`` -- the cited chunks, in full -- opened for
   ``citations_support``. The ordering is the control, and it is the one part of this that depends
   on the annotator doing as asked.

**Percent agreement and Cohen's κ, because four buckets are not balanced.** With 50 of 120 rows in
``fact``, two annotators who both guessed ``fact`` every time would agree about 42% of the time
having done nothing at all. κ divides out that chance floor. Both are reported, per field, because
the *gap* between them is the informative part: high percent agreement with κ near zero is a
guidelines document that only works on the easy majority class.

**Two of the three fields break κ, in two different ways, and the report says which rather than
printing a number next to each.** ``v1.jsonl`` holds only rows the first annotator accepted or
edited -- that is what promotion means -- and that filters the very fields being scored:

- ``answer_correct`` is ``yes`` on every row of the file. A constant marginal makes
  :math:`p_e = p_o`, so κ is **exactly** 0 however well the two annotators agree, and 0/0 when
  the second annotator is also unanimous. The zero is an artifact of the sampling frame and says
  nothing about the guidelines.
- ``citations_support`` is not constant -- it is ``n/a`` exactly where the row carries no gold ids
  -- and that is worse than constant, because it makes the field-level number *look* healthy. A
  row with no citations shows the annotator an empty list, they answer ``n/a``, and annotator 1
  answered ``n/a`` for the same mechanical reason: six of fifteen rows agree by construction and
  carry both :math:`p_o` and κ up with them. Restricted to the rows that cite something, annotator
  1 is back to a constant ``yes`` and κ is 0 again. So the report prints the restricted line too
  (:func:`cited_pairs`), and that is the one to read.

For both fields percent agreement is the number that means something, and what it estimates is
*residual label error in the frozen set* -- how often a second reader would reject a row Day 4 is
about to score -- not inter-annotator reliability. A κ that means reliability here needs the sample
drawn from the **drafts**, before promotion, where annotator 1's verdicts vary; that is a different
measurement and not the one #60 asks for.

**Reported honestly at n=15.** The interval is a Wilson score interval, per field, from the counts
-- not the flat ±25 points, which is the worst case (:math:`\\hat p = 0.5`) and far too wide at the
top of the range where the answer will actually land. 14 of 15 is 93.3% with a Wilson interval of
roughly [70%, 99%], and quoting the 93.3% without it is the kind of precision that gets a
methodology section marked down. No interval is given for κ: its asymptotic standard error is not
trustworthy at fifteen items, and an interval that is wrong is worse than an absent one.

**Disagreements are the useful output.** Each one is itemised with both labels, the guideline rule
the second annotator says they applied, and their note -- which is why the sheet has a ``rule``
column at all. That list is the revision plan for ``GUIDELINES.md`` v2. Questions asked during
annotation go in ``agreement_questions.md`` and are evidence that the guidelines are incomplete;
they are answered by amending the file *afterwards*, and #62 records that the amendment came after
the agreement was measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import REPO_ROOT, Config, file_sha256, load_config, rel_to_repo, report_path
from eval.schema import BUCKETS, RULES, Bucket, EvalItem, row_json
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH, corpus_text
from eval.validate import parse_items, plural

#: Where the packet lives, beside the dataset it is drawn from.
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"


class Packet(NamedTuple):
    """Every file the handoff is made of, rooted in one directory.

    A record rather than seven module constants, so that the commands take the paths they write
    as an argument. A test that had to monkeypatch seven globals to avoid writing into
    ``eval/datasets/`` would be a test that stopped covering the seventh the day an eighth file
    was added -- and one of these files is a person's completed work.
    """

    #: The fifteen rows with the first annotator's labels on them. Committed: it is the answer key
    #: the score is computed against, and a sample nobody can re-read is not evidence.
    sample: Path
    #: Seed, allocation and what was drawn -- the sidecar convention #56 established. "Sampling
    #: reproducible from the recorded seed" is an acceptance line, and this is the record.
    meta: Path
    #: The blind sheet as exported: three columns filled, five empty.
    sheet: Path
    #: The blind sheet as returned. A second path, not the same one: the export must never be
    #: able to overwrite a person's work, and ``--force`` is the obvious way that happens.
    #: Committed **unedited** -- a tidied sheet is not a measurement.
    filled: Path
    #: Pass two: every cited chunk in full, opened only after the sheet's first two columns are
    #: filled in and saved. See the module docstring on what the citations give away.
    citations: Path
    #: What the annotator is asked to do -- in the packet rather than in a message, so that what
    #: they were told is as reconstructable as what they answered.
    instructions: Path
    #: Questions asked during annotation. Written once as a stub and never overwritten: they are
    #: evidence that ``GUIDELINES.md`` is incomplete, and #62 reports them.
    questions: Path

    @property
    def written(self) -> tuple[Path, ...]:
        """What an export writes every time.

        Not :attr:`filled`, which is the annotator's, and not :attr:`questions`, which is written
        once and then belongs to them too.
        """
        return (self.sample, self.meta, self.sheet, self.citations, self.instructions)


def packet(directory: Path | None = None) -> Packet:
    """The packet's paths, in ``eval/datasets`` unless told otherwise."""
    base = DATASETS_DIR if directory is None else Path(directory)
    return Packet(
        sample=base / "agreement_sample.jsonl",
        meta=base / "agreement_sample.meta.json",
        sheet=base / "agreement_sheet.tsv",
        filled=base / "agreement_sheet_filled.tsv",
        citations=base / "agreement_citations.md",
        instructions=base / "agreement_instructions.md",
        questions=base / "agreement_questions.md",
    )


#: The report's stem; :func:`config.report_path` adds the config hash and the extension.
STEM = "day3_agreement"

#: A cell with nothing in it.
MISSING = "—"


# --- the sample -----------------------------------------------------------------------------------

#: How many rows each bucket contributes. See the module docstring: roughly proportional to
#: :data:`eval.schema.RULES`, with ``ambiguous`` rounded up because it is where disagreement is.
ALLOCATION: Mapping[Bucket, int] = {
    "fact": 6,
    "multihop": 3,
    "unanswerable": 3,
    "ambiguous": 3,
}

#: 15. Summed, so the allocation table stays the one place a count lives.
SAMPLE_N = sum(ALLOCATION.values())


class Draw(NamedTuple):
    """One bucket's contribution to the sample: what was asked for, and what there was."""

    bucket: Bucket
    wanted: int
    got: int
    ids: tuple[str, ...]

    @property
    def short(self) -> int:
        return self.wanted - self.got


class Sample(NamedTuple):
    """The fifteen rows in sheet order, and the per-bucket record of how they were drawn."""

    rows: tuple[EvalItem, ...]
    draws: tuple[Draw, ...]
    seed: int

    @property
    def short(self) -> list[Draw]:
        """Buckets that could not fill their allocation, in table order."""
        return [draw for draw in self.draws if draw.short]

    @property
    def by_id(self) -> dict[str, EvalItem]:
        return {row.id: row for row in self.rows}


def stratified_sample(
    rows: Sequence[EvalItem], seed: int, allocation: Mapping[Bucket, int] = ALLOCATION
) -> Sample:
    """Draw the sample: per-bucket streams over sorted ids, then one shuffle of the result.

    Deterministic in ``(seed, the id set, the allocation)`` and nothing else -- not in line order,
    and not across buckets. A bucket short of its allocation contributes what it has and records
    the shortfall; refusing outright is :func:`export`'s call to make, because a twelve-row sample
    is a real measurement and a twelve-row sample reported as fifteen is not.
    """
    pools: dict[Bucket, list[str]] = {bucket: [] for bucket in BUCKETS}
    for row in rows:
        pools[row.bucket].append(row.id)
    by_id = {row.id: row for row in rows}
    draws: list[Draw] = []
    picked: list[str] = []
    for bucket in BUCKETS:
        wanted = allocation.get(bucket, 0)
        pool = sorted(pools[bucket])
        rng = random.Random(f"{seed}:{bucket}")
        ids = sorted(rng.sample(pool, k=min(wanted, len(pool))))
        draws.append(Draw(bucket, wanted, len(ids), tuple(ids)))
        picked.extend(ids)
    random.Random(f"{seed}:order").shuffle(picked)
    return Sample(tuple(by_id[item_id] for item_id in picked), tuple(draws), seed)


def load_dataset(path: Path) -> list[EvalItem]:
    """``v1.jsonl`` as rows, or a :class:`ValueError` naming what is wrong with it.

    Strict where :mod:`eval.validate` is forgiving: that tool exists to report every fault in a
    file being worked on, and this one is sampling the frozen artifact. A row that does not parse
    is a dataset that cannot be sampled, not a warning to carry into a report.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"no dataset at {rel_to_repo(path)}. The agreement sample is drawn from the promoted "
            "set, so #59 has to have run to the end first: draft the remaining buckets "
            "(`python -m eval.scripts.draft_items`), review every row "
            "(`python -m eval.scripts.review`), then `--promote`."
        )
    rows, violations = parse_items(path)
    if violations:
        shown = "; ".join(f"line {v.line}: {v.message}" for v in violations[:5])
        raise ValueError(f"{rel_to_repo(path)} has {plural(len(violations), 'bad row')}: {shown}")
    if not rows:
        raise ValueError(f"{rel_to_repo(path)} is empty")
    return [row.item for row in rows]


def write_sample(path: Path, rows: Sequence[EvalItem]) -> Path:
    """The sampled rows, one object per line, in sheet order -- the same shape as ``v1.jsonl``.

    Sheet order rather than id order, deliberately: this file and the sheet are read side by side
    when the disagreements are talked through, and two orderings of fifteen rows is a way to
    misread which row is which.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row_json(row), ensure_ascii=False) + "\n" for row in rows)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --- the three fields -----------------------------------------------------------------------------

#: What the second annotator decides, in #60's order. The same three checks as #59's verification
#: pass, so the two people are answering the same questions and the numbers below compare.
FIELDS: tuple[str, ...] = ("bucket", "answer_correct", "citations_support")

#: Every value each field may take. Fixed universes, not "whatever came back in the sheet": a
#: confusion matrix over observed values only would drop the row a bucket never received, and
#: :math:`p_e` computed over observed values only would move with the sample.
CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "bucket": BUCKETS,
    "answer_correct": ("yes", "no"),
    "citations_support": ("yes", "no", "n/a"),
    # Not a field anyone fills in -- :data:`DERIVED` is read off ``bucket`` -- but it is scored
    # through the same function, so it needs a universe like the other three.
    "should_abstain": ("true", "false"),
}

#: Spellings accepted on the way in, per field. A human fills this file in a spreadsheet, and
#: refusing ``Yes`` or ``N/A`` would be pedantry that costs a re-read of fifteen rows. Case is
#: folded before the lookup; anything still unrecognised is a :class:`Problem` and not a guess.
ALIASES: Mapping[str, Mapping[str, str]] = {
    "answer_correct": {"y": "yes", "n": "no", "true": "yes", "false": "no"},
    "citations_support": {"na": "n/a", "n.a.": "n/a", "n / a": "n/a", "-": "n/a", "none": "n/a"},
}

#: The derived fourth line of the per-field table: ``should_abstain``, read off ``bucket`` through
#: :data:`eval.schema.RULES`. Not a field the annotator fills -- it is the half of the bucket call
#: the candidate answer's wording already gives away (see the module docstring), and reporting it
#: separately is what keeps the four-way bucket number from being read as four-way blind.
DERIVED = "should_abstain"


def first_annotator(item: EvalItem) -> dict[str, str]:
    """The three labels ``v1.jsonl`` asserts about one row -- annotator 1's side of every pair.

    Two of the three are constants, and that is the point of the paragraph in the module docstring:
    a promoted row is one the first annotator judged answerable-as-written, so ``answer_correct``
    is ``yes`` on every row of the file, and ``citations_support`` is fixed by whether the row
    carries ids at all. Written out as a function anyway, rather than hard-coded into the scorer,
    so that a sample drawn from somewhere else scores through the same path.
    """
    return {
        "bucket": item.bucket,
        "answer_correct": "yes",
        "citations_support": "yes" if item.gold_doc_ids else "n/a",
    }


def abstains(bucket: str) -> str:
    """``should_abstain`` for a bucket, as the string the tables print."""
    return "true" if RULES[bucket].should_abstain else "false"


# --- the sheet ------------------------------------------------------------------------------------

#: The sheet's columns. The first three are exported filled and are never read back -- the ids are
#: matched against the sample, and a returned sheet whose questions were edited is still scored on
#: its ids. The last five are the annotator's.
SHEET_COLUMNS: tuple[str, ...] = (
    "id",
    "question",
    "candidate_answer",
    *FIELDS,
    "rule",
    "note",
)

#: Filled in by the tool, not the annotator -- so ``--score`` can say which columns it expected to
#: find empty and did not.
EXPORTED_COLUMNS: tuple[str, ...] = SHEET_COLUMNS[:3]


def cell(text: str) -> str:
    """One TSV cell: whitespace collapsed to single spaces.

    TSV rather than CSV because every question and answer here contains commas and none contains
    a tab, so the file has no quoting rules to get wrong in a spreadsheet round-trip -- but that
    only holds if nothing written into it can contain a tab or a newline, which is this function.
    """
    return " ".join(str(text).split())


def write_sheet(path: Path, rows: Sequence[EvalItem]) -> Path:
    """The blind sheet: id, question, candidate answer, and five empty columns.

    No ``bucket``, no ``should_abstain``, no ``gold_doc_ids`` -- the first two because showing them
    makes this a review of the first annotator's work, the third because
    :data:`eval.schema.RULES` fixes the id count per bucket tightly enough that listing them
    labels the row. The citations reach the annotator in pass two, through
    :func:`render_citations`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(SHEET_COLUMNS)]
    for row in rows:
        filled = (row.id, cell(row.question), cell(row.gold_answer))
        lines.append("\t".join((*filled, *("" for _ in SHEET_COLUMNS[3:]))))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


class Problem(NamedTuple):
    """Something wrong with the returned sheet: a line, a column, and what is wrong with it."""

    line: int
    id: str | None
    column: str | None
    message: str


class Filled(NamedTuple):
    """One row of the returned sheet: the three judgements, the rule cited, and the note."""

    line: int
    id: str
    labels: dict[str, str]
    rule: str
    note: str


def normalize(field: str, raw: str) -> str | None:
    """One cell's value as a category, or ``None`` when it is not one.

    Case folded and aliased (:data:`ALIASES`); never guessed. An unrecognised value is reported
    against the line it was on, because the alternative -- dropping the row -- changes the
    denominator of the agreement number without saying so.
    """
    text = " ".join(raw.split()).casefold()
    if not text:
        return None
    text = ALIASES.get(field, {}).get(text, text)
    return text if text in CATEGORIES[field] else None


def read_sheet(path: Path, expected: Sequence[str]) -> tuple[list[Filled], list[Problem]]:
    """Read a returned sheet against the ids it should hold, reporting every fault at once.

    Every fault at once for the reason :mod:`eval.validate` does it: the file is fifteen lines
    filled in by hand, and a reader that stopped at the first blank cell would send a person back
    to their spreadsheet once per blank cell.

    A missing id, an unknown id and a duplicated id are all problems rather than silent drops --
    the denominator of a percent agreement is exactly what a sheet with three rows missing gets
    wrong.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"no filled sheet at {rel_to_repo(path)}. Export the packet, have a second person "
            "fill the sheet in, save it under that name, and score that."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[Filled] = []
    problems: list[Problem] = []
    if not lines:
        return rows, [Problem(0, None, None, "the sheet is empty")]
    header = lines[0].split("\t")
    if tuple(header) != SHEET_COLUMNS:
        return rows, [
            Problem(
                1,
                None,
                None,
                "header is "
                + ", ".join(header)
                + "; it must be exactly "
                + ", ".join(SHEET_COLUMNS),
            )
        ]
    seen: dict[str, int] = {}
    for line, text in enumerate(lines[1:], start=2):
        if not text.strip():
            continue
        values = text.split("\t")
        if len(values) != len(SHEET_COLUMNS):
            problems.append(
                Problem(
                    line,
                    None,
                    None,
                    f"has {len(values)} columns, not {len(SHEET_COLUMNS)} -- a tab was added or "
                    "lost; the three exported columns must not be edited",
                )
            )
            continue
        record = dict(zip(SHEET_COLUMNS, values, strict=True))
        item_id = record["id"].strip()
        if item_id in seen:
            problems.append(Problem(line, item_id, "id", f"already judged on line {seen[item_id]}"))
            continue
        seen[item_id] = line
        if item_id not in expected:
            problems.append(Problem(line, item_id, "id", "is not in the sample"))
            continue
        labels: dict[str, str] = {}
        for field in FIELDS:
            value = normalize(field, record[field])
            if value is None:
                allowed = ", ".join(CATEGORIES[field])
                given = record[field].strip()
                what = "is empty" if not given else f"is {given!r}"
                problems.append(Problem(line, item_id, field, f"{what}; one of {allowed}"))
            else:
                labels[field] = value
        if len(labels) == len(FIELDS):
            rows.append(
                Filled(line, item_id, labels, record["rule"].strip(), record["note"].strip())
            )
    for item_id in expected:
        if item_id not in seen:
            problems.append(Problem(0, item_id, None, "is in the sample and not in the sheet"))
    return rows, problems


# --- the packet -----------------------------------------------------------------------------------

#: What the annotator is asked to do, as a template rather than as prose assembled in code: this
#: text is the other half of the measurement -- ``GUIDELINES.md`` says what the labels mean and
#: this says what the exercise is -- and a paragraph that has to be reconstructed by reading a
#: renderer is a paragraph nobody will check. Substitutions are named; there are no other braces.
INSTRUCTIONS = """\
# Second annotator — {n} items

{n} rows from `{dataset}`, drawn with seed `{seed}`. You are the second annotator: another person
has already labelled these rows, and you are not being shown their labels. Please do not ask for
them, and please do not open `{sample}` — it is the answer key.

## Read first, and read only this

`eval/GUIDELINES.md`. One page. It is the whole instruction, and that is the point of the
exercise: what is being measured is whether that page is enough for someone who did not write it,
so anything you have to ask is a finding and not a nuisance.

Log every question in `{questions}`, including the ones you answer yourself. They are the evidence
for what the guidelines leave open, and they are answered by amending the page *after* the
agreement is measured — so a question you asked and worked around is worth more here than a
question you asked and had answered.

## Two passes, in this order

The order is a control, not bureaucracy. The cited documents nearly give the bucket away — a row
with no citations is almost always a refusal, a row with three is almost always a join — so a
bucket judged after reading them measures very little.

**Pass 1 — `{sheet}`, without opening `{citations}`.** Fill in `bucket` and `answer_correct` for
all {n} rows, and save the file.

- `bucket` — one of {buckets}. `GUIDELINES.md` → *Which bucket*, and the eight boundary cases.
- `answer_correct` — `yes` or `no`. Is the candidate answer right? Where the candidate answer is a
  refusal or a clarifying question, the question is *is this the right thing to refuse (or to
  ask)?* — `GUIDELINES.md` → *When a refusal is correct*.

To check an answer, go to the source: read the chunk, or call `get_course`, `get_prereq_chain`,
`check_requirement`. Do **not** run the agent or `retrieve()` to decide a label — what the system
does with a question is the thing under test, and it cannot also be the thing that labels it.

**Pass 2 — `{citations}`, then the last judgement.** It prints, per row, the records the first
annotator cited, in full. Fill in `citations_support`:

- `yes` — every fact the candidate answer asserts is readable in the text of a cited record.
- `no` — it is not. An id that is merely *about* the right course is a `no`: `GUIDELINES.md`
  rule 4.
- `n/a` — the row cites nothing.

Changing a pass-1 answer after reading the citations is allowed. Say so in `note` if you do: that
is exactly the leak the two passes exist to measure, and it is more useful recorded than avoided.

## The last two columns

- `rule` — which part of `GUIDELINES.md` you applied: a boundary case (`F1`, `M2`, `U2`, `A1` …),
  a numbered rule (`rule 3`), or a section name. On a row that was obvious, leave it empty. On a
  row that was not, this is the most useful cell in the sheet — where we disagree it says which
  rule was ambiguous, instead of leaving us to guess which one you were reading.
- `note` — anything else, in your words.

## When you are done

Save the sheet as `{filled}` and hand it back. It is committed exactly as you leave it; nobody
tidies it up afterwards.

Do not edit the `id`, `question` or `candidate_answer` columns, and do not add or remove rows: the
scorer matches on `id`, and it refuses a sheet that has lost one rather than quietly scoring
fourteen.
"""

#: The header of pass two. The per-row blocks are generated; this is the part that says what the
#: narrow question is, and it is narrow on purpose -- "does this text support the claim" is a
#: different question from "is the claim true", which was pass one.
CITATIONS_HEADER = """\
# Cited records — pass two

Open this **after** `bucket` and `answer_correct` are filled in for all {n} rows and the sheet is
saved. See `{instructions}`.

Each row below lists the records the first annotator cited, in full. The question is narrow: is
every fact the candidate answer asserts readable in this text? Not *is the answer right* — that
was pass one — and not *is this the best document for the question*.
"""

#: The questions log, written once as a stub and never overwritten. A file that exists gets used;
#: an instruction to start one does not.
QUESTIONS_STUB = """\
# Questions asked during annotation

Every question the second annotator had, including the ones they answered themselves. A question
here is evidence that `eval/GUIDELINES.md` is incomplete at that point: the file is the whole
instruction, so anything that had to be asked was not in it.

These are answered by **amending `GUIDELINES.md` after** the agreement is measured, and #62
records that the amendment came afterwards. Answering one during annotation would make the
agreement number a measurement of a document that no longer exists.

One per heading, with the row id where there is one.

<!-- e.g.
## v1-0042 — does a cross-listed course count as one record or two?

Guidelines say to count records, not codes (*Which bucket*). ECE 374 / CS 374 is M1, but this row
is a different pair and M1 does not obviously generalise. Called it `multihop`.
-->
"""


def render_instructions(sample: Sample, dataset: Path, pack: Packet) -> str:
    """What the annotator is asked to do. In the packet, not in a message.

    A rule agreed verbally is a rule that does not exist -- ``GUIDELINES.md`` says so about
    itself, and it is as true of the instructions as of the labels. If the number is going to be
    quoted, what the annotator was told has to be as reconstructable as what they answered.
    """
    return INSTRUCTIONS.format(
        n=len(sample.rows),
        dataset=rel_to_repo(dataset),
        seed=sample.seed,
        sample=pack.sample.name,
        sheet=pack.sheet.name,
        filled=pack.filled.name,
        citations=pack.citations.name,
        questions=pack.questions.name,
        buckets=", ".join(f"`{bucket}`" for bucket in BUCKETS),
    )


def render_citations(sample: Sample, texts: Mapping[str, str], pack: Packet) -> str:
    """Pass two: the cited records, in full, one block per row.

    In full and in the packet, for the reason #59 puts the chunk text on the reviewer's screen: an
    annotator who has to open another window to read a cited chunk approves citations by vibe by
    item 40, and a row whose answer is right and whose citation is wrong is worse than a wrong row
    -- it silently rewards a system that retrieved the wrong document.

    Rows that cite nothing are listed too, saying so. Omitting them would make this file a list of
    the non-abstaining rows, which is a label.
    """
    out = [CITATIONS_HEADER.format(n=len(sample.rows), instructions=pack.instructions.name)]
    for row in sample.rows:
        out.append(
            "\n".join(
                [
                    "---",
                    "",
                    f"## `{row.id}`",
                    "",
                    f"**Q.** {cell(row.question)}",
                    "",
                    f"**Candidate answer.** {cell(row.gold_answer)}",
                ]
            )
        )
        if not row.gold_doc_ids:
            out.append("*No cited records — `citations_support` is `n/a` for this row.*")
            continue
        for doc_id in row.gold_doc_ids:
            text = texts.get(doc_id)
            if text is None:
                body = (
                    "*Not in the corpus.* Nothing supports this row, so that is a `no` — and a "
                    "hole in the validator. Please note it."
                )
            else:
                body = "```\n" + text.rstrip() + "\n```"
            out.append(f"### `{doc_id}`\n\n{body}")
    return "\n\n".join(out)


# --- the numbers ----------------------------------------------------------------------------------

#: Two-sided 95%. A constant rather than a lookup: one interval is quoted in this report, and a
#: dependency on ``scipy`` to obtain 1.96 is a dependency for 1.96.
Z = 1.959963985


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """A Wilson score interval for ``k`` of ``n``.

    Wilson and not the normal approximation, because the answer will land near the top of the
    range and that is exactly where the normal approximation breaks: 14 of 15 gives
    ``0.933 ± 0.127``, an upper bound of 1.06, which is not a probability. Wilson stays inside
    [0, 1] by construction and does not collapse to zero width at 15 of 15.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


class Kappa(NamedTuple):
    """Cohen's κ, the chance agreement it divided out, and why it may not mean anything."""

    value: float | None
    p_e: float
    #: Set when the number is undefined, or defined and uninterpretable. See
    #: :func:`cohen_kappa` -- this is the field that keeps the report from printing a bare 0.00.
    note: str | None


def cohen_kappa(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> Kappa:
    """κ over paired labels, with the degenerate cases named rather than rounded.

    :math:`\\kappa = (p_o - p_e) / (1 - p_e)`, with :math:`p_e` from each annotator's own
    marginals. Two degeneracies matter here and both are reached by this sample, not by contrived
    input:

    **Both annotators unanimous on the same category** makes :math:`p_e = 1` and κ a literal 0/0.
    Reported as undefined. Some libraries return 0.0, which reads as "chance-level agreement"
    for fifteen rows two people agreed on completely.

    **One annotator unanimous** makes :math:`p_e = p_o` and so κ **exactly** 0, whatever the other
    annotator did. That is the case for ``answer_correct`` and ``citations_support`` against a
    promoted dataset -- see the module docstring -- and a 0 that is forced by the sampling frame
    is not a finding about the guidelines. The value is returned, since it is what κ is, with a
    note saying it was not free to be anything else.

    Categories are taken as given and unioned with what was observed, so a value outside the
    declared universe still counts toward :math:`p_e` instead of silently deflating it.
    """
    n = len(pairs)
    if not n:
        return Kappa(None, 0.0, "no rows to compare")
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    universe = list(dict.fromkeys([*categories, *first, *second]))
    p_o = sum(1 for a, b in pairs if a == b) / n
    p_e = sum((first[c] / n) * (second[c] / n) for c in universe)
    if p_e >= 1.0:
        return Kappa(
            None,
            p_e,
            "undefined: both annotators used a single category, so chance agreement is 1 and κ "
            "is 0/0",
        )
    value = (p_o - p_e) / (1.0 - p_e)
    note = None
    if len(first) == 1:
        note = (
            "0 by construction: annotator 1 has no variance on this field in a promoted dataset, "
            "which forces p_e = p_o. Read the percent agreement, not this"
        )
    elif len(second) == 1:
        note = (
            "0 by construction: annotator 2 used one category for all "
            f"{n} rows, which forces p_e = p_o"
        )
    return Kappa(value, p_e, note)


class Agreement(NamedTuple):
    """One line of the numbers table: the counts, percent agreement with its interval, and κ."""

    #: Which of :data:`CATEGORIES` the labels came from.
    field: str
    #: How the line is named in the report. Differs from :attr:`field` for a derived or restricted
    #: line -- the same field scored over a subset says something different, and a table that
    #: called both ``citations_support`` would be a table with two contradictory rows in it.
    label: str
    n: int
    agreed: int
    kappa: Kappa
    lo: float
    hi: float

    @property
    def p_o(self) -> float:
        return self.agreed / self.n if self.n else 0.0

    @property
    def pct(self) -> str:
        return MISSING if not self.n else f"{100.0 * self.p_o:.1f}%"

    @property
    def interval(self) -> str:
        if not self.n:
            return MISSING
        return f"[{100.0 * self.lo:.0f}%, {100.0 * self.hi:.0f}%]"

    @property
    def heading(self) -> str:
        """How this line is named in the table *and* in the κ notes under it.

        One property rather than the same conditional twice: a note that named a line differently
        from the row it is about is a note a reader has to match up by guessing.
        """
        return f"`{self.field}`" if self.label == self.field else self.label


def score_field(
    pairs: Sequence[tuple[str, str]], field: str, label: str | None = None
) -> Agreement:
    """Percent agreement, its Wilson interval, and κ for one field."""
    agreed = sum(1 for a, b in pairs if a == b)
    lo, hi = wilson(agreed, len(pairs))
    return Agreement(
        field,
        field if label is None else label,
        len(pairs),
        agreed,
        cohen_kappa(pairs, CATEGORIES[field]),
        lo,
        hi,
    )


def pairs_for(
    field: str, judged: Sequence[Filled], sample: Mapping[str, EvalItem]
) -> list[tuple[str, str]]:
    """``(annotator 1, annotator 2)`` for every judged row, in sheet order."""
    return [(first_annotator(sample[row.id])[field], row.labels[field]) for row in judged]


def derived_pairs(
    judged: Sequence[Filled], sample: Mapping[str, EvalItem]
) -> list[tuple[str, str]]:
    """``should_abstain`` for both annotators, read off each one's bucket through :data:`RULES`."""
    return [(abstains(sample[row.id].bucket), abstains(row.labels["bucket"])) for row in judged]


def cited_pairs(
    judged: Sequence[Filled], sample: Mapping[str, EvalItem]
) -> list[tuple[str, str]]:
    """``citations_support``, over the rows that actually cite something.

    The field-level number is inflated and this one is not. A row with no gold ids shows the
    annotator an empty citation list, they answer ``n/a``, and annotator 1 answered ``n/a`` for
    the same mechanical reason -- so six of fifteen rows agree by construction and carry both
    ``p_o`` and κ up with them. What is left here is the subset where a person actually read a
    chunk and decided whether it supported a claim, which is the check #59 puts the chunk text on
    screen for.
    """
    return [
        (first_annotator(sample[row.id])["citations_support"], row.labels["citations_support"])
        for row in judged
        if sample[row.id].gold_doc_ids
    ]


def confusion(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> list[list[int]]:
    """Rows = annotator 1, columns = annotator 2, over the full category universe.

    The full universe, including categories neither annotator used: an empty row is the statement
    that the sample held none of that bucket, and at n=15 that is a thing a reader needs to see.
    """
    counts = Counter(pairs)
    return [[counts[(a, b)] for b in categories] for a in categories]


class Disagreement(NamedTuple):
    """One row, one field, both labels -- and what the second annotator says they applied."""

    id: str
    field: str
    first: str
    second: str
    rule: str
    note: str
    question: str


def disagreements(judged: Sequence[Filled], sample: Mapping[str, EvalItem]) -> list[Disagreement]:
    """Every field of every row where the two annotators differ, in sheet order.

    The useful output of the whole exercise: this list is the revision plan for ``GUIDELINES.md``
    v2, and a disagreement that reveals the *first* annotator was wrong is a fix to ``v1.jsonl``
    -- logged, and with the agreement number still reported as measured before the fix.
    """
    out: list[Disagreement] = []
    for row in judged:
        item = sample[row.id]
        theirs = first_annotator(item)
        for field in FIELDS:
            if theirs[field] != row.labels[field]:
                out.append(
                    Disagreement(
                        row.id,
                        field,
                        theirs[field],
                        row.labels[field],
                        row.rule,
                        row.note,
                        cell(item.question),
                    )
                )
    return out


# --- the report -----------------------------------------------------------------------------------

def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub markdown table. Cells are written as given -- the caller does its own quoting.

    Local rather than imported from :mod:`eval.scripts.day2_smoke`, which has the same six lines:
    that module imports the agent graph, and ``langgraph`` with it, and nothing here needs an
    agent. A report generator that cannot run without the retrieval stack installed is a report
    generator that stops being runnable the first time a dependency moves.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def kappa_cell(kappa: Kappa) -> str:
    """κ as a table cell: the number, signed, or an em dash where there is no number."""
    return MISSING if kappa.value is None else f"{kappa.value:+.3f}"


def render_header(meta: Mapping[str, Any], judged: Sequence[Filled], sheet: Path) -> str:
    """What was measured, over what, from which seed, against which version of the guidelines."""
    rows = [
        ["sampled from", f"`{meta['dataset']}` ({meta['dataset_rows']} rows)"],
        ["dataset sha256", f"`{meta['dataset_sha256'][:16]}…`"],
        ["seed", f"`{meta['seed']}`"],
        # In bucket order, not the sidecar's: ``write_json`` sorts keys, and an allocation row
        # ordered differently from the "drawn" row below it invites a reader to compare the two
        # cell by cell and find a mismatch that is not there.
        [
            "allocation",
            ", ".join(f"{b} {meta['allocation'][b]}" for b in BUCKETS if b in meta["allocation"]),
        ],
        ["drawn", ", ".join(f"{d['got']} {d['bucket']}" for d in meta["draws"])],
        ["sheet", f"`{rel_to_repo(sheet)}`"],
        ["guidelines sha256", f"`{meta['guidelines_sha256'][:16]}…`"],
    ]
    return "\n".join(
        [
            "# Day 3 — inter-annotator agreement",
            "",
            (
                f"Config hash `{meta['config_hash']}`. {plural(len(judged), 'row')} judged by "
                "a second annotator against `eval/GUIDELINES.md` and nothing else."
            ),
            "",
            md_table(["", ""], rows),
        ]
    )


#: Under the numbers table. Says what ``p_e`` is and why κ has no interval, because a reader who
#: sees one interval and not the other will otherwise assume the second was forgotten.
NUMBERS_NOTE = """\
`p_e` is the chance agreement κ divides out. The interval is on the percent agreement only: κ's
asymptotic standard error is not trustworthy at n={n}, and an interval that is wrong is worse than
an absent one.
"""


def render_numbers(scores: Sequence[Agreement], n: int) -> str:
    """The table #60 asks for: percent agreement **and** Cohen's κ, per field.

    Three fields and two further lines -- ``should_abstain``, which is the half of the bucket call
    the candidate answer gives away, and ``citations_support`` over the cited rows only. Both are
    marked, and both are there because the plain field-level number above them cannot be read
    without them.
    """
    rows = [
        [
            score.heading,
            f"{score.agreed}/{score.n}",
            score.pct,
            score.interval,
            kappa_cell(score.kappa),
            f"{score.kappa.p_e:.3f}",
        ]
        for score in scores
    ]
    notes = [f"- {score.heading} — κ {score.kappa.note}." for score in scores if score.kappa.note]
    out = [
        "## Agreement",
        "",
        md_table(["field", "agreed", "percent", "95% CI (Wilson)", "Cohen's κ", "p_e"], rows),
        "",
        NUMBERS_NOTE.format(n=n).rstrip(),
    ]
    if notes:
        out += ["", "**Where κ does not mean what it usually means:**", "", *notes]
    return "\n".join(out)


#: Under the confusion matrix. The single most misreadable number in this report is a four-way
#: bucket agreement that was not four-way blind, and this is the paragraph that says so.
CONFUSION_NOTE = """\
**Read this as two 2×2 blocks, not one 4×4.** The sheet strips `bucket` and `should_abstain`, but
it has to show the candidate answer — field 2 is *is this answer right* — and a refusal is visibly
a refusal. So the abstain/non-abstain split was largely given away, and the two off-diagonal
blocks (`fact`/`multihop` against `unanswerable`/`ambiguous`) should be near-empty for that reason
rather than because the guidelines are clear. The measurement lives in the two on-diagonal blocks:
`fact` against `multihop`, and `unanswerable` against `ambiguous` — which are also the two calls
`GUIDELINES.md` spends its boundary cases on (**F1**/**M1**, **A2**/**U1**).
"""


def render_confusion(pairs: Sequence[tuple[str, str]]) -> str:
    """The bucket confusion matrix, and what the candidate answers already gave away."""
    matrix = confusion(pairs, BUCKETS)
    rows = [
        [f"**{bucket}**", *(str(count) for count in row)]
        for bucket, row in zip(BUCKETS, matrix, strict=True)
    ]
    return "\n".join(
        [
            "## Bucket confusion",
            "",
            "Rows = annotator 1 (`v1.jsonl`), columns = annotator 2.",
            "",
            md_table(["1 ↓ / 2 →", *BUCKETS], rows),
            "",
            CONFUSION_NOTE.rstrip(),
        ]
    )


#: What "no disagreements" is worth at this sample size. Not nothing, and not what it looks like.
NO_DISAGREEMENTS = """\
## Disagreements

None. At n={n} that is a weaker result than it looks — see the interval above — and two of the
three fields cannot disagree in the direction that would be informative: annotator 1 accepted
every row in this sample by construction.
"""

#: Above the disagreement table. The sentence that matters is the last one: the number is reported
#: as measured, before any row it caused to be fixed.
DISAGREEMENTS_LEAD = """\
{found} over {rows}. This list is the revision plan for `GUIDELINES.md` v2. Where one shows the
**first** annotator was wrong, the fix goes into `v1.jsonl` through #59's log — and the number
above stays as it was measured, before the fix.
"""

#: Printed under a disagreement whose ``rule`` cell is empty.
NO_RULE = (
    "No rule cited — **attribute one by hand before #62**: a disagreement whose guideline rule is "
    "unidentified cannot be fixed by editing the guidelines."
)


def render_disagreements(found: Sequence[Disagreement], n: int) -> str:
    """Every disagreement, itemised, with the rule the second annotator says they applied."""
    if not found:
        return NO_DISAGREEMENTS.format(n=n).rstrip()
    rows = [
        [
            f"`{d.id}`",
            f"`{d.field}`",
            f"`{d.first}`",
            f"`{d.second}`",
            f"`{d.rule}`" if d.rule else MISSING,
            d.note or MISSING,
        ]
        for d in found
    ]
    out = [
        "## Disagreements",
        "",
        DISAGREEMENTS_LEAD.format(
            found=plural(len(found), "disagreement"),
            rows=plural(len({d.id for d in found}), "row"),
        ).rstrip(),
        "",
        md_table(["id", "field", "annotator 1", "annotator 2", "rule cited", "note"], rows),
    ]
    for d in found:
        out += [
            "",
            f"### `{d.id}` — `{d.field}`",
            "",
            f"**Q.** {d.question}",
            "",
            f"Annotator 1 says `{d.first}`, annotator 2 says `{d.second}`.",
            "",
            f"Rule cited: `{d.rule}`." if d.rule else NO_RULE,
        ]
        if d.note:
            out += ["", f"> {d.note}"]
    return "\n".join(out)


#: The section the report is least allowed to skip. Every number above is qualified here, and the
#: two qualifications that matter most are not about the sample size: κ on two of the three fields
#: is an artifact of scoring against a set that was filtered on the field being scored, and the
#: bucket call was never fully blind. Written as one block so it reads as a paragraph a person
#: wrote, rather than as bullets a generator emitted.
CAVEATS = """\
## What this does and does not measure

**n={n}.** A fifteen-item agreement estimate is wide, and wider than the usual rule of thumb
suggests. 14 of 15 is 93.3% with a Wilson interval of **[70%, 99%]** — 23 points of room below the
estimate, not the ±13 a normal approximation gives at that count (and the ±25 quoted for n=15 is
the p̂ = 0.5 worst case). 13 of 15 reaches down to 62%. So this measurement catches a guidelines
document that does not work *at all*; it does not distinguish 85% from 95%, and no point estimate
from the table above should be quoted without its interval beside it.

**κ on `answer_correct` and `citations_support` is not a reliability number, for two different
reasons.** `v1.jsonl` holds only rows the first annotator accepted or edited — that is what
promotion means — so the fields being scored are the fields the set was filtered on.

On `answer_correct` their label is constant `yes`, `p_e` collapses onto `p_o`, and κ is exactly 0
or undefined however well the two annotators agree. On `citations_support` their label is not
constant but it is mechanical: `n/a` exactly where the row carries no gold ids, which the second
annotator reads off the same empty list. Those rows agree by construction and carry the
field-level number up with them, which is why the table also gives `citations_support` over the
cited rows only — and on that subset annotator 1 is a constant `yes` again.

What the percent agreement on both fields estimates is **residual label error in the frozen set**:
how often a second reader would reject a row Day 4 is about to score. That is worth knowing, and
it is a different quantity from inter-annotator reliability. A reliability κ on these fields needs
the sample drawn from `eval/datasets/drafts/` before promotion, where annotator 1's verdicts vary.

**The bucket number is narrower than four-way.** See the note under the confusion matrix: the
candidate answer's wording carries `should_abstain`, so the blind part of the bucket call is the
two within-pair distinctions.

**The two-pass order is an honour system.** Nothing here can prove `agreement_citations.md` was
opened after the sheet's first two columns were filled and saved. The `note` column is where a
changed pass-1 answer is recorded, and a sheet that records none may simply mean there were none.

**One annotator, one sitting.** {rows} read by one other person is a reliability estimate for
*this pair of readers*, not for the guidelines in general. Three annotators and Fleiss' κ would be
the honest version of this measurement; that is not what #60 asks for, and this number should not
be quoted as if it were.

**The guidelines did not move during annotation.** Questions in `agreement_questions.md` are
answered by amending `eval/GUIDELINES.md` afterwards. The page this was measured against is
sha256 `{guidelines}…`; if #62 reports a different one, the amendment happened after the
measurement, which is the intended order.
"""

#: The two commands, and what is and is not a function of what. No timestamp anywhere in the
#: report, so a diff under ``report/`` means the sheet or the dataset moved.
REPRODUCTION = """\
## Reproducing this

```bash
uv run python -m eval.scripts.agreement --seed {seed} --force   # redraws the sample
uv run python -m eval.scripts.agreement --score
```

The sample is a function of the seed, the id set and the allocation alone — per-bucket streams over
sorted ids, so adding or deleting a row in one bucket does not move another bucket's draw. The
report carries no timestamp: nothing in it changes unless the sheet or the dataset did.
"""


def render_report(
    meta: Mapping[str, Any],
    judged: Sequence[Filled],
    sample: Mapping[str, EvalItem],
    sheet: Path,
) -> str:
    """The whole report, in one string."""
    scores = [score_field(pairs_for(field, judged, sample), field) for field in FIELDS]
    scores.append(
        score_field(derived_pairs(judged, sample), DERIVED, "`should_abstain` (derived)")
    )
    cited = cited_pairs(judged, sample)
    if cited:
        scores.append(
            score_field(cited, "citations_support", "`citations_support` (cited rows only)")
        )
    sections = [
        render_header(meta, judged, sheet),
        render_numbers(scores, len(judged)),
        render_confusion(pairs_for("bucket", judged, sample)),
        render_disagreements(disagreements(judged, sample), len(judged)),
        CAVEATS.format(
            n=len(judged),
            rows=plural(len(judged), "row"),
            guidelines=meta["guidelines_sha256"][:16],
        ).rstrip(),
        REPRODUCTION.format(seed=meta["seed"]).rstrip(),
    ]
    return "\n\n".join(sections) + "\n"


# --- the commands ---------------------------------------------------------------------------------

#: The annotator's whole instruction, hashed into the sample's sidecar and into the report.
#: Amending it after the measurement is the intended order (``GUIDELINES.md`` -> *Amending this
#: file*); the hash is what makes "after" checkable rather than asserted.
GUIDELINES_PATH = REPO_ROOT / "eval" / "GUIDELINES.md"


def sample_meta(
    sample: Sample, cfg: Config, dataset: Path, rows: Sequence[EvalItem]
) -> dict[str, Any]:
    """The sidecar: everything the draw was a function of, and what it drew."""
    return {
        "allocation": dict(ALLOCATION),
        "config_hash": cfg.hash,
        "dataset": rel_to_repo(dataset),
        "dataset_rows": len(rows),
        "dataset_sha256": file_sha256(dataset),
        "draws": [
            {"bucket": draw.bucket, "wanted": draw.wanted, "got": draw.got, "ids": list(draw.ids)}
            for draw in sample.draws
        ],
        "guidelines_sha256": file_sha256(GUIDELINES_PATH),
        "sample_n": len(sample.rows),
        "seed": sample.seed,
        "target_n": SAMPLE_N,
    }


def export(
    cfg: Config,
    dataset: Path,
    seed: int,
    pack: Packet,
    force: bool = False,
    allow_short: bool = False,
    texts: Mapping[str, str] | None = None,
    out: TextIO = sys.stdout,
) -> int:
    """Draw the sample and write the packet a second person fills in.

    Refuses to overwrite an existing packet without ``--force``, and refuses even with it while a
    filled sheet is sitting there. Re-drawing the sample under a returned sheet leaves a file of
    judgements about rows that are no longer the sample -- which scores as a pile of unknown ids
    if you are lucky, and as a wrong number if the ids happen to overlap.
    """
    rows = load_dataset(dataset)
    sample = stratified_sample(rows, seed)
    if pack.filled.exists():
        out.write(
            f"{rel_to_repo(pack.filled)} exists -- a sheet has already been filled in against "
            "the current sample. Score it, or move it aside deliberately; this command will not "
            "re-draw the sample out from under it.\n"
        )
        return 2
    existing = [p for p in pack.written if p.exists()]
    if existing and not force:
        names = ", ".join(rel_to_repo(p) for p in existing)
        out.write(f"{names} already exist. Pass --force to overwrite.\n")
        return 2
    out.writelines(
        f"warning: {draw.bucket} wanted {draw.wanted} rows and the dataset has {draw.got}\n"
        for draw in sample.short
    )
    if len(sample.rows) < SAMPLE_N and not allow_short:
        out.write(
            f"drew {len(sample.rows)} rows, not {SAMPLE_N}. The sample size is what the interval "
            "in the report is computed from, so a short draw is a decision and not a default: "
            "finish the dataset, or pass --allow-short and the report will say what it drew.\n"
        )
        return 1
    write_sample(pack.sample, sample.rows)
    write_json(pack.meta, sample_meta(sample, cfg, dataset, rows))
    write_sheet(pack.sheet, sample.rows)
    body = render_citations(sample, corpus_text() if texts is None else texts, pack)
    pack.citations.write_text(body + "\n", encoding="utf-8")
    pack.instructions.write_text(render_instructions(sample, dataset, pack), encoding="utf-8")
    if not pack.questions.exists():
        pack.questions.write_text(QUESTIONS_STUB, encoding="utf-8")
    drawn = ", ".join(f"{draw.got} {draw.bucket}" for draw in sample.draws)
    out.write(
        f"{plural(len(sample.rows), 'row')} ({drawn}) at seed {sample.seed}\n"
        + "".join(f"  {rel_to_repo(p)}\n" for p in (*pack.written, pack.questions))
        + f"\nHand over {rel_to_repo(pack.instructions)}. The filled sheet comes back as "
        f"{rel_to_repo(pack.filled)}, and --score reads it.\n"
    )
    return 0


def score(
    cfg: Config,
    pack: Packet,
    sheet: Path | None = None,
    dry_run: bool = False,
    report: Path | None = None,
    out: TextIO = sys.stdout,
) -> int:
    """Read the returned sheet and write ``report/day3_agreement_{hash}.md``.

    Every fault in the sheet is reported and nothing is written. This is the one place where
    :mod:`eval.scripts.day2_smoke`'s rule -- write the report even when the run failed, because
    the report is the evidence -- does not apply: a sheet with three unreadable cells produces a
    percent agreement over twelve rows, and a twelve-row number in a file named ``day3_agreement``
    is worse than no file at all.

    The first annotator's labels come from the sample, not from ``v1.jsonl``, so that the score is
    against the rows as they were when the sheet went out. A row edited in the meantime is a
    finding for #62, not a silent change of denominator.
    """
    sheet = pack.filled if sheet is None else Path(sheet)
    if not pack.sample.exists():
        out.write(f"no sample at {rel_to_repo(pack.sample)}; run the export first.\n")
        return 2
    sample = {item.id: item for item in load_dataset(pack.sample)}
    meta = json.loads(pack.meta.read_text(encoding="utf-8"))
    judged, problems = read_sheet(sheet, tuple(sample))
    if problems:
        out.write(f"{rel_to_repo(sheet)}: {plural(len(problems), 'problem')}\n")
        for problem in problems:
            where = f"line {problem.line}" if problem.line else "sample"
            parts = [where, problem.id, problem.column]
            out.write("  " + " ".join(p for p in parts if p) + f": {problem.message}\n")
        out.write("\nNothing written: the sheet is the agreement number's denominator.\n")
        return 1
    body = render_report(meta, judged, sample, sheet)
    if dry_run:
        out.write(body)
        return 0
    written = report_path(STEM, cfg) if report is None else Path(report)
    written.write_text(body, encoding="utf-8")
    bucket = score_field(pairs_for("bucket", judged, sample), "bucket")
    out.write(
        f"{rel_to_repo(written)}\n"
        f"bucket: {bucket.agreed}/{bucket.n} ({bucket.pct}, 95% CI {bucket.interval}), "
        f"kappa {kappa_cell(bucket.kappa)}\n"
        f"{plural(len(disagreements(judged, sample)), 'disagreement')}\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.agreement",
        description="Draw a seeded, stratified 15-row sample for a second annotator, and score "
        "the sheet they return: percent agreement and Cohen's kappa per field.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="score the returned sheet instead of exporting a new packet",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="the promoted dataset to sample (default: eval/datasets/v1.jsonl)",
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=None,
        help="the filled sheet to score (default: eval/datasets/agreement_sheet_filled.tsv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="sampling seed; defaults to config.yaml's, and is recorded in the sidecar either way",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing packet")
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help=f"export fewer than {SAMPLE_N} rows when a bucket cannot fill its share",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="with --score: print the report, write nothing"
    )
    args = parser.parse_args(argv)
    cfg = load_config()
    pack = packet()
    try:
        if args.score:
            return score(cfg, pack, args.sheet, args.dry_run, sys.stdout)
        seed = cfg["seed"] if args.seed is None else args.seed
        return export(cfg, args.dataset, seed, pack, args.force, args.allow_short, sys.stdout)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
