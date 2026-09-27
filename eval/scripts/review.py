"""The verification pass: every drafted row read by a person, three verdicts, and a log.

    uv run python -m eval.scripts.review              # the pass, resumable
    uv run python -m eval.scripts.review --stats      # what has been judged, per bucket
    uv run python -m eval.scripts.review --promote    # accepted + edited -> eval/datasets/v1.jsonl

This is the only part of Day 3 that needs a person, so the whole design question is what the
person's minute is spent on. Everything decidable by machine has already run -- #54's schema and
validator over every drafted row, #56's id and overlap checks, #57's and #58's corpus checks --
and what is left is the three questions in #59 that no check can answer: is the ``gold_answer``
right, do the ``gold_doc_ids`` actually support it, and is the bucket label right against
``eval/GUIDELINES.md``.

**The chunk text is on the screen, pulled live, or check 2 does not happen.** A reviewer who has
to open another window to read a cited chunk approves citations by vibe at item 40, and a row
whose answer is right and whose citation is wrong is worse than a wrong row: it silently rewards
a system that retrieved the wrong document. So every cited chunk is printed in full, out of
``data/chunks.jsonl`` -- or rebuilt through :func:`ingest.chunk.render_chunk` from the committed
``data/courses.json`` when the corpus has not been built, the same fallback :mod:`eval.validate`
uses and for the same reason: the machine that has not run the pipeline yet is the machine this
is for.

**The log is the dataset's provenance, not a diary.** ``eval/datasets/review_log.jsonl`` is
append-only, one entry per changed field, and ``v1.jsonl`` is a *function* of the draft files and
that log: :func:`promote` replays it. Nothing else writes the dataset. The alternative -- editing
rows into ``v1.jsonl`` directly and keeping a log beside it -- gives you two files that can
disagree, and the one that would be wrong is the one that documents the work. Here a log that
does not reproduce the dataset is a failure the promote step raises, since every edit records the
value it replaced and :func:`apply_edits` checks it against the draft still on disk.

**Latest verdict wins, and a re-review starts from what the row currently is.** Resumption is by
``id``: a row with a verdict is not shown again unless ``--redo`` asks for it. When it is, the
screen shows the row with its edits already applied, and saving records the whole diff against
the draft again -- so one row has one verdict, however many sittings it took, and the log stays
replayable from the drafts alone.

**Ids are not renumbered.** #56's blocks already keep the four draft files from colliding, so
promotion has nothing to fix, and ``v1-1001`` in ``v1.jsonl`` is the same row as ``v1-1001`` in
the drafts and in the log. Renumbering would buy a contiguous file and cost every join between
the three.

**The queue interleaves buckets.** Fifty ``fact`` rows in a row is how a reviewer stops reading
the chunk text, so the order deals one row from each bucket in turn, ids ascending inside a
bucket -- deterministic, so two people reviewing the same drafts see the same order, and a
resumed sitting picks up where the last one stopped.

**This is the one artifact in the repo that carries a clock.** Everything else (#39's report,
#56's drafts) moves only when the run moved, because it is regenerable. A review log is not: it
is the record of when a person judged what, and #59 wants the timestamps precisely so that a
reader can check for a cluster of late-session accepts rather than take the pass on trust.

**The flags are prompts, never verdicts.** :func:`flags` marks a row that trips something in the
guidelines -- a credit question, more than four gold ids, #56's top-decile overlap, a gold id the
validator rejects. Each one names the rule and stops there. A flag that resolved itself would be
the machine judging the row, which is the thing this pass exists to not do.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, TextIO

from pydantic import ValidationError

from config import REPO_ROOT, load_config, rel_to_repo
from eval.schema import BUCKETS, RULES, Bucket, EvalItem
from eval.scripts import build_multihop, build_unanswerable, draft_items
from eval.scripts.draft_items import DRAFTS_DIR, credit_question, parse_drafts
from eval.validate import Row, Violation, check_row, corpus_doc_ids, render, validate_rows
from ingest.chunk import CHUNKS_PATH, load_chunks, load_courses, render_chunk
from ingest.export_almabot import COURSES_PATH
from ingest.tokenize import CODE_RE, get_tokenizer

#: The log of what a person did, one entry per changed field. Committed: it is the evidence that
#: the verification happened, and #62 reports its counts.
REVIEW_LOG_PATH = REPO_ROOT / "eval" / "datasets" / "review_log.jsonl"

#: What the pass promotes into. Not in ``drafts/``: this is the artifact Day 4 and Day 5 read.
DATASET_PATH = REPO_ROOT / "eval" / "datasets" / "v1.jsonl"

#: The three. ``edit`` promotes the row too -- it is an accept with changes, and the changes are
#: what the log is for.
Verdicts = Literal["accept", "edit", "delete"]
VERDICT_NAMES: tuple[Verdicts, ...] = ("accept", "edit", "delete")

#: Verdicts whose rows reach ``v1.jsonl``.
PROMOTED: frozenset[str] = frozenset({"accept", "edit"})

#: Every key a log entry carries, in #59's order. ``field_changed``, ``before`` and ``after`` are
#: ``null`` on an accept and on a delete: nothing changed, and a log that omitted the keys
#: instead would make "no field changed" and "an old log" the same shape.
LOG_KEYS: tuple[str, ...] = ("id", "verdict", "reason", "field_changed", "before", "after", "ts")

#: What an edit may change. ``id`` is not here -- see the module docstring on renumbering -- and
#: neither is ``provenance``: a draft's record of how it was made is not the reviewer's to
#: rewrite, and what the reviewer did to it is in the log instead.
EDITABLE: tuple[str, ...] = (
    "question",
    "gold_answer",
    "gold_doc_ids",
    "bucket",
    "should_abstain",
)

#: Which provenance set a draft file is held to, longest name prefix first. The three builders
#: record three different things -- a batch and a model (#56), a failure shape and a gate reading
#: (#57), a walk over the graph (#58) -- and a reader that accepted any of them over all three
#: files would accept a file that recorded none of them.
PROVENANCE_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multihop_graph_", tuple(build_multihop.PROVENANCE_KEYS)),
    ("unanswerable_", tuple(build_unanswerable.PROVENANCE_KEYS)),
    ("", tuple(draft_items.PROVENANCE_KEYS)),
)

#: Past this many cited records a row is a list question rather than a join --
#: ``eval/GUIDELINES.md``, "Which bucket". A flag, not a check: the validator has no ceiling on
#: a ``multihop`` row's gold ids.
JOIN_CAP = 4

#: Screen width for the rules between rows. Nothing wraps to it; it is a separator.
RULE = "-" * 96


# --- the queue ------------------------------------------------------------------------------------


def find_drafts(drafts_dir: Path | None = None) -> list[Path]:
    """Every draft file the pass reviews, in name order.

    ``.standin.jsonl`` is skipped. #56 writes canned rows there precisely so that nothing
    downstream reads them, and a verification pass that promoted one would put a row no model
    ever drafted into the dataset under the same six fields as the rest.
    """
    directory = DRAFTS_DIR if drafts_dir is None else Path(drafts_dir)
    if not directory.exists():
        raise FileNotFoundError(f"{rel_to_repo(directory)} does not exist")
    return sorted(p for p in directory.glob("*.jsonl") if not p.name.endswith(".standin.jsonl"))


def provenance_keys(path: Path) -> tuple[str, ...]:
    """The provenance set ``path`` is held to, by its name's prefix. See :data:`PROVENANCE_SETS`."""
    name = Path(path).name
    for prefix, keys in PROVENANCE_SETS:
        if name.startswith(prefix):
            return keys
    raise ValueError(f"no provenance set for {name!r}")  # pragma: no cover - "" matches everything


class Candidate(NamedTuple):
    """One drafted row waiting for a verdict, and where it was read from."""

    source: Path
    line: int
    item: EvalItem
    provenance: dict[str, Any]

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def where(self) -> str:
        return f"{rel_to_repo(self.source)}:{self.line}"


class Found(NamedTuple):
    """A violation and the draft file it is in -- :class:`eval.validate.Violation` has no file."""

    path: Path
    violation: Violation


def load_candidates(paths: Sequence[Path]) -> tuple[list[Candidate], list[Found]]:
    """Read every draft file into candidates, reporting every line that is not a row.

    Rows that do not parse are dropped rather than shown: a reviewer cannot judge a row whose
    ``bucket`` is missing, and the drafting scripts already refuse to write one. Reaching here is
    a file edited by hand, which is why the caller prints these and stops.
    """
    candidates: list[Candidate] = []
    found: list[Found] = []
    for path in paths:
        drafts, violations = parse_drafts(path, provenance_keys(path))
        candidates.extend(Candidate(path, d.line, d.item, d.provenance) for d in drafts)
        found.extend(Found(path, v) for v in violations)
    return candidates, found


def duplicate_ids(candidates: Sequence[Candidate]) -> list[str]:
    """Ids that appear in more than one draft file, which promotion cannot merge.

    #56's blocks make this impossible by construction; it is checked because the alternative to
    checking is a ``v1.jsonl`` with two rows under one id, and #61's split would carry whichever
    one it read last.
    """
    counts = Counter(c.id for c in candidates)
    return sorted(item_id for item_id, n in counts.items() if n > 1)


def interleave(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Round-robin over the buckets, ids ascending inside each. See the module docstring.

    Buckets are dealt in :data:`eval.schema.BUCKETS` order, so the sequence is a function of the
    drafts alone: the same files give the same queue on two machines, and #60's second annotator
    sees the rows in the order the first one did.
    """
    by_bucket: dict[Bucket, list[Candidate]] = {bucket: [] for bucket in BUCKETS}
    for candidate in sorted(candidates, key=lambda c: c.id):
        by_bucket[candidate.item.bucket].append(candidate)
    queue: list[Candidate] = []
    while any(by_bucket.values()):
        for bucket in BUCKETS:
            if by_bucket[bucket]:
                queue.append(by_bucket[bucket].pop(0))
    return queue


# --- the corpus on screen -------------------------------------------------------------------------

_TEXTS: dict[tuple[str, str], Mapping[str, str]] = {}


def corpus_text(
    chunks_path: Path | None = None, courses_path: Path | None = None
) -> Mapping[str, str]:
    """``doc_id`` to the full chunk text, for the cited chunks on screen.

    The twin of :func:`eval.validate.corpus_doc_ids`, with the same fallback and the same
    caching: ``data/chunks.jsonl`` when it is there, and otherwise the text re-rendered from the
    committed ``data/courses.json`` through the one function that built it. A corpus named
    outright is used as given and its absence is an error.
    """
    named = chunks_path is not None
    chunks = CHUNKS_PATH if chunks_path is None else Path(chunks_path)
    courses = COURSES_PATH if courses_path is None else Path(courses_path)
    key = (str(chunks), str(courses))
    if key not in _TEXTS:
        if named or chunks.exists():
            _TEXTS[key] = {row["doc_id"]: row["text"] for row in load_chunks(chunks)}
        else:
            template = load_config()["chunk"]["template_version"]
            loaded = load_courses(courses)
            _TEXTS[key] = {
                f"{c['term']}:{c['subject']}-{c['number']}": render_chunk(c, template)
                for c in loaded
            }
    return _TEXTS[key]


def clear_cache() -> None:
    """Drop the cached corpus text -- for a test that moves the corpus under it."""
    _TEXTS.clear()


def credit_ranges(courses_path: Path | None = None) -> frozenset[str] | None:
    """Codes the catalog gives a credit *range* for, or ``None`` where the export cannot say.

    ``eval/GUIDELINES.md`` deletes any credit row about one of these: the chunk prints
    ``Credits:`` as a single number even where the catalog says "3 OR 4 hours.", so the answer is
    wrong about either the catalog or the corpus. 1,415 of 4,502 courses are affected.

    The fix for the export is #17, which is on ``main`` and not in this stack, so on this branch
    ``data/courses.json`` carries ``credits`` and not ``creditsMin``/``creditsMax`` and the check
    cannot run. Returning ``None`` -- rather than an empty set, which would read as "no course is
    affected" and silently pass 1,415 of them -- is what makes :func:`flags` fall back to naming
    the rule and the command instead of answering it.
    """
    courses = COURSES_PATH if courses_path is None else Path(courses_path)
    loaded = load_courses(courses)
    if not loaded or "creditsMin" not in loaded[0]:
        return None
    return frozenset(c["code"] for c in loaded if c["creditsMin"] != c["creditsMax"])


# --- the log --------------------------------------------------------------------------------------


class Edit(NamedTuple):
    """One field a verdict changed, and the draft value it replaced.

    ``before`` is always the value in the *draft* file, never an intermediate one, so replaying
    a verdict needs only the drafts and the latest verdict. See the module docstring.
    """

    field: str
    before: Any
    after: Any


class Verdict(NamedTuple):
    """What a person decided about one row, in one sitting."""

    id: str
    verdict: Verdicts
    #: One word for an edit, a sentence for a delete, ``None`` for an accept.
    reason: str | None
    #: UTC, ISO 8601 with a ``Z``. Also the key that groups the entries of one verdict.
    ts: str
    edits: tuple[Edit, ...] = ()

    def entries(self) -> list[dict[str, Any]]:
        """This verdict as log lines: one per changed field, or one with the fields null.

        An edit that touched three fields is three lines sharing a ``ts``, because #59 fixes the
        log's shape at seven scalar keys and a ``field_changed`` holding a list would make every
        reader of the file special-case it. The shared ``ts`` is what reassembles them.
        """
        common = {"id": self.id, "verdict": self.verdict, "reason": self.reason, "ts": self.ts}
        if not self.edits:
            return [{**common, "field_changed": None, "before": None, "after": None}]
        return [
            {**common, "field_changed": e.field, "before": e.before, "after": e.after}
            for e in self.edits
        ]


def now() -> str:
    """The timestamp a verdict carries. Seam for the tests, which must not race a clock."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """The log as written, in file order. A missing log is an empty pass, not an error."""
    path = REVIEW_LOG_PATH if path is None else Path(path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line, text in enumerate(f, start=1):
            if not text.strip():
                continue
            entry = json.loads(text)
            missing = [key for key in LOG_KEYS if key not in entry]
            if missing:
                raise ValueError(
                    f"{rel_to_repo(path)}:{line} is missing {', '.join(missing)}; a review log "
                    "entry carries all seven keys"
                )
            entries.append(entry)
    return entries


def append_verdict(path: Path, verdict: Verdict) -> None:
    """Add one verdict's entries to the log and fsync, before the next row is shown.

    Flushed per verdict rather than per sitting for the reason #59 gives for resumability at
    all: losing an hour of judgements to a closed terminal is the way this task gets abandoned.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in verdict.entries():
            f.write(json.dumps({key: entry[key] for key in LOG_KEYS}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def replay(entries: Iterable[dict[str, Any]]) -> dict[str, Verdict]:
    """The log as one verdict per row: entries sharing an ``id`` and ``ts`` are one judgement.

    File order is judgement order -- the log is append-only -- so a later verdict on a row
    replaces an earlier one outright rather than merging with it. That is what makes "one row,
    one verdict" true of a dataset built from a log that may hold three entries for a row.
    """
    latest: dict[str, Verdict] = {}
    for entry in entries:
        item_id, ts = entry["id"], entry["ts"]
        seen = latest.get(item_id)
        if seen is None or seen.ts != ts:
            seen = Verdict(item_id, entry["verdict"], entry["reason"], ts)
        if entry["field_changed"] is not None:
            edit = Edit(entry["field_changed"], entry["before"], entry["after"])
            seen = seen._replace(edits=(*seen.edits, edit))
        latest[item_id] = seen
    return latest


def read_verdicts(path: Path | None = None) -> dict[str, Verdict]:
    """:func:`read_entries` and :func:`replay`, which is how every caller wants the log."""
    return replay(read_entries(path))


def apply_edits(item: EvalItem, edits: Sequence[Edit]) -> EvalItem:
    """The drafted row with a verdict's edits applied, refusing one whose ``before`` has moved.

    The check is the point. ``v1.jsonl`` is the drafts plus this log and nothing else, so a
    draft file edited after it was reviewed must stop the promotion rather than quietly produce
    a row nobody judged -- the reviewer approved the old text, and the new text has never been
    read by anyone.
    """
    data = item.model_dump()
    for edit in edits:
        if edit.field not in EDITABLE:
            raise ValueError(
                f"{item.id}: the log edits {edit.field!r}, which is not one of "
                f"{', '.join(EDITABLE)}"
            )
        if data[edit.field] != edit.before:
            raise ValueError(
                f"{item.id}: the log replaced {edit.field} {edit.before!r}, but the draft now "
                f"reads {data[edit.field]!r}; the row was edited after it was reviewed and has "
                "to be re-reviewed"
            )
        data[edit.field] = edit.after
    return EvalItem.model_validate(data)


def diff_edits(draft: EvalItem, edited: EvalItem) -> tuple[Edit, ...]:
    """Every field of ``edited`` that differs from the draft, in :data:`EDITABLE` order."""
    before, after = draft.model_dump(), edited.model_dump()
    return tuple(
        Edit(field, before[field], after[field])
        for field in EDITABLE
        if before[field] != after[field]
    )


def current(candidate: Candidate, verdict: Verdict | None) -> EvalItem:
    """The row as it stands: the draft, plus the edits of a verdict it already has."""
    if verdict is None or not verdict.edits:
        return candidate.item
    return apply_edits(candidate.item, verdict.edits)


# --- the screen -----------------------------------------------------------------------------------

#: The three questions, on screen above every row, because they are what the pass is.
CHECKS = (
    "1. is the gold_answer right?  against the catalog, not against plausibility",
    "2. do the gold_doc_ids support it?  the cited chunk must contain the fact asserted",
    "3. is the bucket right?  eval/GUIDELINES.md, and its eight boundary cases",
)

#: The one-liner ``eval/GUIDELINES.md`` gives for the variable-credit courses, repeated on screen
#: when a credit question comes up and this branch's export cannot answer it. See
#: :func:`credit_ranges`.
CREDIT_COMMAND = (
    "git show origin/main:data/courses.json | jq -r "
    "'.[]|select(.creditsMin!=.creditsMax)|.code'"
)


def answer_codes(text: str) -> list[str]:
    """Course codes a ``gold_answer`` names, ``CS 225`` form, in order and without repeats."""
    seen: dict[str, None] = {}
    for match in CODE_RE.finditer(text):
        seen.setdefault(f"{match.group('subject').upper()} {match.group('number')}", None)
    return list(seen)


def flags(
    candidate: Candidate,
    item: EvalItem,
    known: frozenset[str],
    ranges: frozenset[str] | None,
    tokenizer: Callable[[str], list[str]],
) -> list[str]:
    """What this row trips, each naming the rule. Prompts for the reviewer, never verdicts.

    Everything here is either a validator error (which would stop the promotion anyway, so the
    reviewer should see it while the row is in front of them) or a rule in the guidelines that a
    machine can spot but not settle.
    """
    out = [
        f"{v.field or 'row'}: {v.message}"
        for v in check_row(Row(candidate.line, item), known)
    ]
    cited = len(item.gold_doc_ids)
    if cited > JOIN_CAP:
        out.append(
            f"{cited} cited records, past the cap of {JOIN_CAP} -- GUIDELINES, \"Which bucket\": "
            "more than that is a list question, so narrow it or drop it"
        )
    if credit_question(item.question, tokenizer):
        named = answer_codes(f"{item.question} {item.gold_answer}")
        if ranges is None:
            out.append(
                "a credit question -- GUIDELINES rule 2 deletes one about a variable-credit "
                "course, and this export cannot say which those are (#17). Check "
                f"{', '.join(named) or 'the course'} against:  {CREDIT_COMMAND}"
            )
        else:
            variable = [code for code in named if code in ranges]
            if variable:
                out.append(
                    f"{', '.join(variable)} has a credit range in the catalog and a single "
                    "number in the chunk -- GUIDELINES rule 2: delete this row, do not repair it"
                )
    if candidate.provenance.get("flagged"):
        out.append(
            "#56 put this question in the top decile of overlap with its own chunk: it may be "
            "retrievable by wording alone, so rewrite it or delete it"
        )
    return out


def indent(text: str, by: str = "      ") -> str:
    """Every line of ``text`` shifted right, so chunk text reads as quoted and not as output."""
    return "\n".join(by + line if line else line for line in text.splitlines())


def render_row(
    candidate: Candidate,
    item: EvalItem,
    texts: Mapping[str, str],
    marks: Sequence[str],
    position: int,
    total: int,
    verdict: Verdict | None = None,
) -> str:
    """One row, whole, on one screen: the question, the answer, the flags, every cited chunk."""
    spec = RULES[item.bucket]
    lines = [
        RULE,
        f"{item.id}  {item.bucket} -- {spec.gloss}"
        + ("  (edited)" if verdict and verdict.edits else ""),
        f"{candidate.where}   row {position} of {total}",
        "",
        f"Q  {item.question}",
        f"A  {item.gold_answer}",
        f"   should_abstain: {str(item.should_abstain).lower()}"
        + f"   gold_doc_ids: {len(item.gold_doc_ids)}",
    ]
    if verdict is not None:
        was = f"{verdict.verdict}" + (f" ({verdict.reason})" if verdict.reason else "")
        lines += ["", f"already judged {was} at {verdict.ts}"]
    lines += ["", *(f"   {check}" for check in CHECKS)]
    if marks:
        lines += ["", "flags"]
        lines += [f" !  {mark}" for mark in marks]
    note = candidate.provenance.get("note")
    shape = candidate.provenance.get("shape")
    if shape or note:
        lines += ["", f"drafted as  {shape or '?'}" + (f" -- {note}" if note else "")]
    lines += ["", "cited chunks"]
    if not item.gold_doc_ids:
        lines.append(
            f"      none, and {spec.gloss}. What to check is the claim of absence: open the "
            "catalog, not the agent."
        )
    for n, gold in enumerate(item.gold_doc_ids, start=1):
        text = texts.get(gold)
        lines.append(f"  [{n}] {gold}")
        lines.append(indent(text) if text is not None else "      -- not in the corpus --")
    named = [code for code in answer_codes(item.gold_answer) if item.gold_doc_ids]
    if named:
        cited_text = "\n".join(texts.get(gold, "") for gold in item.gold_doc_ids)
        missing = [code for code in named if code not in cited_text]
        shown = ", ".join(
            code + (" (in no cited chunk)" if code in missing else "") for code in named
        )
        lines += ["", f"   the answer names {shown}"]
    return "\n".join(lines)


# --- the three verdicts ---------------------------------------------------------------------------

PROMPT = "verdict?  [a]ccept  [e]dit  [d]elete  [q]uit > "

#: What to type for each verdict, and the word the log records. The full word is accepted too,
#: because a reviewer who types "delete" means it and should not be told to press "d".
KEYS: Mapping[str, Verdicts] = {
    "a": "accept",
    "accept": "accept",
    "e": "edit",
    "edit": "edit",
    "d": "delete",
    "delete": "delete",
}

Ask = Callable[[str], str | None]


def reader(out: TextIO, input_fn: Callable[[str], str]) -> Ask:
    """An ``ask`` that returns ``None`` at end of input, so a closed stdin quits rather than loops.

    The prompt is written to ``out`` rather than passed to ``input`` so that a transcript of a
    sitting -- which is what the tests read -- holds the question as well as the answer.
    """

    def ask(prompt: str) -> str | None:
        out.write(prompt)
        out.flush()
        try:
            return input_fn("")
        except EOFError:
            out.write("\n")
            return None

    return ask


def render_fields(data: Mapping[str, Any]) -> str:
    """The editable fields, numbered, with what they currently hold."""
    lines = ["", "which field?"]
    for n, field in enumerate(EDITABLE, start=1):
        value = data[field]
        shown = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        lines.append(f"  {n}. {field:<15} {shown}")
    return "\n".join(lines) + "\n"


def resolve_field(choice: str) -> str | None:
    """A field named or numbered, or ``None`` if it is neither."""
    choice = choice.strip()
    if choice in EDITABLE:
        return choice
    if choice.isdigit() and 1 <= int(choice) <= len(EDITABLE):
        return EDITABLE[int(choice) - 1]
    return None


def parse_value(field: str, text: str) -> Any:
    """What the reviewer typed, as the field's type. Raises :class:`ValueError` with the options.

    ``gold_doc_ids`` splits on commas and whitespace and is the one field where typing nothing
    means the empty list -- which is what an ``unanswerable`` row wants and what relabelling a
    row into that bucket needs.
    """
    text = text.strip()
    if field == "gold_doc_ids":
        return [part for part in text.replace(",", " ").split() if part]
    if field == "should_abstain":
        if text.lower() in {"true", "t", "yes", "y", "1"}:
            return True
        if text.lower() in {"false", "f", "no", "n", "0"}:
            return False
        raise ValueError("should_abstain is true or false")
    if field == "bucket":
        if text not in BUCKETS:
            raise ValueError(f"bucket is one of {', '.join(BUCKETS)}")
        return text
    return text


def edit_row(
    item: EvalItem, known: frozenset[str], line: int, out: TextIO, ask: Ask
) -> EvalItem | None:
    """Field by field until ``done``, re-checked before it is accepted. ``None`` if nothing changed.

    A row is not allowed out of here with a violation in it. Every check :func:`check_row` runs
    is fixable by editing -- a bucket that disagrees with ``should_abstain``, too few or too many
    gold ids, an id with a space in it -- and a reviewer who cannot fix the row has ``cancel``,
    which is how a row that should not be repaired gets deleted instead.
    """
    data = item.model_dump()
    while True:
        out.write(render_fields(data))
        answer = ask("field to change, or 'done' / 'cancel' > ")
        if answer is None or answer.strip() == "cancel":
            return None
        choice = answer.strip()
        if choice == "done":
            if data == item.model_dump():
                out.write("nothing changed.\n")
                return None
            try:
                edited = EvalItem.model_validate(data)
            except ValidationError as exc:
                out.write("\n".join(f"  {e['loc'][0]}: {e['msg']}" for e in exc.errors()) + "\n")
                continue
            violations = list(check_row(Row(line, edited), known))
            if violations:
                out.write(
                    "\n".join(f"  {v.field or 'row'}: {v.message}" for v in violations)
                    + "\nfix those, or cancel and delete the row instead.\n"
                )
                continue
            return edited
        field = resolve_field(choice)
        if field is None:
            out.write(f"no field {choice!r}; name one or give its number.\n")
            continue
        typed = ask(f"new {field} > ")
        if typed is None:
            return None
        try:
            data[field] = parse_value(field, typed)
        except ValueError as exc:
            out.write(f"  {exc}\n")


def one_word(text: str) -> str | None:
    """A reason for an edit: one word, so that #62 can count them. ``None`` if it is not one."""
    words = text.split()
    return words[0] if len(words) == 1 else None


def judge(
    candidate: Candidate,
    item: EvalItem,
    known: frozenset[str],
    out: TextIO,
    ask: Ask,
) -> Verdict | None:
    """One row's verdict, or ``None`` to stop. Prompts until it gets one of the three."""
    while True:
        answer = ask(PROMPT)
        if answer is None or answer.strip().lower() in {"q", "quit"}:
            return None
        verdict = KEYS.get(answer.strip().lower())
        if verdict is None:
            out.write("a, e, d or q.\n")
            continue
        if verdict == "accept":
            return Verdict(item.id, "accept", None, now())
        if verdict == "delete":
            reason = ask("why delete? > ")
            if reason is None:
                return None
            if not reason.strip():
                out.write("a delete needs a reason -- it is the one record the row leaves.\n")
                continue
            return Verdict(item.id, "delete", reason.strip(), now())
        edited = edit_row(item, known, candidate.line, out, ask)
        if edited is None:
            continue
        reason = ask("one-word reason for the edit > ")
        if reason is None:
            return None
        word = one_word(reason.strip())
        if word is None:
            out.write("one word, so the reasons can be counted. The row is unchanged.\n")
            continue
        return Verdict(item.id, "edit", word, now(), diff_edits(candidate.item, edited))


# --- the sitting ----------------------------------------------------------------------------------


def run(
    queue: Sequence[Candidate],
    known: frozenset[str],
    texts: Mapping[str, str],
    ranges: frozenset[str] | None,
    tokenizer: Callable[[str], list[str]],
    log_path: Path,
    judged: dict[str, Verdict],
    out: TextIO,
    ask: Ask,
    limit: int | None = None,
) -> Counter[str]:
    """Walk the queue, one row and one verdict at a time, writing each one before the next screen.

    Returns what this sitting judged. Stopping is not an error and not a state: the log holds
    every verdict already given, so the next run picks the queue up where this one left it.
    """
    made: Counter[str] = Counter()
    total = len(queue)
    for position, candidate in enumerate(queue, start=1):
        seen = judged.get(candidate.id)
        item = current(candidate, seen)
        marks = flags(candidate, item, known, ranges, tokenizer)
        out.write(render_row(candidate, item, texts, marks, position, total, seen) + "\n\n")
        verdict = judge(candidate, item, known, out, ask)
        if verdict is None:
            break
        append_verdict(log_path, verdict)
        judged[candidate.id] = verdict
        made[verdict.verdict] += 1
        if limit is not None and sum(made.values()) >= limit:
            out.write(f"\nstopping at {limit}, as asked.\n")
            break
    return made


# --- what the pass has done -----------------------------------------------------------------------


def tally(
    candidates: Sequence[Candidate], judged: Mapping[str, Verdict]
) -> dict[str, Counter[str]]:
    """Per bucket: how many rows were drafted, and what became of each."""
    counts = {bucket: Counter[str]() for bucket in BUCKETS}
    for candidate in candidates:
        bucket = candidate.item.bucket
        counts[bucket]["drafted"] += 1
        verdict = judged.get(candidate.id)
        counts[bucket][verdict.verdict if verdict else "unreviewed"] += 1
    return counts


def events(entries: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    """One ``(ts, id, verdict)`` per judgement, in log order -- entries of one edit collapse."""
    out: list[tuple[str, str, str]] = []
    for entry in entries:
        key = (entry["ts"], entry["id"], entry["verdict"])
        if not out or out[-1] != key:
            out.append(key)
    return out


def accept_rate(judgements: Sequence[tuple[str, str, str]]) -> float | None:
    """Share of ``judgements`` that were accepts, or ``None`` for none at all."""
    if not judgements:
        return None
    return sum(1 for _, _, verdict in judgements if verdict == "accept") / len(judgements)


def fatigue(judgements: Sequence[tuple[str, str, str]]) -> str:
    """Accept rate over the first half of the judgements against the second.

    #59 asks for the timestamps so a reader can check whether judgement degraded over a session
    rather than take it on trust, and a rising accept rate is what that looks like. It is a
    prompt to go and read the late rows, not a test: a queue that interleaves buckets can still
    deal an easy stretch, and two halves of 25 are not a measurement.
    """
    if len(judgements) < 4:
        return "too few judgements to say anything about fatigue."
    half = len(judgements) // 2
    first, second = accept_rate(judgements[:half]), accept_rate(judgements[half:])
    assert first is not None and second is not None
    drift = "  <- read the late rows" if second - first >= 0.2 else ""
    return (
        f"accept rate: {first:.0%} over the first {half} judgements, "
        f"{second:.0%} over the last {len(judgements) - half}{drift}"
    )


def stats(
    candidates: Sequence[Candidate], entries: Sequence[Mapping[str, Any]], log_path: Path
) -> str:
    """The table #62 reports: per bucket accepted, edited and deleted, and the top edit reasons.

    The most honest thing Day 3 produces, which is why it is printed rather than summarised: a
    bucket with a 70% edit rate tells Day 4 how much to trust the drafting pipeline it inherits,
    and hiding it makes the dataset look cleaner and the methodology look worse.
    """
    judged = replay(entries)
    counts = tally(candidates, judged)
    columns = ("drafted", "accept", "edit", "delete", "unreviewed")
    header = f"{'bucket':<14}" + "".join(f"{c:>12}" for c in columns)
    lines = [f"{rel_to_repo(log_path)}: {len(entries)} entries", "", header]
    totals: Counter[str] = Counter()
    for bucket in BUCKETS:
        row = counts[bucket]
        totals.update(row)
        lines.append(f"{bucket:<14}" + "".join(f"{row[c]:>12}" for c in columns))
    lines.append(f"{'total':<14}" + "".join(f"{totals[c]:>12}" for c in columns))

    reasons = Counter(
        verdict.reason
        for verdict in judged.values()
        if verdict.verdict == "edit" and verdict.reason
    )
    if reasons:
        lines += ["", "top edit reasons"]
        lines += [f"  {n:>3}  {reason}" for reason, n in reasons.most_common(10)]
    deletes = Counter(
        verdict.reason for verdict in judged.values() if verdict.verdict == "delete"
    )
    if deletes:
        lines += ["", "deletes"]
        lines += [f"  {n:>3}  {reason}" for reason, n in deletes.most_common(10)]

    judgements = events(entries)
    if judgements:
        stamps = sorted(ts for ts, _, _ in judgements)
        lines += ["", f"judged between {stamps[0]} and {stamps[-1]}", fatigue(judgements)]
    return "\n".join(lines)


# --- promotion ------------------------------------------------------------------------------------


class Promotion(NamedTuple):
    """What ``--promote`` found: the rows it would write, and every reason it may not."""

    rows: list[EvalItem]
    #: Drafted rows with no verdict. Any at all stops the write -- #59's first acceptance line is
    #: that no row reaches ``v1.jsonl`` unreviewed, and a partial dataset that looks whole is the
    #: one failure mode a reader cannot see.
    unreviewed: list[str]
    #: :func:`eval.validate.validate_rows` over the promoted rows. Errors stop the write.
    violations: list[Violation]
    #: Verdicts naming a row no draft file holds -- a draft deleted or renamed under the log.
    orphans: list[str]

    @property
    def ok(self) -> bool:
        return not self.unreviewed and not self.orphans and not any(
            v.level == "error" for v in self.violations
        )


def promote(
    candidates: Sequence[Candidate], judged: Mapping[str, Verdict], known: frozenset[str]
) -> Promotion:
    """The accepted and edited rows, re-validated, sorted by id -- the dataset as a function.

    Re-validated and not trusted: the rows were checked when they were drafted and again when
    they were edited, but what lands in ``v1.jsonl`` is the *composition* of a draft file and a
    log, and the composition is what Day 4 reads. A row is only as checked as the last thing
    that touched it.

    Raises :class:`ValueError` -- through :func:`apply_edits` -- when a draft moved under a
    verdict that edited it. That one is not a :class:`~eval.validate.Violation` because the row
    it would describe does not exist: there is no answering what the reviewer approved.
    """
    rows: list[EvalItem] = []
    unreviewed: list[str] = []
    for candidate in candidates:
        verdict = judged.get(candidate.id)
        if verdict is None:
            unreviewed.append(candidate.id)
        elif verdict.verdict in PROMOTED:
            rows.append(current(candidate, verdict))
    rows.sort(key=lambda item: item.id)
    drafted = {candidate.id for candidate in candidates}
    orphans = sorted(item_id for item_id in judged if item_id not in drafted)
    violations = validate_rows([Row(n, item) for n, item in enumerate(rows, start=1)], known)
    return Promotion(rows, sorted(unreviewed), violations, orphans)


def write_dataset(path: Path, rows: Sequence[EvalItem]) -> Path:
    """``v1.jsonl``: one object per line, schema field order, no trailing whitespace.

    Through a temporary file, like every other artifact here, so a crash cannot leave Day 4 half
    a dataset that still parses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in rows
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def promotion_report(promotion: Promotion, path: Path) -> str:
    """What the promote step did, or every reason it did not."""
    lines: list[str] = []
    if promotion.unreviewed:
        shown = ", ".join(promotion.unreviewed[:10])
        rest = len(promotion.unreviewed) - 10
        more = "" if rest <= 0 else f", and {rest} more"
        lines.append(
            f"{len(promotion.unreviewed)} drafted rows have no verdict ({shown}{more}). "
            "No row reaches the dataset unreviewed; run the pass to the end first."
        )
    if promotion.orphans:
        lines.append(
            f"the log judges {', '.join(promotion.orphans)}, which no draft file holds -- a "
            "draft was deleted or renamed after it was reviewed"
        )
    errors = [v for v in promotion.violations if v.level == "error"]
    if errors:
        lines.append(render(path, errors))
    warnings = [v for v in promotion.violations if v.level == "warning"]
    if warnings:
        lines.append(render(path, warnings))
    counts = Counter(item.bucket for item in promotion.rows)
    lines.append(
        f"{len(promotion.rows)} rows: "
        + ", ".join(f"{counts[bucket]} {bucket}" for bucket in BUCKETS)
    )
    return "\n".join(lines)


# --- the command ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.review",
        description="Read every drafted eval row, judge it, log what changed, and promote what "
        "survives into eval/datasets/v1.jsonl.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stats",
        action="store_true",
        help="print the per-bucket accept/edit/delete counts and the top edit reasons, and stop",
    )
    mode.add_argument(
        "--promote",
        action="store_true",
        help=f"replay the log over the drafts and write {rel_to_repo(DATASET_PATH)}; refuses "
        "while any drafted row is unreviewed or any promoted row fails eval.validate",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="show rows that already have a verdict too, with their edits applied; a new "
        "verdict replaces the old one",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many judgements -- a sitting, not the whole pass",
    )
    parser.add_argument(
        "--drafts-dir",
        default=None,
        help=f"where the draft files are (default: {rel_to_repo(DRAFTS_DIR)})",
    )
    parser.add_argument(
        "--log", default=None, help=f"the review log (default: {rel_to_repo(REVIEW_LOG_PATH)})"
    )
    parser.add_argument(
        "--out", default=None, help=f"where --promote writes (default: {rel_to_repo(DATASET_PATH)})"
    )
    parser.add_argument(
        "--chunks", default=None, help=f"the corpus (default: {rel_to_repo(CHUNKS_PATH)})"
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log) if args.log else REVIEW_LOG_PATH
    out_path = Path(args.out) if args.out else DATASET_PATH
    paths = find_drafts(Path(args.drafts_dir) if args.drafts_dir else None)
    if not paths:
        where = Path(args.drafts_dir) if args.drafts_dir else DRAFTS_DIR
        print(
            f"no draft files in {rel_to_repo(where)} -- #56, #57 and #58 write them.",
            file=sys.stderr,
        )
        return 1
    candidates, found = load_candidates(paths)
    if found:
        print(
            "\n".join(render(f.path, [f.violation]) for f in found)
            + "\nThose lines are not rows; the drafting scripts do not write them.",
            file=sys.stderr,
        )
        return 1
    repeated = duplicate_ids(candidates)
    if repeated:
        print(
            f"{', '.join(repeated)} appears in more than one draft file; ids are minted from "
            "eval.scripts.draft_items.ID_BLOCKS so that they cannot",
            file=sys.stderr,
        )
        return 1

    known = corpus_doc_ids(Path(args.chunks)) if args.chunks else corpus_doc_ids()
    entries = read_entries(log_path)
    judged = replay(entries)

    if args.stats:
        print(stats(candidates, entries, log_path))
        return 0

    if args.promote:
        try:
            promotion = promote(candidates, judged, known)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            print(f"{rel_to_repo(out_path)} not written.", file=sys.stderr)
            return 1
        if promotion.ok:
            write_dataset(out_path, promotion.rows)
            print(f"wrote {rel_to_repo(out_path)}")
            print(promotion_report(promotion, out_path))
            return 0
        print(promotion_report(promotion, out_path), file=sys.stderr)
        print(f"{rel_to_repo(out_path)} not written.", file=sys.stderr)
        return 1

    queue = interleave([c for c in candidates if args.redo or c.id not in judged])
    if not queue:
        print(
            f"every one of the {len(candidates)} drafted rows has a verdict. "
            "--stats for the counts, --promote to write the dataset, --redo to re-judge."
        )
        return 0
    texts = corpus_text(Path(args.chunks) if args.chunks else None)
    ranges = credit_ranges()
    tokenizer = get_tokenizer(load_config())
    ask = reader(sys.stdout, lambda prompt: input(prompt))
    made = run(
        queue, known, texts, ranges, tokenizer, log_path, judged, sys.stdout, ask, args.limit
    )
    left = sum(1 for c in candidates if c.id not in judged)
    print(
        f"\n{sum(made.values())} judged this sitting "
        f"({', '.join(f'{n} {verdict}' for verdict, n in sorted(made.items())) or 'none'}); "
        f"{left} of {len(candidates)} drafted rows still unreviewed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
