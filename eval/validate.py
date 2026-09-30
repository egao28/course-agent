"""The runnable check over a file of eval rows: every violation, then a non-zero exit.

    uv run python -m eval.validate eval/datasets/v1.jsonl

:mod:`eval.schema` says what a row is; this says whether a file of them is one. It runs before
the drafting harness (#56) writes anything and again over whatever the verification pass (#59)
leaves behind, and :mod:`tests.test_eval_validate` runs it over the committed dataset so an edit
cannot land without having been re-validated.

**Every violation, not the first.** A 120-row file with eight typos should be one pass with eight
messages, not eight runs. Every check below yields a :class:`Violation` and the run ends by
printing all of them, sorted by line; nothing raises on the first bad row. Pydantic already works
this way within a row -- a row that is missing ``bucket`` *and* carries an extra ``difficulty``
reports both -- and the cross-field checks here match it.

**The gold ids are most of the job.** A hand-typed ``2026-sp:MATH 241`` is the single likeliest
error in this dataset, because ``metadata.code`` really is ``MATH 241`` with a space and only the
doc id uses a dash. It must fail rather than warn -- a dangling id makes a row unscoreable, and a
row that cannot be scored in a set built to score things is worse than a missing row, since it
still occupies a slot in the per-bucket counts. :func:`suggest` turns each one into the id it was
probably meant to be, because "not in the corpus" is a true message that leaves the reader to
find the dash themselves.

**Counts warn, everything else fails.** :data:`eval.schema.RULES` carries a target per bucket and
the totals are checked against it, but a short bucket exits 0: deleting a bad item during
verification is what verification is for, and a check that fails on 49 facts is a check that
argues for keeping a bad row.

**The split is checked for shape here and for correctness in #61.** A file is either frozen or it
is not (:func:`check_split`), and a half-frozen one is an error: a loader asking for the ``test``
half of a file where sixty rows carry no half would hand Day 4 whatever happened to be labelled
and call it the test set. Whether the *right* rows are in each half is not decidable here -- it
takes the seed -- and ``python -m eval.scripts.split --verify`` is the check that re-derives it.
What this file adds is the one stratification property that needs no seed: within a bucket the
two halves differ by at most one row, a warning like every other count.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import ValidationError

from config import rel_to_repo
from eval.schema import BUCKETS, RULES, SPLITS, TARGET_TOTAL, EvalItem, normalize_question, rule
from ingest.chunk import CHUNKS_PATH, doc_id, load_chunks, load_courses
from ingest.export_almabot import COURSES_PATH

# --- what the corpus holds ------------------------------------------------------------------------

_DOC_IDS: dict[tuple[str, str], frozenset[str]] = {}


def corpus_doc_ids(
    chunks_path: Path | None = None, courses_path: Path | None = None
) -> frozenset[str]:
    """Every id a ``gold_doc_ids`` entry may name: 4,502 of them, ``2026-sp:MATH-241`` form.

    Read from ``data/chunks.jsonl`` when it is there, since that is the file the indexes were
    built from and the one the issue names. It is gitignored, though -- rebuildable from the
    committed JSON, and 3 MB of churn on every template change -- so on a fresh clone the ids are
    derived instead from ``data/courses.json`` through :func:`ingest.chunk.doc_id`, the same
    function that minted them into the chunk file. The two sets are equal by construction, and
    :mod:`tests.test_eval_validate` checks that they are in fact equal here. Without the
    fallback, the test over the committed dataset would skip on exactly the machine that has not
    run the pipeline yet, which is the machine it is for.

    A corpus named outright -- ``--chunks``, or this argument -- is used as given and its
    absence is an error: the fallback is what the *default* means, not a second guess at a path
    somebody typed.

    Cached, because the graph is not the only thing that pays for a 3 MB read: the validator
    checks one file and the test suite checks several.
    """
    named = chunks_path is not None
    chunks = CHUNKS_PATH if chunks_path is None else Path(chunks_path)
    courses = COURSES_PATH if courses_path is None else Path(courses_path)
    key = (str(chunks), str(courses))
    if key not in _DOC_IDS:
        if named or chunks.exists():
            ids = {chunk["doc_id"] for chunk in load_chunks(chunks)}
        else:
            ids = {doc_id(course) for course in load_courses(courses)}
        _DOC_IDS[key] = frozenset(ids)
    return _DOC_IDS[key]


def clear_cache() -> None:
    """Drop the cached id set -- for a test that moves the corpus under it, and for a re-export."""
    _DOC_IDS.clear()


def suggest(bad: str, known: frozenset[str]) -> str | None:
    """The id ``bad`` was probably meant to be, or ``None`` if nothing obvious is in the corpus.

    Three repairs, tried in order and composed: spaces to dashes (``2026-sp:MATH 241``), the
    term prefix a bare code is missing (``MATH 241``), and upper case (``math 241``). Anything
    further -- a nearest-neighbour over 4,502 strings -- would start guessing at *which course*
    was meant, and that is the reviewer's call, not the validator's.
    """
    dashed = "-".join(bad.split())
    candidates: list[str] = []
    for base in dict.fromkeys([bad, dashed, dashed.upper()]):
        candidates.append(base)
        if ":" not in base:
            candidates.extend(f"{term}:{base}" for term in sorted({i.split(":")[0] for i in known}))
    for candidate in candidates:
        if candidate != bad and candidate in known:
            return candidate
    return None


# --- wording --------------------------------------------------------------------------------------


def plural(n: int, noun: str) -> str:
    """``1 id`` / ``2 ids`` -- a validator that says "1 row(s)" reads like a form."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def article(bucket: str) -> str:
    """``a`` or ``an`` -- so a message about the unanswerable bucket reads like a sentence."""
    return "an" if bucket[0] in "aeiou" else "a"


# --- what a violation is --------------------------------------------------------------------------

#: ``error`` exits non-zero; ``warning`` is printed and does not. The only warnings are counts.
Level = Literal["error", "warning"]


class Violation(NamedTuple):
    """One thing wrong with one row, or with the file as a whole."""

    #: 1-based line in the dataset, or ``None`` for a whole-file finding such as a bucket count.
    line: int | None
    #: The row's ``id`` as written, or ``None`` when the row has none to report.
    item_id: str | None
    #: Which field is wrong, dotted for a list element (``gold_doc_ids.1``); ``None`` for a
    #: finding that is not about one field.
    field: str | None
    #: What is wrong, and where possible what it should have been instead.
    message: str
    #: See :data:`Level`.
    level: Level = "error"


class Row(NamedTuple):
    """A row that parsed, and the line it came from."""

    line: int
    item: EvalItem


# --- reading the file -----------------------------------------------------------------------------


def parse_items(path: Path) -> tuple[list[Row], list[Violation]]:
    """Read a JSONL file into rows, reporting every row that is not one.

    A row that fails here is dropped rather than half-repaired, so the cross-row checks
    (duplicate ids, duplicate questions) run over what actually parsed. Blank lines are skipped:
    a trailing newline is not a row.
    """
    rows: list[Row] = []
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
            try:
                rows.append(Row(line, EvalItem.model_validate(raw)))
            except ValidationError as exc:
                violations.extend(pydantic_violations(line, found, exc))
    return rows, violations


def pydantic_violations(
    line: int, item_id: str | None, exc: ValidationError
) -> Iterator[Violation]:
    """One :class:`Violation` per pydantic error, so a row with three faults reports three.

    Public because the drafting harness (#56) validates the same six fields out of a row that
    also carries provenance, and a second rendering of pydantic's errors would be a second set
    of messages for the same fault.
    """
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or None
        message = error["msg"]
        if error["type"] == "extra_forbidden":
            message = f"{message}; the row has six fields, and `split` once #61 freezes it"
        yield Violation(line, item_id, field, message)


# --- the checks -----------------------------------------------------------------------------------


def check_row(row: Row, known: frozenset[str]) -> Iterator[Violation]:
    """Everything decidable from one row: the bucket's promises, and the gold ids.

    ``bucket`` is a :data:`eval.schema.Bucket` by the time a row exists, so every rule below can
    be read straight out of the table.
    """
    item = row.item
    spec = rule(item.bucket)

    if item.should_abstain != spec.should_abstain:
        yield Violation(
            row.line,
            item.id,
            "should_abstain",
            f"is {item.should_abstain}, but every {item.bucket} row is "
            f"{spec.should_abstain}; one of the two is the mislabel",
        )

    count = len(item.gold_doc_ids)
    if count < spec.min_gold_ids:
        yield Violation(
            row.line,
            item.id,
            "gold_doc_ids",
            f"{plural(count, 'id')}; {article(item.bucket)} {item.bucket} row needs at "
            f"least {spec.min_gold_ids} -- {spec.gloss}",
        )
    if spec.max_gold_ids is not None and count > spec.max_gold_ids:
        allowed = "no gold ids at all" if spec.max_gold_ids == 0 else f"at most {spec.max_gold_ids}"
        yield Violation(
            row.line,
            item.id,
            "gold_doc_ids",
            f"{plural(count, 'id')}; {article(item.bucket)} {item.bucket} row carries "
            f"{allowed} -- {spec.gloss}",
        )

    seen: dict[str, int] = {}
    for position, gold in enumerate(item.gold_doc_ids):
        field = f"gold_doc_ids.{position}"
        first = seen.get(gold)
        if first is not None:
            yield Violation(
                row.line,
                item.id,
                field,
                f"repeats {gold!r} from position {first}; a document cited twice is "
                "still one document, and in a multihop row it is one hop wearing two ids",
            )
            continue
        seen[gold] = position
        if gold not in known:
            meant = suggest(gold, known)
            hint = f"; did you mean {meant!r}?" if meant else ""
            yield Violation(row.line, item.id, field, f"{gold!r} is not in the corpus{hint}")


def check_across_rows(rows: Sequence[Row]) -> Iterator[Violation]:
    """Ids unique, and no question asked twice once whitespace and case stop counting."""
    ids: dict[str, int] = {}
    questions: dict[str, Row] = {}
    for row in rows:
        first = ids.get(row.item.id)
        if first is not None:
            yield Violation(
                row.line,
                row.item.id,
                "id",
                f"already used on line {first}; #60's sample and #61's split name rows by id",
            )
        else:
            ids[row.item.id] = row.line
        key = normalize_question(row.item.question)
        twin = questions.get(key)
        if twin is not None:
            yield Violation(
                row.line,
                row.item.id,
                "question",
                f"repeats line {twin.line} ({twin.item.id}) once whitespace and case are "
                "normalised",
            )
        else:
            questions[key] = row


def check_split(rows: Sequence[Row]) -> Iterator[Violation]:
    """One error if the file is frozen for some rows and not others. Nothing otherwise.

    The value itself is :data:`eval.schema.Split`, so ``"cal"`` never reaches here -- pydantic
    has already rejected the row. What is left is the state no single row can be wrong about:
    a file where the freeze half ran. It is reported once, with the ids, because the fix is to
    re-run ``python -m eval.scripts.split`` over the whole file and not to edit sixty lines.
    """
    missing = [row for row in rows if row.item.split is None]
    if not missing or len(missing) == len(rows):
        return
    shown = ", ".join(row.item.id for row in missing[:5])
    rest = len(missing) - 5
    more = "" if rest <= 0 else f", and {rest} more"
    yield Violation(
        None,
        None,
        "split",
        f"{plural(len(rows) - len(missing), 'row')} carry a split and "
        f"{len(missing)} do not ({shown}{more}); a dataset is frozen or it is not -- "
        "re-run `python -m eval.scripts.split`",
    )


def check_split_counts(rows: Sequence[Row]) -> Iterator[Violation]:
    """Per bucket, the two halves within one row of each other -- when the file is frozen at all.

    The half of #61's stratification that is decidable without the seed. A bucket split 17/8 is
    reproducible and seeded and still not stratified, and at these sizes that is the difference
    between calibrating on ten ambiguous rows and calibrating on three.
    """
    frozen = [row for row in rows if row.item.split is not None]
    if not frozen or len(frozen) != len(rows):
        return
    for bucket in BUCKETS:
        halves = {
            half: sum(1 for r in frozen if r.item.bucket == bucket and r.item.split == half)
            for half in SPLITS
        }
        if abs(halves["calib"] - halves["test"]) > 1:
            yield Violation(
                None,
                None,
                bucket,
                f"{halves['calib']} calib / {halves['test']} test; a stratified half-and-half "
                "split leaves at most one row between them",
                "warning",
            )


def check_counts(rows: Sequence[Row]) -> Iterator[Violation]:
    """The per-bucket targets, as warnings. A short bucket is a decision, not a corruption."""
    counts = {bucket: 0 for bucket in BUCKETS}
    for row in rows:
        counts[row.item.bucket] += 1
    for bucket in BUCKETS:
        spec = RULES[bucket]
        if counts[bucket] != spec.target_n:
            direction = "short of" if counts[bucket] < spec.target_n else "over"
            yield Violation(
                None,
                None,
                bucket,
                f"{plural(counts[bucket], 'row')}, {direction} the planned {spec.target_n}",
                "warning",
            )
    if len(rows) != TARGET_TOTAL:
        yield Violation(
            None,
            None,
            None,
            f"{plural(len(rows), 'row')} in total, not {TARGET_TOTAL}",
            "warning",
        )


def validate_rows(
    rows: Sequence[Row], known: frozenset[str], unparsed: Sequence[Violation] = ()
) -> list[Violation]:
    """Every check over a parsed file, in file order, warnings last.

    ``unparsed`` is whatever :func:`parse_items` could not turn into a row. It is merged in
    rather than printed first so that the output reads down the file: a reader fixing line 40
    should not have to look in two places for what is wrong with line 12.
    """
    violations = list(unparsed)
    violations.extend(v for row in rows for v in check_row(row, known))
    violations.extend(check_across_rows(rows))
    violations.extend(check_split(rows))
    violations.sort(key=lambda v: (v.line or 0, v.field or ""))
    violations.extend(check_counts(rows))
    violations.extend(check_split_counts(rows))
    return violations


def validate_file(path: Path, known: frozenset[str] | None = None) -> list[Violation]:
    """Parse ``path`` and run every check over it. An empty file is an error, not an empty pass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{rel_to_repo(path)} does not exist")
    known = corpus_doc_ids() if known is None else known
    rows, unparsed = parse_items(path)
    if not rows and not unparsed:
        return [Violation(None, None, None, "holds no rows")]
    return validate_rows(rows, known, unparsed)


# --- saying so ------------------------------------------------------------------------------------


def render(path: Path, violations: Iterable[Violation]) -> str:
    """One line per violation: where it is, which field, and what is wrong.

    Prefixed with the file and line so an editor can jump to it, which is the whole reason a
    validator prints line numbers rather than ids.
    """
    lines = []
    for violation in violations:
        where = f"{rel_to_repo(path)}:{violation.line}" if violation.line else rel_to_repo(path)
        what = " ".join(part for part in (violation.item_id, violation.field) if part is not None)
        mark = "warning: " if violation.level == "warning" else ""
        lines.append(f"{mark}{where}  {what}{'  ' if what else ''}{violation.message}")
    return "\n".join(lines)


def summary(path: Path, violations: Sequence[Violation]) -> str:
    """The count line: how many errors, how many warnings, and whether the file stands."""
    errors = sum(1 for v in violations if v.level == "error")
    warnings = len(violations) - errors
    verdict = "ok" if not errors else "FAILED"
    return f"{rel_to_repo(path)}: {verdict} -- {errors} error(s), {warnings} warning(s)"


def main(argv: Sequence[str] | None = None) -> int:
    """Check one JSONL eval set and print every violation; exit non-zero if any was an error."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.validate",
        description="Check a JSONL eval set against eval/schema.py; print every violation.",
    )
    parser.add_argument("dataset", help="the .jsonl file to check")
    parser.add_argument(
        "--chunks",
        default=None,
        help=f"corpus whose doc ids the gold ids must name (default: {rel_to_repo(CHUNKS_PATH)}, "
        f"or {rel_to_repo(COURSES_PATH)} when it has not been built)",
    )
    args = parser.parse_args(argv)

    path = Path(args.dataset)
    known = corpus_doc_ids(Path(args.chunks)) if args.chunks else corpus_doc_ids()
    violations = validate_file(path, known)
    if violations:
        print(render(path, violations))
    print(summary(path, violations))
    return 1 if any(v.level == "error" for v in violations) else 0


if __name__ == "__main__":
    sys.exit(main())
