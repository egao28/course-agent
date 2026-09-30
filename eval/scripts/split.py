"""The frozen 50/50 split: stratified, seeded, written into the dataset, and checkable forever.

    uv run python -m eval.scripts.split                # freeze eval/datasets/v1.jsonl
    uv run python -m eval.scripts.split --dry-run      # the table, written nowhere
    uv run python -m eval.scripts.split --verify       # re-derive and compare; 1 on drift

Day 5's conformal prediction calibrates on one half and Day 4 reports every final number on the
other. The two halves have to be fixed *before* anything is measured, because a threshold tuned on
the rows it is later evaluated on produces a coverage number that describes the tuning rather than
predicting anything, and a split adjusted after the test half has been read is not a split -- it
is a fit with an extra step.

**Stratified, not uniform.** 50/50 *within each bucket*, so both halves have the composition of
the whole. A uniform draw of 60 from 120 puts the twenty ``ambiguous`` rows somewhere other than
10/10 about **81%** of the time, and splits them 6/14 or worse **8.5%** of the time
(hypergeometric, both asserted in the tests) -- the case #61 names. At twenty rows that is not a
rounding difference: it decides how much of Day 5's calibration is spent on the bucket where
abstention is hardest. Odd buckets alternate which half takes the extra row (:func:`derive`), so
neither half is systematically the larger one -- at the target counts ``multihop`` gives
calibration 13 of 25 and ``unanswerable`` gives test 13 of 25.

**Reproducible and frozen are two properties, and this file needs both.**

- *Reproducible* is ``config.yaml``'s ``eval.split_seed`` plus :func:`derive`, which draws each
  bucket from ``random.Random(f"{seed}:{bucket}")`` over that bucket's **sorted** ids. Sorted, and
  per bucket, for the two reasons #60's sampler gives: a split derived from file order re-splits
  the dataset the first time someone sorts the file, and one shared stream moves the ``ambiguous``
  assignment whenever a ``fact`` row is deleted.
- *Frozen* is the ``split`` field committed on every row. That is the contract -- what Day 4 and
  Day 5 actually read -- and the seed is only its provenance. :func:`verify` holds the two
  together by re-deriving from the seed and comparing id by id, and ``tests/test_split.py`` runs
  it over the committed file, so drift fails the suite rather than turning up in week three.

**The digest makes an edit to the eval set a reviewable event.** :func:`freeze` writes
``v1.meta.json`` beside the dataset with its sha256, the seed, the config hash and the per-bucket
counts. A later edit to ``v1.jsonl`` that does not update the sidecar in the same commit fails the
test, and :func:`load_split` refuses to hand out rows from a file whose digest has moved. The
issue asks for the digest in the Day 3 report too; the report is #62 and does not exist yet, so
the sidecar is where it lives and where #62 will read it from.

**The loader takes a required argument and there is no way to ask for both halves.** Day 5 calls
``load_split("calib")``; Day 4 calls ``load_split("test")`` when it is ready to report. There is
deliberately no ``load_all()``: reading the whole file is what the freeze, the validator and #62's
counting table do, and each of those reaches for :func:`eval.validate.parse_items` by name --
which is a validation entry point, not something tuning code reaches for by accident.

**One number Day 5 cannot get around.** Split conformal prediction takes the
:math:`\\lceil (n+1)(1-\\alpha) \\rceil`-th smallest calibration score as its threshold, and that
index has to exist -- :math:`\\lceil (n+1)(1-\\alpha) \\rceil \\le n` -- so
:math:`\\alpha \\ge 1/(n+1)`.
With ``n = 60`` calibration rows that is :math:`\\alpha \\ge 1/61 \\approx 0.0164` -- **98.4%
coverage is the ceiling and 99% is unreachable at any threshold**, not because the model is weak
but because sixty rows cannot express a finer quantile. :func:`conformal_limit` computes it, the
freeze prints it and the sidecar records it. The lever is the split ratio, and the time to move it
is before anything is measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, file_sha256, load_config, rel_to_repo
from eval.schema import BUCKETS, SPLITS, Bucket, EvalItem, Split, row_json
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.validate import corpus_doc_ids, parse_items, plural, render, validate_rows

#: Seed, digest and counts, beside the dataset they describe -- the sidecar convention #56
#: established. ``eval/datasets/v1.meta.json`` for ``eval/datasets/v1.jsonl``.
META_SUFFIX = ".meta.json"


def meta_path(dataset: Path = DATASET_PATH) -> Path:
    """The sidecar that belongs to ``dataset``."""
    return Path(dataset).with_suffix(META_SUFFIX)


# --- deriving the split ---------------------------------------------------------------------------


class Stratum(NamedTuple):
    """One bucket's half-and-half, and which half took the odd row if there was one."""

    bucket: Bucket
    #: Calibration ids, sorted. Sorted for reading; the assignment does not depend on the order.
    calib: tuple[str, ...]
    #: Test ids, sorted.
    test: tuple[str, ...]
    #: ``calib``, ``test``, or ``None`` when the bucket held an even number of rows.
    extra: Split | None

    @property
    def n(self) -> int:
        """How many rows the bucket holds across both halves."""
        return len(self.calib) + len(self.test)


class Assignment(NamedTuple):
    """Every row's half, the per-bucket record of how it was drawn, and the seed it came from."""

    halves: Mapping[str, Split]
    strata: tuple[Stratum, ...]
    seed: int

    def counts(self, half: Split) -> int:
        """How many rows landed in ``half``."""
        return sum(1 for value in self.halves.values() if value == half)

    @property
    def totals(self) -> dict[str, int]:
        """Rows per half, both halves always present even at zero."""
        return {half: self.counts(half) for half in SPLITS}


def derive(rows: Sequence[EvalItem], seed: int) -> Assignment:
    """The assignment, as a function of ``(seed, the id set, the bucket of each id)`` alone.

    Not of file order, and not of anything a row says beyond its bucket. Each bucket draws from
    its own stream, so a ``fact`` row deleted during a late review re-draws ``fact`` -- a
    different pool is a different draw, and no seeded scheme avoids that -- and leaves the other
    three buckets' *draws* untouched.

    Odd buckets alternate in :data:`eval.schema.BUCKETS` order: the first odd bucket gives its
    extra row to ``calib``, the second to ``test``, and so on. Alternating over *the odd ones*
    rather than over all four is what keeps the two halves within one row of each other however
    many buckets are odd; at the target counts it reproduces the table in #61 exactly (fact
    25/25, multihop 13/12, unanswerable 12/13, ambiguous 10/10).

    That alternation is the one place buckets are **not** independent, and it is worth stating
    plainly rather than discovering: deleting a single ``fact`` row takes that bucket from 50 to
    49, which makes it the *first* odd bucket, which hands ``multihop`` and ``unanswerable`` the
    other half's extra row than before. Deleting two leaves the other three exactly as they were.
    A dataset whose id set moves after the freeze therefore has to be re-frozen deliberately --
    see :func:`freeze`, which refuses to do it quietly.
    """
    pools: dict[Bucket, list[str]] = {bucket: [] for bucket in BUCKETS}
    for row in rows:
        pools[row.bucket].append(row.id)
    halves: dict[str, Split] = {}
    strata: list[Stratum] = []
    odd_seen = 0
    for bucket in BUCKETS:
        ids = sorted(pools[bucket])
        extra: Split | None = None
        n_calib = len(ids) // 2
        if len(ids) % 2:
            extra = SPLITS[odd_seen % len(SPLITS)]
            odd_seen += 1
            if extra == "calib":
                n_calib += 1
        rng = random.Random(f"{seed}:{bucket}")
        calib = frozenset(rng.sample(ids, k=n_calib))
        for item_id in ids:
            halves[item_id] = "calib" if item_id in calib else "test"
        strata.append(
            Stratum(
                bucket=bucket,
                calib=tuple(i for i in ids if i in calib),
                test=tuple(i for i in ids if i not in calib),
                extra=extra,
            )
        )
    return Assignment(halves, tuple(strata), seed)


def assign(rows: Sequence[EvalItem], assignment: Assignment) -> list[EvalItem]:
    """``rows`` in the order given, each carrying its half.

    File order is preserved rather than sorted: the assignment does not depend on it (see
    :func:`derive`), so re-ordering the dataset is a diff nobody has to think about, and a freeze
    that also re-ordered 120 lines would bury the one field it changed.
    """
    return [row.model_copy(update={"split": assignment.halves[row.id]}) for row in rows]


def conformal_limit(n_calib: int) -> tuple[float, float]:
    """``(alpha_floor, coverage_ceiling)`` for a calibration set of ``n_calib`` rows.

    Split conformal prediction sorts the calibration scores and takes the
    ``ceil((n+1)(1-alpha))``-th of them; that index cannot exceed ``n``, which bounds ``alpha``
    below by ``1/(n+1)`` and coverage above by ``n/(n+1)``. Returned as a pair rather than
    printed in two places, because the ceiling is the number Day 5 will quote and the floor is
    the one that explains it.
    """
    if n_calib < 1:
        raise ValueError(f"a calibration set needs at least one row, got {n_calib}")
    return 1 / (n_calib + 1), n_calib / (n_calib + 1)


# --- reading the dataset --------------------------------------------------------------------------


def read_rows(path: Path, known: frozenset[str] | None = None) -> list[EvalItem]:
    """The dataset as rows, or a :class:`ValueError` naming what stops it being one.

    Strict, like #60's reader and for the same reason: :mod:`eval.validate` exists to report every
    fault in a file somebody is working on, and this is the frozen artifact. Errors stop the
    freeze -- a dataset that cannot be validated cannot be split, since the split is a claim about
    rows Day 4 will score. Warnings do not: a short bucket is verification having deleted a bad
    row, which is what verification is for.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"no dataset at {rel_to_repo(path)}. The split is over the promoted set, so #59 has "
            "to have run to the end first: draft the remaining buckets "
            "(`python -m eval.scripts.draft_items`), review every row "
            "(`python -m eval.scripts.review`), then `--promote`."
        )
    rows, unparsed = parse_items(path)
    known = corpus_doc_ids() if known is None else known
    violations = validate_rows(rows, known, unparsed)
    errors = [v for v in violations if v.level == "error"]
    if errors:
        raise ValueError(
            f"{rel_to_repo(path)} does not validate, so it cannot be frozen:\n"
            + render(path, errors)
        )
    if not rows:
        raise ValueError(f"{rel_to_repo(path)} holds no rows")
    return [row.item for row in rows]


def write_rows(path: Path, rows: Sequence[EvalItem]) -> Path:
    """The dataset back out, one object per line, through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row_json(row), ensure_ascii=False) + "\n" for row in rows)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def frozen(rows: Sequence[EvalItem]) -> bool:
    """True when every row carries a half.

    A partly-split file is :func:`eval.validate.check_split`'s error, not a third state this
    module reasons about.
    """
    return bool(rows) and all(row.split is not None for row in rows)


# --- the sidecar ----------------------------------------------------------------------------------


def sidecar(path: Path, rows: Sequence[EvalItem], assignment: Assignment, cfg: Config) -> dict:
    """What the freeze records about itself: the digest, the seed, and the counts.

    The digest is of the file *as written*, so it is computed after :func:`write_rows`, never
    over the rows in memory -- the point of a digest is that it answers for the bytes.
    """
    n_calib = assignment.counts("calib")
    alpha, coverage = conformal_limit(n_calib)
    return {
        "dataset": rel_to_repo(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "seed": assignment.seed,
        "config_hash": cfg.hash,
        "totals": assignment.totals,
        "buckets": {
            stratum.bucket: {"calib": len(stratum.calib), "test": len(stratum.test)}
            for stratum in assignment.strata
        },
        # Day 5's ceiling, recorded where #62 can print it without re-deriving it. See
        # conformal_limit for where the bound comes from.
        "conformal": {
            "n_calib": n_calib,
            "alpha_floor": alpha,
            "coverage_ceiling": coverage,
        },
    }


def read_sidecar(path: Path) -> dict[str, Any] | None:
    """The sidecar as written, or ``None`` when there is none."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --- checking it ----------------------------------------------------------------------------------


def digest_problem(dataset: Path, meta: Mapping[str, Any] | None) -> str | None:
    """What is wrong between the file and its recorded digest, or ``None``.

    A missing sidecar is a problem in its own right: an eval set whose digest nothing records can
    be edited without the edit being visible, which is the whole thing #61 is buying.
    """
    dataset = Path(dataset)
    if meta is None:
        return (
            f"{rel_to_repo(dataset)} has no sidecar at {rel_to_repo(meta_path(dataset))}; "
            "re-run `python -m eval.scripts.split` so the digest is recorded"
        )
    recorded = meta.get("sha256")
    actual = file_sha256(dataset)
    if recorded != actual:
        return (
            f"{rel_to_repo(dataset)} has sha256 {actual}, but "
            f"{rel_to_repo(meta_path(dataset))} records {recorded}. The eval set changed after it "
            "was frozen: re-run `python -m eval.scripts.split --verify`, and commit the sidecar "
            "with the change so it is reviewable."
        )
    return None


def drift(rows: Sequence[EvalItem], assignment: Assignment) -> list[str]:
    """Every row whose committed half is not the one the seed derives, as a sentence each."""
    out: list[str] = []
    for row in rows:
        want = assignment.halves[row.id]
        if row.split != want:
            held = "no split" if row.split is None else row.split
            out.append(f"{row.id} ({row.bucket}) is {held}, derived {want}")
    return out


class Verification(NamedTuple):
    """What ``--verify`` found. ``ok`` is the exit code."""

    problems: tuple[str, ...]
    rows: int
    assignment: Assignment

    @property
    def ok(self) -> bool:
        """True when the committed split still re-derives from the seed, problems and all."""
        return not self.problems


def verify(path: Path, seed: int, known: frozenset[str] | None = None) -> Verification:
    """Re-derive the split from the seed and hold the committed file to it, digest included.

    Three ways this fails, and they are different failures: the file is not frozen at all (the
    freeze never ran), a row's half is not the one the seed gives (somebody hand-edited a split,
    or the id set moved and nobody re-froze), or the digest does not match (the rows themselves
    changed). All three are reported together -- a verify that stopped at the first would hide the
    second on every run until the first was fixed.
    """
    path = Path(path)
    rows = read_rows(path, known)
    assignment = derive(rows, seed)
    problems: list[str] = []
    if not frozen(rows):
        missing = [row.id for row in rows if row.split is None]
        shown = ", ".join(missing[:5])
        rest = len(missing) - 5
        more = "" if rest <= 0 else f", and {rest} more"
        problems.append(
            f"{plural(len(missing), 'row')} carry no split ({shown}{more}); "
            "run `python -m eval.scripts.split` to freeze the file"
        )
    problems.extend(drift(rows, assignment))
    problem = digest_problem(path, read_sidecar(meta_path(path)))
    if problem:
        problems.append(problem)
    return Verification(tuple(problems), len(rows), assignment)


# --- the loader Day 4 and Day 5 use -------------------------------------------------------------


def load_split(
    which: Split, path: Path = DATASET_PATH, known: frozenset[str] | None = None
) -> list[EvalItem]:
    """One half of the frozen dataset, by name. There is no default and no way to ask for both.

    ``load_split("calib")`` is Day 5's; ``load_split("test")`` is the one Day 4 calls once it is
    reporting, and the rule is in the README: the test half is read once per milestone, not once
    per experiment. The argument is required because the failure this guards against is not
    someone typing the wrong half -- it is tuning code that never chose one.

    The digest is checked on the way through. A row edited after the freeze is exactly the event
    that silently invalidates a number, and the alternative -- a loader that reads whatever is
    there and a test that notices in CI an hour later -- is how a stale set gets a result written
    about it first.
    """
    if which not in SPLITS:
        raise ValueError(
            f"load_split takes {' or '.join(repr(s) for s in SPLITS)}, got {which!r}; "
            "there is no 'all' -- calibrating and reporting are different halves on purpose"
        )
    path = Path(path)
    rows = read_rows(path, known)
    if not frozen(rows):
        raise ValueError(
            f"{rel_to_repo(path)} is not frozen: run `python -m eval.scripts.split` first"
        )
    problem = digest_problem(path, read_sidecar(meta_path(path)))
    if problem:
        raise ValueError(problem)
    return [row for row in rows if row.split == which]


# --- saying what happened -------------------------------------------------------------------------

#: Widths for the per-bucket table: the longest bucket name is ``unanswerable``.
_BUCKET_W = max(len(b) for b in BUCKETS) + 2


def table(assignment: Assignment) -> str:
    """The per-bucket split, and the totals, as the freeze and ``--dry-run`` both print it."""
    lines = [f"{'bucket':<{_BUCKET_W}}{'n':>5}{'calib':>8}{'test':>7}"]
    for stratum in assignment.strata:
        mark = "" if stratum.extra is None else f"   (odd row -> {stratum.extra})"
        lines.append(
            f"{stratum.bucket:<{_BUCKET_W}}{stratum.n:>5}{len(stratum.calib):>8}"
            f"{len(stratum.test):>7}{mark}"
        )
    totals = assignment.totals
    n = totals["calib"] + totals["test"]
    lines.append(f"{'total':<{_BUCKET_W}}{n:>5}{totals['calib']:>8}{totals['test']:>7}")
    return "\n".join(lines)


def ceiling_line(n_calib: int) -> str:
    """The sentence about what ``n`` calibration rows can and cannot be asked for.

    The unreachable figure is the next whole percent *above* the ceiling -- 99% at n=60 -- and
    not the ceiling rounded up, which at n=99 would claim 100% was the first thing out of reach
    when 99% already is.
    """
    alpha, coverage = conformal_limit(n_calib)
    out_of_reach = (math.floor(coverage * 100) + 1) / 100
    return (
        f"calibration n={n_calib}: coverage 1-alpha is attainable only for "
        f"alpha >= 1/{n_calib + 1} = {alpha:.4f},\nso {coverage:.1%} is the ceiling and "
        f"{out_of_reach:.0%} is out of reach at any threshold."
    )


def summary(path: Path, rows: Sequence[EvalItem], assignment: Assignment, cfg: Config) -> str:
    """The header every command prints: what file, how many rows, which seed, which config."""
    return (
        f"{rel_to_repo(path)}  {plural(len(rows), 'row')}  seed {assignment.seed}  "
        f"config {cfg.hash}"
    )


# --- the commands ---------------------------------------------------------------------------------


def freeze(
    cfg: Config,
    path: Path,
    seed: int,
    out: TextIO,
    dry_run: bool = False,
    refreeze: bool = False,
    known: frozenset[str] | None = None,
) -> int:
    """Derive the split, write it into the dataset, and record the digest beside it.

    Re-running over an already-frozen file is a no-op when the assignment is unchanged and is
    **refused** when it is not. That refusal is the point of the command: an id set that has moved
    since the freeze means rows are about to change halves, and a row that moves from ``test`` to
    ``calib`` after Day 4 has reported on it is a silent invalidation of the number, not a
    re-split. ``--refreeze`` does it anyway, having printed every row that moves.
    """
    rows = read_rows(path, known)
    assignment = derive(rows, seed)
    out.write(summary(path, rows, assignment, cfg) + "\n\n")
    out.write(table(assignment) + "\n\n")
    out.write(ceiling_line(assignment.counts("calib")) + "\n")

    moved = drift(rows, assignment) if frozen(rows) else []
    if moved and not refreeze:
        shown = "\n  ".join(moved[:10])
        rest = len(moved) - 10
        more = "" if rest <= 0 else f"\n  and {rest} more"
        out.write(
            f"\n{rel_to_repo(path)} is already frozen and {plural(len(moved), 'row')} would "
            f"change halves:\n  {shown}{more}\n"
            "Nothing written. A row that changes halves after a number has been reported on it "
            "invalidates that number; pass --refreeze if the dataset really is being re-frozen, "
            "and say so in the commit.\n"
        )
        return 1
    if dry_run:
        out.write("\n--dry-run: nothing written.\n")
        return 0

    written = write_rows(path, assign(rows, assignment))
    meta = meta_path(path)
    write_json(meta, sidecar(written, rows, assignment, cfg))
    out.write(f"\nwrote  {rel_to_repo(written)}  sha256 {file_sha256(written)}\n")
    out.write(f"       {rel_to_repo(meta)}\n")
    return 0


def report_verification(path: Path, result: Verification, cfg: Config, out: TextIO) -> int:
    """``--verify``'s output: the table when it holds, every problem when it does not."""
    out.write(summary(path, range(result.rows), result.assignment, cfg) + "\n\n")
    out.write(table(result.assignment) + "\n")
    if result.ok:
        out.write(
            f"\nok -- the committed split is the one seed {result.assignment.seed} derives, and "
            f"{rel_to_repo(meta_path(path))} records the file as it stands.\n"
        )
        return 0
    out.write("\n" + "\n".join(f"  {problem}" for problem in result.problems) + "\n")
    out.write(f"\nFAILED -- {plural(len(result.problems), 'problem')}.\n")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze the 50/50 split, or with ``--verify`` check the committed one still re-derives.

    A row changing halves on an already-frozen dataset needs ``--refreeze``: the halves are what
    every later run is reported against, so moving one silently would re-scope those numbers.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.split",
        description="Freeze the eval set into a stratified, seeded 50/50 calibration/test split, "
        "or check that the committed split is still the one the seed derives.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive the split and compare it, and the file's digest, against what is "
        "committed; writes nothing and exits 1 on any difference",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the split that would be written, write nothing",
    )
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="allow rows to change halves on an already-frozen dataset (they are listed first)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"the dataset to split (default: {rel_to_repo(DATASET_PATH)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="split seed; defaults to config.yaml's eval.split_seed, and is recorded in the "
        "sidecar either way. Overriding it re-splits the dataset -- see --refreeze",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    path = Path(args.dataset) if args.dataset else DATASET_PATH
    seed = cfg["eval"]["split_seed"] if args.seed is None else args.seed
    try:
        if args.verify:
            return report_verification(path, verify(path, seed), cfg, sys.stdout)
        return freeze(cfg, path, seed, sys.stdout, args.dry_run, args.refreeze)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
