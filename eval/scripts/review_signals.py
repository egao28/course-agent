"""The human pass over a signals run: a person's ``correct`` on every row, and the CSV Day 5 reads.

    uv run python -m eval.scripts.review_signals --split calib --reviewer NAME   # resumable
    uv run python -m eval.scripts.review_signals --split calib --stats           # judge vs human
    uv run python -m eval.scripts.review_signals --split calib --write-csv       # -> signals CSV

``eval/scripts/judge.py`` made the first call on every row. This tool shows each row -- question,
gold answer, the system's answer, how the agent exited, and the judge's verdict with its reason --
and records the reviewer's verdict to ``eval/runs/correctness_log.jsonl``. The CSV writer then
joins #87's signals, the judge's verdicts and the log into
``eval/runs/signals_{split}_{signals_key}.csv``, the ``signals.csv`` in the plan.

**The tool does not complete the pass for anyone.** Nothing here writes a verdict a person did not
type, and :func:`write_csv` refuses to emit ``correct`` until every row of the signals run has one.
Agreeing with the judge takes one key (``a``), and the log records it as ``accepted_judge``. That
is a reviewed row, and it says so. It is not the same as a row with no entry, which is unreviewed
however confident the judge was. Overturning the judge needs a one-line reason, because the
overturns are the evidence on whether the judge can be used without a person on Day 6.

**The log is append-only and the latest verdict wins**, the pattern of Day 3's ``review_log.jsonl``
(``eval/scripts/review.py``). Each entry carries a timestamp and the reviewer's name, and is fsynced
before the next row is shown, so a closed terminal loses at most the row on screen. Resumption is by
``id``: a row with a current verdict is not shown again unless ``--redo`` asks.

**A verdict belongs to one answer text, not to one run.** Each entry records the sha256 of the
answer it judged, and the signals key of the run it was given in. A verdict applies to a row when
the row's ``id`` and answer hash both match, whichever run it came from. A rerun of #87 changes
some answers and not others, even at temperature 0: when #87 moved from signals key ``40070d8d``
to ``37164723``, 37 of 46 calib answers came back word for word and 9 did not. A person's verdict
on a text is still that person's verdict on the same text, so the 37 keep theirs. The 9 are
reported as stale and shown again. The CSV sidecar counts how many verdicts were carried over from
another run.

**The CSV writes booleans as 1 and 0, and a missing value as an empty cell.** ``judge_correct`` is
empty where the judge's reply could not be read. ``correct`` is never empty, because a file with an
empty ``correct`` cell is not written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, file_sha256, load_config, rel_to_repo, signals_key
from eval.schema import BUCKETS, SPLITS, EvalItem, Split
from eval.scripts.draft_items import write_json
from eval.scripts.judge import gold_by_id, judge_path, read_jsonl, read_run
from eval.scripts.review import DATASET_PATH, indent, reader
from eval.scripts.signals import RUNS_DIR, run_path, sha256_text
from eval.scripts.split import load_split

#: The reviewers' verdicts, one line each. Committed: it is where every ``correct`` comes from.
LOG_PATH = RUNS_DIR / "correctness_log.jsonl"

#: Every key a log entry carries. ``judge_correct`` is what the reviewer was shown, so an overturn
#: can be counted from the log alone, even after the judge file has moved on.
LOG_KEYS: tuple[str, ...] = (
    "id",
    "split",
    "signals_key",
    "answer_sha256",
    "judge_correct",
    "correct",
    "action",
    "reviewer",
    "note",
    "ts",
)

#: What the reviewer did, derived from the two verdicts rather than typed.
ACTIONS = ("accepted_judge", "overturned", "no_judge_verdict")

#: The CSV's columns, in the plan's order.
CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "bucket",
    "split",
    "retrieval_score",
    "dense_top1",
    "verbalized",
    "self_consistency",
    "judge_correct",
    "correct",
    "reviewer",
)

RULE = "-" * 96


def csv_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/signals_{split}_{signals_key}.csv``, beside the run it labels."""
    return run_path(split, cfg, runs_dir).with_suffix(".csv")


# --- the log --------------------------------------------------------------------------------------


def now() -> str:
    """The timestamp a verdict carries. Seam for the tests, which must not race a clock."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def action_of(judge_correct: bool | None, correct: bool) -> str:
    """What a human verdict did to the judge's: accepted, overturned, or had none to act on."""
    if judge_correct is None:
        return "no_judge_verdict"
    return "accepted_judge" if judge_correct == correct else "overturned"


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    """The log as written, in file order. A missing log is a pass not yet started."""
    path = LOG_PATH if path is None else Path(path)
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
                    f"{rel_to_repo(path)}:{line} is missing {', '.join(missing)}; a correctness "
                    f"log entry carries all {len(LOG_KEYS)} keys"
                )
            if not isinstance(entry["correct"], bool) or not entry["reviewer"]:
                raise ValueError(
                    f"{rel_to_repo(path)}:{line} has no boolean `correct` or no reviewer; every "
                    "entry is a person's verdict"
                )
            entries.append(entry)
    return entries


def append_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """One verdict, flushed and fsynced before the next row is shown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({key: entry[key] for key in LOG_KEYS}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def latest(entries: Sequence[Mapping[str, Any]], split: Split) -> dict[tuple[str, str], dict]:
    """The latest verdict per ``(id, answer_sha256)`` in one half. File order is judgement order."""
    found: dict[tuple[str, str], dict] = {}
    for entry in entries:
        if entry["split"] == split:
            found[entry["id"], entry["answer_sha256"]] = dict(entry)
    return found


# --- the rows -------------------------------------------------------------------------------------


class Row(NamedTuple):
    """One row of the signals run, with its gold answer and the judge's verdict beside it."""

    line: dict[str, Any]
    item: EvalItem
    judged: dict[str, Any]

    @property
    def id(self) -> str:
        """The eval row's id, as the run line carries it."""
        return self.line["id"]

    @property
    def answer_sha256(self) -> str:
        """The digest of the answer text, which is what a stored verdict is pinned to."""
        return sha256_text(self.line["answer"])


def load_rows(
    split: Split,
    cfg: Config,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
) -> list[Row]:
    """The signals run, joined to the dataset and the judge file. Every mismatch is refused."""
    lines = read_run(run_path(split, cfg, runs_dir))
    source = judge_path(split, cfg, runs_dir)
    if not source.exists():
        raise ValueError(
            f"{rel_to_repo(source)} does not exist: run `python -m eval.scripts.judge --split "
            f"{split}` first, so the reviewer sees the judge's call"
        )
    judged = {line["id"]: line for line in read_jsonl(source)}
    items = gold_by_id(load_split(split, dataset, known))
    rows: list[Row] = []
    problems: list[str] = []
    for line in lines:
        item, verdict = items.get(line["id"]), judged.get(line["id"])
        if item is None:
            problems.append(f"{line['id']} is not in the {split} half of the dataset")
        elif verdict is None:
            problems.append(f"{line['id']} has no line in {rel_to_repo(source)}")
        elif verdict["answer_sha256"] != sha256_text(line["answer"]):
            problems.append(f"{line['id']}: the judge graded a different answer")
        else:
            rows.append(Row(line, item, verdict))
    if problems:
        raise ValueError("; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))
    return rows


def interleave(rows: Sequence[Row]) -> list[Row]:
    """One row from each bucket in turn, ids ascending inside a bucket, as Day 3's queue did."""
    by_bucket = {b: sorted((r for r in rows if r.line["bucket"] == b), key=lambda r: r.id)
                 for b in BUCKETS}  # fmt: skip
    queue: list[Row] = []
    while any(by_bucket.values()):
        for bucket in BUCKETS:
            if by_bucket[bucket]:
                queue.append(by_bucket[bucket].pop(0))
    return queue


class Status(NamedTuple):
    """Which rows have a current verdict, and which carry one for an answer that has changed."""

    verdicts: dict[str, dict]
    stale: list[str]


def status(rows: Sequence[Row], entries: Sequence[Mapping[str, Any]], split: Split) -> Status:
    """Rows whose current answer has a verdict, and rows with verdicts only for other answers."""
    found = latest(entries, split)
    judged_ids = {item_id for item_id, _ in found}
    current: dict[str, dict] = {}
    stale: list[str] = []
    for row in rows:
        entry = found.get((row.id, row.answer_sha256))
        if entry is not None:
            current[row.id] = entry
        elif row.id in judged_ids:
            stale.append(row.id)
    return Status(current, stale)


# --- the screen -----------------------------------------------------------------------------------


def word(verdict: bool | None) -> str:
    """A verdict as the word the reviewer reads; ``None`` is an unparsed judgement, not a "no"."""
    return "unparsed" if verdict is None else "correct" if verdict else "incorrect"


def render_row(row: Row, position: int, total: int, seen: Mapping[str, Any] | None) -> str:
    """One row as the reviewer sees it: question, gold, answer, the judge, and any earlier verdict.

    A draft the citation check discarded is flagged, because the reviewer is then judging text the
    user never saw and should know that before calling it correct.
    """
    line = row.line
    lines = [
        RULE,
        f"[{position}/{total}] {row.id}   bucket: {line['bucket']}   split: {line['split']}",
        "",
        "question:",
        indent(line["question"]),
        "",
        f"gold ({'clarifying question' if line['bucket'] == 'ambiguous' else 'reference'}):",
        indent(row.item.gold_answer or "(empty)"),
        "",
        "system answer:",
        indent(line["answer"]),
    ]
    if line["exit"] != "answer":
        lines += [
            "",
            (
                f" !  the citation check discarded this draft; the user saw the {line['exit']!r} "
                f"exit (unknown citations: {line.get('citations_unknown')})"
            ),
        ]
    judged = row.judged
    lines += [
        "",
        f"judge: {word(judged['judge_correct'])}"
        + (f" -- {judged['rationale']}" if judged.get("rationale") else ""),
    ]
    if seen is not None:
        lines.append(
            f"your earlier verdict: {word(seen['correct'])} ({seen['action']}, {seen['reviewer']})"
        )
    return "\n".join(lines)


def prompt_for(judge_correct: bool | None) -> str:
    """The input line, offering ``[a]gree`` only when there is a judge verdict to agree with."""
    agree = "[a]gree with judge  " if judge_correct is not None else ""
    return f"correct?  {agree}[y]es  [n]o  [q]uit > "


Ask = Callable[[str], str | None]


def ask_verdict(row: Row, out: TextIO, ask: Ask) -> tuple[bool, str | None] | None:
    """``(correct, note)``, or ``None`` to stop. An overturn is not taken without a reason."""
    judge_correct = row.judged["judge_correct"]
    while True:
        answer = ask(prompt_for(judge_correct))
        if answer is None:
            return None
        choice = answer.strip().lower()
        if choice in {"q", "quit"}:
            return None
        if choice in {"a", "agree"} and judge_correct is not None:
            return judge_correct, None
        if choice in {"y", "yes", "n", "no"}:
            correct = choice.startswith("y")
            if judge_correct is None or correct == judge_correct:
                return correct, None
            while True:
                note = ask("the judge said otherwise -- one line on why > ")
                if note is None:
                    return None
                if note.strip():
                    return correct, note.strip()
                out.write("an overturn needs a reason: it is what Day 6 reads about the judge.\n")
        out.write(("a, " if judge_correct is not None else "") + "y, n or q.\n")


def run(
    rows: Sequence[Row],
    done: Mapping[str, Mapping[str, Any]],
    key: str,
    reviewer: str,
    log_path: Path,
    out: TextIO,
    ask: Ask,
    redo: bool = False,
) -> Counter[str]:
    """Walk the queue, writing each verdict before the next row. Returns this sitting's actions."""
    queue = [row for row in interleave(rows) if redo or row.id not in done]
    made: Counter[str] = Counter()
    for position, row in enumerate(queue, start=1):
        out.write(render_row(row, position, len(queue), done.get(row.id)) + "\n\n")
        verdict = ask_verdict(row, out, ask)
        if verdict is None:
            break
        correct, note = verdict
        judge_correct = row.judged["judge_correct"]
        entry = {
            "id": row.id,
            "split": row.line["split"],
            "signals_key": key,
            "answer_sha256": row.answer_sha256,
            "judge_correct": judge_correct,
            "correct": correct,
            "action": action_of(judge_correct, correct),
            "reviewer": reviewer,
            "note": note,
            "ts": now(),
        }
        append_entry(log_path, entry)
        made[entry["action"]] += 1
    return made


# --- judge against human --------------------------------------------------------------------------


def agreement(rows: Sequence[Row], verdicts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per bucket: rows, reviewed, overturned; and judge x human over the reviewed rows.

    The judge side is the judge file's verdict, which is what the reviewer was shown for this
    answer. ``unparsed`` is a column of its own: a reply the judge could not give is not a
    disagreement and not an agreement.
    """
    per_bucket: dict[str, Counter[str]] = {bucket: Counter() for bucket in BUCKETS}
    matrix = {
        j: {h: 0 for h in ("correct", "incorrect")} for j in ("correct", "incorrect", "unparsed")
    }
    for row in rows:
        counts = per_bucket[row.line["bucket"]]
        counts["rows"] += 1
        entry = verdicts.get(row.id)
        if entry is None:
            continue
        counts["reviewed"] += 1
        judge_correct = row.judged["judge_correct"]
        counts[action_of(judge_correct, entry["correct"])] += 1
        matrix[word(judge_correct)][word(entry["correct"])] += 1
    buckets = {
        bucket: {k: counts[k] for k in ("rows", "reviewed", *ACTIONS)}
        for bucket, counts in per_bucket.items()
        if counts["rows"]
    }
    return {"by_bucket": buckets, "judge_x_human": matrix}


def render_agreement(result: Mapping[str, Any]) -> str:
    """Progress per bucket and the judge-by-human confusion, as fixed-width terminal text."""
    lines = [
        (
            f"{'bucket':<14}{'rows':>6}{'reviewed':>10}{'accepted':>10}{'overturned':>12}"
            f"{'no judge':>10}"
        )
    ]
    total: Counter[str] = Counter()
    for bucket, c in result["by_bucket"].items():
        total.update(c)
        lines.append(
            f"{bucket:<14}{c['rows']:>6}{c['reviewed']:>10}{c['accepted_judge']:>10}"
            f"{c['overturned']:>12}{c['no_judge_verdict']:>10}"
        )
    lines.append(
        f"{'all':<14}{total['rows']:>6}{total['reviewed']:>10}{total['accepted_judge']:>10}"
        f"{total['overturned']:>12}{total['no_judge_verdict']:>10}"
    )
    lines += ["", "judge (rows) x human (columns), reviewed rows only:",
              f"{'':<14}{'correct':>10}{'incorrect':>11}"]  # fmt: skip
    for judge_word, by in result["judge_x_human"].items():
        lines.append(f"{judge_word:<14}{by['correct']:>10}{by['incorrect']:>11}")
    return "\n".join(lines)


# --- the CSV --------------------------------------------------------------------------------------


def cell(value: Any) -> str:
    """One CSV value: ``None`` as empty, a bool as 1/0, anything else as ``str``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_csv(
    rows: Sequence[Row],
    verdicts: Mapping[str, Mapping[str, Any]],
    stale: Sequence[str],
    path: Path,
    cfg: Config,
    log_path: Path,
    judge_file: Path,
    signals_file: Path,
) -> Path:
    """The signals CSV, or a refusal naming the rows no person has labelled."""
    missing = [row.id for row in rows if row.id not in verdicts]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        detail = f"; {len(stale)} of them have a verdict for an older answer" if stale else ""
        raise ValueError(
            f"refusing to write {rel_to_repo(path)}: {len(missing)} of {len(rows)} rows have no "
            f"human verdict ({shown}){detail}. "
            "`correct` comes only from the review log -- finish the pass with --reviewer NAME."
        )
    body = [
        {
            "id": row.id,
            "bucket": row.line["bucket"],
            "split": row.line["split"],
            "retrieval_score": row.line["retrieval_score"],
            "dense_top1": row.line["dense_top1"],
            "verbalized": row.line["verbalized"],
            "self_consistency": row.line["self_consistency"],
            "judge_correct": row.judged["judge_correct"],
            "correct": verdicts[row.id]["correct"],
            "reviewer": verdicts[row.id]["reviewer"],
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for line in body:
            writer.writerow([cell(line[column]) for column in CSV_COLUMNS])
    os.replace(tmp, path)
    write_json(
        path.with_name(path.name + ".meta.json"),
        {
            "csv": rel_to_repo(path),
            "sha256": file_sha256(path),
            "signals_key": signals_key(cfg),
            "signals_run": rel_to_repo(signals_file),
            "signals_run_sha256": file_sha256(signals_file),
            "judge_run": rel_to_repo(judge_file),
            "judge_run_sha256": file_sha256(judge_file),
            "review_log": rel_to_repo(log_path),
            "review_log_sha256": file_sha256(log_path),
            "n_rows": len(body),
            "verdicts_carried_over": sum(
                1 for row in rows if verdicts[row.id]["signals_key"] != signals_key(cfg)
            ),
            "reviewers": dict(sorted(Counter(line["reviewer"] for line in body).items())),
            **agreement(rows, verdicts),
        },
    )
    return path


# --- the command ----------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    input_fn: Callable[[str], str] = input,
    runs_dir: Path = RUNS_DIR,
    log_path: Path | None = None,
    known: frozenset[str] | None = None,
) -> int:
    """Review one half's answers, or with ``--stats`` / ``--write-csv`` report on what is reviewed.

    ``--reviewer`` is required for a reviewing sitting and not for the two reporting modes: every
    verdict records who gave it, and a report of existing verdicts adds none.

    ``out``, ``input_fn``, ``runs_dir``, ``log_path`` and ``known`` are injected so a test can run
    the whole loop without a terminal or the committed runs.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS, help="which half to review")
    parser.add_argument("--reviewer", help="your name, recorded on every verdict you give")
    parser.add_argument("--redo", action="store_true", help="show rows already reviewed, too")
    parser.add_argument("--stats", action="store_true", help="progress and judge vs human")
    parser.add_argument("--write-csv", action="store_true", help="write the signals CSV")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="a frozen dataset")
    args = parser.parse_args(argv)
    log_path = LOG_PATH if log_path is None else Path(log_path)
    cfg = load_config()
    try:
        key = signals_key(cfg)
        rows = load_rows(args.split, cfg, args.dataset, runs_dir, known)
        current = status(rows, read_log(log_path), args.split)
        if current.stale:
            print(
                f"{len(current.stale)} verdict(s) are for an answer that has since changed and "
                f"count as unreviewed: {', '.join(current.stale[:5])}",
                file=out,
            )
        if args.stats:
            print(render_agreement(agreement(rows, current.verdicts)), file=out)
            return 0
        if args.write_csv:
            path = write_csv(
                rows,
                current.verdicts,
                current.stale,
                csv_path(args.split, cfg, runs_dir),
                cfg,
                log_path,
                judge_path(args.split, cfg, runs_dir),
                run_path(args.split, cfg, runs_dir),
            )
            print(f"wrote {rel_to_repo(path)} ({len(rows)} rows)", file=out)
            print(render_agreement(agreement(rows, current.verdicts)), file=out)
            return 0
        if not args.reviewer or not args.reviewer.strip():
            raise ValueError("--reviewer NAME is required: every verdict records who gave it")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"{args.split}: {len(current.verdicts)} of {len(rows)} rows reviewed; "
        f"verdicts go to {rel_to_repo(log_path)}\n",
        file=out,
    )
    made = run(
        rows,
        current.verdicts,
        key,
        args.reviewer.strip(),
        log_path,
        out,
        reader(out, input_fn),
        redo=args.redo,
    )
    done = status(rows, read_log(log_path), args.split)
    print(
        f"\nthis sitting: {sum(made.values())} verdict(s) "
        f"({', '.join(f'{n} {a}' for a, n in sorted(made.items())) or 'none'}); "
        f"{len(done.verdicts)} of {len(rows)} reviewed.",
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
