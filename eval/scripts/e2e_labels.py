"""Correctness labels for #103's end-to-end run: #88's judge first, then a person on every row.

    uv run --env-file .env python -m eval.scripts.e2e_labels --split test --judge
    uv run python -m eval.scripts.e2e_labels --split test --review --reviewer NAME  # resumable
    uv run python -m eval.scripts.e2e_labels --split test --stats
    uv run python -m eval.scripts.e2e_labels --split test --write-csv  # -> e2e_{split}_{hash}.csv

The pass is #88's, unchanged: the judge prompts are ``correctness.judge_prompts``, one per bucket,
the reply is parsed by :func:`eval.scripts.judge.parse_reply`, and the label is a person's verdict
-- never the judge's. Every function that decides anything here is imported from
``eval/scripts/judge.py`` or ``eval/scripts/review_signals.py`` rather than written again, so the
Day 6 labels mean what the Day 5 labels mean.

**Only the answered rows are labelled.** An ``answer`` exit is a text the model wrote and a person
can be right or wrong about; the other two exits are fixed sentences
(``agent/nodes/exits.py``) and judging a constant would put the judge's agreement rate at the mercy
of how often the gate fired. Whether a refusal was the *right* thing to do is #104's question, and
it reads the exits straight off the run file.

**A verdict belongs to one answer text.** As in #88, an entry records the sha256 of the answer it
was given on, and it applies wherever ``(id, answer_sha256)`` matches -- so an end-to-end answer
that came back byte-identical to the one in ``signals_{split}_{signals_key}.jsonl`` keeps the
verdict a person already gave it, and only the answers that changed are asked about again. The
carry-over is counted in the CSV's sidecar and printed before the queue starts.

**This pass has its own log**, ``eval/runs/e2e_correctness_log.jsonl``, in #88's format plus the
run each verdict was given on. Two reasons, both about not editing history: #88's
``correctness_log.jsonl`` is hashed into the sidecar of a CSV that is already committed, and a
verdict typed while looking at the end-to-end run came from a different run than one typed on the
signals run, which is worth being able to tell later. Both logs are read for carry-over; only this
one is appended to.

**Nothing here writes a verdict a person did not type.** :func:`write_csv` refuses to emit the CSV
until every answered row has one, and says how many are left. That refusal is the point of the
file: an end-to-end correctness number with a machine-filled label in it is not one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, file_sha256, load_config, rel_to_repo, signals_key
from eval.schema import BUCKETS, SPLITS, Split
from eval.scripts import review_signals as rs
from eval.scripts.ablation import write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.e2e import RUNS_DIR, run_path
from eval.scripts.judge import gold_by_id, judge_line, prompts, read_jsonl
from eval.scripts.judge import tally as judge_tally
from eval.scripts.review import DATASET_PATH, reader
from eval.scripts.signals import sha256_text
from eval.scripts.split import META_SUFFIX, load_split, meta_path, read_sidecar
from llm import Chat, check_llm_config, make_chat

#: This pass's verdicts, one line each. #88's ten keys, plus the run the verdict was given on --
#: so ``eval.scripts.review_signals.read_log`` can still read this file, and a reader can still
#: tell which run put the answer on the screen.
LOG_KEYS: tuple[str, ...] = (*rs.LOG_KEYS, "run", "run_key")

#: Committed, like #88's: it is where every ``correct`` in the CSV comes from.
LOG_PATH = RUNS_DIR / "e2e_correctness_log.jsonl"

#: The CSV's columns. The signals CSV's shape with the end-to-end run's own columns in place of
#: the three confidence signals: what the gate saw, and what it did.
CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "bucket",
    "split",
    "intent",
    "exit",
    "top_score",
    "gate_ok",
    "judge_correct",
    "correct",
    "reviewer",
)


def judge_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/e2e_judge_{split}_{config_hash}.jsonl``: the judge's first call on this run."""
    return Path(runs_dir) / f"e2e_judge_{split}_{cfg.hash}.jsonl"


def csv_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/e2e_{split}_{config_hash}.csv``, beside the run it labels."""
    return run_path(split, cfg, runs_dir).with_suffix(".csv")


# --- the rows -------------------------------------------------------------------------------------


def read_run(path: Path) -> list[dict[str, Any]]:
    """The end-to-end run file, refused by name if nobody has made it yet."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{rel_to_repo(path)} does not exist: run `python -m eval.scripts.e2e --split <half>` "
            "first. There is nothing to label until the system has answered."
        )
    return read_jsonl(path)


def answered(lines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows that left through the ``answer`` exit -- the ones a verdict is about."""
    return [dict(line) for line in lines if line["exit"] == "answer"]


def load_rows(
    split: Split,
    cfg: Config,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
    judged: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[rs.Row]:
    """The answered rows, each with its gold answer and the judge's call. Mismatches are refused.

    ``judged`` is read from the judge file unless it is handed in. A row the judge graded under a
    different answer text is refused rather than shown with a verdict about something else.
    """
    lines = answered(read_run(run_path(split, cfg, runs_dir)))
    source = judge_path(split, cfg, runs_dir)
    if judged is None:
        if not source.exists():
            raise ValueError(
                f"{rel_to_repo(source)} does not exist: run `python -m eval.scripts.e2e_labels "
                f"--split {split} --judge` first, so the reviewer sees the judge's call"
            )
        judged = {line["id"]: line for line in read_jsonl(source)}
    items = gold_by_id(load_split(split, dataset, known))
    rows: list[rs.Row] = []
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
            rows.append(rs.Row(line, item, dict(verdict)))
    if problems:
        raise ValueError("; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))
    return rows


# --- #88's judge, on this run ---------------------------------------------------------------------


def judge_answers(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
    chat: Chat | None = None,
) -> Path:
    """The judge's first call on every answered row, written beside the run it grades."""
    templates = prompts(cfg)
    source = run_path(split, cfg, runs_dir)
    lines = answered(read_run(source))
    items = gold_by_id(load_split(split, dataset, known))
    unknown = [line["id"] for line in lines if line["id"] not in items]
    if unknown:
        raise ValueError(
            f"{rel_to_repo(source)} has {len(unknown)} id(s) the {split} half of "
            f"{rel_to_repo(Path(dataset))} does not: {', '.join(unknown[:5])}"
        )
    if chat is None:
        check_llm_config(cfg)
        chat = make_chat(cfg)
    print(f"{split}: judging {len(lines)} answered rows of {rel_to_repo(source)}", file=out)
    judged: list[dict[str, Any]] = []
    for n, line in enumerate(lines, start=1):
        result = judge_line(line, items[line["id"]], templates, chat)
        judged.append(result)
        print(
            f"  [{n:>3}/{len(lines)}] {line['id']} {line['bucket']:<12} "
            f"{rs.word(result['judge_correct'])}",
            file=out,
        )

    path = write_lines(judge_path(split, cfg, runs_dir), judged)
    dataset_meta = read_sidecar(meta_path(Path(dataset))) or {}
    write_json(
        path.with_suffix(META_SUFFIX),
        {
            "run": rel_to_repo(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "e2e_run": rel_to_repo(source),
            "e2e_run_sha256": file_sha256(source),
            "dataset": rel_to_repo(Path(dataset)),
            "dataset_sha256": dataset_meta.get("sha256"),
            "split": split,
            "model": chat.model,
            "prompts_sha256": {b: sha256_text(templates[b]) for b in BUCKETS},
            "note": (
                "judge_correct is the LLM judge's first call on the answered rows, not the label; "
                "`correct` comes only from a person's verdict in the review log"
            ),
            "n_lines": len(judged),
            "by_bucket": judge_tally(judged),
        },
    )
    for bucket, by in judge_tally(judged).items():
        print(f"  {bucket:<12} " + ", ".join(f"{n} {k}" for k, n in sorted(by.items())), file=out)
    print(
        f"wrote {rel_to_repo(path)} ({len(judged)} lines). Not labels: run --review next.",
        file=out,
    )
    return path


# --- the human pass -------------------------------------------------------------------------------


class Logs(NamedTuple):
    """The two verdict logs: #88's, read for the carry-over, and this pass's, which is written to.

    Both are read and only ``own`` is appended to -- see "This pass has its own log" above.
    """

    prior: Path = rs.LOG_PATH
    own: Path = LOG_PATH


def entries(logs: Logs) -> list[dict[str, Any]]:
    """Every verdict a person has given, #88's log first and this pass's second.

    Order is precedence: :func:`eval.scripts.review_signals.latest` keeps the last entry for an
    ``(id, answer_sha256)``, so a verdict re-typed on this run wins over the one carried over.
    """
    return [*rs.read_log(logs.prior), *rs.read_log(logs.own)]


def append_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """One verdict, flushed and fsynced before the next row is shown (#88's discipline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({key: entry[key] for key in LOG_KEYS}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def review(
    rows: Sequence[rs.Row],
    done: Mapping[str, Mapping[str, Any]],
    cfg: Config,
    split: Split,
    reviewer: str,
    logs: Logs,
    run: Path,
    out: TextIO,
    ask: rs.Ask,
    redo: bool = False,
) -> Counter[str]:
    """Walk the queue, writing each verdict before the next row. Returns this sitting's actions."""
    queue = [row for row in rs.interleave(rows) if redo or row.id not in done]
    made: Counter[str] = Counter()
    for position, row in enumerate(queue, start=1):
        out.write(rs.render_row(row, position, len(queue), done.get(row.id)) + "\n\n")
        verdict = rs.ask_verdict(row, out, ask)
        if verdict is None:
            break
        correct, note = verdict
        judge_correct = row.judged["judge_correct"]
        entry = {
            "id": row.id,
            "split": split,
            "signals_key": signals_key(cfg),
            "answer_sha256": row.answer_sha256,
            "judge_correct": judge_correct,
            "correct": correct,
            "action": rs.action_of(judge_correct, correct),
            "reviewer": reviewer,
            "note": note,
            "ts": rs.now(),
            "run": rel_to_repo(run),
            "run_key": cfg.hash,
        }
        append_entry(logs.own, entry)
        made[entry["action"]] += 1
    return made


# --- the CSV --------------------------------------------------------------------------------------


def carried_over(
    rows: Sequence[rs.Row], verdicts: Mapping[str, Mapping[str, Any]], run: Path
) -> list[str]:
    """The ids whose verdict was given on another run: #88's pass, or an earlier end-to-end one."""
    return [row.id for row in rows if verdicts[row.id].get("run") != rel_to_repo(run)]


def write_csv(
    rows: Sequence[rs.Row],
    verdicts: Mapping[str, Mapping[str, Any]],
    stale: Sequence[str],
    path: Path,
    cfg: Config,
    logs: Logs,
    judge_file: Path,
    run: Path,
) -> Path:
    """The end-to-end CSV, or a refusal naming the rows no person has labelled."""
    missing = [row.id for row in rows if row.id not in verdicts]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        detail = f"; {len(stale)} of them have a verdict for an older answer" if stale else ""
        raise ValueError(
            f"refusing to write {rel_to_repo(path)}: {len(missing)} of {len(rows)} answered rows "
            f"have no human verdict ({shown}){detail}. `correct` comes only from the review log -- "
            "finish the pass with --review --reviewer NAME."
        )
    body = [
        {
            "id": row.id,
            "bucket": row.line["bucket"],
            "split": row.line["split"],
            "intent": row.line["intent"],
            "exit": row.line["exit"],
            "top_score": row.line["top_score"],
            "gate_ok": row.line["gate_ok"],
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
            writer.writerow([rs.cell(line[column]) for column in CSV_COLUMNS])
    os.replace(tmp, path)
    over = carried_over(rows, verdicts, run)
    write_json(
        path.with_name(path.name + ".meta.json"),
        {
            "csv": rel_to_repo(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "e2e_run": rel_to_repo(run),
            "e2e_run_sha256": file_sha256(run),
            "judge_run": rel_to_repo(judge_file),
            "judge_run_sha256": file_sha256(judge_file),
            "review_logs": {rel_to_repo(p): file_sha256(p) for p in logs if p.exists()},
            "n_answered_rows": len(body),
            "verdicts_carried_over": len(over),
            "carried_over_ids": over,
            "reviewers": dict(sorted(Counter(line["reviewer"] for line in body).items())),
            **rs.agreement(rows, verdicts),
        },
    )
    return path


# --- the command ----------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    input_fn: Callable[[str], str] = input,
    runs_dir: Path = RUNS_DIR,
    logs: Logs | None = None,
    known: frozenset[str] | None = None,
    chat: Chat | None = None,
) -> int:
    """Label one half's answers: exactly one of ``--judge``, ``--review``, ``--stats``, CSV.

    Requiring exactly one is the point -- the judge pass and the human pass write different logs,
    and a run that did both would leave no record of which verdict came first.

    ``out``, ``input_fn``, ``runs_dir``, ``logs``, ``known`` and ``chat`` are injected so a test
    can drive the whole loop without a terminal, a model or the committed runs.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS, help="which half to label")
    parser.add_argument("--judge", action="store_true", help="the LLM judge's first pass")
    parser.add_argument("--review", action="store_true", help="the human pass, resumable")
    parser.add_argument("--reviewer", help="your name, recorded on every verdict you give")
    parser.add_argument("--redo", action="store_true", help="show rows already reviewed, too")
    parser.add_argument("--stats", action="store_true", help="progress and judge vs human")
    parser.add_argument("--write-csv", action="store_true", help="write the end-to-end CSV")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="a frozen dataset")
    args = parser.parse_args(argv)
    logs = Logs() if logs is None else logs
    cfg = load_config()
    run = run_path(args.split, cfg, runs_dir)
    chosen = [name for name in ("judge", "review", "stats", "write_csv") if getattr(args, name)]
    try:
        if len(chosen) != 1:
            raise ValueError(
                "choose exactly one of --judge, --review, --stats, --write-csv; "
                f"got {', '.join('--' + c.replace('_', '-') for c in chosen) or 'none'}"
            )
        if args.judge:
            judge_answers(cfg, args.split, out, args.dataset, runs_dir, known, chat)
            return 0
        rows = load_rows(args.split, cfg, args.dataset, runs_dir, known)
        current = rs.status(rows, entries(logs), args.split)
        if current.stale:
            print(
                f"{len(current.stale)} verdict(s) are for an answer that has since changed and "
                f"count as unreviewed: {', '.join(current.stale[:5])}",
                file=out,
            )
        if args.stats:
            over = [r for r in rows if r.id in current.verdicts]
            print(
                f"{len(current.verdicts)} of {len(rows)} answered rows reviewed; "
                f"{len(carried_over(over, current.verdicts, run))} verdict(s) carried over",
                file=out,
            )
            print(rs.render_agreement(rs.agreement(rows, current.verdicts)), file=out)
            return 0
        if args.write_csv:
            path = write_csv(
                rows,
                current.verdicts,
                current.stale,
                csv_path(args.split, cfg, runs_dir),
                cfg,
                logs,
                judge_path(args.split, cfg, runs_dir),
                run,
            )
            print(f"wrote {rel_to_repo(path)} ({len(rows)} answered rows)", file=out)
            print(rs.render_agreement(rs.agreement(rows, current.verdicts)), file=out)
            return 0
        if not args.reviewer or not args.reviewer.strip():
            raise ValueError("--reviewer NAME is required: every verdict records who gave it")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    reviewed = [r for r in rows if r.id in current.verdicts]
    print(
        f"{args.split}: {len(current.verdicts)} of {len(rows)} answered rows reviewed "
        f"({len(carried_over(reviewed, current.verdicts, run))} carried over from another run); "
        f"new verdicts go to {rel_to_repo(logs.own)}\n",
        file=out,
    )
    made = review(
        rows,
        current.verdicts,
        cfg,
        args.split,
        args.reviewer.strip(),
        logs,
        run,
        out,
        reader(out, input_fn),
        redo=args.redo,
    )
    done = rs.status(rows, entries(logs), args.split)
    print(
        f"\nthis sitting: {sum(made.values())} verdict(s) "
        f"({', '.join(f'{n} {a}' for a, n in sorted(made.items())) or 'none'}); "
        f"{len(done.verdicts)} of {len(rows)} reviewed.",
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
