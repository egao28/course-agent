"""Day 6's four end-to-end numbers, off #103's run and #102's gold intents (#104).

    uv sync --extra analysis        # scipy (via scikit-learn), for the Clopper-Pearson interval
    uv run python -m eval.scripts.e2e_metrics

Reads four committed files and writes ``eval/runs/e2e_metrics_{config_hash}.json``: the run
(``eval/runs/e2e_test_{config_hash}.jsonl``), its human-verdict CSV, the gold intents
(``eval/datasets/v1_intents.jsonl``) and the frozen dataset. It calls no model, reads no label
that #103 did not already read, and fits nothing -- so unlike ``eval/scripts/conformal_eval.py``
it does not refuse a second run. Delete the JSON, change nothing, run it again and the same
numbers come back.

**The test half, and only it.** There is no ``--split``: the gold intents are labelled for the
test half alone (#102), and a calib row has none. A number over the rows that happened to have a
label is a number about the labelling.

**Each metric is k of n with a 95% Clopper-Pearson interval**, the level and the method #91 used
(``eval.bootstrap.ci``, and :func:`eval.scripts.conformal_eval.clopper_pearson` itself -- imported,
not written again). Per bucket the same k and n are given as point estimates with no interval: at
n = 10 to 12 an exact interval spans most of the unit interval, and printing one invites a
comparison the half cannot support.

**The four.**

* *intent accuracy* -- the router's ``intent`` is in the row's ``gold_intents``. A set, not a
  label (#102): a fact question about a prerequisite is legitimately either reading.
* *end-to-end answer correctness* -- the row's outcome is right for its bucket, below.
* *false release* -- a ``should_abstain`` row that answered anyway.
* *false block* -- a ``should_abstain = false`` row that did not answer.

**Correct per bucket**, as #88's judge prompts put it: a ``fact`` or ``multihop`` row is correct
when a person judged its answer correct, an ``unanswerable`` row when the system declined, and an
``ambiguous`` row when it asked which course was meant. Two of those are read off the exit and one
off a verdict, so the three exits (:data:`agent.state.EXITS`) and :data:`eval.schema.RULES` do the
work: a row that did not answer is correct exactly when its exit is one its bucket accepts, and a
row that answered is correct exactly when a person said so.

**There is no clarify exit**, so an abstaining row that was handled well can still leave through
``answer`` -- that is what #88's unanswerable and ambiguous prompts judge. Both correctness and
false release are therefore reported two ways, and the gap is a finding rather than noise:

* *exit-only*: an abstaining row that left through ``answer`` released, whatever it said.
* *verdict-backed*: it released unless the person who read the text said it declined or asked for
  clarification. On this run that is 2 rows rather than 4, so the exits alone overstate the
  release rate by a factor of two.

Nothing else differs between the two. A ``fact`` or ``multihop`` row has no reading but its
verdict, and an abstaining row that never answered has no verdict to read.

**One caveat has to travel with the ambiguous bucket.** Nine of its ten rows ended in
``refuse_handoff``, which :data:`eval.schema.RULES` accepts, and this file counts them correct on
that. But the hand-off is fixed text (``agent/nodes/exits.py``) and #88's ambiguous prompt calls a
reply that "declines without asking" incorrect -- and nobody read those nine, because
``eval/scripts/e2e_labels.py`` labels answered rows only. Some of them do point at what is missing
("name it by code"), others only report a score under a threshold, and which of those counts as
asking is a person's call, not a rule's. So the headline is an upper bound: the JSON carries the
nine ids and the correctness the run would show if none of them counted, and #106 reports the
range, not the top of it.

**A run with an errored row is refused.** #103 says such a run is an infrastructure failure and
not a result; a denominator that quietly drops a row is how it would become one anyway.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from agent.state import INTENTS, Exit, Intent
from config import Config, file_sha256, load_config, rel_to_repo
from eval import intents as gold_intents
from eval.schema import BUCKETS, Bucket, rule
from eval.scripts.conformal_eval import clopper_pearson
from eval.scripts.draft_items import write_json
from eval.scripts.e2e import RUNS_DIR, run_path
from eval.scripts.e2e_labels import CSV_COLUMNS, csv_path
from eval.scripts.judge import read_jsonl
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX, load_split, read_sidecar

#: The half this reads. Fixed, like ``eval/scripts/conformal_eval.py``'s: the gold intents exist
#: for the test half only, and every final number is reported on it.
SPLIT = "test"

#: The two readings of an abstaining row that left through ``answer``. See the module docstring.
READINGS: tuple[str, ...] = ("verdict_backed", "exit_only")


# --- one row, joined ------------------------------------------------------------------------------


class Row(NamedTuple):
    """One test row as the four metrics see it: what was asked of it, and what the system did.

    ``verdict`` is the person's call from the CSV and is ``None`` on every row that did not leave
    through ``answer`` -- those were never labelled, by #88's rule that judging a fixed sentence
    measures how often the gate fired rather than how good the reply was.
    """

    id: str
    bucket: Bucket
    exit: Exit
    intent: Intent | None
    gold_intents: frozenset[str]
    should_abstain: bool
    verdict: bool | None


def intent_hit(row: Row) -> bool:
    """The router put the row in one of the intents its labeller accepts."""
    return row.intent in row.gold_intents


def answered(row: Row) -> bool:
    """It left through ``answer``: the user saw a written reply, not a fixed refusal."""
    return row.exit == "answer"


def correct(row: Row, reading: str = "verdict_backed") -> bool:
    """Was the outcome right for this row's bucket? See "Correct per bucket" above.

    The two readings differ on one case only -- an abstaining row that answered -- so the branch
    for it is the only place ``reading`` appears.
    """
    if reading not in READINGS:
        raise ValueError(f"reading must be one of {', '.join(READINGS)}, got {reading!r}")
    if not answered(row):
        return row.exit in rule(row.bucket).exits
    if not row.should_abstain:
        return row.verdict is True
    return reading == "verdict_backed" and row.verdict is True


def false_release(row: Row, reading: str = "verdict_backed") -> bool:
    """It should have declined and did not. The complement of :func:`correct` on answered rows."""
    return row.should_abstain and answered(row) and not correct(row, reading)


def false_block(row: Row) -> bool:
    """It should have answered and refused. One reading: a fact row that did not answer is one."""
    return not row.should_abstain and not answered(row)


def unread_handoffs(rows: Sequence[Row]) -> list[str]:
    """The ambiguous rows counted correct on an exit whose fixed text nobody read.

    The caveat in the module docstring, as ids. Only ``ambiguous``: what an unanswerable row's
    bucket asks for is a decline, and both fixed messages plainly decline, so there is no gap
    there to report.
    """
    return [row.id for row in rows if row.bucket == "ambiguous" and not answered(row)]


# --- the numbers ----------------------------------------------------------------------------------


def score(
    numerator: Sequence[Row],
    denominator: Sequence[Row],
    level: float | None = None,
    denominator_ids: bool = True,
) -> dict[str, Any]:
    """``k`` of ``n`` with the ids behind ``k``, and the interval when a level is given.

    Every number this file reports goes through here, so every one of them carries the rows it
    was counted from: that is #104's acceptance, and #107 checks the report against these lists.
    A per-bucket block leaves the denominator out -- it is its metric's denominator intersected
    with ``rows_by_bucket``, and the same 63 ids twenty times over is a file nobody reads.
    """
    k, n = len(numerator), len(denominator)
    if k > n:
        raise ValueError(f"{k} rows in a numerator of {n}: the two sets are not nested")
    block: dict[str, Any] = {
        "k": k,
        "n": n,
        "rate": k / n if n else None,
        "numerator_ids": [row.id for row in numerator],
    }
    if denominator_ids:
        block["denominator_ids"] = [row.id for row in denominator]
    if level is not None:
        lo, hi = clopper_pearson(k, n, level)
        block["ci"] = [lo, hi]
    return block


#: A predicate over one row: the numerator of a metric, per bucket and over the whole half.
Hit = Callable[[Row], bool]


def by_bucket(rows: Sequence[Row], hit: Hit) -> dict[str, Any]:
    """The same count per bucket, point estimates only. A bucket with no row is left out."""
    out: dict[str, Any] = {}
    for bucket in BUCKETS:
        rows_in = [row for row in rows if row.bucket == bucket]
        if rows_in:
            out[bucket] = score(
                [row for row in rows_in if hit(row)], rows_in, denominator_ids=False
            )
    return out


def metric(among: Sequence[Row], hit: Hit, level: float) -> dict[str, Any]:
    """One metric: the interval over ``among``, and the same fraction per bucket beneath it."""
    return {
        **score([row for row in among if hit(row)], among, level),
        "by_bucket": by_bucket(among, hit),
    }


def misses(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Every row the router put outside its gold set, with both sides: #105 attributes from it."""
    return [
        {
            "id": row.id,
            "bucket": row.bucket,
            "predicted": row.intent,
            "gold_intents": sorted(row.gold_intents),
            "exit": row.exit,
            "outcome_correct_verdict_backed": correct(row),
        }
        for row in rows
        if not intent_hit(row)
    ]


def caveat(rows: Sequence[Row], level: float) -> dict[str, Any]:
    """The ambiguous hand-offs, and what end-to-end correctness would be if none of them counted.

    A bound, not a second result. The ids are here so a person can settle it by reading nine
    replies; until someone does, the true number is between this and the headline.
    """
    unread = unread_handoffs(rows)
    kept = [row for row in rows if row.id not in set(unread)]
    return {
        "ids": unread,
        "n": len(unread),
        "why": (
            "ambiguous rows counted correct because refuse_handoff is an exit eval.schema.RULES "
            "accepts for the bucket. The hand-off is fixed text and #88's ambiguous prompt calls "
            "a reply that declines without asking incorrect, and no person read these -- "
            "e2e_labels.py labels answered rows only. Whether a hand-off that says 'name it by "
            "code' asks the question is a judgement, so this is the lower bound, not a verdict"
        ),
        "correctness_if_none_counted": score(
            [row for row in kept if correct(row)], rows, level
        ),
    }


def measure(rows: Sequence[Row], level: float) -> dict[str, Any]:
    """The four metrics, both readings of the two that have two, and the caveat."""
    abstaining = [row for row in rows if row.should_abstain]
    answerable = [row for row in rows if not row.should_abstain]
    ids_by_bucket = {
        b: [row.id for row in rows if row.bucket == b]
        for b in BUCKETS
        if any(row.bucket == b for row in rows)
    }
    return {
        "n": len(rows),
        "buckets": {b: len(ids) for b, ids in ids_by_bucket.items()},
        "rows_by_bucket": ids_by_bucket,
        "ci_level": level,
        "ci_method": "Clopper-Pearson, two-sided",
        "metrics": {
            "intent_accuracy": {
                **metric(rows, intent_hit, level),
                "predicted": {i: sum(row.intent == i for row in rows) for i in INTENTS},
                "misses": misses(rows),
            },
            "answer_correctness": {
                reading: metric(rows, lambda row, r=reading: correct(row, r), level)
                for reading in READINGS
            },
            "false_release": {
                reading: metric(
                    abstaining, lambda row, r=reading: false_release(row, r), level
                )
                for reading in READINGS
            },
            "false_block": metric(answerable, false_block, level),
        },
        "caveats": {"ambiguous_handoff": caveat(rows, level)},
    }


# --- the inputs -----------------------------------------------------------------------------------


def read_run(cfg: Config, runs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """#103's run file and its sidecar, refused unless it is the file the sidecar describes."""
    path = run_path(SPLIT, cfg, runs_dir)
    meta = read_sidecar(path.with_suffix(META_SUFFIX))
    if not path.exists() or meta is None:
        raise ValueError(
            f"no end-to-end run at {rel_to_repo(path)}: run "
            f"`python -m eval.scripts.e2e --split {SPLIT}` first -- there is nothing to measure "
            "until the system has answered the half"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("config_hash") != cfg.hash:
        problems.append(f"it was run at config {meta.get('config_hash')}, not {cfg.hash}")
    if meta.get("split") != SPLIT:
        problems.append(f"it is the {meta.get('split')} half, not {SPLIT}")
    if meta.get("errors"):
        problems.append(
            f"{meta['errors']} of its rows failed to run, so it is not a result: delete it and "
            "run it again (#103)"
        )
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))
    return read_jsonl(path), meta


def read_verdicts(cfg: Config, runs_dir: Path, run: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    """The human verdict per answered row, from #103's CSV, tied to the run file on disk."""
    path = csv_path(SPLIT, cfg, runs_dir)
    meta = read_sidecar(path.with_name(path.name + ".meta.json"))
    if not path.exists() or meta is None:
        raise ValueError(
            f"no verdict CSV at {rel_to_repo(path)}: finish the human pass and run "
            f"`python -m eval.scripts.e2e_labels --split {SPLIT} --write-csv` first"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("e2e_run_sha256") != file_sha256(run):
        problems.append(f"it labels another copy of {rel_to_repo(run)} than the one on disk")
    if meta.get("config_hash") != cfg.hash:
        problems.append(f"it was written at config {meta.get('config_hash')}, not {cfg.hash}")
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"{rel_to_repo(path)}: columns are {reader.fieldnames}, "
                             f"not #103's {list(CSV_COLUMNS)}")
        lines = list(reader)
    bad = [line["id"] for line in lines if line["correct"] not in ("0", "1")]
    if bad:
        raise ValueError(f"{rel_to_repo(path)}: rows without a 0/1 correct {bad[:5]}")
    return {line["id"]: line["correct"] == "1" for line in lines}, meta


def read_intents(
    sidecar: Path, dataset: Path, guidelines: Path
) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    """#102's labels, refused unless :mod:`eval.intents` passes them at their stamped digests."""
    violations = [v for v in gold_intents.validate(sidecar, dataset, guidelines) if
                  v.level == "error"]
    if violations:
        raise ValueError(
            f"{rel_to_repo(sidecar)} does not pass `python -m eval.intents`: "
            + "; ".join(str(v.message) for v in violations[:5])
        )
    rows, _ = gold_intents.parse_sidecar(sidecar)
    meta = gold_intents.read_meta(sidecar) or {}
    return {r.row.id: frozenset(r.row.gold_intents) for r in rows}, meta


def join(
    lines: Sequence[Mapping[str, Any]],
    items: Mapping[str, bool],
    labels: Mapping[str, frozenset[str]],
    verdicts: Mapping[str, bool],
) -> list[Row]:
    """One :class:`Row` per run line, or a refusal naming every row the four files disagree on."""
    problems: list[str] = []
    rows: list[Row] = []
    for line in lines:
        item_id = line["id"]
        if item_id not in items:
            problems.append(f"{item_id} is in the run and not in the {SPLIT} half")
            continue
        if item_id not in labels:
            problems.append(f"{item_id} has no gold intent")
            continue
        if (line["exit"] == "answer") != (item_id in verdicts):
            problems.append(
                f"{item_id} left through {line['exit']} and the CSV "
                + ("has a verdict for it" if item_id in verdicts else "has none for it")
                + "; the CSV labels the rows that answered, and only those"
            )
            continue
        rows.append(
            Row(
                id=item_id,
                bucket=line["bucket"],
                exit=line["exit"],
                intent=line["intent"],
                gold_intents=labels[item_id],
                should_abstain=items[item_id],
                verdict=verdicts.get(item_id),
            )
        )
    unrun = [i for i in items if i not in {line["id"] for line in lines}]
    if unrun:
        problems.append(f"{len(unrun)} row(s) of the half have no line in the run: {unrun[:5]}")
    if problems:
        raise ValueError("refusing to measure: " + "; ".join(problems[:5])
                         + (" ..." if len(problems) > 5 else ""))
    return rows


def result_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/e2e_metrics_{config_hash}.json``, named after the run it measures."""
    return Path(runs_dir) / f"e2e_metrics_{cfg.hash}.json"


# --- saying what happened ---------------------------------------------------------------------


def _rate(block: Mapping[str, Any]) -> str:
    if block["rate"] is None:
        return "- (no rows)"
    ci = block.get("ci")
    interval = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{block['rate']:.3f}{interval}"


def _cell(metric_block: Mapping[str, Any], bucket: str) -> str:
    block = metric_block["by_bucket"].get(bucket)
    return "-" if block is None else f"{block['k']}/{block['n']} ({block['rate']:.2f})"


def summary(result: Mapping[str, Any]) -> str:
    """The headline table, the per-bucket table, and the caveat -- what #106 quotes."""
    ci = round(result["ci_level"] * 100)
    m = result["metrics"]
    named: list[tuple[str, Mapping[str, Any]]] = [
        ("intent accuracy", m["intent_accuracy"]),
        ("end-to-end correctness (verdict-backed)", m["answer_correctness"]["verdict_backed"]),
        ("end-to-end correctness (exit-only)", m["answer_correctness"]["exit_only"]),
        ("false release (verdict-backed)", m["false_release"]["verdict_backed"]),
        ("false release (exit-only)", m["false_release"]["exit_only"]),
        ("false block", m["false_block"]),
    ]
    out = [
        f"{result['split']} n = {result['n']} {result['buckets']}, {ci}% Clopper-Pearson",
        (
            f"run {result['run']}, labels {result['csv']}, "
            f"gold intents {result['intents']} by {result['intents_labeller']}"
        ),
        "",
        f"| metric | k / n | rate [{ci}% CI] |",
        "|---|---|---|",
    ]
    out += [f"| {name} | {b['k']} / {b['n']} | {_rate(b)} |" for name, b in named]
    buckets = list(result["buckets"])
    out += [
        "",
        (
            "Per bucket: point estimates, no interval -- at these n an exact interval spans "
            "most of the unit interval."
        ),
        "",
        (
            "| bucket | n | intent acc | correct (verdict) | correct (exit) | false release "
            "| false block |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for bucket in buckets:
        out.append(
            f"| {bucket} | {result['buckets'][bucket]} "
            f"| {_cell(m['intent_accuracy'], bucket)} "
            f"| {_cell(m['answer_correctness']['verdict_backed'], bucket)} "
            f"| {_cell(m['answer_correctness']['exit_only'], bucket)} "
            f"| {_cell(m['false_release']['verdict_backed'], bucket)} "
            f"| {_cell(m['false_block'], bucket)} |"
        )
    empty = [i for i, n in m["intent_accuracy"]["rows_with_gold_label"].items() if not n]
    if empty:
        out += [
            "",
            (
                f"no test row is labelled {', '.join(empty)}: the half carries no sample of "
                "that intent, and #106 says so rather than reporting 0"
            ),
        ]
    for miss in m["intent_accuracy"]["misses"]:
        out.append(
            f"  intent miss: {miss['id']} ({miss['bucket']}) -> {miss['predicted']}, gold "
            f"{'/'.join(miss['gold_intents'])}; its outcome was "
            + ("right anyway" if miss["outcome_correct_verdict_backed"] else "wrong too")
        )
    amb = result["caveats"]["ambiguous_handoff"]
    bound = amb["correctness_if_none_counted"]
    out += [
        "",
        (
            f"caveat: {amb['n']} ambiguous row(s) are counted correct on their exit and nobody "
            f"read the text ({', '.join(amb['ids'])})."
        ),
        (
            f"If none of them counted, end-to-end correctness would be {bound['k']} / "
            f"{bound['n']} ({_rate(bound)}) instead. The report gives the range."
        ),
    ]
    return "\n".join(out)


def run(
    cfg: Config,
    out: TextIO,
    runs_dir: Path = RUNS_DIR,
    dataset: Path = DATASET_PATH,
    sidecar: Path = gold_intents.SIDECAR_PATH,
    guidelines: Path = gold_intents.GUIDELINES_PATH,
) -> Path:
    """Check every input against its sidecar, count, write the JSON and print the tables."""
    lines, run_meta = read_run(cfg, runs_dir)
    source = run_path(SPLIT, cfg, runs_dir)
    verdicts, csv_meta = read_verdicts(cfg, runs_dir, source)
    labels, intents_meta = read_intents(Path(sidecar), Path(dataset), Path(guidelines))
    items = {item.id: item.should_abstain for item in load_split(SPLIT, dataset)}
    rows = join(lines, items, labels, verdicts)

    level = float(cfg["eval"]["bootstrap"]["ci"])
    result = measure(rows, level)
    result["metrics"]["intent_accuracy"]["rows_with_gold_label"] = {
        intent: intents_meta.get("rows_with_label", {}).get(intent, 0) for intent in INTENTS
    }
    result = {
        "split": SPLIT,
        "config_hash": cfg.hash,
        "run": rel_to_repo(source),
        "run_sha256": run_meta["sha256"],
        "csv": rel_to_repo(csv_path(SPLIT, cfg, runs_dir)),
        "csv_sha256": csv_meta["sha256"],
        "intents": rel_to_repo(Path(sidecar)),
        "intents_sha256": intents_meta.get("sha256"),
        "intents_labeller": intents_meta.get("labeller"),
        "guidelines_intent_sha256": intents_meta.get("guidelines_intent_sha256"),
        "dataset": rel_to_repo(Path(dataset)),
        "dataset_sha256": run_meta.get("dataset_sha256"),
        **result,
    }
    path = write_json(result_path(cfg, runs_dir), result)
    print(summary(result), file=out)
    print(f"\nwrote {rel_to_repo(path)}", file=out)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Join the run, its labels and the gold intents into the metrics JSON; a refusal exits 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="where the run, its CSV and the JSON this writes live")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="the frozen dataset the run and the gold intents were made on")
    parser.add_argument("--intents", type=Path, default=gold_intents.SIDECAR_PATH,
                        help="#102's gold intents for the test half")
    parser.add_argument("--guidelines", type=Path, default=gold_intents.GUIDELINES_PATH,
                        help="the rules the intents were labelled under; its digest is checked")
    args = parser.parse_args(argv)
    try:
        run(load_config(), sys.stdout, args.runs_dir, args.dataset, args.intents, args.guidelines)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
