"""RQ3: #90's calib-fit conformal threshold against a fixed 0.7, applied once to the test half.

    uv sync --extra analysis        # scipy (via scikit-learn), for the Clopper-Pearson interval
    uv run python -m eval.scripts.conformal_eval

Reads two things only: #88's test CSV, ``eval/runs/signals_test_{signals_key}.csv``, and #90's
fit, ``eval/runs/conformal_{conformal_key}.json``. It writes
``eval/runs/conformal_eval_{conformal_key}.json`` and prints the tables. It calls no model and
fits nothing: every threshold is fixed before the test labels are opened.

**Provenance is checked before any label is read.** The run is refused unless all of these hold:
the test CSV matches its sidecar and this config's signals key (#89's
:func:`~eval.scripts.risk_coverage.load_table`); it grades this config's judge run, so its
correctness key is this config's; the fit is named after, and records, this config's conformal,
correctness and signals keys and alpha; the fit's calib CSV is still the file it was fitted on;
and the signals runs behind *both* halves were collected on the dataset whose sha256 is the
committed ``eval/datasets/v1.jsonl``'s.

**The test half is read once.** If the result file exists, the run is refused. Deleting it to
re-run after a change of inputs is allowed; re-running to get a different number is not, and there
is nothing to tune here anyway.

**The two methods.** Both answer the rows with ``signal >= t`` and skip the rest:

- *conformal*: ``t`` is #90's, read by :func:`eval.conformal.threshold`. ``t = +inf`` answers
  nothing, and then there is no realized error: the table says ``-``, not 0.
- *fixed 0.7*: the template's threshold, put on each signal's scale without looking at data.
  ``verbalized`` is 0-100, so it is compared as ``verbalized / 100 >= 0.7``. ``self_consistency``
  takes the values 0, 1/3, 2/3 and 1, so 0.7 means all 3 pairs agree. ``retrieval_score`` is raw
  BM25 top-1 and unbounded, so 0.7 is compared as is; the output gives the signal's test range
  beside it, because an unnormalized fixed threshold failing is what this comparison shows.

**Realized error** is wrong / answered on test, with a two-sided Clopper-Pearson interval at the
``eval.bootstrap.ci`` level (0.95), the same level as every other interval in the report. Coverage
is given as a count, answered of the signal's non-null rows, and as a fraction. A null row
(``retrieval_score`` on tool rows) cannot clear any threshold, so it is not answered. What the gate
does on it in production is #92's decision.

**Nothing here decides whether conformal "worked".** Its guarantee holds in expectation over calib
draws, not on this one test half, and about 40 answered rows put one standard error near 5 points.
The file records the numbers and ``above_target``, a comparison, not a verdict.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from config import (
    Config,
    conformal_key,
    correctness_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval.conformal import result_path as fit_path
from eval.conformal import threshold
from eval.scripts.draft_items import write_json
from eval.scripts.judge import judge_path
from eval.scripts.review import DATASET_PATH
from eval.scripts.review_signals import csv_path
from eval.scripts.risk_coverage import SIGNALS, Table, load_table
from eval.scripts.signals import RUNS_DIR, run_path
from eval.scripts.split import META_SUFFIX, read_sidecar

#: The half this reads. Fixed: the thresholds were fitted on the other one.
SPLIT = "test"

#: The template's threshold, on a 0-1 scale.
FIXED = 0.7

#: How 0.7 is put on each signal's own scale: ``signal >= FIXED * SCALE[signal]``.
SCALE: Mapping[str, float] = {"retrieval_score": 1.0, "verbalized": 100.0, "self_consistency": 1.0}

#: What 0.7 means on each signal, as the tables print it.
MEANING: Mapping[str, str] = {
    "retrieval_score": "raw BM25 top-1 >= 0.7 (unbounded score, not normalized)",
    "verbalized": "verbalized / 100 >= 0.7, i.e. verbalized >= 70",
    "self_consistency": "between 2/3 and 1: all 3 pairs agree",
}

METHODS: tuple[str, ...] = ("conformal", "fixed")


# --- the numbers: pure, no I/O ------------------------------------------------------------------


def clopper_pearson(k: int, n: int, level: float) -> tuple[float, float]:
    """Exact two-sided interval for ``k`` successes of ``n``; ``(0, 1)`` when ``n = 0``."""
    from scipy.stats import beta

    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k = {k}, n = {n}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    tail = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lo, hi


def apply(
    scores: Sequence[float | None], correct: Sequence[int], t: float, level: float
) -> dict[str, Any]:
    """Answer the rows with ``score >= t``: coverage as counts, realized error with its interval."""
    if len(scores) != len(correct):
        raise ValueError(f"{len(scores)} scores for {len(correct)} labels")
    present = [(s, c) for s, c in zip(scores, correct, strict=True) if s is not None]
    answered = [c for s, c in present if s >= t]
    n, k = len(answered), len(answered) - sum(answered)
    lo, hi = clopper_pearson(k, n, level)
    return {
        "t": None if math.isinf(t) else t,
        "answers_nothing": n == 0,
        "n_non_null": len(present),
        "n_null": len(scores) - len(present),
        "answered": n,
        "wrong": k,
        "coverage": n / len(present) if present else 0.0,
        "realized_error": k / n if n else None,
        "ci": [lo, hi],
    }


def evaluate(
    table: Table, fit: Mapping[str, Any], level: float, dense_top1: Sequence[float | None] = ()
) -> dict[str, Any]:
    """Both methods on every signal, with the fit's chosen signal on top."""
    alpha = float(fit["alpha"])
    per_signal: dict[str, Any] = {}
    for name in SIGNALS:
        values = table.signals[name]
        conformal = apply(values, table.correct, threshold(fit["signals"][name]), level)
        conformal["target_error"] = alpha
        conformal["above_target"] = (None if conformal["realized_error"] is None
                                     else conformal["realized_error"] > alpha)
        fixed = apply(values, table.correct, FIXED * SCALE[name], level)
        fixed["target_error"] = None
        fixed["meaning"] = MEANING[name]
        present = [v for v in values if v is not None]
        per_signal[name] = {
            "conformal": conformal,
            "fixed": fixed,
            "test_range": [min(present), max(present)] if present else None,
        }
    # Diagnostic only: dense cosine top-1 is in the CSV but is not a candidate signal. It shows
    # where a raw 0.7 would fall had the agent retrieved with dense.
    dense = [v for v in dense_top1 if v is not None]
    wrong = len(table.correct) - sum(table.correct)
    lo, hi = clopper_pearson(wrong, len(table.correct), level)
    return {
        "split": SPLIT,
        "n": len(table.ids),
        "buckets": {b: table.buckets.count(b) for b in sorted(set(table.buckets))},
        "alpha": alpha,
        "fixed": FIXED,
        "ci_level": level,
        "ci_method": "Clopper-Pearson, two-sided",
        "chosen": fit["chosen"]["signal"],
        "answer_everything": {"answered": len(table.correct), "wrong": wrong,
                              "realized_error": wrong / len(table.correct), "ci": [lo, hi]},
        "dense_top1": {
            "n": len(dense),
            "max": max(dense) if dense else None,
            "answered_at_0.7": sum(v >= FIXED for v in dense),
        },
        "signals": per_signal,
    }


# --- the inputs ---------------------------------------------------------------------------------


def _signals_run_meta(split: str, cfg: Config, runs_dir: Path) -> dict[str, Any] | None:
    return read_sidecar(run_path(split, cfg, runs_dir).with_suffix(META_SUFFIX))


def check_provenance(
    table: Table, fit: Mapping[str, Any], cfg: Config, runs_dir: Path, dataset: Path
) -> dict[str, Any]:
    """Every hash that ties the test CSV and the fit to this config and dataset, or ValueError."""
    problems: list[str] = []
    alpha = float(cfg["conformal"]["alpha"])
    for field, want in (("conformal_key", conformal_key(cfg)),
                        ("correctness_key", correctness_key(cfg)),
                        ("signals_key", signals_key(cfg))):
        if fit.get(field) != want:
            problems.append(f"the fit has {field} {fit.get(field)}, this config {want}")
    if fit.get("split") != "calib":
        problems.append(f"the fit was made on {fit.get('split')!r}, not calib")
    if fit.get("alpha") != alpha:
        problems.append(f"the fit has alpha {fit.get('alpha')}, this config {alpha}")
    calib_csv = csv_path("calib", cfg, runs_dir)
    if not calib_csv.exists() or file_sha256(calib_csv) != fit.get("csv_sha256"):
        problems.append(f"{rel_to_repo(calib_csv)} is not the calib CSV the fit was made on")

    judge = judge_path(SPLIT, cfg, runs_dir)
    if Path(str(table.meta.get("judge_run", ""))).name != judge.name:
        problems.append(f"the test CSV grades {table.meta.get('judge_run')}, not {judge.name}, "
                        "so its labels are not this config's correctness key")

    dataset_sha = file_sha256(dataset) if Path(dataset).exists() else None
    if dataset_sha is None:
        problems.append(f"no dataset at {rel_to_repo(dataset)}")
    runs: dict[str, Any] = {}
    calib_meta = read_sidecar(calib_csv.with_name(calib_csv.name + ".meta.json")) or {}
    for split, csv_meta in (("calib", calib_meta), (SPLIT, table.meta)):
        path = run_path(split, cfg, runs_dir)
        meta = _signals_run_meta(split, cfg, runs_dir)
        if meta is None:
            problems.append(f"no sidecar for the {split} signals run {rel_to_repo(path)}")
            continue
        if csv_meta.get("signals_run_sha256") != meta.get("sha256"):
            problems.append(f"the {split} CSV was not built from the {split} signals run on disk")
        if meta.get("dataset_sha256") != dataset_sha:
            problems.append(f"the {split} signals run was collected on dataset "
                            f"{str(meta.get('dataset_sha256'))[:12]}, not "
                            f"{rel_to_repo(dataset)} ({str(dataset_sha)[:12]})")
        runs[split] = {"run": rel_to_repo(path), "sha256": meta.get("sha256")}
    if problems:
        raise ValueError("refusing to read the test labels: " + "; ".join(problems))
    return {
        "config_hash": cfg.hash,
        "conformal_key": conformal_key(cfg),
        "correctness_key": correctness_key(cfg),
        "signals_key": signals_key(cfg),
        "dataset": rel_to_repo(dataset),
        "dataset_sha256": dataset_sha,
        "fit": rel_to_repo(fit_path(cfg, runs_dir)),
        "fit_calib_csv_sha256": fit["csv_sha256"],
        "csv": rel_to_repo(table.path),
        "csv_sha256": table.meta["sha256"],
        "signals_runs": runs,
    }


def read_dense_top1(table: Table) -> list[float | None]:
    """The ``dense_top1`` column, which :class:`Table` leaves out. Read after its sha check."""
    with table.path.open(encoding="utf-8", newline="") as f:
        return [None if line["dense_top1"] == "" else float(line["dense_top1"])
                for line in csv.DictReader(f)]


def result_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/conformal_eval_{conformal_key}.json``: one test reading per fit."""
    return Path(runs_dir) / f"conformal_eval_{conformal_key(cfg)}.json"


def load_fit(cfg: Config, runs_dir: Path) -> dict[str, Any]:
    path = fit_path(cfg, runs_dir)
    fit = read_sidecar(path)
    if fit is None:
        raise ValueError(f"no conformal fit at {rel_to_repo(path)}: run "
                         "`python -m eval.conformal` on the calib half first")
    return fit


# --- saying what happened -----------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _error(block: Mapping[str, Any]) -> str:
    if block["realized_error"] is None:
        return "- (nothing answered)"
    lo, hi = block["ci"]
    return f"{block['realized_error']:.3f} [{lo:.3f}, {hi:.3f}] ({block['wrong']} wrong)"


def _coverage(block: Mapping[str, Any]) -> str:
    return f"{block['answered']} / {block['n_non_null']} ({_pct(block['coverage'])})"


def _t(block: Mapping[str, Any]) -> str:
    return "+inf" if block["t"] is None else f"{block['t']:g}"


def summary(result: Mapping[str, Any]) -> str:
    """The main table for the chosen signal, and the appendix of all signals x both methods."""
    ci = round(result["ci_level"] * 100)
    every = result["answer_everything"]
    lo, hi = every["ci"]
    chosen = result["chosen"]
    block = result["signals"][chosen]
    out = [
        (
            f"test n = {result['n']} {result['buckets']}, alpha = {result['alpha']:.2f}, "
            f"{ci}% Clopper-Pearson; signal chosen on calib: {chosen}"
        ),
        "",
        f"| method | t | target error | realized error [{ci}% CI] | coverage (answered / n) |",
        "|---|---|---|---|---|",
        (
            f"| conformal (calib-fit t) | {_t(block['conformal'])} | {result['alpha']:.2f} "
            f"| {_error(block['conformal'])} | {_coverage(block['conformal'])} |"
        ),
        (
            f"| fixed t = {result['fixed']} | {_t(block['fixed'])} | - (no guarantee) "
            f"| {_error(block['fixed'])} | {_coverage(block['fixed'])} |"
        ),
        (
            f"| answer everything (reference) | - | - | {every['realized_error']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] ({every['wrong']} wrong) "
            f"| {every['answered']} / {result['n']} (100.0%) |"
        ),
        "",
        "Appendix: every signal x both methods",
        "",
        (
            f"| signal | method | t | realized error [{ci}% CI] | coverage | null | test range "
            "| what the threshold means |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in SIGNALS:
        b = result["signals"][name]
        rng = "-" if b["test_range"] is None else f"{b['test_range'][0]:g}-{b['test_range'][1]:g}"
        for method in METHODS:
            m = b[method]
            meaning = (m["meaning"] if method == "fixed"
                       else f"calib fit at alpha = {result['alpha']:.2f}")
            out.append(f"| {name} | {method} | {_t(m)} | {_error(m)} | {_coverage(m)} "
                       f"| {m['n_null']} | {rng} | {meaning} |")
    dense = result["dense_top1"]
    if dense["n"]:
        out += ["", (f"dense_top1 (diagnostic, not a signal): max {dense['max']:.3f} on "
                     f"{dense['n']} rows; a raw 0.7 would answer {dense['answered_at_0.7']}")]
    return "\n".join(out)


def run(cfg: Config, out: TextIO, runs_dir: Path = RUNS_DIR,
        dataset: Path = DATASET_PATH) -> Path:
    """Check every input, apply both thresholds to the test half once, write and print."""
    path = result_path(cfg, runs_dir)
    if path.exists():
        raise ValueError(f"{rel_to_repo(path)} exists: the test half has been read for this fit. "
                         "Delete it only if an input changed, never to get another number")
    fit = load_fit(cfg, runs_dir)
    table = load_table(SPLIT, cfg, runs_dir)
    provenance = check_provenance(table, fit, cfg, runs_dir, dataset)
    level = float(cfg["eval"]["bootstrap"]["ci"])
    result = {**provenance, **evaluate(table, fit, level, read_dense_top1(table))}
    write_json(path, result)
    print(summary(result), file=out)
    print(f"\nwrote {rel_to_repo(path)}", file=out)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="where the CSVs, the signals runs and the fit are; the JSON goes here")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="the frozen dataset whose sha256 both signals runs must record")
    args = parser.parse_args(argv)
    try:
        run(load_config(), sys.stdout, args.runs_dir, args.dataset)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
