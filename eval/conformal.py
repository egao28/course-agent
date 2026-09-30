"""RQ3: an answering threshold per confidence signal, fitted on the calib half only.

    uv sync --extra analysis        # scikit-learn, for the AUROC that picks the signal
    uv run python -m eval.conformal

Reads #88's calib CSV, ``eval/runs/signals_calib_{signals_key}.csv`` (checked against its sidecar by
#89's :func:`eval.scripts.risk_coverage.load_table`), and writes
``eval/runs/conformal_{conformal_key}.json``: per signal the threshold ``t``, how many calib rows
it answers and how many of those are wrong, the whole scan that produced it, and the one signal
#91 and #92 use. There is no ``--split``. The test half is never read here, because a threshold
fitted on the rows it is scored on is a description, not a guarantee.

**The rule.** For a candidate ``t`` the gate answers the rows with ``signal >= t``: ``n_t`` rows,
``k_t`` of them wrong. ``t`` is acceptable when ``(k_t + 1) / (n_t + 1) <= alpha``. That is a
finite-sample-corrected risk estimate, not a high-probability bound. A bound would be a
Clopper-Pearson upper limit at some delta, and nothing here computes one.

**The scan is fixed-sequence.** Candidates are the signal's distinct values, strictest first, so
tied rows are answered or skipped together. The scan stops at the first candidate that fails, and
``t`` is the last one that passed. It does not take the loosest ``t`` that passes anywhere. Risk
is not monotone in ``t``, and on about 50 rows it zig-zags, so the loosest passing ``t`` can sit
past one that failed. Picking it would be a selection over every candidate, and the ``+1``
correction does not cover that selection. This is the Learn-then-Test ordering.

**A candidate too small to pass on any labels is skipped, not failed.** A candidate answering ``n``
rows cannot pass unless ``1 / (n + 1) <= alpha``, even with no errors, so at alpha = 0.10 it needs
``n >= 9``. The scan passes over the strict candidates below that count and starts at the first one
that can be tested. Without the skip, a signal whose top values are distinct (``retrieval_score``,
say) would fail on its single strictest row, at ``1/2``, and could never be accepted whatever its
labels were. The skip reads only counts, never ``correct``, so it selects nothing. The file records
how many candidates were skipped.

**No acceptable t is a result.** It is ``t = +inf``: answer nothing, coverage 0. At alpha = 0.10
nothing is accepted on fewer than 9 answered rows, and 9 needs 0 errors. JSON has no infinity, so
the file writes ``"t": null`` with ``"answers_nothing": true``, and :func:`threshold` reads it back
as ``math.inf``.

**Null rows are left out, and counted.** ``retrieval_score`` is null on tool rows, and
``verbalized`` where the reply had no number. Each signal is fitted on its own non-null rows, and
the file records how many it dropped. What the gate does on a null row is #92's decision.

**Which signal goes on.** The one with the highest calib AUROC (#89's
:func:`~eval.scripts.risk_coverage.auroc`). A tie on the point estimate goes to the signal fitted on
more non-null rows, then to the order of :data:`~eval.scripts.risk_coverage.SIGNALS`. A signal with
no AUROC (constant, or one class on its rows) cannot be chosen. The choice uses calib only. Taking
the best test AUROC would reuse the test half for selection, the max-of-means bias Day 4 avoided.
Every calib row counts, in all four buckets, because the gate in production sees every bucket.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, conformal_key, correctness_key, load_config, rel_to_repo, signals_key
from eval.scripts.draft_items import write_json
from eval.scripts.risk_coverage import SIGNALS, Table, auroc, load_table
from eval.scripts.signals import RUNS_DIR

#: The half the threshold is fitted on. Fixed: see the module docstring.
SPLIT = "calib"


# --- the procedure: pure, no I/O ----------------------------------------------------------------


class Step(NamedTuple):
    """One candidate of the scan: answer every row with ``signal >= t``.

    ``tested`` is False where ``n`` is too small to pass even with ``k = 0``; such a step is
    skipped, and ``acceptable`` is then False without ending the scan.
    """

    t: float
    n: int
    k: int
    tested: bool
    acceptable: bool


def _acceptable(n: int, k: int, alpha: float) -> bool:
    # Exact: alpha comes from YAML as a float, and a ratio that equals alpha, such as 1/10 at
    # alpha = 0.10, must pass or fail on the arithmetic rather than on how a float rounds.
    return Fraction(k + 1, n + 1) <= Fraction(str(alpha))


def _present(scores: Sequence[float | None], correct: Sequence[int]) -> list[tuple[float, int]]:
    if len(scores) != len(correct):
        raise ValueError(f"{len(scores)} scores for {len(correct)} labels")
    bad = [c for c in correct if c not in (0, 1)]
    if bad:
        raise ValueError(f"correct must be 0 or 1, got {bad[:5]}")
    return [(float(s), int(c)) for s, c in zip(scores, correct, strict=True) if s is not None]


def scan(scores: Sequence[float | None], correct: Sequence[int], alpha: float) -> list[Step]:
    """Every distinct value, strictest first, up to and including the first tested one that fails.

    Null scores are left out. An empty result means there were no non-null rows to scan.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rows = _present(scores, correct)
    steps = []
    for t in sorted({s for s, _ in rows}, reverse=True):
        answered = [c for s, c in rows if s >= t]
        n, k = len(answered), len(answered) - sum(answered)
        # Whether n rows could pass with no errors: a count, so skipping on it reads no label.
        tested = _acceptable(n, 0, alpha)
        steps.append(Step(t, n, k, tested, tested and _acceptable(n, k, alpha)))
        if tested and not steps[-1].acceptable:
            break
    return steps


def select_threshold(
    scores: Sequence[float | None], correct: Sequence[int], alpha: float
) -> tuple[float, int, int]:
    """``(t, n_answered, k_wrong)`` from the fixed-sequence scan; ``(inf, 0, 0)`` if none passes."""
    passed = [step for step in scan(scores, correct, alpha) if step.acceptable]
    if not passed:
        return math.inf, 0, 0
    last = passed[-1]
    return last.t, last.n, last.k


def lowest_risk_hat(scores: Sequence[float | None], correct: Sequence[int]) -> float | None:
    """``min_t (k_t + 1) / (n_t + 1)`` over every candidate, *not* a threshold rule.

    Reported only to show how far the best candidate anywhere is from alpha. Choosing ``t`` by it
    is exactly the global selection :func:`scan` refuses.
    """
    rows = _present(scores, correct)
    risks = []
    for t in {s for s, _ in rows}:
        answered = [c for s, c in rows if s >= t]
        risks.append((len(answered) - sum(answered) + 1) / (len(answered) + 1))
    return min(risks, default=None)


def choose_signal(candidates: Mapping[str, tuple[float | None, int]]) -> str:
    """The name with the highest AUROC; ties go to more rows, then to :data:`SIGNALS` order.

    ``candidates`` maps a signal to ``(auroc, n_non_null)``, with ``auroc`` None where it has none.
    """
    order = {name: i for i, name in enumerate(SIGNALS)}
    eligible = [name for name, (a, _) in candidates.items() if a is not None]
    if not eligible:
        raise ValueError("no signal has an AUROC on calib: there is nothing to choose between")
    return min(eligible, key=lambda name: (-candidates[name][0], -candidates[name][1],
                                           order.get(name, len(order)), name))


def threshold(block: Mapping[str, Any]) -> float:
    """A signal's ``t`` as written by :func:`fit`, with ``null`` read back as ``math.inf``."""
    return math.inf if block["t"] is None else float(block["t"])


# --- the fit ------------------------------------------------------------------------------------


def _auroc_or_none(values: Sequence[float | None], correct: Sequence[int]) -> tuple[float | None,
                                                                                    str | None]:
    try:
        return auroc(values, correct), None
    except ValueError as error:
        return None, str(error)


def fit(table: Table, cfg: Config) -> dict[str, Any]:
    """Every signal's threshold on the calib table, and the one chosen by calib AUROC."""
    alpha = float(cfg["conformal"]["alpha"])
    per_signal: dict[str, Any] = {}
    for name in SIGNALS:
        values = table.signals[name]
        n = sum(v is not None for v in values)
        steps = scan(values, table.correct, alpha)
        t, n_answered, k_wrong = select_threshold(values, table.correct, alpha)
        a, why = _auroc_or_none(values, table.correct)
        per_signal[name] = {
            "n": n,
            "n_null": len(values) - n,
            "t": None if math.isinf(t) else t,
            "answers_nothing": math.isinf(t),
            "n_answered": n_answered,
            "k_wrong": k_wrong,
            "coverage": n_answered / n if n else 0.0,
            "risk_hat": (k_wrong + 1) / (n_answered + 1) if n_answered else None,
            "scan": [
                {**step._asdict(), "risk_hat": (step.k + 1) / (step.n + 1)} for step in steps
            ],
            "stopped_at": next((step.t for step in steps if step.tested and not step.acceptable),
                               None),
            "n_skipped_too_few_rows": sum(not step.tested for step in steps),
            "lowest_risk_hat_any_t": lowest_risk_hat(values, table.correct),
            "auroc": a,
            "auroc_refused": why,
        }
    chosen = choose_signal({name: (b["auroc"], b["n"]) for name, b in per_signal.items()})
    wrong = len(table.correct) - sum(table.correct)
    return {
        "split": SPLIT,
        "alpha": alpha,
        "conformal_key": conformal_key(cfg),
        "correctness_key": correctness_key(cfg),
        "signals_key": signals_key(cfg),
        "config_hash": cfg.hash,
        "csv": rel_to_repo(table.path),
        "csv_sha256": table.meta["sha256"],
        "n": len(table.ids),
        "buckets": {b: table.buckets.count(b) for b in sorted(set(table.buckets))},
        "answer_everything": {"n": len(table.correct), "k": wrong,
                              "risk_hat": (wrong + 1) / (len(table.correct) + 1)},
        "min_answered_to_accept": math.ceil(1 / alpha - 1),
        "signals": per_signal,
        "chosen": {
            "signal": chosen,
            "rule": "highest calib AUROC; ties to more non-null rows, then to SIGNALS order",
            "t": per_signal[chosen]["t"],
            "answers_nothing": per_signal[chosen]["answers_nothing"],
        },
    }


def result_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/conformal_{conformal_key}.json``: the labels and alpha decide it and name it."""
    return Path(runs_dir) / f"conformal_{conformal_key(cfg)}.json"


def summary(result: Mapping[str, Any]) -> str:
    """The per-signal thresholds as Markdown, with n everywhere."""
    every = result["answer_everything"]
    out = [
        (
            f"calib n = {result['n']} {result['buckets']}, alpha = {result['alpha']:.2f}; "
            f"answering everything: {every['k']} wrong of {every['n']}, "
            f"(k+1)/(n+1) = {every['risk_hat']:.3f}"
        ),
        "",
        (
            "| signal | n | null | AUROC | t | answered | wrong | coverage | stopped at "
            "| lowest (k+1)/(n+1), any t |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SIGNALS:
        b = result["signals"][name]
        a = "-" if b["auroc"] is None else f"{b['auroc']:.3f}"
        t = "+inf" if b["answers_nothing"] else f"{b['t']:g}"
        stop = "-" if b["stopped_at"] is None else f"{b['stopped_at']:g}"
        low = b["lowest_risk_hat_any_t"]
        out.append(
            f"| {name} | {b['n']} | {b['n_null']} | {a} | {t} | {b['n_answered']} "
            f"| {b['k_wrong']} | {b['coverage']:.3f} | {stop} "
            f"| {'-' if low is None else f'{low:.3f}'} |"
        )
    chosen = result["chosen"]
    t = "+inf (answer nothing)" if chosen["answers_nothing"] else chosen["t"]
    out += ["", f"chosen for #91/#92: {chosen['signal']} ({chosen['rule']}), t = {t}"]
    return "\n".join(out)


def run(cfg: Config, out: TextIO, runs_dir: Path = RUNS_DIR) -> Path:
    """Fit on the calib CSV, write the JSON, print the table."""
    result = fit(load_table(SPLIT, cfg, runs_dir), cfg)
    path = write_json(result_path(cfg, runs_dir), result)
    print(summary(result), file=out)
    print(f"\nwrote {rel_to_repo(path)} (conformal key {result['conformal_key']})", file=out)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the threshold on the calib half and write the JSON; a refusal exits 1 with its reason."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="where the calib CSV is, and where the JSON goes")
    args = parser.parse_args(argv)
    try:
        run(load_config(), sys.stdout, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
