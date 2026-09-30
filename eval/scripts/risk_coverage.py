"""RQ2: risk-coverage curves for the three confidence signals, and their AUROC with paired CIs.

    uv sync --extra analysis        # scikit-learn and matplotlib; the agent never imports them
    uv run python -m eval.scripts.risk_coverage                  # test: the reported numbers
    uv run python -m eval.scripts.risk_coverage --split calib    # development only

Reads #88's ``eval/runs/signals_{split}_{signals_key}.csv`` -- refused unless its sidecar's
sha256 and signals key are this file's and this config's -- and writes one figure,
``report/figures/risk_coverage_{split}_{signals_key}.png``, and the numbers behind it,
``eval/runs/risk_coverage_{split}_{signals_key}.json``. It calls no model: the CSV is the sample.

**Risk-coverage.** For a signal and a threshold ``t`` the agent answers the rows with
``signal >= t``: coverage is the fraction answered, risk the error rate among them. The sweep
runs over the signal's *distinct* values, strictest first, so tied rows are answered or skipped
together and form one step. ``self_consistency`` takes four values and so has at most four points;
breaking its ties by row order would draw operating points no threshold can reach.

**Null rows are dropped per signal, and the legend says how many are left.** ``retrieval_score``
is null on tool rows, and ``verbalized`` where the reply had no number. Each curve's coverage is
over its own non-null rows, so the three curves can have different denominators; the legend gives
each n rather than hiding that.

**AURC** is the area under the step function the sweep draws: risk ``r_j`` over coverage
``(c_{j-1}, c_j]``, with ``c_0 = 0``, which is ``sum_j (c_j - c_{j-1}) * r_j``. Lower is better.
With no ties it is the usual mean of the risks at each row's coverage. With ties it gives a
group one risk, the one its threshold actually has.

**AUROC** is ``sklearn.metrics.roc_auc_score(correct, signal)`` on the non-null rows: the chance
that a correct row scores above an incorrect one, with ties counting half. A signal that is
constant on the sample, or a sample with one class only, has no AUROC and is refused, not scored
0.5.

**Paired bootstrap.** One draw of ``B`` index vectors over *all* rows is shared by the three
signals (Day 4's :func:`eval.scripts.bootstrap.resample_indices`, and its ``eval.bootstrap``
settings: B, seed, ci). Each signal then keeps the drawn rows it is non-null on, so the three
intervals, and each pairwise difference, come from the same resampled questions. A resample in
which a signal's rows are all correct or all incorrect, or that drew none of its non-null rows,
has no AUROC. It is skipped and counted,
not scored 0.5, and the count is reported. At n of about 63 the intervals are wide, and the table
shows them as they are.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import numpy as np

from config import REPO_ROOT, Config, file_sha256, load_config, rel_to_repo, signals_key
from eval.schema import SPLITS, Split
from eval.scripts.bootstrap import resample_indices
from eval.scripts.draft_items import write_json
from eval.scripts.review_signals import CSV_COLUMNS, csv_path
from eval.scripts.signals import RUNS_DIR
from eval.scripts.split import read_sidecar

#: The three signals, in the order the plan names them. ``dense_top1`` is in the CSV but is a
#: recorded diagnostic, not a candidate signal.
SIGNALS: tuple[str, ...] = ("retrieval_score", "verbalized", "self_consistency")

FIGURES_DIR = REPO_ROOT / "report" / "figures"

#: Categorical slots 1-3 of the dataviz reference palette, validated for CVD separation on a light
#: surface. The third is below 3:1 contrast, so every curve also has its own marker and a legend.
STYLE: Mapping[str, tuple[str, str]] = {
    "retrieval_score": ("#2a78d6", "o"),
    "verbalized": ("#eb6834", "s"),
    "self_consistency": ("#1baf7a", "^"),
}


# --- the procedure: pure, no I/O ----------------------------------------------------------------


class Point(NamedTuple):
    """One operating point: answer every row with ``signal >= threshold``."""

    threshold: float
    coverage: float
    risk: float
    answered: int


def _present(signal: Sequence[float | None], correct: Sequence[int]) -> tuple[np.ndarray, ...]:
    """The rows where the signal is not null, as ``(signal, correct)`` arrays."""
    if len(signal) != len(correct):
        raise ValueError(f"{len(signal)} signal values for {len(correct)} labels")
    keep = [i for i, value in enumerate(signal) if value is not None]
    return (
        np.array([signal[i] for i in keep], dtype=float),
        np.array([correct[i] for i in keep], dtype=int),
    )


def risk_coverage(signal: Sequence[float | None], correct: Sequence[int]) -> list[Point]:
    """One point per distinct signal value, strictest threshold first; null rows are left out."""
    s, y = _present(signal, correct)
    if not len(s):
        raise ValueError("the signal is null on every row: there is no curve to draw")
    points = []
    for t in np.unique(s)[::-1]:
        answered = s >= t
        points.append(
            Point(float(t), float(answered.mean()), float(1 - y[answered].mean()),
                  int(answered.sum()))
        )
    return points


def aurc(points: Sequence[Point]) -> float:
    """``sum_j (c_j - c_{j-1}) * r_j``: the area under the step :func:`risk_coverage` draws."""
    area, previous = 0.0, 0.0
    for point in points:
        area += (point.coverage - previous) * point.risk
        previous = point.coverage
    return area


def auroc(signal: Sequence[float | None], correct: Sequence[int]) -> float:
    """``roc_auc_score(correct, signal)`` on the non-null rows, refusing a sample it cannot rank."""
    from sklearn.metrics import roc_auc_score

    s, y = _present(signal, correct)
    if len(np.unique(y)) < 2:
        raise ValueError(f"all {len(y)} non-null rows have correct = {y[0] if len(y) else '-'}: "
                         "an AUROC needs both classes")
    if len(np.unique(s)) < 2:
        raise ValueError(f"the signal is constant ({s[0]}) on all {len(s)} non-null rows: it "
                         "ranks nothing, and 0.5 would read as a measured chance level")
    return float(roc_auc_score(y, s))


class Resampled(NamedTuple):
    """Per signal, the ``B`` resample AUROCs, NaN where the resample had one class only."""

    indices: np.ndarray
    aurocs: Mapping[str, np.ndarray]


def bootstrap_auroc(
    signals: Mapping[str, Sequence[float | None]],
    correct: Sequence[int],
    n_resamples: int,
    seed: int,
) -> Resampled:
    """Resample all rows once and take every signal's AUROC on the same draw."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(correct, dtype=int)
    indices = resample_indices(len(y), n_resamples, seed)
    aurocs: dict[str, np.ndarray] = {}
    for name, values in signals.items():
        if len(values) != len(y):
            raise ValueError(f"{name}: {len(values)} values for {len(y)} labels")
        present = np.array([v is not None for v in values])
        s = np.array([np.nan if v is None else v for v in values], dtype=float)
        out = np.full(n_resamples, np.nan)
        for b, idx in enumerate(indices):
            idx = idx[present[idx]]
            yb = y[idx]
            # No non-null rows drawn at all is skipped the same way as one class.
            if len(yb) and yb.min() != yb.max():
                out[b] = roc_auc_score(yb, s[idx])
        aurocs[name] = out
    return Resampled(indices, aurocs)


def percentile_interval(values: np.ndarray, ci: float) -> tuple[float, float, int]:
    """The ``ci`` percentile interval over the finite values, and how many were NaN (skipped)."""
    finite = values[np.isfinite(values)]
    skipped = int(len(values) - len(finite))
    if not len(finite):
        raise ValueError("every resample was skipped: no interval can be drawn")
    tail = (1 - ci) / 2 * 100
    lo, hi = np.percentile(finite, [tail, 100 - tail])
    return float(lo), float(hi), skipped


# --- the CSV ------------------------------------------------------------------------------------


class Table(NamedTuple):
    """One half's signals CSV, checked against its sidecar."""

    path: Path
    meta: dict[str, Any]
    ids: list[str]
    buckets: list[str]
    signals: dict[str, list[float | None]]
    correct: list[int]


def _number(value: str) -> float | None:
    return None if value == "" else float(value)


def load_table(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Table:
    """Read #88's CSV, refusing it unless it is the file its sidecar describes, for this config."""
    path = csv_path(split, cfg, runs_dir)
    meta = read_sidecar(path.with_name(path.name + ".meta.json"))
    if meta is None or not path.exists():
        raise ValueError(
            f"no {split} signals CSV at {rel_to_repo(path)}: finish the human pass and run "
            f"`python -m eval.scripts.review_signals --split {split} --write-csv` first"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("signals_key") != signals_key(cfg):
        problems.append(f"it was written for signals {meta.get('signals_key')}, "
                        f"not {signals_key(cfg)}")
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"{rel_to_repo(path)}: columns are {reader.fieldnames}, "
                             f"not #88's {list(CSV_COLUMNS)}")
        lines = list(reader)
    if len(lines) != meta.get("n_rows"):
        raise ValueError(f"{rel_to_repo(path)}: {len(lines)} rows, sidecar says "
                         f"{meta.get('n_rows')}")
    wrong_split = [line["id"] for line in lines if line["split"] != split]
    bad = [line["id"] for line in lines if line["correct"] not in ("0", "1")]
    if wrong_split or bad:
        raise ValueError(f"{rel_to_repo(path)}: rows from another split {wrong_split[:5]}, "
                         f"rows without a 0/1 correct {bad[:5]}")
    return Table(
        path,
        meta,
        [line["id"] for line in lines],
        [line["bucket"] for line in lines],
        {name: [_number(line[name]) for line in lines] for name in SIGNALS},
        [int(line["correct"]) for line in lines],
    )


# --- the numbers --------------------------------------------------------------------------------


def _interval(point: float, resamples: np.ndarray, ci: float) -> dict[str, Any]:
    lo, hi, skipped = percentile_interval(resamples, ci)
    return {"point": point, "lo": lo, "hi": hi, "skipped_resamples": skipped}


def analyse(table: Table, split: Split, cfg: Config) -> dict[str, Any]:
    """Curves, AURC and AUROC per signal, and the pairwise AUROC differences, on one draw."""
    params = cfg["eval"]["bootstrap"]
    n_resamples, seed, ci = int(params["n_resamples"]), int(params["seed"]), float(params["ci"])
    boot = bootstrap_auroc(table.signals, table.correct, n_resamples, seed)

    per_signal: dict[str, Any] = {}
    for name in SIGNALS:
        values = table.signals[name]
        points = risk_coverage(values, table.correct)
        n = sum(v is not None for v in values)
        per_signal[name] = {
            "n": n,
            "n_null": len(values) - n,
            "points": [point._asdict() for point in points],
            "aurc": aurc(points),
            "auroc": _interval(auroc(values, table.correct), boot.aurocs[name], ci),
        }

    differences = {}
    for a, b in itertools.combinations(SIGNALS, 2):
        both = np.isfinite(boot.aurocs[a]) & np.isfinite(boot.aurocs[b])
        gap = np.where(both, boot.aurocs[a] - boot.aurocs[b], np.nan)
        block = _interval(per_signal[a]["auroc"]["point"] - per_signal[b]["auroc"]["point"],
                          gap, ci)
        block["significant"] = block["lo"] > 0 or block["hi"] < 0
        differences[f"{a} - {b}"] = block

    return {
        "split": split,
        "signals_key": signals_key(cfg),
        "config_hash": cfg.hash,
        "csv": rel_to_repo(table.path),
        "csv_sha256": table.meta["sha256"],
        "n": len(table.ids),
        "buckets": {b: table.buckets.count(b) for b in sorted(set(table.buckets))},
        "answer_everything_risk": 1 - float(np.mean(table.correct)),
        # The risk = alpha reference line: #90's conformal.alpha, which the figure only draws.
        "reference_risk": float(cfg["conformal"]["alpha"]),
        "n_resamples": n_resamples,
        "seed": seed,
        "ci": ci,
        "signals": per_signal,
        "differences": differences,
    }


def figure_path(split: Split, cfg: Config, figures_dir: Path = FIGURES_DIR) -> Path:
    return Path(figures_dir) / f"risk_coverage_{split}_{signals_key(cfg)}.png"


def result_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    return Path(runs_dir) / f"risk_coverage_{split}_{signals_key(cfg)}.json"


def plot(result: Mapping[str, Any], path: Path) -> Path:
    """All three curves on one axis, with the alpha and answer-everything references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=150)
    for name in SIGNALS:
        block = result["signals"][name]
        colour, marker = STYLE[name]
        cov = [p["coverage"] for p in block["points"]]
        risk = [p["risk"] for p in block["points"]]
        label = f"{name} (n = {block['n']}, AURC {block['aurc']:.3f})"
        # where="pre": risk r_j holds over (c_{j-1}, c_j], the step AURC integrates.
        ax.step([0.0, *cov], [risk[0], *risk], where="pre", color=colour, linewidth=2)
        ax.plot(cov, risk, linestyle="none", marker=marker, markersize=6, color=colour,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
    ax.axhline(result["reference_risk"], color="#6b6a64", linestyle="--", linewidth=1,
               label=f"risk = alpha = {result['reference_risk']:.2f}")
    ax.axhline(result["answer_everything_risk"], color="#6b6a64", linestyle=":", linewidth=1,
               label=f"answer everything (n = {result['n']}): "
                     f"{result['answer_everything_risk']:.3f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(0.5, *(p["risk"] for s in result["signals"].values()
                               for p in s["points"])) + 0.05)
    ax.set_xlabel("coverage (fraction of non-null rows answered)")
    ax.set_ylabel("risk (error rate among answered rows)")
    ax.set_title(f"Risk-coverage, {result['split']} half, signals {result['signals_key']}",
                 fontsize=10)
    ax.grid(color="#e4e3dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Below the axes: in the top left it would cover the strictest points, where curves start.
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # No Software/date metadata, so the same CSV gives the same bytes.
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def summary(result: Mapping[str, Any]) -> str:
    """The AUROC table as Markdown, with n everywhere."""
    ci = round(result["ci"] * 100)
    out = [
        (
            f"{result['split']} n = {result['n']} {result['buckets']}, "
            f"B = {result['n_resamples']}, seed {result['seed']}, {ci}% paired percentile; "
            f"answer-everything risk {result['answer_everything_risk']:.3f}"
        ),
        "",
        f"| signal | n | null | AUROC [{ci}% CI] | skipped resamples | AURC |",
        "|---|---|---|---|---|---|",
    ]
    for name in SIGNALS:
        block = result["signals"][name]
        a = block["auroc"]
        out.append(
            f"| {name} | {block['n']} | {block['n_null']} | {a['point']:.3f} "
            f"[{a['lo']:.3f}, {a['hi']:.3f}] | {a['skipped_resamples']} | {block['aurc']:.3f} |"
        )
    out += ["", "| AUROC difference | point [CI] | skipped | excludes 0 |", "|---|---|---|---|"]
    for pair, d in result["differences"].items():
        out.append(f"| {pair} | {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                   f"{d['skipped_resamples']} | {'yes' if d['significant'] else 'no'} |")
    return "\n".join(out)


def run(
    split: Split,
    cfg: Config,
    out: TextIO,
    runs_dir: Path = RUNS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Read one half's CSV, compute everything, write the figure and the JSON, print the table."""
    result = analyse(load_table(split, cfg, runs_dir), split, cfg)
    figure = plot(result, figure_path(split, cfg, figures_dir))
    result["figure"] = rel_to_repo(figure)
    result["figure_sha256"] = file_sha256(figure)
    numbers = write_json(result_path(split, cfg, runs_dir), result)
    print(summary(result), file=out)
    print(f"\nwrote {rel_to_repo(figure)} and {rel_to_repo(numbers)}", file=out)
    return figure, numbers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=SPLITS,
                        help="test is reported; calib is for development (default: test)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where the CSV is")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="where the PNG goes")
    args = parser.parse_args(argv)
    try:
        run(args.split, load_config(), sys.stdout, args.runs_dir, args.figures_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
