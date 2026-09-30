"""RQ1's intervals: a paired bootstrap over questions, from the two committed run files.

    uv run python -m eval.scripts.bootstrap

Reads ``eval/runs/ablation_{calib,test}_{hash}.jsonl`` (#75) and writes
``eval/runs/bootstrap_{bootstrap_key}.json`` -- named after ``config.bootstrap_key``: ``cfg.hash``,
the ``eval.bootstrap`` section and the ``rerank`` section, which is everything a resample reads
(see ``config.AFTER_THE_RUNS``). It never calls ``retrieve()``: the run
files are the sample, so re-drawing an interval never re-queries an index -- and cannot, since
the dense side of a live run does not reproduce.

**Paired.** One draw of ``B`` index vectors, ``idx ~ randint(0, n, n)``, is shared by every mode.
Each resample asks all three retrievers the same questions, so the difference between two modes
is computed per resample from the same ``idx`` and the between-question variance -- most of the
variance with n = 41 -- cancels instead of adding. Two independent draws would give a wider
interval around the same point, and a test that almost never says anything.

**Percentile intervals.** The ``(1 - ci) / 2`` and ``(1 + ci) / 2`` quantiles of the ``B``
resample means. A difference is *significant iff its interval excludes 0*; the point and the
interval are reported either way, because "not significant" is a result.

**"Strongest single" is chosen on calib.** Picking the better of dense and bm25 on the same 41
test questions the gap is then tested on would compare hybrid with the larger of two noisy means,
which is biased upward, and quietly shrink the gap. So the choice is made per metric on ``calib``
-- its means are recorded beside it -- and only the difference is computed on ``test``. If the two
metrics pick different winners, each gets its own comparison. A tie on calib is broken in
:data:`retrieval.MODES` order and recorded as a tie.

**What n = 41 buys.** Recall@5 on the 29 ``fact`` rows moves in steps of 1/29, and a percentile
interval on a mean of 0/1 values near 1 collapses against the ceiling: a mode that scores 1 on
every question has the interval [1, 1] however few questions there are. That is recorded as
``degenerate``, and must be reported as such -- it is the absence of evidence of variance, not
certainty.

**The stretch row (#79).** When ``config.yaml`` has a ``rerank`` section, the test half's rerank run
(``eval.scripts.rerank``) is a fourth score vector in the same :func:`paired_bootstrap` call, so it
is resampled with the very indices the three modes are -- adding it moves none of their intervals
-- and ``rerank - hybrid`` gets an interval by the same rule. It is required then, not picked up
when present: a table that quietly lost a row would be the silent filter #75 guards against.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import numpy as np

from config import Config, bootstrap_key, file_sha256, load_config, rel_to_repo
from eval.schema import Split
from eval.scripts import rerank as rr
from eval.scripts.ablation import RUNS_DIR, K, run_path
from eval.scripts.draft_items import write_json
from eval.scripts.split import META_SUFFIX, read_sidecar
from retrieval import MODES

METRICS = (f"recall_at_{K}", f"mrr_at_{K}")

#: The fused mode, and the two it is compared against.
HYBRID = "hybrid"
SINGLES = tuple(mode for mode in MODES if mode != HYBRID)


# --- the procedure: pure, no I/O ----------------------------------------------------------------


class Interval(NamedTuple):
    """A point estimate on the full sample and its percentile interval."""

    point: float
    lo: float
    hi: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    @property
    def degenerate(self) -> bool:
        """Every resample gave the same mean: no variance was seen, which is not certainty."""
        return self.lo == self.hi


def resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """``(B, n)`` row indices drawn with replacement: the one draw every mode shares."""
    if n < 1 or n_resamples < 1:
        raise ValueError(f"need n >= 1 and B >= 1, got n = {n}, B = {n_resamples}")
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


@dataclass(frozen=True)
class Bootstrap:
    """One paired bootstrap: the shared indices and, per mode, the ``B`` resample means."""

    indices: np.ndarray
    scores: Mapping[str, np.ndarray]
    means: Mapping[str, np.ndarray]
    ci: float

    def _interval(self, point: float, resamples: np.ndarray) -> Interval:
        tail = (1 - self.ci) / 2 * 100
        lo, hi = np.percentile(resamples, [tail, 100 - tail])
        return Interval(float(point), float(lo), float(hi))

    def interval(self, mode: str) -> Interval:
        """The mean of ``mode`` over all n questions, and its interval."""
        return self._interval(self.scores[mode].mean(), self.means[mode])

    def difference(self, minuend: str, subtrahend: str) -> Interval:
        """``minuend - subtrahend``, per resample from the same indices, and its interval."""
        point = self.scores[minuend].mean() - self.scores[subtrahend].mean()
        return self._interval(point, self.means[minuend] - self.means[subtrahend])


def paired_bootstrap(
    scores: Mapping[str, np.ndarray], n_resamples: int, seed: int, ci: float = 0.95
) -> Bootstrap:
    """Resample the questions ``n_resamples`` times, once, and take every mode's mean on each.

    ``scores[mode]`` is the per-question score vector, rows aligned across modes. Vectorised: one
    fancy index per mode, no loop over resamples.
    """
    if not scores:
        raise ValueError("no score vectors to resample")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    vectors = {mode: np.asarray(s, dtype=float) for mode, s in scores.items()}
    shapes = {v.shape for v in vectors.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError(f"score vectors must be 1-D and the same length, got {sorted(shapes)}")
    indices = resample_indices(len(next(iter(vectors.values()))), n_resamples, seed)
    means = {mode: v[indices].mean(axis=1) for mode, v in vectors.items()}
    return Bootstrap(indices, vectors, means, ci)


def strongest_single(means: Mapping[str, float]) -> tuple[str, bool]:
    """The single mode with the highest mean, and whether it tied; ties go in MODES order."""
    best = max(means[mode] for mode in SINGLES)
    winners = [mode for mode in SINGLES if means[mode] == best]
    return winners[0], len(winners) > 1


# --- the run files ------------------------------------------------------------------------------


class Run(NamedTuple):
    """One half's run file, checked against its sidecar, as score vectors aligned by id."""

    path: Path
    meta: dict[str, Any]
    ids: list[str]
    scores: dict[str, dict[str, np.ndarray]]  # metric -> mode -> (n,)


def load_run(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Run:
    """Read one half's run file, refusing it unless it is the file its sidecar describes.

    The digest, the config it was run under, the modes and k, and exactly one line per
    (id, mode) for the ids the sidecar lists -- a missing line would silently shrink n for one
    mode, and the pairing would then be over different questions.
    """
    path = run_path(split, cfg, runs_dir)
    meta = read_sidecar(path.with_suffix(META_SUFFIX))
    if meta is None or not path.exists():
        raise ValueError(
            f"no {split} run at {rel_to_repo(path)}: run "
            f"`python -m eval.scripts.ablation --split {split}` first"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("config_hash") != cfg.hash:
        problems.append(f"it was run under {meta.get('config_hash')}, not {cfg.hash}")
    if meta.get("split") != split or meta.get("k") != K or meta.get("modes") != list(MODES):
        problems.append("its split, k or modes are not this script's")
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))

    ids = list(meta["ids"])
    lines: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = json.loads(raw)
            key = (line["id"], line["mode"])
            if key in lines:
                raise ValueError(f"{rel_to_repo(path)}: {key} appears twice")
            lines[key] = line
    expected = {(i, mode) for i in ids for mode in MODES}
    if set(lines) != expected:
        raise ValueError(
            f"{rel_to_repo(path)}: lines do not cover the sidecar's {len(ids)} ids x "
            f"{len(MODES)} modes exactly ({len(set(lines) - expected)} extra, "
            f"{len(expected - set(lines))} missing)"
        )
    scores = {
        metric: {mode: np.array([lines[i, mode][metric] for i in ids]) for mode in MODES}
        for metric in METRICS
    }
    return Run(path, meta, ids, scores)


# --- calib chooses, test measures ---------------------------------------------------------------


def _interval_json(interval: Interval) -> dict[str, Any]:
    return {
        "point": interval.point,
        "lo": interval.lo,
        "hi": interval.hi,
        "degenerate": interval.degenerate,
    }


def analyse(calib: Run, test: Run, cfg: Config, rerank: rr.Rerank | None = None) -> dict[str, Any]:
    """Per metric: the strongest single on calib, every interval on test, and the gap.

    With ``rerank``, also its interval and ``rerank - hybrid``, from the same resample indices.
    """
    for field in ("dense_key", "bm25_key", "dataset_sha256"):
        if calib.meta.get(field) != test.meta.get(field):
            raise ValueError(f"the two halves were run with different {field}s")
    params = cfg["eval"]["bootstrap"]
    n_resamples, seed, ci = int(params["n_resamples"]), int(params["seed"]), float(params["ci"])

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        calib_means = {mode: float(calib.scores[metric][mode].mean()) for mode in SINGLES}
        chosen, tied = strongest_single(calib_means)
        vectors = dict(test.scores[metric])
        if rerank is not None:
            vectors[rr.MODE] = rerank.scores[metric]
        boot = paired_bootstrap(vectors, n_resamples, seed, ci)
        metrics[metric] = {
            "strongest_single": {"mode": chosen, "tied": tied, "calib_means": calib_means},
            "test": {mode: _interval_json(boot.interval(mode)) for mode in vectors},
            "difference": _difference_json(boot, HYBRID, chosen),
        }
        if rerank is not None:
            metrics[metric]["rerank_difference"] = _difference_json(boot, rr.MODE, rr.BASE)
    extra = {}
    if rerank is not None:
        meta = rerank.meta
        extra["rerank"] = {
            "run": meta["run"],
            "sha256": meta["sha256"],
            "rerank_key": meta["rerank_key"],
            "model": meta["rerank"]["model"],
            "revision": meta["rerank"]["revision"],
            "pool": meta["pool"],
        }
    return {
        "config_hash": cfg.hash,
        "bootstrap_key": bootstrap_key(cfg),
        "full_hash": cfg.full_hash,
        "n_resamples": n_resamples,
        "seed": seed,
        "ci": ci,
        "k": K,
        "dense_key": test.meta["dense_key"],
        "bm25_key": test.meta["bm25_key"],
        "dataset_sha256": test.meta["dataset_sha256"],
        "runs": {
            run.meta["split"]: {"run": rel_to_repo(run.path), "sha256": run.meta["sha256"]}
            for run in (calib, test)
        },
        "n": {"calib": len(calib.ids), "test": len(test.ids)},
        "buckets": test.meta.get("buckets", {}),
        "metrics": metrics,
        **extra,
    }


def _difference_json(boot: Bootstrap, minuend: str, subtrahend: str) -> dict[str, Any]:
    gap = boot.difference(minuend, subtrahend)
    return {
        "minuend": minuend,
        "subtrahend": subtrahend,
        **_interval_json(gap),
        "significant": gap.excludes_zero,
    }


def result_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/bootstrap_{bootstrap_key}.json`` -- see :func:`config.bootstrap_key`."""
    return Path(runs_dir) / f"bootstrap_{bootstrap_key(cfg)}.json"


# --- saying what happened -----------------------------------------------------------------------


def _fmt(interval: Mapping[str, Any]) -> str:
    text = f"{interval['point']:.3f} [{interval['lo']:.3f}, {interval['hi']:.3f}]"
    return text + (" degenerate" if interval["degenerate"] else "")


def summary(result: Mapping[str, Any]) -> str:
    """The intervals as lines of text; the table itself is #77's."""
    n = result["n"]["test"]
    ci = round(result["ci"] * 100)
    out = [f"test n = {n}, B = {result['n_resamples']}, seed {result['seed']}, {ci}% percentile"]
    for metric, block in result["metrics"].items():
        choice = block["strongest_single"]
        calib = ", ".join(f"{m} {v:.3f}" for m, v in choice["calib_means"].items())
        tie = ", a tie" if choice["tied"] else ""
        out.append(f"\n{metric}: strongest single on calib = {choice['mode']} ({calib}{tie})")
        out.extend(f"  {mode:<13} {_fmt(interval)}" for mode, interval in block["test"].items())
        for key in ("difference", "rerank_difference"):
            if key not in block:
                continue
            gap = block[key]
            verdict = "significant" if gap["significant"] else "not significant: the CI contains 0"
            out.append(f"  {gap['minuend']} - {gap['subtrahend']}  {_fmt(gap)}  {verdict}")
    return "\n".join(out)


def bootstrap(cfg: Config, out: TextIO, runs_dir: Path = RUNS_DIR) -> Path:
    """Read both halves' runs, compute the intervals, write them, and print them."""
    test = load_run("test", cfg, runs_dir)
    rerank = rr.load_rerank(cfg, test, runs_dir) if "rerank" in cfg else None
    result = analyse(load_run("calib", cfg, runs_dir), test, cfg, rerank)
    path = write_json(result_path(cfg, runs_dir), result)
    print(summary(result), file=out)
    print(
        f"\nwrote {rel_to_repo(path)} (bootstrap key {bootstrap_key(cfg)}, config {cfg.hash})",
        file=out,
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where the run files are")
    args = parser.parse_args(argv)
    try:
        bootstrap(load_config(), sys.stdout, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
