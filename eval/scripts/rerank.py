"""The stretch row (#79): hybrid's top 20, reranked by a cross-encoder, cut to 5.

    uv run --env-file .env --extra local python -m eval.scripts.rerank --split calib   # develop
    uv run --env-file .env --extra local python -m eval.scripts.rerank --split test    # reported

A step after #75's run, over the same rows: it reads ``ablation_{split}_{hash}.jsonl`` for the ids
and the hybrid lines it is paired with, and writes ``eval/runs/rerank_{split}_{rerank_key}.jsonl``
plus a sidecar. #76's bootstrap reads the test file with the same resample indices as the three
modes; nothing downstream calls the reranker again.

**One variable.** Each line is ``retrieve(q, "hybrid", k=pool)`` -- the same call the ablation
makes, deeper -- then :func:`retrieval.rerank.rerank` on those hits, cut to 5. ``hybrid`` itself is
not touched, and neither is :data:`retrieval.MODES`: this is ``hybrid_rerank`` in the files and
the report, never an argument to ``retrieve()``.

**The pool is a second live draw.** The run file keeps only hybrid's top 5, so the pool of 20 has
to be retrieved again, and the dense side does not reproduce (see ``eval.scripts.ablation``). Each
line records ``pool_top5_matches_run``: whether the pool's first five are the committed hybrid top
5 the reranked list is compared against. The sidecar lists the ids where it is ``false``.

**The ceiling is printed with the row.** The reranker only reorders the pool, so its Recall@5 is
at most the pool's Recall@20; each line carries ``pool_recall_at_20`` and the sidecar its mean.

**Latency is per question, and the clock excludes loading.** ``pool_ms`` is ``retrieve()``'s own
``latency_ms``; ``rerank_ms`` is the cross-encoder over the pool's (query, chunk) pairs, one batch.
The model is loaded and run once on a dummy pair before the first question, for the reason
``retrieval`` excludes index loading: a first number that paid for the load, and forty that did
not, would not be the latency of anything. The timings make the run file differ on every run,
which it did already, through the dense side.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, TextIO

import numpy as np

from config import Config, file_sha256, load_config, rel_to_repo, rerank_key
from eval.metrics import mrr_at_k, recall_at_k
from eval.schema import SPLITS, EvalItem, Split
from eval.scripts import bootstrap as bs
from eval.scripts.ablation import RUNS_DIR, K, select, write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX, load_split, read_sidecar
from retrieval import retrieve
from retrieval.rerank import Scorer, load_scorer, rerank

#: What the lines, the bootstrap and the report call this row.
MODE = "hybrid_rerank"
#: The mode it reranks, and is compared against.
BASE = "hybrid"


def run_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/rerank_{split}_{rerank_key}.jsonl``."""
    return Path(runs_dir) / f"rerank_{split}_{rerank_key(cfg)}.jsonl"


def pool_depth(cfg: Config) -> int:
    depth = int(cfg["rerank"]["pool"])
    if depth < K:
        raise ValueError(
            f"rerank.pool is {depth}; a pool shallower than k = {K} cannot be cut to k"
        )
    return depth


# --- the run ------------------------------------------------------------------------------------


def score(
    row: EvalItem, base: Mapping[str, Any], cfg: Config, scorer: Scorer, depth: int
) -> dict[str, Any]:
    """One row: the pool, the reranked top 5, both scores, the ceiling, and the two timings."""
    pool = retrieve(row.question, BASE, k=depth, cfg=cfg)
    start = perf_counter()
    hits = rerank(row.question, pool.hits, K, scorer)
    rerank_ms = (perf_counter() - start) * 1000
    pool_ids = [hit.doc_id for hit in pool.hits]
    ranked = [hit.doc_id for hit in hits]
    return {
        "id": row.id,
        "bucket": row.bucket,
        "mode": MODE,
        "ranked": ranked,
        "scores": [round(hit.score, 6) for hit in hits],
        "gold": list(row.gold_doc_ids),
        f"recall_at_{K}": recall_at_k(ranked, row.gold_doc_ids, K),
        f"mrr_at_{K}": mrr_at_k(ranked, row.gold_doc_ids, K),
        "pool": pool_ids,
        f"pool_recall_at_{depth}": recall_at_k(pool_ids, row.gold_doc_ids, depth),
        "pool_top5_matches_run": pool_ids[:K] == list(base["ranked"])[:K],
        "pool_ms": round(pool.latency_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
    }


def base_lines(run: bs.Run) -> dict[str, dict[str, Any]]:
    """The run's hybrid lines by id -- :func:`bs.load_run` has already checked the file."""
    with run.path.open(encoding="utf-8") as f:
        return {line["id"]: line for line in map(json.loads, f) if line["mode"] == BASE}


def spread(values: Sequence[float]) -> dict[str, float]:
    """Median, 90th percentile, mean and max, in ms, to one place."""
    return {
        "median": round(statistics.median(values), 1),
        "p90": round(float(np.percentile(values, 90)), 1),
        "mean": round(statistics.fmean(values), 1),
        "max": round(max(values), 1),
    }


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def sidecar(
    path: Path, run: bs.Run, lines: Sequence[Mapping[str, Any]], cfg: Config, depth: int
) -> dict[str, Any]:
    ids = [line["id"] for line in lines]
    return {
        "run": rel_to_repo(path),
        "sha256": file_sha256(path),
        "config_hash": cfg.hash,
        "rerank_key": rerank_key(cfg),
        "rerank": dict(cfg["rerank"]),
        "versions": {p: version(p) for p in ("sentence-transformers", "torch", "transformers")},
        "paired_with": {"run": rel_to_repo(run.path), "sha256": file_sha256(run.path)},
        "dataset_sha256": run.meta.get("dataset_sha256"),
        "split": run.meta["split"],
        "mode": MODE,
        "base": BASE,
        "k": K,
        "pool": depth,
        "n": len(lines),
        "buckets": run.meta.get("buckets", {}),
        "ids": ids,
        f"pool_recall_at_{depth}": statistics.fmean(ln[f"pool_recall_at_{depth}"] for ln in lines),
        "pool_top5_differs": [ln["id"] for ln in lines if not ln["pool_top5_matches_run"]],
        "latency_ms": {
            "pool": spread([ln["pool_ms"] for ln in lines]),
            "rerank": spread([ln["rerank_ms"] for ln in lines]),
        },
    }


def summary(lines: Sequence[Mapping[str, Any]], base: Mapping[str, Mapping[str, Any]]) -> str:
    n = len(lines)
    out = [f"{'':<14}  {'n':>3}  recall@{K}  mrr@{K}"]
    for name, rows in ((BASE, [base[ln["id"]] for ln in lines]), (MODE, lines)):
        recall = sum(r[f"recall_at_{K}"] for r in rows) / n
        mrr = sum(r[f"mrr_at_{K}"] for r in rows) / n
        out.append(f"{name:<14}  {n:>3}  {recall:>8.3f}  {mrr:>5.3f}")
    return "\n".join(out)


def write_rerank(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    scorer: Scorer | None = None,
    known: frozenset[str] | None = None,
) -> Path:
    """Rerank every row of one half's run, write the run file and its sidecar."""
    run = bs.load_run(split, cfg, runs_dir)
    if run.meta.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError(
            f"{rel_to_repo(run.path)} was made from a different {rel_to_repo(Path(dataset))} "
            "than the one on disk: re-run the ablation over the frozen dataset"
        )
    rows = {row.id: row for row in select(load_split(split, dataset, known)).rows}
    if list(rows) != run.ids:
        raise ValueError(f"{rel_to_repo(run.path)} does not score this half's gold-bearing rows")
    depth = pool_depth(cfg)
    base = base_lines(run)

    if scorer is None:
        scorer = load_scorer(cfg)
    scorer([("warm-up", "warm-up")])  # load and first-call cost, before any clock starts
    lines = [score(rows[i], base[i], cfg, scorer, depth) for i in run.ids]

    path = write_lines(run_path(split, cfg, runs_dir), lines)
    meta = sidecar(path, run, lines, cfg, depth)
    write_json(path.with_suffix(META_SUFFIX), meta)

    rerank_ms, drifted = meta["latency_ms"]["rerank"], meta["pool_top5_differs"]
    print(
        f"{split}: n = {len(lines)}, pool = hybrid@{depth}, model {cfg['rerank']['model']}",
        file=out,
    )
    print(f"\n{summary(lines, base)}\n", file=out)
    print(f"  ceiling: hybrid Recall@{depth} = {meta[f'pool_recall_at_{depth}']:.3f}", file=out)
    print(
        f"  rerank latency per question: median {rerank_ms['median']} ms, "
        f"p90 {rerank_ms['p90']} ms ({cfg['rerank']['device']})",
        file=out,
    )
    print(
        f"  live pool top {K} differs from the run on {len(drifted)}"
        + (f" ({', '.join(drifted)})" if drifted else ""),
        file=out,
    )
    print(f"wrote {rel_to_repo(path)} ({len(lines)} lines, rerank key {rerank_key(cfg)})", file=out)
    return path


# --- reading it back ----------------------------------------------------------------------------


class Rerank(NamedTuple):
    """One half's rerank run, checked, as score vectors aligned with the run it is paired with."""

    path: Path
    meta: dict[str, Any]
    scores: dict[str, np.ndarray]  # metric -> (n,)


def load_rerank(cfg: Config, run: bs.Run, runs_dir: Path = RUNS_DIR) -> Rerank:
    """Refuse the rerank run unless it is its sidecar's file, over ``run``'s rows, under ``cfg``."""
    split = run.meta["split"]
    path = run_path(split, cfg, runs_dir)
    meta = read_sidecar(path.with_suffix(META_SUFFIX))
    if meta is None or not path.exists():
        raise ValueError(
            f"no {split} rerank run at {rel_to_repo(path)}: run "
            f"`python -m eval.scripts.rerank --split {split}` first"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("rerank_key") != rerank_key(cfg) or meta.get("config_hash") != cfg.hash:
        problems.append(f"it was run under {meta.get('config_hash')} / {meta.get('rerank_key')}")
    if meta.get("paired_with", {}).get("sha256") != file_sha256(run.path):
        problems.append(f"it was not paired with the {split} run on disk: re-run it")
    if meta.get("ids") != run.ids or meta.get("k") != K:
        problems.append(f"its ids or k are not the {split} run's")
    lines: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in map(json.loads, f):
            if line["id"] in lines:
                problems.append(f"{line['id']} appears twice")
            lines[line["id"]] = line
    if set(lines) != set(run.ids):
        problems.append(f"its lines do not cover the {len(run.ids)} ids exactly")
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))
    scores = {m: np.array([lines[i][m] for i in run.ids]) for m in bs.METRICS}
    return Rerank(path, meta, scores)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS, help="which half's run to rerank")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="the frozen dataset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="runs and the rerank run")
    args = parser.parse_args(argv)
    try:
        write_rerank(load_config(), args.split, sys.stdout, args.dataset, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
