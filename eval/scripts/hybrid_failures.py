"""Hybrid's misses on the test half: every gold id not in its top 5, and where each side ranked it.

    uv run --env-file .env python -m eval.scripts.hybrid_failures   # after `ablation --split test`

A second step after #75's test run, for Day 6's error analysis. It writes
``eval/runs/hybrid_failures_{hash}.jsonl``, one line per question whose hybrid Recall@5 is below 1,
and a sidecar beside it with the counts by bucket and the digest of the run it was read from.

**Which questions fail is read, not re-run.** The failing rows, their gold ids and hybrid's top 5
come out of the committed ``ablation_test_{hash}.jsonl`` -- the file the Day 4 table is computed
from -- so the log is about the same 41 answers the report scores. Re-running ``k=5`` here would be
a second draw of the dense side, and it could fail a different set of questions.

**The deeper ranks are a second, live pass.** For each failing row only, ``retrieve(q, mode, k=20)``
in all three modes, reusing the cached retrievers. 20 is ``fusion.top_k_each``: it is how deep
hybrid looks on each side before fusing, so a gold id that neither ``dense@20`` nor ``bm25@20``
holds is one fusion never saw (recall), one that a side holds but ``hybrid@20`` ranks past 5 is
one fusion buried, and one at hybrid rank 6-8 is a question of k. Nothing here says which --
that is Day 6.

**The live pass can disagree with the run.** Dense query vectors jitter between calls (see
``eval.scripts.ablation``), so each line records ``live_top5_matches_run``: whether the first five
of the live ``hybrid@20`` are the committed top 5. Where it is ``false`` the ranks describe a
slightly different retrieval than the one the table scored, and the sidecar lists those ids.

**Titles** are the first line of each document's chunk text -- what ``Hit.text`` carries -- looked
up in ``data/chunks.jsonl`` by id, so a reader sees what came back instead of the gold without
opening the corpus, and a committed top-5 id has a title even when the live pass did not return it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from config import Config, file_sha256, load_config, rel_to_repo
from eval.schema import BUCKETS
from eval.scripts import bootstrap as bs
from eval.scripts.ablation import RUNS_DIR, K, write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX, load_split, read_sidecar
from ingest.chunk import load_chunks
from retrieval import MODES, retrieve
from retrieval.types import top_k_each

#: The half the log is written for: the one the Day 4 table reports.
SPLIT = "test"


def failures_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/hybrid_failures_{hash}.jsonl`` -- under the same config hash as the table."""
    return Path(runs_dir) / f"hybrid_failures_{cfg.hash}.jsonl"


def titles_from(chunks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """doc id -> the chunk's first line (``CS 374 — Introduction to Algorithms ...``)."""
    return {c["doc_id"]: (c["text"].splitlines() or [c["doc_id"]])[0] for c in chunks}


def rank_of(doc_id: str, ranked: Sequence[str]) -> int | None:
    """1-based position of ``doc_id``, or ``None`` when it is not there."""
    return ranked.index(doc_id) + 1 if doc_id in ranked else None


def run_lines(run: bs.Run) -> dict[tuple[str, str], dict[str, Any]]:
    """The run file's lines by ``(id, mode)`` -- :func:`bs.load_run` has already checked them."""
    with run.path.open(encoding="utf-8") as f:
        return {(line["id"], line["mode"]): line for line in map(json.loads, f)}


def failing(run: bs.Run) -> list[dict[str, Any]]:
    """The run's hybrid lines with Recall@5 < 1, in the run's id order."""
    lines = run_lines(run)
    return [lines[i, "hybrid"] for i in run.ids if lines[i, "hybrid"][f"recall_at_{K}"] < 1]


def log_line(
    line: Mapping[str, Any],
    question: str,
    deep: Mapping[str, Sequence[str]],
    titles: Mapping[str, str],
) -> dict[str, Any]:
    """One failing question: what hybrid returned, what it missed, and where each mode had it."""
    top5 = list(line["ranked"])[:K]
    missing = [g for g in dict.fromkeys(line["gold"]) if g not in top5]
    return {
        "id": line["id"],
        "bucket": line["bucket"],
        "question": question,
        "gold": list(line["gold"]),
        "missing": missing,
        "hybrid_top5": [{"doc_id": d, "title": titles.get(d)} for d in top5],
        "missing_ranks": {g: {mode: rank_of(g, deep[mode]) for mode in MODES} for g in missing},
        "live_top5_matches_run": list(deep["hybrid"])[:K] == top5,
    }


def deep_ranks(question: str, cfg: Config, depth: int) -> dict[str, list[str]]:
    """The question through ``retrieve(q, mode, k=depth)`` in every mode, as id lists."""
    return {
        mode: [hit.doc_id for hit in retrieve(question, mode, k=depth, cfg=cfg).hits]
        for mode in MODES
    }


def write_failures(
    cfg: Config,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    chunks: Sequence[Mapping[str, Any]] | None = None,
    known: frozenset[str] | None = None,
) -> Path:
    """Read the test run, re-query the failing rows at depth, write the log and its sidecar."""
    run = bs.load_run(SPLIT, cfg, runs_dir)
    if run.meta.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError(
            f"{rel_to_repo(run.path)} was made from a different {rel_to_repo(Path(dataset))} "
            "than the one on disk: re-run the ablation over the frozen dataset"
        )
    questions = {row.id: row.question for row in load_split(SPLIT, dataset, known)}
    titles = titles_from(load_chunks() if chunks is None else chunks)
    depth = top_k_each(cfg)

    lines = [
        log_line(line, questions[line["id"]], deep_ranks(questions[line["id"]], cfg, depth), titles)
        for line in failing(run)
    ]
    path = write_lines(failures_path(cfg, runs_dir), lines)

    bucket_of = {i: line["bucket"] for (i, _), line in run_lines(run).items()}
    scored = Counter(bucket_of[i] for i in run.ids)
    failed = Counter(line["bucket"] for line in lines)
    drifted = [line["id"] for line in lines if not line["live_top5_matches_run"]]
    write_json(
        path.with_suffix(META_SUFFIX),
        {
            "log": rel_to_repo(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "run": rel_to_repo(run.path),
            "run_sha256": file_sha256(run.path),
            "k": K,
            "depth": depth,
            "n": len(run.ids),
            "scored": {b: scored[b] for b in BUCKETS if scored[b]},
            "failing": {b: failed[b] for b in BUCKETS if scored[b]},
            "live_top5_differs": drifted,
        },
    )

    counts = ", ".join(f"{failed[b]} of {scored[b]} {b}" for b in BUCKETS if scored[b])
    print(
        f"{SPLIT}: hybrid misses a gold id in the top {K} on {len(lines)} of {len(run.ids)}",
        file=out,
    )
    print(f"  by bucket: {counts}", file=out)
    print(
        f"  live hybrid@{depth} top {K} differs from the run on {len(drifted)}"
        + (f" ({', '.join(drifted)})" if drifted else ""),
        file=out,
    )
    print(f"wrote {rel_to_repo(path)} ({len(lines)} lines, config {cfg.hash})", file=out)
    return path


def load_failures(cfg: Config, run_path: Path, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """The log's sidecar, refused unless the log and the run it names are the files on disk."""
    path = failures_path(cfg, runs_dir)
    meta = read_sidecar(path.with_suffix(META_SUFFIX))
    if meta is None or not path.exists():
        raise ValueError(
            f"no failure log at {rel_to_repo(path)}: run "
            "`python -m eval.scripts.hybrid_failures` first"
        )
    problems = []
    if file_sha256(path) != meta.get("sha256"):
        problems.append("its sha256 is not the one its sidecar recorded")
    if meta.get("config_hash") != cfg.hash:
        problems.append(f"it was written under {meta.get('config_hash')}, not {cfg.hash}")
    if meta.get("run_sha256") != file_sha256(run_path):
        problems.append("it was not read from the test run on disk: re-run it")
    if problems:
        raise ValueError(f"{rel_to_repo(path)}: " + "; ".join(problems))
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Log the rows where hybrid misses a gold id, with ranks at depth; a refusal exits 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="the frozen dataset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="runs and the log")
    args = parser.parse_args(argv)
    try:
        write_failures(load_config(), sys.stdout, args.dataset, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
