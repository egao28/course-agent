"""RQ1's run: every gold-bearing row of one half, through ``retrieve(q, mode, k=5)`` per mode.

    uv run --env-file .env python -m eval.scripts.ablation --split calib   # develop against this
    uv run --env-file .env python -m eval.scripts.ablation --split test    # the reported number

One line per (row, mode) goes to ``eval/runs/ablation_{split}_{hash}.jsonl``, and a sidecar
beside it records what the run was. **The run file is the artifact**: the bootstrap (#76) and the
table (#77) read it and never call ``retrieve()`` themselves, so re-drawing a CI never re-queries
an index, and the numbers in the report are the ones in a file somebody can diff.

**Same call as the pipeline, one variable.** Every line comes from :func:`retrieval.retrieve`
with ``k=5``, and ``mode`` is the only argument that changes. Nothing here reaches past it into
``DenseRetriever``, ``BM25Retriever`` or ``fuse``: a different code path is a different pipeline,
and the ablation would then be measuring the difference between two programs rather than between
two modes. Hybrid still asks each side for ``fusion.top_k_each`` (20) before fusing, because
``retrieve`` does.

**Which rows.** Only rows with ``gold_doc_ids`` can be scored -- ``fact`` and ``multihop`` --
and the rest are *skipped, counted by bucket, and the count printed and recorded*. A silent
filter is how a denominator goes wrong, and ``n`` is in every line this prints.

**Which half.** ``--split`` is required and has no default, for the reason
:func:`eval.scripts.split.load_split` gives: the failure to guard against is a run that never
chose one. Both halves get a file of their own -- the split is in the name -- because #76 needs
both: the strongest single mode is chosen on ``calib`` and the gap is tested on ``test``. One name
for both would let a debugging run over ``calib`` overwrite the number that goes in the report.

**The runner is deterministic; the dense side of a live run is not.** Rows are taken in file
order and modes in :data:`retrieval.MODES` order, and neither file carries a timestamp or a
latency, so with deterministic retrievers a re-run rewrites byte-identical files -- the tests hold
it to that with fakes. BM25 is deterministic. The dense query vector is not: the embeddings API
returns vectors that differ by ~1e-4 in cosine from call to call, and two live ``--split calib``
runs on 2026-09-27 differed in one of 41 dense lines (``v1-0007``, MRR@5 0.5 vs 1/3, where TE 461
and MATH 461 sit 0.002 apart). A diff under ``eval/runs/`` is therefore a change in what was
retrieved, never in the clock -- but it is not always a change in the pipeline. The committed run
file is the one draw the report is computed from, which is why nothing downstream re-queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import REPO_ROOT, Config, bm25_key, dense_key, file_sha256, load_config, rel_to_repo
from eval.metrics import mrr_at_k, recall_at_k
from eval.schema import BUCKETS, SPLITS, EvalItem, Split
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX, load_split, meta_path, read_sidecar
from ingest.chunk import chunk_key
from retrieval import MODES, retrieve

#: The cut-off, in every mode. RQ1 asks about the five hits the answer node is handed.
K = 5

#: Where run files go. Committed: they are what the Day 4 report is computed from.
RUNS_DIR = REPO_ROOT / "eval" / "runs"


def run_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/ablation_{split}_{hash}.jsonl`` -- one file per half, per config."""
    return Path(runs_dir) / f"ablation_{split}_{cfg.hash}.jsonl"


# --- which rows ---------------------------------------------------------------------------------


class Selection(NamedTuple):
    """The rows that can be scored, and how many of the rest were left out, by bucket."""

    rows: list[EvalItem]
    skipped: Counter[str]

    def counts(self) -> dict[str, int]:
        """Scored rows per bucket, buckets with none left out."""
        scored = Counter(row.bucket for row in self.rows)
        return {bucket: scored[bucket] for bucket in BUCKETS if scored[bucket]}


def select(rows: Sequence[EvalItem]) -> Selection:
    """Rows with gold ids, in file order; every other row counted under its bucket."""
    scored = [row for row in rows if row.gold_doc_ids]
    skipped = Counter(row.bucket for row in rows if not row.gold_doc_ids)
    if not scored:
        raise ValueError("no row in this half carries gold_doc_ids; there is nothing to score")
    return Selection(scored, skipped)


# --- the run ------------------------------------------------------------------------------------


def score(row: EvalItem, mode: str, cfg: Config) -> dict[str, Any]:
    """One (row, mode) line: what came back, what should have, and the two per-question scores."""
    result = retrieve(row.question, mode, k=K, cfg=cfg)  # type: ignore[arg-type]
    ranked = [hit.doc_id for hit in result.hits]
    return {
        "id": row.id,
        "bucket": row.bucket,
        "mode": mode,
        "ranked": ranked,
        "gold": list(row.gold_doc_ids),
        f"recall_at_{K}": recall_at_k(ranked, row.gold_doc_ids, K),
        f"mrr_at_{K}": mrr_at_k(ranked, row.gold_doc_ids, K),
    }


def run(rows: Sequence[EvalItem], cfg: Config) -> list[dict[str, Any]]:
    """Every row through every mode, row-major: a row's three lines sit together."""
    return [score(row, mode, cfg) for row in rows for mode in MODES]


def write_lines(path: Path, lines: Sequence[Mapping[str, Any]]) -> Path:
    """The run file, one object per line, through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def sidecar(
    path: Path, dataset: Path, split: Split, selection: Selection, cfg: Config
) -> dict[str, Any]:
    """What the run was: the settings, both index keys, the dataset digest, and the rows used.

    The dataset digest is the one #61's freeze recorded, which :func:`load_split` has just checked
    against the file -- so it names the frozen set, not merely whatever was on disk. The run's own
    digest is taken from the file as written, never from the lines in memory.
    """
    key = chunk_key(cfg)
    dataset_meta = read_sidecar(meta_path(dataset)) or {}
    return {
        "run": rel_to_repo(path),
        "sha256": file_sha256(path),
        "config_hash": cfg.hash,
        "dense_key": dense_key(cfg, key),
        "bm25_key": bm25_key(cfg, key),
        "dataset": rel_to_repo(dataset),
        "dataset_sha256": dataset_meta.get("sha256"),
        "split": split,
        "k": K,
        "modes": list(MODES),
        "n": len(selection.rows),
        "lines": len(selection.rows) * len(MODES),
        "buckets": selection.counts(),
        "skipped": dict(sorted(selection.skipped.items())),
        "ids": [row.id for row in selection.rows],
    }


# --- saying what happened -----------------------------------------------------------------------


def summary(lines: Sequence[Mapping[str, Any]], n: int) -> str:
    """Mean Recall@5 and MRR@5 per mode, with n -- a check that the run ran, not the table."""
    out = [f"{'mode':<8}  {'n':>3}  recall@{K}  mrr@{K}"]
    for mode in MODES:
        mine = [line for line in lines if line["mode"] == mode]
        recall = sum(line[f"recall_at_{K}"] for line in mine) / len(mine)
        mrr = sum(line[f"mrr_at_{K}"] for line in mine) / len(mine)
        out.append(f"{mode:<8}  {n:>3}  {recall:>8.3f}  {mrr:>5.3f}")
    return "\n".join(out)


def ablate(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
) -> Path:
    """Run one half, write the run file and its sidecar, and say what was scored and skipped."""
    selection = select(load_split(split, dataset, known))
    counts = ", ".join(f"{n} {bucket}" for bucket, n in selection.counts().items())
    skipped = ", ".join(f"{n} {bucket}" for bucket, n in sorted(selection.skipped.items()))
    print(f"{split}: scoring n = {len(selection.rows)} ({counts})", file=out)
    print(
        f"{split}: skipped {sum(selection.skipped.values())} without gold ids "
        f"({skipped or 'none'})",
        file=out,
    )

    lines = run(selection.rows, cfg)
    path = write_lines(run_path(split, cfg, runs_dir), lines)
    write_json(path.with_suffix(META_SUFFIX), sidecar(path, Path(dataset), split, selection, cfg))

    print(f"\n{summary(lines, len(selection.rows))}\n", file=out)
    print(f"wrote {rel_to_repo(path)} ({len(lines)} lines, config {cfg.hash})", file=out)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run one half through all three modes and write the run file; a refusal exits 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        required=True,
        choices=SPLITS,
        help="which half to run; required, because a run that never chose one is the failure",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="a frozen dataset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where the run file goes")
    args = parser.parse_args(argv)
    try:
        ablate(load_config(), args.split, sys.stdout, args.dataset, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
