"""RQ1's table: three modes x two metrics on the test half, a paired CI in every cell.

    uv run python -m eval.scripts.day4_report --check     # what exists, and what makes the rest
    uv run python -m eval.scripts.day4_report             # write report/day4_ablation_{hash}.md
    uv run python -m eval.scripts.day4_report --dry-run   # print it, write nothing

Every number is read out of a committed artifact: the intervals and the strongest-single choice
out of ``eval/runs/bootstrap_{full_hash}.json`` (#76), the per-bucket points and the multi-gold
count out of the test run file (#75). Nothing here resamples or re-queries -- a second bootstrap
inside the report would be a second set of intervals that could disagree with the first, and the
dense side of a live run does not reproduce.

**It refuses to write rather than print a table from files that do not belong together.** The
bootstrap result must name the run files on disk by digest, both runs must pass #76's own checks,
and the test run must have been made from the ``v1.jsonl`` that is committed now. A table whose
intervals came from one draw and whose bucket split came from another would be two reports in one
file. ``--check`` lists every input and the command that makes each missing one.

**No clock in the body**, as in #62: a re-run over unchanged inputs rewrites a byte-identical file,
so a diff under ``report/`` is a run file or the bootstrap having moved.

**Intervals only where n carries them.** The main table is the whole test half (n = 41). The
bucket split is point estimates only: at n = 12 one multihop question moves a mean by up to 1/12,
and an interval that wide printed beside a number reads as a finding it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, file_sha256, load_config, rel_to_repo, report_path
from eval.schema import BUCKETS
from eval.scripts import bootstrap as bs
from eval.scripts import hybrid_failures as hf
from eval.scripts.ablation import RUNS_DIR, K, run_path
from eval.scripts.agreement import md_table
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX
from eval.validate import plural
from retrieval import MODES

STEM = "day4_ablation"

#: The metric names as the table prints them.
LABELS = {f"recall_at_{K}": f"Recall@{K}", f"mrr_at_{K}": f"MRR@{K}"}


# --- what the report is built from ----------------------------------------------------------------


class Sources(NamedTuple):
    """The dataset and the directory the runs and the bootstrap result live in."""

    dataset: Path = DATASET_PATH
    runs_dir: Path = RUNS_DIR


class Input(NamedTuple):
    """One artifact the report needs, and the command that produces it when it is missing."""

    name: str
    path: Path
    how: str

    @property
    def exists(self) -> bool:
        return self.path.exists()


def inputs(cfg: Config, src: Sources) -> list[Input]:
    """Every artifact, in the order the pipeline makes them."""
    found = [Input("dataset", src.dataset, "uv run python -m eval.scripts.review --promote")]
    for split in ("calib", "test"):
        run = run_path(split, cfg, src.runs_dir)
        how = f"uv run --env-file .env python -m eval.scripts.ablation --split {split}"
        found += [
            Input(f"{split} run", run, how),
            Input(f"{split} sidecar", run.with_suffix(META_SUFFIX), how),
        ]
    found.append(
        Input(
            "bootstrap",
            bs.result_path(cfg, src.runs_dir),
            "uv run python -m eval.scripts.bootstrap",
        )
    )
    found.append(
        Input(
            "failure log",
            hf.failures_path(cfg, src.runs_dir),
            "uv run --env-file .env python -m eval.scripts.hybrid_failures",
        )
    )
    return found


def missing(cfg: Config, src: Sources) -> list[Input]:
    return [one for one in inputs(cfg, src) if not one.exists]


def render_check(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--check``: the inputs, one per line, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    out.write(
        f"report {rel_to_repo(report_path(STEM, cfg))}  config {cfg.hash}  full {cfg.full_hash}\n\n"
    )
    for one in found:
        out.write(f"  {'ok     ' if one.exists else 'MISSING'}  {one.name:<{width}}  ")
        out.write(rel_to_repo(one.path) + "\n")
        if not one.exists:
            out.write(f"  {'':7}  {'':<{width}}  {one.how}\n")
    gaps = missing(cfg, src)
    if not gaps:
        out.write("\nEvery input is present; the report can be written.\n")
        return 0
    out.write(f"\n{plural(len(gaps), 'input')} missing, so the report is not written.\n")
    return 1


# --- reading them ---------------------------------------------------------------------------------


class Facts(NamedTuple):
    """Everything the report prints, read and cross-checked."""

    cfg: Config
    src: Sources
    result: dict[str, Any]
    test: bs.Run
    #: bucket -> mode -> metric -> mean over that bucket's test rows.
    by_bucket: dict[str, dict[str, dict[str, float]]]
    bucket_n: dict[str, int]
    #: How many test rows carry more than one gold id, per bucket.
    multi_gold: dict[str, int]
    #: #78's failure-log sidecar: hybrid's misses on the same test run, by bucket.
    failures: dict[str, Any]


def read_facts(cfg: Config, src: Sources) -> Facts:
    """Load the bootstrap result and the runs, refusing any pair that does not belong together."""
    calib = bs.load_run("calib", cfg, src.runs_dir)
    test = bs.load_run("test", cfg, src.runs_dir)
    path = bs.result_path(cfg, src.runs_dir)
    result = json.loads(path.read_text(encoding="utf-8"))

    problems = []
    if result.get("config_hash") != cfg.hash or result.get("full_hash") != cfg.full_hash:
        problems.append(
            f"{rel_to_repo(path)} was computed under {result.get('config_hash')} / "
            f"{result.get('full_hash')}, not {cfg.hash} / {cfg.full_hash}"
        )
    for run in (calib, test):
        split = run.meta["split"]
        if result.get("runs", {}).get(split, {}).get("sha256") != file_sha256(run.path):
            problems.append(
                f"{rel_to_repo(path)} was not computed from the {split} run on disk: "
                "re-run `python -m eval.scripts.bootstrap`"
            )
    dataset_sha = file_sha256(src.dataset)
    if test.meta.get("dataset_sha256") != dataset_sha:
        problems.append(
            f"the test run was made from a different {rel_to_repo(src.dataset)} than the one "
            "on disk: re-run the ablation over the frozen dataset"
        )
    if result.get("dataset_sha256") != dataset_sha:
        problems.append(f"{rel_to_repo(path)} names a different dataset digest")
    if result.get("n", {}).get("test") != len(test.ids):
        problems.append(f"{rel_to_repo(path)}'s test n is not the test run's")
    try:
        failures = hf.load_failures(cfg, test.path, src.runs_dir)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        raise ValueError("refusing to write the report:\n  " + "\n  ".join(problems))

    lines: dict[tuple[str, str], dict[str, Any]] = {}
    with test.path.open(encoding="utf-8") as f:
        for raw in f:
            line = json.loads(raw)
            lines[line["id"], line["mode"]] = line
    bucket_of = {i: lines[i, MODES[0]]["bucket"] for i in test.ids}
    buckets = [b for b in BUCKETS if b in bucket_of.values()]
    by_bucket: dict[str, dict[str, dict[str, float]]] = {}
    bucket_n: dict[str, int] = {}
    multi_gold: dict[str, int] = defaultdict(int)
    for bucket in buckets:
        ids = [i for i in test.ids if bucket_of[i] == bucket]
        bucket_n[bucket] = len(ids)
        multi_gold[bucket] = sum(len(set(lines[i, MODES[0]]["gold"])) > 1 for i in ids)
        by_bucket[bucket] = {
            mode: {m: sum(lines[i, mode][m] for i in ids) / len(ids) for m in bs.METRICS}
            for mode in MODES
        }
    return Facts(cfg, src, result, test, by_bucket, bucket_n, dict(multi_gold), failures)


# --- saying it ------------------------------------------------------------------------------------


def num(x: float, signed: bool = False) -> str:
    """Three places, a real minus sign, and never ``-0.000``."""
    x = round(x, 3) + 0.0
    text = f"{x:+.3f}" if signed else f"{x:.3f}"
    return text.replace("-", "−")


def cell(interval: Mapping[str, Any], signed: bool = False) -> str:
    text = f"{num(interval['point'], signed)} [{num(interval['lo'])}, {num(interval['hi'])}]"
    return text + (" (degenerate)" if interval["degenerate"] else "")


def verdict(gap: Mapping[str, Any]) -> str:
    return (
        "significant (CI excludes 0)" if gap["significant"] else "not significant (CI contains 0)"
    )


def subtrahends(result: Mapping[str, Any]) -> list[str]:
    return [result["metrics"][m]["difference"]["subtrahend"] for m in bs.METRICS]


def render_header(facts: Facts) -> str:
    r = facts.result
    n = r["n"]["test"]
    split = ", ".join(f"{b} {facts.bucket_n[b]}" for b in facts.bucket_n)
    runs = r["runs"]
    return "\n".join(
        [
            "# Day 4 — RQ1: retrieval ablation",
            "",
            (
                f"Recall@{K} and MRR@{K} for `dense`, `bm25` and `hybrid` through "
                f"`retrieve(q, mode, k={K})`, on the frozen `test` half."
            ),
            "",
            f"- **n** = {n} test rows ({split}) — the rows that carry gold doc ids",
            f"- **dataset** `{rel_to_repo(facts.src.dataset)}` sha256 `{r['dataset_sha256']}`",
            (
                f"- **config** `{r['config_hash']}` (dense index `{r['dense_key']}`, bm25 index "
                f"`{r['bm25_key']}`)"
            ),
            (
                f"- **runs** `{runs['test']['run']}` (reported), `{runs['calib']['run']}` "
                "(chooses the strongest single mode only)"
            ),
            (
                f"- **bootstrap** `{rel_to_repo(bs.result_path(facts.cfg, facts.src.runs_dir))}` — "
                f"paired, B = {r['n_resamples']:,}, seed {r['seed']}, "
                f"{round(r['ci'] * 100)}% percentile intervals (full hash `{r['full_hash']}`)"
            ),
        ]
    )


def render_table(facts: Facts) -> str:
    r = facts.result
    metrics = r["metrics"]
    rows = [[f"`{mode}`", *(cell(metrics[m]["test"][mode]) for m in bs.METRICS)] for mode in MODES]
    subs = subtrahends(r)
    same = len(set(subs)) == 1
    label = f"*hybrid − {subs[0]}*" if same else "*hybrid − strongest single*"
    gaps = []
    for m in bs.METRICS:
        gap = metrics[m]["difference"]
        text = f"{cell(gap, signed=True)}, {verdict(gap)}"
        gaps.append(f"*{text}*" if same else f"*vs {gap['subtrahend']}: {text}*")
    rows.append([label, *gaps])
    table = md_table(["mode", *(LABELS[m] for m in bs.METRICS)], rows)

    reading = []
    for m in bs.METRICS:
        gap = metrics[m]["difference"]
        sub = gap["subtrahend"]
        if gap["significant"]:
            higher = "hybrid" if gap["point"] > 0 else sub
            tail = f"the interval excludes 0: on this metric `{higher}` is higher."
        else:
            tail = (
                f"the interval contains 0: these {r['n']['test']} questions do not separate them."
            )
        reading.append(
            f"- **{LABELS[m]}**: hybrid − {sub} = {num(gap['point'], signed=True)} "
            f"[{num(gap['lo'])}, {num(gap['hi'])}]; {tail}"
        )
    degenerate = [
        f"`{mode}` {LABELS[m]}"
        for m in bs.METRICS
        for mode in MODES
        if metrics[m]["test"][mode]["degenerate"]
    ]
    if degenerate:
        reading.append(
            f"- Degenerate intervals ({', '.join(degenerate)}): every resample gave the same "
            "mean. That is no variance observed, not certainty."
        )
    return "\n".join(
        [
            "## Result",
            "",
            (
                f"Test half, n = {r['n']['test']}. Point estimate on all n rows, "
                f"{round(r['ci'] * 100)}% paired-bootstrap interval in brackets."
            ),
            "",
            table,
            "",
            *reading,
        ]
    )


def render_strongest(facts: Facts) -> str:
    out = [
        "## Strongest single mode",
        "",
        (
            "Chosen per metric on `calib`, then tested on `test`. Choosing it on the test rows "
            "would compare hybrid with the larger of two noisy means, which biases the gap "
            "against hybrid."
        ),
        "",
    ]
    for m in bs.METRICS:
        choice = facts.result["metrics"][m]["strongest_single"]
        means = ", ".join(f"`{mode}` {num(v)}" for mode, v in choice["calib_means"].items())
        tie = f"; a tie, broken in `{'`, `'.join(MODES)}` order" if choice["tied"] else ""
        out.append(
            f"- **{LABELS[m]}**: `{choice['mode']}` (calib n = {facts.result['n']['calib']}: "
            f"{means}{tie})"
        )
    return "\n".join(out)


def render_buckets(facts: Facts) -> str:
    headers = ["mode"] + [
        f"{b} {LABELS[m]} (n = {facts.bucket_n[b]})" for b in facts.by_bucket for m in bs.METRICS
    ]
    rows = [
        [f"`{mode}`"]
        + [num(facts.by_bucket[b][mode][m]) for b in facts.by_bucket for m in bs.METRICS]
        for mode in MODES
    ]
    smallest = min(facts.bucket_n, key=facts.bucket_n.get)
    k = facts.bucket_n[smallest]
    return "\n".join(
        [
            "## By bucket (point estimates only)",
            "",
            md_table(headers, rows),
            "",
            (
                f"No intervals here, on purpose. At n = {k} one `{smallest}` question moves a "
                f"mean by up to 1/{k} = {num(1 / k)}, so an interval on this split is too wide to "
                "be read as a finding. The split describes where the rows in the main table come "
                "from. It is not a second test."
            ),
        ]
    )


def render_failures(facts: Facts) -> str:
    f = facts.failures
    total = sum(f["failing"].values())
    counts = ", ".join(f"{f['failing'][b]} of {f['scored'][b]} `{b}`" for b in f["scored"])
    drifted = f["live_top5_differs"]
    return "\n".join(
        [
            "## Where hybrid misses (for Day 6)",
            "",
            (
                f"Hybrid leaves at least one gold id out of its top {f['k']} on {total} of "
                f"{f['n']} test rows: {counts}. Each is logged in `{f['log']}` with the ids it "
                f"missed, the {f['k']} it returned, and the rank each missing id has in "
                f"`dense@{f['depth']}`, `bm25@{f['depth']}` and `hybrid@{f['depth']}`."
            ),
            "",
            (
                f"The failing rows are read from `{f['run']}`. The ranks at depth come from a "
                "second, live pass over those rows only, and its top "
                f"{f['k']} differs from the run's on {len(drifted)} of {total}"
                + (f" ({', '.join(drifted)})" if drifted else "")
                + ". The log does not sort the misses by cause. That is Day 6."
            ),
        ]
    )


def render_definitions(facts: Facts) -> str:
    multi = ", ".join(f"{facts.multi_gold[b]} of {facts.bucket_n[b]} `{b}`" for b in facts.bucket_n)
    return "\n".join(
        [
            "## Definitions",
            "",
            (
                f"- **MRR is MRR@{K}.** 1 / rank of the first gold id in the top {K}, and 0 if "
                "none is there (`eval/metrics.py`, #74)."
            ),
            (
                f"- **Recall@{K} is fractional on multi-gold rows.** It is "
                f"|gold ∩ top-{K}| / |gold|, not Hit@{K}. Rows with more than one gold id: {multi}."
            ),
            (
                f"- The ranked list is cut to {K} before anything is counted, and a repeated id "
                "in the top k counts once."
            ),
        ]
    )


REPRODUCTION = """\
## Reproduction

```bash
uv run --env-file .env python -m eval.scripts.ablation --split calib
uv run --env-file .env python -m eval.scripts.ablation --split test
uv run python -m eval.scripts.bootstrap
uv run --env-file .env python -m eval.scripts.hybrid_failures
uv run python -m eval.scripts.day4_report
```

This file is a function of the two committed run files, the bootstrap result computed from them,
the failure log's sidecar, the dataset, and `config.yaml`. It has no timestamp, so re-running the
last command over unchanged inputs rewrites it byte for byte. Re-running the first two does not
reproduce the run files: the embeddings API returns slightly different query vectors from call to
call, and two live calib runs differed in 1 of 41 dense lines. The committed run files are the
sample this table describes.
"""


def render_report(facts: Facts) -> str:
    sections = [
        render_header(facts),
        render_table(facts),
        render_strongest(facts),
        render_buckets(facts),
        render_failures(facts),
        render_definitions(facts),
        REPRODUCTION.rstrip(),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def summary(facts: Facts, path: Path) -> str:
    metrics = facts.result["metrics"]
    out = [f"config hash {facts.cfg.hash}, test n = {facts.result['n']['test']}"]
    for m in bs.METRICS:
        gap = metrics[m]["difference"]
        out.append(
            f"{LABELS[m]:<9} hybrid − {gap['subtrahend']} {cell(gap, signed=True)}  {verdict(gap)}"
        )
    out.append(f"wrote {rel_to_repo(path)}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.day4_report",
        description="Write report/day4_ablation_{hash}.md from the committed run files and the "
        "paired bootstrap over them.",
    )
    parser.add_argument(
        "--check", action="store_true", help="list the inputs and what makes each; write nothing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the report instead of writing it"
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="the frozen dataset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="runs and bootstrap")
    args = parser.parse_args(argv)

    cfg = load_config()
    src = Sources(args.dataset, args.runs_dir)
    if args.check:
        return render_check(cfg, src, sys.stdout)
    if missing(cfg, src):
        render_check(cfg, src, sys.stderr)
        return 1
    try:
        facts = read_facts(cfg, src)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    body = render_report(facts)
    if args.dry_run:
        sys.stdout.write(body)
        return 0
    path = report_path(STEM, cfg)
    path.write_text(body, encoding="utf-8")
    print(summary(facts, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
