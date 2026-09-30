"""``report/README.md``'s tables, and the check that every number in it traces to a file.

    uv run python -m eval.scripts.final_report            # rewrite the generated blocks
    uv run python -m eval.scripts.final_report --check     # verify it; exit 1 if it does not
    uv run python -m eval.scripts.final_report --dry-run   # print the rewritten file, write nothing
    uv run python -m eval.scripts.final_report --inputs    # list the artifacts it reads

#106's report is the one report in this repo whose prose is *written*, not formatted -- that is
what makes it half a day's work rather than a generator. So the Day 3-5 rule ("every number is
formatted from the data") is replaced here by a lighter one that a machine can still hold:

* **A table or a figure is generated.** Each sits between ``<!-- BEGIN gen:name -->`` and
  ``<!-- END gen:name -->``, and :data:`BLOCKS` is the only thing that writes between them. A
  table outside a marker pair is an error, not an exemption: it is a number nothing regenerates.
* **A number in the prose carries its source.** ``0.405 [src](../eval/runs/x.json "a.key.path")``
  -- the link names the file, its title names the path inside it, and ``--check`` reads the value
  there and compares it against the digits, at the precision they are printed to.
* **Everything else is an allowlist**, :data:`ALLOWED`, and it is meant to stay short: issue
  numbers, section numbers, RQ and day labels, interval levels, and the two formulas. A number
  that wants a new entry almost always wants a link instead.

``--check`` exits non-zero when a block is stale, a prose number has no link, a linked value is
not in the linked file at the printed precision, a hex key is not one this config produces, a
table sits outside a block, or any input fails the sha256 chain its sidecars record. That last one
is not this script's own work: :func:`eval.scripts.day4_report.read_facts` and
:func:`eval.scripts.day5_report.read_facts` already refuse to build a report out of files that do
not belong together, and this one runs both of them and adds Day 6's chain to it. A roll-up report
that re-implemented those checks would be a second opinion nobody asked for.

**It never calls a model and never resamples.** Every interval it prints was computed by the
script that owns it, read back out of the committed JSON. The Day 4 blocks come through
:mod:`eval.scripts.day4_report`'s own facts (including its per-bucket means, computed from the
committed ablation run), so the front page and the Day 4 report cannot disagree about a number:
two renderings of one table, yes -- the front page's are terser, and #106 chose that -- but of one
set of values.

**The architecture block is drawn from the wiring, not from a picture.** The nodes and edges come
from :func:`agent.graph.build_graph` under this config and under the one config that builds the
dormant gate (:func:`eval.scripts.day5_report.switched_config`); an edge only the second one has
is dashed. The graph is built with stub nodes, so drawing the diagram opens no index and reads no
key. What stays hand-written is the *wording* on each box and arrow (:data:`LABELS`,
:data:`EDGE_LABELS`) -- with the parts a config decides interpolated, so the model name, the
retrieval mode and ``k`` in the picture move when ``config.yaml`` does.

**No clock anywhere.** A re-run over unchanged inputs rewrites the same bytes, which is what makes
``--check`` a check rather than a diff against a timestamp.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from agent.graph import CONDITIONAL, NODE_BUILDERS, build_graph, node_names
from agent.state import EXITS, INTENTS
from config import (
    REPORT_DIR,
    Config,
    bootstrap_key,
    conformal_key,
    correctness_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval.schema import BUCKETS, RULES
from eval.scripts import (
    agreement,
    attribute,
    conformal_eval,
    day3_report,
    e2e,
    e2e_metrics,
    risk_coverage,
)
from eval.scripts import day4_report as d4
from eval.scripts import day5_report as d5
from eval.scripts.agreement import md_table
from eval.scripts.day4_report import num
from eval.scripts.review import DATASET_PATH
from eval.scripts.signals import RUNS_DIR
from eval.scripts.split import META_SUFFIX
from eval.validate import plural
from retrieval.fusion import top_k_final

#: The file this script owns. Not named after a hash: it is the front page, and a front page that
#: changes its name when a threshold moves is a front page nobody can link to.
REPORT = REPORT_DIR / "README.md"

#: ``<!-- BEGIN gen:name -->`` / ``<!-- END gen:name -->``, at the start of a line.
MARKER_RE = re.compile(r"^<!-- (?P<edge>BEGIN|END) gen:(?P<name>[a-z0-9_]+) -->$", re.MULTILINE)

#: ``0.405 [src](../eval/runs/x.json "a.key.path")`` -- a number's file, and where in it to look.
#: The title may name several paths, comma-separated, for a sentence that reads three numbers out
#: of one file; the two halves may sit on either side of a line break, because the prose is
#: wrapped by hand.
SRC_RE = re.compile(r'\[src\]\((?P<path>[^)\s]+)\s+"(?P<key>[^"]+)"\s*\)')

#: The gold intent sidecar (#102), read for the label counts the intent block prints.
INTENTS_PATH = DATASET_PATH.with_name("v1_intents.jsonl")


# --- what it reads ------------------------------------------------------------------------------


class Sources(NamedTuple):
    """Where the report reads from, rooted so a test can point it at a copy."""

    report: Path = REPORT
    runs_dir: Path = RUNS_DIR
    dataset: Path = DATASET_PATH
    intents: Path = INTENTS_PATH
    figures_dir: Path = risk_coverage.FIGURES_DIR
    report_dir: Path = REPORT_DIR


class Input(NamedTuple):
    """One artifact the report needs, and the command that produces it when it is missing."""

    name: str
    path: Path
    how: str


def sibling(stem: str, cfg: Config, src: Sources) -> Path:
    """``report/{stem}_{config hash}.md`` -- a day report this one links to and quotes from."""
    return src.report_dir / f"{stem}_{cfg.hash}.md"


def day6_inputs(cfg: Config, src: Sources) -> list[Input]:
    """Day 6's own artifacts. Day 4's and Day 5's are their reports' lists, not this one's."""
    runs = src.runs_dir
    run = e2e.run_path("test", cfg, runs)
    metrics = e2e_metrics.result_path(cfg, runs)
    attribution = attribute.attribution_path(cfg, runs)
    return [
        Input("gold intents", src.intents,
              "uv run python -m eval.intents --stamp --labeller NAME"),
        Input("e2e run", run,
              "uv run --env-file .env python -m eval.scripts.e2e --split test"),
        Input("e2e run sidecar", run.with_suffix(META_SUFFIX),
              "uv run --env-file .env python -m eval.scripts.e2e --split test"),
        Input("e2e metrics", metrics, "uv run python -m eval.scripts.e2e_metrics"),
        Input("attribution", attribution,
              "uv run python -m eval.scripts.attribute --review --reviewer NAME"),
        Input("attribution sidecar", attribution.with_suffix(META_SUFFIX),
              "uv run python -m eval.scripts.attribute --review --reviewer NAME"),
    ]


def inputs(cfg: Config, src: Sources) -> list[Input]:
    """Every artifact the front page is a function of, in the order its sections read them."""
    d4_src = d4.Sources(runs_dir=src.runs_dir, dataset=src.dataset)
    d5_src = d5.Sources(runs_dir=src.runs_dir, dataset=src.dataset, figures_dir=src.figures_dir)
    day4 = [one for one in d4.inputs(cfg, d4_src) if one.name != "dataset"]
    day5 = [one for one in d5.inputs(cfg, d5_src) if one.name != "dataset"]
    return [
        Input("dataset", src.dataset, "uv run python -m eval.scripts.review --promote"),
        *(Input(f"day 4: {one.name}", one.path, one.how) for one in day4),
        *(Input(f"day 5: {one.name}", one.path, one.how) for one in day5),
        *(Input(f"day 6: {one.name}", one.path, one.how) for one in day6_inputs(cfg, src)),
        Input("day 3 report", sibling(day3_report.STEM, cfg, src),
              "uv run python -m eval.scripts.day3_report"),
        Input("day 3 agreement", sibling(agreement.STEM, cfg, src),
              "uv run python -m eval.scripts.agreement --read SHEET"),
        Input("day 4 report", sibling(d4.STEM, cfg, src),
              "uv run python -m eval.scripts.day4_report"),
        Input("day 5 report", d5.report_path(cfg, src.report_dir),
              "uv run python -m eval.scripts.day5_report"),
        Input("the report itself", src.report, "this script, with no flags"),
    ]


def missing(cfg: Config, src: Sources) -> list[Input]:
    """The inputs that are not there. Empty is the condition for rewriting anything."""
    return [one for one in inputs(cfg, src) if not one.path.exists()]


def render_inputs(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--inputs``: every artifact, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    out.write(f"report {rel_to_repo(src.report)}  config {cfg.hash}  full {cfg.full_hash}\n\n")
    for one in found:
        out.write(f"  {'ok     ' if one.path.exists() else 'MISSING'}  {one.name:<{width}}  "
                  f"{rel_to_repo(one.path)}\n")
        if not one.path.exists():
            out.write(f"  {'':7}  {'':<{width}}  {one.how}\n")
    gaps = missing(cfg, src)
    if not gaps:
        out.write("\nEvery input is present.\n")
        return 0
    out.write(f"\n{plural(len(gaps), 'input')} missing, so the blocks are not rewritten.\n")
    return 1


# --- reading them, and checking they belong together --------------------------------------------


class Facts(NamedTuple):
    """Everything the blocks print, read once and checked before a line is rendered."""

    cfg: Config
    src: Sources
    #: Day 4's own facts, from its own reader: the bootstrap result and the per-bucket means.
    day4: d4.Facts
    #: Day 5's own facts: the risk-coverage numbers, the calib fit and its one reading on test.
    day5: d5.Facts
    dataset: list[dict[str, Any]]
    intents: list[dict[str, Any]]
    run: list[dict[str, Any]]
    run_meta: dict[str, Any]
    metrics: dict[str, Any]
    attribution: list[dict[str, Any]]
    attribution_meta: dict[str, Any]
    #: ``(name, repo path, sha256)`` for the traceability block, in reading order.
    digests: list[tuple[str, str, str]]
    #: Every hex key the report is allowed to name, and what produces it.
    keys: dict[str, str]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_facts(cfg: Config, src: Sources) -> Facts:
    """Read every input, or raise ``ValueError`` naming every link in the chain that has moved."""
    d4_src = d4.Sources(runs_dir=src.runs_dir, dataset=src.dataset)
    d5_src = d5.Sources(runs_dir=src.runs_dir, dataset=src.dataset, figures_dir=src.figures_dir)
    day4 = d4.read_facts(cfg, d4_src)  # its own sha256 chain, or ValueError
    day5 = d5.read_facts(cfg, d5_src)  # likewise, including the dormant gate

    problems: list[str] = []
    runs = src.runs_dir
    dataset_sha = file_sha256(src.dataset)
    intents_sha = file_sha256(src.intents)

    run_path = e2e.run_path("test", cfg, runs)
    run_meta = _json(run_path.with_suffix(META_SUFFIX))
    run_sha = file_sha256(run_path)
    metrics_path = e2e_metrics.result_path(cfg, runs)
    metrics = _json(metrics_path)
    metrics_sha = file_sha256(metrics_path)
    attribution_path = attribute.attribution_path(cfg, runs)
    attribution = _jsonl(attribution_path)
    attribution_meta = _json(attribution_path.with_suffix(META_SUFFIX))
    csv_path = runs / metrics["csv"].rsplit("/", 1)[-1]

    if run_meta.get("sha256") != run_sha:
        problems.append(f"{rel_to_repo(run_path)} is not the file its sidecar describes")
    if run_meta.get("errors"):
        problems.append(f"{rel_to_repo(run_path)} has {run_meta['errors']} errored row(s); the "
                        "front page reports it as a clean run")
    for name, record in ((rel_to_repo(metrics_path), metrics),
                         (rel_to_repo(attribution_path), attribution_meta)):
        if record.get("config_hash") != cfg.hash:
            problems.append(f"{name} was made under {record.get('config_hash')}, not {cfg.hash}")
        if record.get("run_sha256") != run_sha:
            problems.append(f"{name} does not describe the e2e run on disk")
        if record.get("dataset_sha256") != dataset_sha:
            problems.append(f"{name} was made on another dataset")
        if record.get("intents_sha256") != intents_sha:
            problems.append(f"{name} was made on another gold intent sidecar")
        if record.get("csv_sha256") != file_sha256(csv_path):
            problems.append(f"{name} does not describe the labelled CSV on disk")
    if attribution_meta.get("metrics_sha256") != metrics_sha:
        problems.append(f"{rel_to_repo(attribution_path)} was attributed against another "
                        f"{rel_to_repo(metrics_path)}")
    if attribution_meta.get("sha256") != file_sha256(attribution_path):
        problems.append(f"{rel_to_repo(attribution_path)} is not the file its sidecar describes")
    if not attribution_meta.get("complete"):
        problems.append(f"{rel_to_repo(attribution_path)} is an unfinished pass: the front page "
                        "prints the table only when every failure has a confirmed class")
    undecided = [row["id"] for row in attribution if not row.get("final")]
    if undecided:
        problems.append(f"{plural(len(undecided), 'attributed row')} with no `final` class: "
                        f"{', '.join(undecided)}")
    if problems:
        raise ValueError("refusing to rewrite the report:\n  - " + "\n  - ".join(problems))

    result = day4.result
    figure = risk_coverage.figure_path(d5.REPORTED, cfg, src.figures_dir)
    digests = [
        ("dataset", rel_to_repo(src.dataset), dataset_sha),
        ("gold intents", rel_to_repo(src.intents), intents_sha),
        ("ablation run (test)", result["runs"]["test"]["run"], result["runs"]["test"]["sha256"]),
        ("paired bootstrap", rel_to_repo(d4.bs.result_path(cfg, runs)),
         file_sha256(d4.bs.result_path(cfg, runs))),
        ("risk-coverage numbers", rel_to_repo(risk_coverage.result_path(d5.REPORTED, cfg, runs)),
         file_sha256(risk_coverage.result_path(d5.REPORTED, cfg, runs))),
        ("risk-coverage figure", rel_to_repo(figure), day5.rc["figure_sha256"]),
        ("conformal fit", rel_to_repo(d5.conformal.result_path(cfg, runs)),
         file_sha256(d5.conformal.result_path(cfg, runs))),
        ("conformal reading (test)", rel_to_repo(conformal_eval.result_path(cfg, runs)),
         file_sha256(conformal_eval.result_path(cfg, runs))),
        ("e2e run", rel_to_repo(run_path), run_sha),
        ("e2e labels", rel_to_repo(csv_path), file_sha256(csv_path)),
        ("e2e metrics", rel_to_repo(metrics_path), metrics_sha),
        ("error attribution", rel_to_repo(attribution_path), attribution_meta["sha256"]),
    ]
    keys = {
        cfg.hash: "config.yaml, the sections every run file is named after",
        cfg.full_hash: "config.yaml including the sections read after the runs",
        bootstrap_key(cfg): "the bootstrap's own settings, and the runs it resamples",
        d5.switched_config(cfg).hash: "config.yaml with evidence.method: conformal",
        run_meta["dense_key"]: "the Chroma collection this config builds",
        run_meta["bm25_key"]: "the BM25 index this config builds",
        result["rerank"]["rerank_key"]: "the cross-encoder rerank of hybrid's pool",
        signals_key(cfg): "the confidence signals section",
        correctness_key(cfg): "the correctness section behind the labels",
        conformal_key(cfg): "the conformal section, and Day 5's report",
    }
    return Facts(cfg, src, day4, day5, _jsonl(src.dataset), _jsonl(src.intents),
                 _jsonl(run_path), run_meta, metrics, attribution, attribution_meta,
                 digests, keys)


# --- saying it: the header, the dataset, the seeds -----------------------------------------------


def to_report(path: Path | str, src: Sources) -> str:
    """A path as the report links to it: relative to ``report/``, so a sibling is a bare name."""
    import os

    return os.path.relpath(Path(path).resolve(), src.report.parent.resolve())


def link(path: Path | str, src: Sources, text: str | None = None) -> str:
    """``[text](relative/path)``, defaulting to the repo-relative path in backticks as the text."""
    return f"[{text or f'`{rel_to_repo(path)}`'}]({to_report(path, src)})"


def rows_table(rows: Sequence[Sequence[str]]) -> str:
    """A two-column table with no header, the shape #106's header block uses."""
    return "\n".join(["| | |", "| --- | --- |", *("| " + " | ".join(r) + " |" for r in rows)])


def block_overview(f: Facts) -> str:
    """What the study is a function of: the dataset, the split, the config, the day reports."""
    cfg, src = f.cfg, f.src
    calib = sum(1 for row in f.dataset if row["split"] == "calib")
    test = sum(1 for row in f.dataset if row["split"] == "test")
    sha = {name: digest for name, _, digest in f.digests}
    runs = src.runs_dir
    return rows_table([
        ["dataset", (f"{link(src.dataset, src)}, {plural(len(f.dataset), 'row')}, "
                     f"sha256 `{sha['dataset'][:12]}`")],
        ["split", f"{calib} calib / {test} test, seed `{cfg['eval']['split_seed']}`"],
        ["config", (f"`{cfg.hash}` (dense index `{f.run_meta['dense_key']}`, "
                    f"bm25 index `{f.run_meta['bm25_key']}`)")],
        ["day 3", link(sibling(day3_report.STEM, cfg, src), src, "the eval set")
                  + " · " + link(sibling(agreement.STEM, cfg, src), src, "second annotator")],
        ["day 4", link(sibling(d4.STEM, cfg, src), src, "retrieval ablation")],
        ["day 5", link(d5.report_path(cfg, src.report_dir), src,
                       "confidence signals and conformal gating")],
        ["day 6", "this file, from "
                  + link(e2e_metrics.result_path(cfg, runs), src)
                  + " and " + link(attribute.attribution_path(cfg, runs), src)],
    ])


def block_dataset_buckets(f: Facts) -> str:
    """The four buckets, their halves, and what each one is for -- ``eval.schema.RULES``' gloss."""
    counts = Counter((row["bucket"], row["split"]) for row in f.dataset)
    rows = []
    for bucket in BUCKETS:
        calib, test = counts[bucket, "calib"], counts[bucket, "test"]
        rows.append([f"`{bucket}`", str(calib + test), str(calib), str(test),
                     RULES[bucket].gloss])
    total = ["**total**", f"**{len(f.dataset)}**",
             f"**{sum(counts[b, 'calib'] for b in BUCKETS)}**",
             f"**{sum(counts[b, 'test'] for b in BUCKETS)}**", ""]
    return md_table(["bucket", "rows", "calib", "test", "what it tests"], [*rows, total])


def block_seeds(f: Facts) -> str:
    """Three constants, each an issue number rather than a tuned value, and what each one names."""
    cfg = f.cfg
    return md_table(
        ["what", "seed", "key", "where"],
        [
            ["the frozen split", f"`{cfg['eval']['split_seed']}`", f"config `{cfg.hash}`",
             "`eval.split_seed`"],
            ["the paired bootstrap", f"`{cfg['eval']['bootstrap']['seed']}`",
             f"bootstrap key `{bootstrap_key(cfg)}`", "`eval.bootstrap`"],
            ["the second-annotator sample", f"`{cfg['seed']}`", f"config `{cfg.hash}`",
             "top-level `seed`"],
        ],
    )


# --- saying it: the diagram, drawn from the wiring -----------------------------------------------

#: Numbers spelled out, for a label that reads as a sentence rather than as a table cell.
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")

#: The query and the reply: where a question enters and where a reply leaves. Not graph nodes --
#: LangGraph's own ``__start__`` and ``__end__``, which :func:`agent.graph.node_names` leaves out.
PSEUDO = {"__start__": "query", "__end__": "reply"}

#: What each box says. The ``{}`` fields are filled from the config, so the model, the retrieval
#: mode and ``k`` in the picture move when ``config.yaml`` moves. The words around them are the
#: one hand-written thing in a generated block, and they are prose, not numbers.
LABELS: dict[str, str] = {
    "query": "a student's question",
    "route_intent": "route_intent — one of {intents} intents ({model})",
    "dispatch": "dispatch — a catalog tool, or retrieve(q, {mode}, k={k})",
    "check_evidence": "check_evidence — gate: enough here to answer from?",
    "answer": "answer — grounded draft, then a citation check",
    "confidence_gate": "confidence_gate — built but not wired (#92)",
    "out_of_scope": "out_of_scope — exit: not a catalog question",
    "refuse_handoff": "refuse_handoff — exit: hand it to an advisor",
    "reply": "one reply",
}

#: What each arrow says; an arrow with nothing to say is drawn bare.
EDGE_LABELS: dict[tuple[str, str], str] = {
    ("route_intent", "dispatch"): "any other intent",
    ("route_intent", "out_of_scope"): "intent is out_of_scope",
    ("check_evidence", "answer"): "evidence_ok",
    ("check_evidence", "refuse_handoff"): "not evidence_ok",
    ("answer", "refuse_handoff"): "cited a doc it was never given",
    ("answer", "confidence_gate"): "only under evidence.method conformal",
    ("confidence_gate", "reply"): "signal over threshold",
    ("confidence_gate", "refuse_handoff"): "signal under threshold",
}

#: Said of a node this config does not build (:data:`agent.graph.CONDITIONAL`).
DORMANT = "dormant, "


def stub_edges(cfg: Config) -> list[tuple[str, str]]:
    """``(source, target)`` for the graph this config wires, in no particular order.

    Built with stub nodes: :func:`agent.graph.build_nodes` uses an override as given rather than
    building it, so the topology comes back without a retriever, an index or an API key -- and the
    topology is what the picture is. A node's *behaviour* is not in the diagram.
    """
    names = node_names(cfg)
    stubs = {name: (lambda state: {}) for name in names}
    graph = build_graph(cfg, stubs).get_graph()
    return [(PSEUDO.get(e.source, e.source), PSEUDO.get(e.target, e.target)) for e in graph.edges]


def block_architecture(f: Facts) -> str:
    """The mermaid diagram: the nodes and edges of the wiring, dashed where a config decides them.

    Two graphs are drawn at once -- this config's, and the one that builds the dormant gate -- so
    the edges only the second has are exactly the dashed ones. Drawing only the live graph would
    leave #92's node out of the report that explains why it is off.
    """
    cfg = f.cfg
    live = stub_edges(cfg)
    order = ["query", *NODE_BUILDERS, "reply"]
    at = {name: i for i, name in enumerate(order)}
    edges = sorted(set(live) | set(stub_edges(d5.switched_config(cfg))),
                   key=lambda e: (at[e[0]], at[e[1]]))
    drawn = [name for name in order if any(name in e for e in edges)]
    fill = {
        "intents": WORDS[len(INTENTS)],
        "model": cfg["llm"]["model"],
        "mode": cfg["agent"]["retrieval_mode"],
        "k": top_k_final(cfg),
    }
    lines = ["```mermaid", "flowchart TD"]
    for name in drawn:
        label = LABELS[name].format(**fill)
        if name in CONDITIONAL and name not in node_names(cfg):
            label = label.replace("— ", f"— {DORMANT}", 1)
        shape = f'(["{label}"])' if name in PSEUDO.values() else f'["{label}"]'
        lines.append(f"    {name}{shape}")
    lines.append("")
    for src_name, dst in edges:
        arrow = "-->" if (src_name, dst) in live else "-.->"
        label = EDGE_LABELS.get((src_name, dst))
        lines.append(f"    {src_name} {arrow}{f'|{label}|' if label else ''} {dst}")
    return "\n".join([*lines, "```"])


# --- saying it: Day 4, through Day 4's own facts --------------------------------------------------

#: The two metrics, as the front page heads them.
METRIC_LABELS = {"recall_at_5": "Recall@5", "mrr_at_5": "MRR@5"}


def _verdict(gap: Mapping[str, Any]) -> str:
    """The front page's short form of :func:`eval.scripts.day4_report.verdict`."""
    return "significant" if gap["significant"] else "not significant"


def block_day4_ablation(f: Facts) -> str:
    """Three modes and the rerank row, two metrics, and the two differences that answer RQ1."""
    result, modes = f.day4.result, d4.rows_of(f.day4)
    metrics = [result["metrics"][m] for m in d4.bs.METRICS]
    rows = [[f"`{mode}`", *(d4.cell(m["test"][mode]) for m in metrics)] for mode in modes]
    for key, minuend, subtrahend in (("difference", "hybrid", "bm25"),
                                     ("rerank_difference", d4.rr.MODE, d4.rr.BASE)):
        if key not in metrics[0]:
            continue
        cells = []
        for m in metrics:
            gap = m[key]
            cells.append(f"*{d4.cell(gap, signed=True)}*, {_verdict(gap)}")
        rows.append([f"*{minuend} − {subtrahend}*", *cells])
    headers = ["mode", *(METRIC_LABELS[m] for m in d4.bs.METRICS)]
    return md_table(headers, rows)


def block_day4_buckets(f: Facts) -> str:
    """The same modes, split by bucket: point estimates, no intervals, ``n`` on the first column."""
    by_bucket = f.day4.by_bucket
    headers = ["mode"]
    for bucket in by_bucket:
        for i, metric in enumerate(d4.bs.METRICS):
            n = f" (n = {f.day4.bucket_n[bucket]})" if i == 0 else ""
            headers.append(f"{bucket} {METRIC_LABELS[metric]}{n}")
    rows = [
        [f"`{mode}`", *(num(by_bucket[b][mode][m]) for b in by_bucket for m in d4.bs.METRICS)]
        for mode in d4.rows_of(f.day4)
    ]
    return md_table(headers, rows)


# --- saying it: Day 5, through Day 5's own facts --------------------------------------------------


def block_day5_auroc(f: Facts) -> str:
    """How well each signal ranks correct answers above wrong ones, and on how many rows."""
    rc = f.day5.rc
    ci = round(rc["ci"] * 100)
    rows = []
    for name in risk_coverage.SIGNALS:
        block = rc["signals"][name]
        a = block["auroc"]
        rows.append([f"`{name}`", str(block["n"]), str(block["n_null"]),
                     f"{a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]", f"{block['aurc']:.3f}"])
    return md_table(["signal", "n", "null", f"AUROC [{ci}% CI]", "AURC"], rows)


def block_day5_conformal(f: Facts) -> str:
    """The calib-fit threshold against the fixed one, on the test half: RQ3, in three rows."""
    reading = f.day5.reading
    ci = round(reading["ci_level"] * 100)
    block = reading["signals"][reading["chosen"]]
    every = reading["answer_everything"]
    alpha = f"{reading['alpha']:.2f}"

    def error(part: Mapping[str, Any]) -> str:
        if part["realized_error"] is None:
            return "— (nothing answered)"
        lo, hi = part["ci"]
        return f"{part['realized_error']:.3f} [{lo:.3f}, {hi:.3f}]"

    def coverage(part: Mapping[str, Any]) -> str:
        return (f"{part['answered']} / {part['n_non_null']} "
                f"({100 * part['coverage']:.1f}%)")

    lo, hi = every["ci"]
    rows = [
        ["conformal (calib-fit t)", conformal_eval._t(block["conformal"]), alpha,
         error(block["conformal"]), coverage(block["conformal"])],
        [f"fixed t = {reading['fixed']}", conformal_eval._t(block["fixed"]), "— (no guarantee)",
         error(block["fixed"]), coverage(block["fixed"])],
        ["answer everything (reference)", "—", "—",
         f"{every['realized_error']:.3f} [{lo:.3f}, {hi:.3f}]",
         f"{every['answered']} / {reading['n']} (100.0%)"],
    ]
    headers = ["method", "t", "target error", f"realized error [{ci}% CI]", "coverage"]
    return md_table(headers, rows)


# --- saying it: Day 6 ----------------------------------------------------------------------------


def block_day6_exits(f: Facts) -> str:
    """Where the 63 test questions came out. The run's own sidecar counts them."""
    exits = f.run_meta["exits"]
    rows = [[f"`{name}`", str(exits[name])] for name in EXITS if name in exits]
    rows.append(["**total**", f"**{sum(exits.values())}**"])
    return md_table(["exit", "rows"], rows)


def block_day6_metrics(f: Facts) -> str:
    """Every headline rate, each with the rows behind it and an exact interval."""
    m = f.metrics["metrics"]
    caveat = f.metrics["caveats"]["ambiguous_handoff"]["correctness_if_none_counted"]
    named: list[tuple[str, Mapping[str, Any]]] = [
        ("intent accuracy", m["intent_accuracy"]),
        ("answer correctness (verdict-backed)", m["answer_correctness"]["verdict_backed"]),
        ("answer correctness (exit-only)", m["answer_correctness"]["exit_only"]),
        ("answer correctness (ambiguous hand-offs not counted)", caveat),
        ("false release (verdict-backed)", m["false_release"]["verdict_backed"]),
        ("false release (exit-only)", m["false_release"]["exit_only"]),
        ("false block", m["false_block"]),
    ]
    rows = []
    for label, block in named:
        lo, hi = block["ci"]
        rows.append([label, f"{block['k']} / {block['n']}", f"{block['rate']:.3f}",
                     f"[{lo:.3f}, {hi:.3f}]"])
    ci = round(f.metrics["ci_level"] * 100)
    return md_table(["metric", "k / n", "rate", f"{ci}% CI"], rows)


def block_day6_buckets(f: Facts) -> str:
    """Correctness per bucket, under both readings of an unread hand-off."""
    correctness = f.metrics["metrics"]["answer_correctness"]
    rows = []
    for bucket in [b for b in BUCKETS if b in f.metrics["buckets"]]:
        cells = []
        for reading in ("verdict_backed", "exit_only"):
            block = correctness[reading]["by_bucket"][bucket]
            cells.append(f"{block['k']} ({block['rate']:.3f})")
        n = correctness["verdict_backed"]["by_bucket"][bucket]["n"]
        rows.append([f"`{bucket}`", str(n), *cells])
    totals = [correctness[r]["k"] for r in ("verdict_backed", "exit_only")]
    rows.append(["**total**", f"**{f.metrics['n']}**", *(f"**{k}**" for k in totals)])
    headers = ["bucket", "n", "correct (verdict-backed)", "correct (exit-only)"]
    return md_table(headers, rows)


def block_day6_intents(f: Facts) -> str:
    """Two counts per intent: the rows whose gold label set accepts it, and the rows routed to it.

    Ordered by the number of rows the labels accept, then by :data:`agent.state.INTENTS`, so the
    intent with no test row at all (the report says which) is last rather than hidden in the middle.
    """
    accuracy = f.metrics["metrics"]["intent_accuracy"]
    gold, predicted = accuracy["rows_with_gold_label"], accuracy["predicted"]
    order = sorted(INTENTS, key=lambda name: (-gold.get(name, 0), INTENTS.index(name)))
    rows = [[f"`{name}`", str(gold.get(name, 0)), str(predicted.get(name, 0))] for name in order]
    return md_table(["intent", "rows the labels accept", "rows routed there"], rows)


def block_attribution(f: Facts) -> str:
    """One cause per failed row, in the rule order that assigned them -- not in count order."""
    bucket_of = {row["id"]: row["bucket"] for row in f.attribution}
    total = len(f.attribution)
    rows = []
    for cls in attribute.FINALS:
        ids = [row["id"] for row in f.attribution if row["final"] == cls]
        counts = Counter(bucket_of[i] for i in ids)
        where = ", ".join(f"`{b}` {counts[b]}" for b in BUCKETS if counts[b])
        rows.append([cls.replace("_", " "), str(len(ids)), f"{len(ids) / total:.2f}",
                     where or "—"])
    rows.append(["**total**", f"**{total}**", "", ""])
    return md_table(["class", "rows", "share", "where they are"], rows)


def block_false_blocks(f: Facts) -> str:
    """The answerable rows that were handed off, with the question each one asked."""
    question = {row["id"]: row["question"] for row in f.run}
    ids = f.metrics["metrics"]["false_block"]["numerator_ids"]
    return md_table(["id", "question"], [[f"`{i}`", question[i]] for i in ids])


def block_inputs(f: Facts) -> str:
    """Every file the front page is a function of, and the sha256 it was read at."""
    rows = [[link(f.src.report.parent.parent / path, f.src), f"`{digest[:12]}`"]
            for _, path, digest in f.digests]
    return md_table(["input", "sha256"], rows)


#: Every generated block, in the order the report uses them. The one place that writes between a
#: ``BEGIN gen:`` and its ``END``; a name here with no markers in the file, or markers with no
#: name here, is an error rather than a no-op.
BLOCKS: dict[str, Callable[[Facts], str]] = {
    "overview": block_overview,
    "architecture": block_architecture,
    "dataset_buckets": block_dataset_buckets,
    "seeds": block_seeds,
    "day4_ablation": block_day4_ablation,
    "day4_buckets": block_day4_buckets,
    "day5_auroc": block_day5_auroc,
    "day5_conformal": block_day5_conformal,
    "day6_exits": block_day6_exits,
    "day6_metrics": block_day6_metrics,
    "day6_buckets": block_day6_buckets,
    "day6_intents": block_day6_intents,
    "attribution": block_attribution,
    "false_blocks": block_false_blocks,
    "inputs": block_inputs,
}


# --- the blocks, in the file ---------------------------------------------------------------------


class Marked(NamedTuple):
    """One ``BEGIN gen:name`` / ``END`` pair, and the span between them."""

    name: str
    #: The first character after the BEGIN marker's newline.
    start: int
    #: The first character of the END marker's line.
    end: int


def marked(text: str) -> list[Marked]:
    """Every generated block in the file, checking the markers pair up and name a renderer."""
    out: list[Marked] = []
    open_name: str | None = None
    start = 0
    for m in MARKER_RE.finditer(text):
        name = m.group("name")
        if m.group("edge") == "BEGIN":
            if open_name is not None:
                raise ValueError(f"gen:{name} opens inside gen:{open_name}")
            if any(one.name == name for one in out):
                raise ValueError(f"gen:{name} is used twice; a block name names one table")
            open_name, start = name, m.end() + 1
        else:
            if open_name != name:
                raise ValueError(f"gen:{name} closes, but gen:{open_name} is open")
            out.append(Marked(name, start, m.start()))
            open_name = None
    if open_name is not None:
        raise ValueError(f"gen:{open_name} is never closed")
    unknown = [one.name for one in out if one.name not in BLOCKS]
    if unknown:
        raise ValueError(f"no renderer for gen:{', gen:'.join(unknown)}; "
                         f"eval/scripts/final_report.py has {', '.join(BLOCKS)}")
    absent = [name for name in BLOCKS if not any(one.name == name for one in out)]
    if absent:
        raise ValueError(f"the report has no gen:{' or gen:'.join(absent)} markers to write into")
    return out


def render_blocks(facts: Facts) -> dict[str, str]:
    """Every block, rendered. The bodies, without the markers around them."""
    return {name: render(facts).rstrip("\n") + "\n" for name, render in BLOCKS.items()}


def splice(text: str, bodies: Mapping[str, str]) -> str:
    """``text`` with every generated block replaced by its rendered body."""
    out = text
    for one in reversed(marked(text)):
        out = out[: one.start] + bodies[one.name] + out[one.end :]
    return out


# --- the prose -----------------------------------------------------------------------------------

#: A number in the prose: digits, with thousands separators and decimals, and a real or ASCII
#: minus. Anything matched here needs a ``[src]`` link or an :data:`ALLOWED` entry.
NUMBER_RE = re.compile(r"(?<![\w.-])[−-]?\d(?:[\d,]*\d)?(?:\.\d+)?")

#: A hex key or the prefix of a sha256: 8 or more hex characters with at least one digit in them,
#: which is what keeps ordinary words out (``defaced`` is hex; ``c73b197d`` is a config).
HEX_RE = re.compile(r"(?=[0-9a-f]*\d)\b[0-9a-f]{8,64}\b")

#: ``[text](target)`` or ``[text](target "title")``: masked before numbers are looked for, because
#: a file name carries the digits of whatever hash it is named after.
TARGET_RE = re.compile(r"\]\([^)]*\)")

#: `` `code` ``: a literal -- an id, a course code, a config value, a key. Masked for the same
#: reason: ``config.yaml``'s own tests hold those, and the hex ones are checked separately.
CODE_RE = re.compile(r"`[^`]*`")

#: The numbers that carry no source, with the reason each one does not need one. Meant to stay
#: short: a number that wants a new line here almost always wants a ``[src]`` link instead.
ALLOWED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"#\d+"), "an issue number"),
    (re.compile(r"\bRQ\d\b"), "a research question's label"),
    (re.compile(r"\b[Dd]ay \d\b"), "a day of the plan"),
    (re.compile(r"\b[Ss]ection \d(?:\.\d)?\b"), "a cross-reference to a section of this report"),
    (re.compile(r"\b95%"), "the interval level, `eval.bootstrap.ci`, which every run file records"),
    (re.compile(r"(?m)^\s*\d+\.\s"), "an ordered list marker"),
    (re.compile(r"\b[Rr]ule \d\b"), "a rule in the order `eval.scripts.attribute.RULES` sets"),
    (re.compile(r"\b(Recall|MRR)@\d+"), "a metric's name; its k is `fusion.top_k_final`"),
    (re.compile(r"\b[A-Za-z]{2,4} \d{3}\b"), "a course code, quoted from a question or an answer"),
    (re.compile(r"\(k \+ 1\) / \(n \+ 1\)"), "the conformal acceptance formula"),
    (re.compile(r"\b1/\d+\b"), "one row's weight in a mean, written as a fraction"),
    (re.compile(r"\b20\d\d\b"), "a year"),
)


class Problem(NamedTuple):
    """One thing ``--check`` found, and where in the file to look for it."""

    line: int
    message: str

    def __str__(self) -> str:
        """``path:line: message`` -- the form an editor can jump to."""
        return f"{rel_to_repo(REPORT)}:{self.line}: {self.message}"


def _leaves(node: Any) -> list[Any]:
    """Every scalar under ``node``, the size of every container, and every total they can be read
    as.

    A container of numbers stands for its cells *and* for their sum, because that is how the report
    reads one: ``exits`` is where the 63 rows went and also how many there were, and
    ``gate_refused_reasons`` is 15 and 5 and the 20 hand-offs they add up to. A container of ids
    stands for how many there are.
    """
    if isinstance(node, Mapping):
        children = list(node.values())
    elif isinstance(node, (list, tuple)):
        children = list(node)
    else:
        return [node]
    out: list[Any] = [len(children)]
    if children and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in children):
        out.append(sum(children))
    return [*out, *(v for child in children for v in _leaves(child))]


def _numbers(values: Iterable[Any]) -> set[float]:
    out: set[float] = set()
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out.add(float(value))
    return out


def _walk(data: Any, key: str) -> Any:
    node = data
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _section_numbers(text: str, heading: str) -> set[float] | None:
    """Every number under the Markdown heading that starts with ``heading``, or None if absent."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        level = len(line) - len(line.lstrip("#"))
        if not line.lstrip("# ").startswith(heading):
            continue
        body = []
        for later in lines[i + 1 :]:
            if later.startswith("#") and len(later) - len(later.lstrip("#")) <= level:
                break
            body.append(later)
        found = NUMBER_RE.findall("\n".join(body))
        return {float(one.replace(",", "").replace("−", "-")) for one in found}
    return None


def candidates(target: Path, title: str) -> set[float] | None:
    """Every number the whole ``[src]`` title explains: the union over its comma-separated paths.

    ``None`` if any one of them does not resolve -- a title that names a key path the file does not
    have is exactly the mistake this check exists to catch, and the other paths do not excuse it.
    """
    found: set[float] = set()
    for key in (part.strip() for part in title.split(",")):
        one = _candidates(target, key)
        if one is None:
            return None
        found |= one
    return found


def _candidates(target: Path, key: str) -> set[float] | None:
    """Every number a ``[src]`` link to ``target`` with title ``key`` is allowed to explain.

    A JSON key path stands for the node at it -- a scalar, or every scalar under a container, plus
    the size of each container, so ``caveats.ambiguous_handoff.ids`` explains "nine rows". A JSONL
    column stands for its values and their counts, so ``evidence_source`` explains "32 of 37 are
    tool rows" without a second file. A Markdown heading stands for the numbers printed under it,
    which is how a number quoted from the Day 4 or Day 5 report is held to that report.
    """
    if target.suffix == ".json":
        node = _walk(json.loads(target.read_text(encoding="utf-8")), key)
        return None if node is None else _numbers(_leaves(node))
    if target.suffix == ".jsonl":
        rows = _jsonl(target)
        if not rows or key not in rows[0]:
            return None
        column = [row.get(key) for row in rows]
        counts = Counter(v for v in column if isinstance(v, (str, int, float, bool)))
        return _numbers([
            len(rows),
            *counts.values(),
            sum(1 for v in column if v not in (None, "", [], {})),
            *column,
        ])
    if target.suffix == ".md":
        return _section_numbers(target.read_text(encoding="utf-8"), key)
    return None


def printed(number: str) -> tuple[float, int]:
    """A printed number as a value and the number of decimals it was printed to."""
    plain = number.replace(",", "").replace("−", "-")
    return float(plain), len(plain.partition(".")[2])


def explains(number: str, values: set[float]) -> bool:
    """Is ``number`` one of ``values``, compared at the precision it is printed to?"""
    value, places = printed(number)
    return any(f"{one:.{places}f}" == f"{value:.{places}f}" for one in values)


def mask(text: str) -> str:
    """``text`` with everything that is not a prose number blanked out, length for length."""
    out = text
    for pattern in (TARGET_RE, CODE_RE, *(p for p, _ in ALLOWED)):
        out = pattern.sub(lambda m: " " * len(m.group(0)), out)
    return out


def paragraphs(text: str) -> list[tuple[int, str]]:
    """``(offset, paragraph)`` for the prose: no generated block, no fenced code, no heading."""
    spans = [(one.start, one.end) for one in marked(text)]
    in_fence = False
    out: list[tuple[int, str]] = []
    offset = 0
    current: list[str] = []
    start = 0
    for line in text.splitlines(keepends=True):
        here, offset = offset, offset + len(line)
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        inside = in_fence or any(a <= here < b for a, b in spans)
        if inside or not line.strip() or line.startswith("#"):
            if current:
                out.append((start, "".join(current)))
                current = []
            continue
        if not current:
            start = here
        current.append(line)
    if current:
        out.append((start, "".join(current)))
    return out


# --- the check ------------------------------------------------------------------------------------


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_blocks(text: str, bodies: Mapping[str, str]) -> list[Problem]:
    """Every generated block is what the generator writes today."""
    return [
        Problem(_line_of(text, one.start),
                f"gen:{one.name} is out of date; run `python -m eval.scripts.final_report`")
        for one in marked(text)
        if text[one.start : one.end] != bodies[one.name]
    ]


def check_tables(text: str) -> list[Problem]:
    """A table in the prose is a number nothing regenerates, so it is an error."""
    spans = [(one.start, one.end) for one in marked(text)]
    problems = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        here, offset = offset, offset + len(line)
        if line.startswith("```"):
            in_fence = not in_fence
        if in_fence or any(a <= here < b for a, b in spans):
            continue
        if line.startswith("|"):
            problems.append(Problem(_line_of(text, here),
                                    "a table outside a gen: block; give it a generator"))
    return problems


def check_hex(text: str, facts: Facts) -> list[Problem]:
    """Every hex key the report names is one this config produces, or an input's sha256."""
    digests = [digest for _, _, digest in facts.digests]
    problems = []
    for m in HEX_RE.finditer(text):
        token = m.group(0)
        if token in facts.keys or any(digest.startswith(token) for digest in digests):
            continue
        problems.append(Problem(
            _line_of(text, m.start()),
            f"`{token}` is not a key this config produces or a prefix of an input's sha256; "
            f"this config's are {', '.join(sorted(facts.keys))}",
        ))
    return problems


def check_prose(text: str, facts: Facts) -> list[Problem]:
    """Every number in the prose is explained by a ``[src]`` link in its own paragraph."""
    problems: list[Problem] = []
    cache: dict[tuple[str, str], set[float] | None] = {}
    for start, paragraph in paragraphs(text):
        links = [(m.group("path"), m.group("key")) for m in SRC_RE.finditer(paragraph)]
        values: set[float] = set()
        for path, key in links:
            target = (facts.src.report.parent / path).resolve()
            line = _line_of(text, start + paragraph.find(path))
            if not target.exists():
                problems.append(Problem(line, f"[src] link to a missing file: {path}"))
                continue
            if (path, key) not in cache:
                cache[path, key] = candidates(target, key)
            found = cache[path, key]
            if found is None:
                problems.append(Problem(
                    line, f"{path} has no key path {key!r} to read a value from"))
                continue
            values |= found
        for m in NUMBER_RE.finditer(mask(paragraph)):
            number = m.group(0)
            line = _line_of(text, start + m.start())
            if not links:
                problems.append(Problem(line, f"{number} has no [src] link in its paragraph"))
            elif not explains(number, values):
                named = ", ".join(sorted({path for path, _ in links}))
                problems.append(Problem(
                    line, f"{number} is not a value in {named} at the precision it is printed to"))
    return problems


def check(text: str, facts: Facts, bodies: Mapping[str, str]) -> list[Problem]:
    """Everything ``--check`` checks, in the order a reader would notice it."""
    return sorted(
        [*check_blocks(text, bodies), *check_tables(text), *check_prose(text, facts),
         *check_hex(text, facts)],
        key=lambda one: one.line,
    )


def render_check(text: str, facts: Facts, bodies: Mapping[str, str], out: TextIO) -> int:
    """``--check``: what does not trace, one per line, or a count of what does."""
    problems = check(text, facts, bodies)
    out.writelines(f"{problem}\n" for problem in problems)
    numbers = sum(len(NUMBER_RE.findall(mask(p))) for _, p in paragraphs(text))
    links = len(SRC_RE.findall(text))
    if problems:
        out.write(f"\n{plural(len(problems), 'problem')} in {rel_to_repo(facts.src.report)}.\n")
        return 1
    out.write(f"{rel_to_repo(facts.src.report)}: {len(bodies)} generated blocks are current, "
              f"{numbers} prose numbers carry {links} source links, and every input matches the "
              "sha256 its sidecar records.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the report's generated blocks, or with ``--check`` verify every number traces.

    ``--check`` writes nothing and exits 1 on a failure, which is what CI runs; ``--inputs``
    lists the artifacts behind the report and ``--dry-run`` prints it instead of writing it.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.final_report",
        description="Rewrite report/README.md's generated blocks, or check that every number in "
                    "it traces to a committed file.",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the committed report and write nothing; exit 1 if it fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rewritten report instead of writing it")
    parser.add_argument("--inputs", action="store_true",
                        help="list the artifacts the report is built from; write nothing")
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    src = Sources()
    if args.inputs:
        return render_inputs(cfg, src, sys.stdout)
    if missing(cfg, src):
        render_inputs(cfg, src, sys.stderr)
        return 1
    try:
        facts = read_facts(cfg, src)
        bodies = render_blocks(facts)
        text = src.report.read_text(encoding="utf-8")
        if args.check:
            return render_check(text, facts, bodies, sys.stdout)
        body = splice(text, bodies)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.write(body)
        return 0
    if body == text:
        print(f"{rel_to_repo(src.report)} is already current")
        return 0
    src.report.write_text(body, encoding="utf-8")
    print(f"rewrote {plural(len(bodies), 'block')} in {rel_to_repo(src.report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
