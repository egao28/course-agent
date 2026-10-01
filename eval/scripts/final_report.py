"""The two front pages' tables, and the check that every number in them traces to a file.

    uv run python -m eval.scripts.final_report            # rewrite the generated blocks
    uv run python -m eval.scripts.final_report --check     # verify them; exit 1 if it does not
    uv run python -m eval.scripts.final_report --dry-run   # print the rewritten pages, write none
    uv run python -m eval.scripts.final_report --inputs    # list the artifacts it reads

Three files are held to the rules, and :data:`Page` is the difference between them.
``report/README.md`` (#106) is the study's report, and all of it is checked. The README (#117) is
checked down to its ``## Reference`` heading and no further: above it sits a front page that states
the three RQ conclusions with their intervals, below it the 2,600-line component reference that
predates the rule and that nothing here generates. ``report/day7_claims.md`` (#120) is the claims
page: one row per claim this study can support, the file and key path it comes from, the rows it
rests on, and what it does not say.

These are pages whose prose is *written*, not formatted -- that is what makes #106 half a day's
work rather than a generator. So the Day 3-5 rule ("every number is formatted from the data") is
replaced here by a lighter one that a machine can still hold:

* **A table or a figure is generated.** Each sits between ``<!-- BEGIN gen:name -->`` and
  ``<!-- END gen:name -->``, and :data:`BLOCKS` is the only thing that writes between them. A
  table outside a marker pair is an error, not an exemption: it is a number nothing regenerates.
* **A number in the prose carries its source.** ``0.405 [src](../eval/runs/x.json "a.key.path")``
  -- the link names the file, its title names the path inside it, and ``--check`` reads the value
  there and compares it against the digits, at the precision they are printed to.
* **Everything else is an allowlist**, :data:`ALLOWED`, and it is meant to stay short: issue
  numbers, section numbers, RQ and day labels, interval levels, and the two formulas. A number
  that wants a new entry almost always wants a link instead.
* **A claim is read back through its own row.** The claims page *is* a table, so its numbers are
  not prose and the rule above would not see them. :func:`check_claims` applies the same rule to
  a row: the ``[src]`` links in the row name the files, and every number in the row -- in the
  claim, in its interval, in its caveat -- is read back out of them. Nothing there is typed in
  either: :data:`CLAIMS` is hand-written wording with the numbers interpolated by
  :func:`claim_values`, and a test refuses a digit in the wording.

``--check`` exits non-zero when a block is stale, a prose number has no link, a linked value is
not in the linked file at the printed precision, a claim row quotes a number its own sources do
not have, a hex key is not one this config produces, a table sits outside a block, or any input
fails the sha256 chain its sidecars record. That last one is not this script's own work:
:func:`eval.scripts.day4_report.read_facts` and
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
retrieval mode and ``k`` in the picture move when ``config.yaml`` does. Both pages carry that one
block, rendered once and spliced into each: #117 asked for one diagram with one owner, and the
owner is this script rather than either file, so the two copies cannot drift.

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
    REPO_ROOT,
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
from ingest.export_almabot import COURSES_PATH, PREREQS_PATH
from retrieval.fusion import top_k_final

#: The two front pages this script owns. Neither is named after a hash: they are front pages, and
#: a front page that changes its name when a threshold moves is one nobody can link to.
REPORT = REPORT_DIR / "README.md"
FRONT = REPO_ROOT / "README.md"

#: The claims page (#120): one row per claim, and the file and key path behind each number in it.
#: Not named after a hash either -- it is the page a claim is answered from, and a page that
#: renames itself when a threshold moves is one nobody can be pointed at twice.
CLAIMS_PAGE = REPORT_DIR / "day7_claims.md"

#: Where the repo's README stops being a front page. Everything below is the component reference.
REFERENCE = "## Reference"

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


class Page(NamedTuple):
    """One file this script writes into, and the part of it the rules apply to."""

    path: Path
    #: The ``gen:`` blocks this page carries. A name here with no markers in the region, or a
    #: marker naming a block the page does not carry, is an error rather than a no-op.
    blocks: tuple[str, ...]
    #: The heading the checked region ends at, or ``None`` for the whole file.
    upto: str | None = None

    def region(self, text: str) -> str:
        """The part of ``text`` held to the rules.

        Always a prefix, so an offset into it is an offset into the file and a reported line
        number is the one an editor jumps to. Give it the whole file: it is not idempotent, and
        it refuses a text with no divider in it rather than guessing that one was already cut.
        """
        if self.upto is None:
            return text
        cut = text.find(f"\n{self.upto}\n")
        if cut < 0:
            raise ValueError(
                f"{rel_to_repo(self.path)} has no `{self.upto}` heading: that heading is where "
                "the front page ends and the reference begins, and without it there is no front "
                "page to check")
        return text[: cut + 1]


class Sources(NamedTuple):
    """Where the pages read from, rooted so a test can point them at a copy."""

    report: Path = REPORT
    front: Path = FRONT
    claims: Path = CLAIMS_PAGE
    runs_dir: Path = RUNS_DIR
    dataset: Path = DATASET_PATH
    intents: Path = INTENTS_PATH
    figures_dir: Path = risk_coverage.FIGURES_DIR
    report_dir: Path = REPORT_DIR
    courses: Path = COURSES_PATH
    prereqs: Path = PREREQS_PATH


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
        Input("corpus: courses", src.courses, "committed; re-exported by "
              "`uv run python -m ingest.export_almabot` from the private CSV"),
        Input("corpus: prerequisites", src.prereqs, "committed; re-exported by "
              "`uv run python -m ingest.export_almabot` from the private CSV"),
        Input("the report itself", src.report, "this script, with no flags"),
        Input("the front page", src.front, "this script, with no flags"),
        Input("the claims page", src.claims, "this script, with no flags"),
    ]


def missing(cfg: Config, src: Sources) -> list[Input]:
    """The inputs that are not there. Empty is the condition for rewriting anything."""
    return [one for one in inputs(cfg, src) if not one.path.exists()]


def render_inputs(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--inputs``: every artifact, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    named = ", ".join(rel_to_repo(page.path) for page in pages(src))
    out.write(f"pages {named}  config {cfg.hash}  full {cfg.full_hash}\n\n")
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
    #: What the corpus claim counts: courses exported, and prerequisite edges among them.
    corpus: dict[str, int]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus(src: Sources) -> dict[str, int]:
    """The two counts the corpus claim prints: courses exported, and prerequisite edges in them.

    Counted from the committed export rather than quoted from a README, while the claim's own
    ``[src]`` link points at the README section that documents them -- so a corpus that moves
    without its documentation moving fails ``--check`` instead of passing quietly.
    """
    courses = json.loads(src.courses.read_text(encoding="utf-8"))
    prereqs = json.loads(src.prereqs.read_text(encoding="utf-8"))
    return {"courses": len(courses),
            "prereq_edges": sum(len(row["prereqs"]) for row in prereqs)}


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
                 digests, keys, _corpus(src))


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


# --- saying it: the claims, and what each one does not say ---------------------------------------

#: Where a claim may read from: the name a row uses, and the file it is under these
#: :class:`Sources`. A row names a file from this table rather than a path of its own, so a run
#: file that moves moves in one place, and so a claim cannot cite a file nothing above checked.
CLAIM_FILES: dict[str, Callable[[Facts], Path]] = {
    "dataset": lambda f: f.src.dataset,
    "bootstrap": lambda f: d4.bs.result_path(f.cfg, f.src.runs_dir),
    "risk_coverage": lambda f: risk_coverage.result_path(d5.REPORTED, f.cfg, f.src.runs_dir),
    "conformal": lambda f: d5.conformal.result_path(f.cfg, f.src.runs_dir),
    "conformal_eval": lambda f: conformal_eval.result_path(f.cfg, f.src.runs_dir),
    "metrics": lambda f: e2e_metrics.result_path(f.cfg, f.src.runs_dir),
    "attribution": lambda f: attribute.attribution_path(f.cfg, f.src.runs_dir).with_suffix(
        META_SUFFIX),
    "front": lambda f: f.src.front,
}


class Src(NamedTuple):
    """One file a claim's numbers come from, and the key paths inside it they come from."""

    #: A key of :data:`CLAIM_FILES`.
    file: str
    #: A JSON key path, a JSONL column, or a Markdown heading -- whatever :func:`candidates`
    #: reads out of that kind of file.
    keys: tuple[str, ...]


class Claim(NamedTuple):
    """One claim, where its numbers come from, what it rests on, and what it does not say.

    ``line``, ``rests_on`` and ``unsaid`` are format strings over :func:`claim_values`: the
    wording is hand-written and every number in it is interpolated from the file the row names.
    ``tests/test_day7_claims.py`` refuses a digit in any of the three, which is what makes the
    first column a quotation of the data rather than a sentence about it.
    """

    #: ``C1``, so a bullet above the table can name the row it quotes.
    id: str
    #: The claim, in the register of the measurement rather than of a resume.
    line: str
    sources: tuple[Src, ...]
    #: The rows behind the number, and how the interval was made.
    rests_on: str
    #: The caveat that travels with the claim. #120's fourth column, and the reason this file is
    #: not simply a list of numbers.
    unsaid: str


class Refused(NamedTuple):
    """A line this work may not be described with, and what the committed numbers say instead."""

    line: str
    instead: str
    sources: tuple[Src, ...] = ()


#: One row per claim this study can support. The order is the order the study was done in: the
#: eval set, then RQ1, RQ2 and RQ3, then the end-to-end run, then the corpus under all of it.
CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C1",
        "A hand-verified evaluation set, {rows} questions in four buckets, split {calib} "
        "calibration / {test} test and frozen before anything was measured, with a second "
        "annotator on a sample of it.",
        (Src("dataset", ("split",)),),
        "{rows} rows; counts rather than estimates, so no interval. Digest `{dataset_sha}`.",
        "Not a double-annotated set. The second annotator read a fifteen-row sample, reported it "
        "verbally for transcription rather than filling the sheet in, and κ on `answer_correct` "
        "is zero there by construction — the percent agreement is the number to read "
        "({day3_agreement}). The questions were drafted from the catalog by the people who built "
        "the system.",
    ),
    Claim(
        "C2",
        "A tuned hybrid retriever did not beat `bm25` on a code-heavy catalog: Recall@5 "
        "hybrid − bm25 = {recall_gap}, MRR@5 = {mrr_gap} — the first interval containing zero, "
        "the second excluding it *against* hybrid.",
        (Src("bootstrap", ("metrics.recall_at_5.difference", "metrics.mrr_at_5.difference",
                           "n.test", "n_resamples")),),
        "n = {n_ablation} test rows, paired bootstrap over {resamples} resamples, 95% "
        "percentile intervals.",
        "Not an improvement in either direction that a reader should hear as one. This is why "
        "the agent ships `bm25`, and a cross-encoder rerank of hybrid's pool separates from "
        "neither metric ({day4_report}). One corpus, where course codes make a lexical index "
        "hard to beat.",
    ),
    Claim(
        "C3",
        "The model's verbalized self-confidence ranks correct answers above wrong ones: test "
        "AUROC {auroc_test}, for the signal chosen on the calibration half at {auroc_calib}.",
        (Src("risk_coverage", ("signals.verbalized.auroc", "n", "n_resamples")),
         Src("conformal", ("signals.verbalized.auroc",))),
        "n = {n_signals} test rows, {rc_resamples} bootstrap resamples, 95% percentile "
        "interval; the test half was read once.",
        "Ranking, not calibration: AUROC says nothing about whether a stated confidence matches "
        "an observed rate, and the gate fitted on this signal accepts no threshold at all (C4). "
        "These rows were collected with the evidence gate bypassed, so they describe answers the "
        "deployed agent might never have released.",
    ),
    Claim(
        "C4",
        "Conformal calibration at α = {alpha} accepts no threshold on a calibration half of "
        "{n_calib} rows, so the fitted gate answers nothing; the fixed {fixed_t} threshold it "
        "was built to replace answers {gate_answered} of {n_signals} at a realized error of "
        "{gate_error}.",
        (Src("conformal", ("alpha", "n", "min_answered_to_accept")),
         Src("conformal_eval", ("fixed", "n", "signals.verbalized.fixed"))),
        "Calibration n = {n_calib} for the fit, by a fixed-sequence scan; n = {n_signals} test "
        "rows for the reading, Clopper-Pearson 95%.",
        "Not a reduction in anything. The conformal gate is built and dormant (#92) and the live "
        "config still runs the Day 2 `top_score` rule. The guarantee is marginal over "
        "calibration draws rather than conditional on this half, and at this size the scan "
        "cannot accept a threshold that answers fewer than {min_answered} rows — so the finding "
        "is \"not measurable here\", not \"conformal does not work\".",
    ),
    Claim(
        "C5",
        "The intent router is right on {intent_k} of {intent_n} test rows, {intent}.",
        (Src("metrics", ("metrics.intent_accuracy",)),),
        "{intent_k} / {intent_n}, Clopper-Pearson 95%, on one end-to-end run.",
        "The most forgiving number in the study: the gold label is a set per row, and the labels "
        "were written after the run existed, so nothing in the process proves the labeller had "
        "not seen the router's output. One of the five intents, `degree_requirement`, has no "
        "test row at all.",
    ),
    Claim(
        "C6",
        "{correct_k} of {correct_n} test rows end in the outcome their bucket calls correct, "
        "{correct} — or {low_k} of {correct_n}, {low}, if the {unread} ambiguous hand-offs "
        "nobody read do not count.",
        (Src("metrics", ("metrics.answer_correctness", "caveats.ambiguous_handoff")),),
        "{correct_k} / {correct_n} and {low_k} / {correct_n}, Clopper-Pearson 95%, on one run "
        "read once. A correct outcome is an answer a person judged right, a decline on an "
        "unanswerable row, or a clarifying question on an ambiguous one.",
        "Three readings, not one: exit-only is {exit_k} / {correct_n} ({exit_only}), and "
        "{correct_k} / {correct_n} is the upper bound of the range rather than the number — the "
        "report asks for it never to be quoted alone. Correctness is one person's verdict and "
        "the judge's call was displayed before they made it, so the agreement between them is "
        "not independent confirmation.",
    ),
    Claim(
        "C7",
        "False release {release_k} of {release_n}, {release}; false block {block_k} of "
        "{block_n}, {block}.",
        (Src("metrics", ("metrics.false_release", "metrics.false_block")),),
        "{release_k} / {release_n} rows that should not have been answered and {block_k} / "
        "{block_n} that should have been, Clopper-Pearson 95%.",
        "Read two ways, because the graph has no clarify exit: a reply that declines without "
        "asking the question counts as a release on the exit-only reading, which is "
        "{release_exit_k} / {release_n} ({release_exit}). The false-block rate is what the gate "
        "costs — {block_k} answerable questions were handed to an advisor instead.",
    ),
    Claim(
        "C8",
        "Every one of the {failures} failures carries a cause, assigned by a fixed rule order "
        "and confirmed by a reviewer: {retrieval} retrieval, {misread} evidence misread, "
        "{overconfident} overconfident, {intent_misjudged} intent, {other} other.",
        (Src("attribution", ("by_class", "actions", "n_rows", "n_failures")),),
        "{failures} failure rows out of {rows_e2e}; a classification, so no interval. All "
        "{confirmed} confirmed, none overridden.",
        "Agreement with a rule, not a second opinion: the reviewer saw the rule's proposal on "
        "every row. The order charges each row to a single cause, so a row with two causes shows "
        "only the first, and `retrieval` leading is partly a property of that order.",
    ),
    Claim(
        "C9",
        "One university catalog, one term: {courses} courses and {edges} prerequisite edges, "
        "re-exported from the catalog CSV and pinned by digest.",
        (Src("front", ("Data",)),),
        "Counts over the committed export rather than estimates; the input CSV is pinned by "
        "`data.source_sha256` and checked on every run.",
        "One institution and one term (`2026-sp`), so \"the catalog cannot answer this\" is a "
        "property of this export rather than of the world. The edges are a reading of catalog "
        "prose, with the known failures written down ({data_readme}). Nothing was deployed, and "
        "no question here came from a real student.",
    ),
)

#: The lines this work may not be described with. #120 names them, and each one is refused by a
#: number rather than by taste — which is why they are a generated table and not a warning.
REFUSED: tuple[Refused, ...] = (
    Refused(
        "Improved retrieval by X%",
        "RQ1 found no gap on Recall@5 ({recall_gap}) and a gap *against* hybrid on MRR@5 "
        "({mrr_gap}). The finding is that a tuned hybrid did not beat `bm25` on this corpus, "
        "which is the stronger line because it is the true one.",
        (Src("bootstrap", ("metrics.recall_at_5.difference", "metrics.mrr_at_5.difference")),),
    ),
    Refused(
        "Deployed / in production / served N users / p99 latency",
        "Nothing was deployed. No user has asked this agent anything; there is no load, no cost "
        "per session and no log of what anyone asks. The only timing in the repo is a "
        "cross-encoder's median per question on one laptop cpu ({day4_report}), measured once, "
        "and a clean machine measured it differently (#118).",
    ),
    Refused(
        "Calibrated confidence gating reduced hallucination",
        "The calibrated gate answers nothing at α = {alpha} on {n_calib} calibration rows, and "
        "the live config still runs `top_score`. What the calibration produced is a measurement "
        "of the calibration half, not a shipped improvement.",
        (Src("conformal", ("alpha", "n")),),
    ),
    Refused(
        "A rate without its n, or a point estimate with the interval dropped",
        "Every rate above is printed as k / n with a 95% interval. At these sample sizes the "
        "interval *is* the finding: several of the comparisons have intervals wide enough to "
        "contain the opposite conclusion, and dropping one turns a measurement into an "
        "assertion.",
    ),
    Refused(
        "{correct_k} of {correct_n} correct, with the lower reading left out",
        "Both readings or neither: {low_k} of {correct_n} ({low}) is the same run with the "
        "{unread} unread ambiguous hand-offs counted wrong. Which one is right is a judgement "
        "about what a hand-off that declines without asking is worth, and that judgement belongs "
        "beside the number rather than behind it.",
        (Src("metrics", ("metrics.answer_correctness", "caveats.ambiguous_handoff")),),
    ),
)


def claim_values(f: Facts) -> dict[str, str]:
    """Every number and link the claims print, formatted once, out of the facts.

    One namespace for every row, so two claims that name the same number print the same digits
    and the place a number is formatted is the place to look when it moves. The rates are
    formatted as :func:`block_day6_metrics` formats them and the differences through Day 4's own
    :func:`eval.scripts.day4_report.cell`, so a claim and the table it is a quotation of cannot
    print one value two ways.
    """
    cfg, src = f.cfg, f.src
    m, caveat = f.metrics["metrics"], f.metrics["caveats"]["ambiguous_handoff"]
    gaps = f.day4.result["metrics"]
    rc, fit, reading = f.day5.rc, f.day5.fit, f.day5.reading
    chosen = fit["chosen"]["signal"]
    auroc = rc["signals"][chosen]["auroc"]
    gate = reading["signals"][reading["chosen"]]["fixed"]
    by_class, actions = f.attribution_meta["by_class"], f.attribution_meta["actions"]
    sha = {name: digest for name, _, digest in f.digests}
    split = Counter(row["split"] for row in f.dataset)

    def rate(block: Mapping[str, Any]) -> str:
        """``0.683 [0.553, 0.794]`` -- a rate and its interval, as the report's tables print it."""
        lo, hi = block["ci"]
        return f"{block['rate']:.3f} [{lo:.3f}, {hi:.3f}]"

    return {
        "day3_agreement": link(sibling(agreement.STEM, cfg, src), src, "Day 3's agreement report"),
        "day4_report": link(sibling(d4.STEM, cfg, src), src, "Day 4's report"),
        "data_readme": link(src.courses.parent / "README.md", src, "`data/README.md`"),
        "rows": str(len(f.dataset)),
        "calib": str(split["calib"]),
        "test": str(split["test"]),
        "dataset_sha": sha["dataset"][:12],
        "recall_gap": d4.cell(gaps["recall_at_5"]["difference"], signed=True),
        "mrr_gap": d4.cell(gaps["mrr_at_5"]["difference"], signed=True),
        "n_ablation": str(f.day4.result["n"]["test"]),
        "resamples": f"{f.day4.result['n_resamples']:,}",
        "auroc_test": f"{auroc['point']:.3f} [{auroc['lo']:.3f}, {auroc['hi']:.3f}]",
        "auroc_calib": f"{fit['signals'][chosen]['auroc']:.3f}",
        "n_signals": str(rc["n"]),
        "rc_resamples": f"{rc['n_resamples']:,}",
        "alpha": f"{reading['alpha']:.2f}",
        "n_calib": str(fit["n"]),
        "min_answered": str(fit["min_answered_to_accept"]),
        "fixed_t": str(reading["fixed"]),
        "gate_answered": str(gate["answered"]),
        "gate_error": rate({"rate": gate["realized_error"], "ci": gate["ci"]}),
        "intent": rate(m["intent_accuracy"]),
        "intent_k": str(m["intent_accuracy"]["k"]),
        "intent_n": str(m["intent_accuracy"]["n"]),
        "correct": rate(m["answer_correctness"]["verdict_backed"]),
        "correct_k": str(m["answer_correctness"]["verdict_backed"]["k"]),
        "correct_n": str(m["answer_correctness"]["verdict_backed"]["n"]),
        "exit_only": rate(m["answer_correctness"]["exit_only"]),
        "exit_k": str(m["answer_correctness"]["exit_only"]["k"]),
        "low": rate(caveat["correctness_if_none_counted"]),
        "low_k": str(caveat["correctness_if_none_counted"]["k"]),
        "unread": str(caveat["n"]),
        "release": rate(m["false_release"]["verdict_backed"]),
        "release_k": str(m["false_release"]["verdict_backed"]["k"]),
        "release_n": str(m["false_release"]["verdict_backed"]["n"]),
        "release_exit": rate(m["false_release"]["exit_only"]),
        "release_exit_k": str(m["false_release"]["exit_only"]["k"]),
        "block": rate(m["false_block"]),
        "block_k": str(m["false_block"]["k"]),
        "block_n": str(m["false_block"]["n"]),
        "failures": str(f.attribution_meta["n_failures"]),
        "rows_e2e": str(f.attribution_meta["n_rows"]),
        "confirmed": str(actions["confirmed"]),
        "retrieval": str(by_class["retrieval_failure"]),
        "misread": str(by_class["evidence_misread"]),
        "overconfident": str(by_class["overconfident"]),
        "intent_misjudged": str(by_class["intent_misjudged"]),
        "other": str(by_class["other"]),
        "courses": f"{f.corpus['courses']:,}",
        "edges": f"{f.corpus['prereq_edges']:,}",
    }


def claim_source(src: Src, f: Facts) -> str:
    """One source cell: the file, the key paths in it, and the ``[src]`` link ``--check`` reads.

    The path is printed as well as linked because #120 asks for the file and the key path to be
    *in* the row -- a reader answering "where is that from?" should not have to hover over a link
    to find out.
    """
    path = CLAIM_FILES[src.file](f)
    keys = ", ".join(src.keys)
    return (f"`{rel_to_repo(path)}` → {', '.join(f'`{key}`' for key in src.keys)} "
            f'[src]({to_report(path, f.src)} "{keys}")')


def block_claims(f: Facts) -> str:
    """#120's four columns, one row per claim, every number formatted from the file it names."""
    values = claim_values(f)
    rows = [[f"**{claim.id}** — {claim.line.format(**values)}",
             " · ".join(claim_source(one, f) for one in claim.sources),
             claim.rests_on.format(**values),
             claim.unsaid.format(**values)]
            for claim in CLAIMS]
    return md_table(["the claim", "where every number in it comes from", "what it rests on",
                     "what it does not say"], rows)


def block_claims_refused(f: Facts) -> str:
    """The lines the numbers refuse, each with what they say instead, and where that is."""
    values = claim_values(f)
    rows = [[f"*{one.line.format(**values)}*", one.instead.format(**values),
             " · ".join(claim_source(src, f) for src in one.sources) or "—"]
            for one in REFUSED]
    return md_table(["the line", "what the committed numbers say instead", "where that is"], rows)


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
    "claims": block_claims,
    "claims_refused": block_claims_refused,
}

#: The blocks the repo's README carries. One name, and it is also the report's: the diagram is
#: rendered once and spliced into both, so the two copies are the same bytes by construction.
FRONT_BLOCKS: tuple[str, ...] = ("architecture",)

#: The claims page's blocks (#120). A row of either one carries its own sources, which is what
#: :func:`check_claims` reads them back through.
CLAIMS_BLOCKS: tuple[str, ...] = ("claims", "claims_refused")

#: Everything else, which is the report's.
REPORT_BLOCKS: tuple[str, ...] = tuple(name for name in BLOCKS if name not in CLAIMS_BLOCKS)


def pages(src: Sources) -> list[Page]:
    """The files this script writes into, in the order a reader meets them."""
    return [Page(src.front, FRONT_BLOCKS, REFERENCE), Page(src.report, REPORT_BLOCKS),
            Page(src.claims, CLAIMS_BLOCKS)]


# --- the blocks, in the file ---------------------------------------------------------------------


class Marked(NamedTuple):
    """One ``BEGIN gen:name`` / ``END`` pair, and the span between them."""

    name: str
    #: The first character after the BEGIN marker's newline.
    start: int
    #: The first character of the END marker's line.
    end: int


def marked(text: str, names: Sequence[str]) -> list[Marked]:
    """Every generated block in ``text``, checking the markers pair up and name a renderer.

    ``names`` is the page's own blocks. A page carries some of :data:`BLOCKS`, not all of them,
    so a marker for a block this page does not carry is as much an error as one for nothing.
    """
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
    unknown = [one.name for one in out if one.name not in names]
    if unknown:
        raise ValueError(f"no renderer for gen:{', gen:'.join(unknown)} on this page; "
                         f"it carries {', '.join(names)}")
    absent = [name for name in names if not any(one.name == name for one in out)]
    if absent:
        raise ValueError(f"the page has no gen:{' or gen:'.join(absent)} markers to write into")
    return out


def blocks_in(page: Page, text: str) -> list[Marked]:
    """Every generated block on ``page``: its own names, inside its own region."""
    return marked(page.region(text), page.blocks)


def render_blocks(facts: Facts) -> dict[str, str]:
    """Every block, rendered. The bodies, without the markers around them."""
    return {name: render(facts).rstrip("\n") + "\n" for name, render in BLOCKS.items()}


def splice(page: Page, text: str, bodies: Mapping[str, str]) -> str:
    """``text`` with every one of ``page``'s generated blocks replaced by its rendered body.

    The spans come from the page's region, which is a prefix, so they index ``text`` unchanged
    and everything below the region is copied through untouched.
    """
    out = text
    for one in reversed(blocks_in(page, text)):
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
    """One thing ``--check`` found, and where to look for it: three pages, so the file is part of
    the answer."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        """``path:line: message`` -- the form an editor can jump to."""
        return f"{rel_to_repo(self.path)}:{self.line}: {self.message}"


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


def paragraphs(page: Page, text: str) -> list[tuple[int, str]]:
    """``(offset, paragraph)`` for the page's prose: no generated block, no fence, no heading."""
    spans = [(one.start, one.end) for one in blocks_in(page, text)]
    text = page.region(text)
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


def check_blocks(page: Page, text: str, bodies: Mapping[str, str]) -> list[Problem]:
    """Every generated block is what the generator writes today."""
    return [
        Problem(page.path, _line_of(text, one.start),
                f"gen:{one.name} is out of date; run `python -m eval.scripts.final_report`")
        for one in blocks_in(page, text)
        if text[one.start : one.end] != bodies[one.name]
    ]


def check_tables(page: Page, text: str) -> list[Problem]:
    """A table in the prose is a number nothing regenerates, so it is an error."""
    spans = [(one.start, one.end) for one in blocks_in(page, text)]
    text = page.region(text)
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
            problems.append(Problem(page.path, _line_of(text, here),
                                    "a table outside a gen: block; give it a generator"))
    return problems


def check_hex(page: Page, text: str, facts: Facts) -> list[Problem]:
    """Every hex key the page names is one this config produces, or an input's sha256."""
    digests = [digest for _, _, digest in facts.digests]
    problems = []
    for m in HEX_RE.finditer(page.region(text)):
        token = m.group(0)
        if token in facts.keys or any(digest.startswith(token) for digest in digests):
            continue
        problems.append(Problem(
            page.path, _line_of(text, m.start()),
            f"`{token}` is not a key this config produces or a prefix of an input's sha256; "
            f"this config's are {', '.join(sorted(facts.keys))}",
        ))
    return problems


def trace(page: Page, text: str, start: int, chunk: str,
          cache: dict[tuple[str, str], set[float] | None], where: str = "paragraph",
          ) -> list[Problem]:
    """Every number in ``chunk`` read back through the ``[src]`` links ``chunk`` carries.

    The whole rule, in one place: a paragraph of prose (#107) and a row of the claims table
    (#120) are held to the same thing, so they are held to it by the same function. ``start`` is
    where ``chunk`` begins in ``text``, which is what makes the reported line the one an editor
    jumps to, and ``where`` is the word for the unit in the message.

    A link is relative to the page that carries it, so the same run file is ``../eval/runs/x`` from
    a page in ``report/`` and ``eval/runs/x`` from the repo's README, and both resolve to the one
    file.
    """
    problems: list[Problem] = []
    links = [(m.group("path"), m.group("key")) for m in SRC_RE.finditer(chunk)]
    values: set[float] = set()
    for path, key in links:
        target = (page.path.parent / path).resolve()
        line = _line_of(text, start + chunk.find(path))
        if not target.exists():
            problems.append(Problem(page.path, line, f"[src] link to a missing file: {path}"))
            continue
        if (path, key) not in cache:
            cache[path, key] = candidates(target, key)
        found = cache[path, key]
        if found is None:
            problems.append(Problem(
                page.path, line, f"{path} has no key path {key!r} to read a value from"))
            continue
        values |= found
    for m in NUMBER_RE.finditer(mask(chunk)):
        number = m.group(0)
        line = _line_of(text, start + m.start())
        if not links:
            problems.append(Problem(page.path, line,
                                    f"{number} has no [src] link in its {where}"))
        elif not explains(number, values):
            named = ", ".join(sorted({path for path, _ in links}))
            problems.append(Problem(
                page.path, line,
                f"{number} is not a value in {named} at the precision it is printed to"))
    return problems


def check_prose(page: Page, text: str) -> list[Problem]:
    """Every number in the prose is explained by a ``[src]`` link in its own paragraph."""
    cache: dict[tuple[str, str], set[float] | None] = {}
    return [problem for start, paragraph in paragraphs(page, text)
            for problem in trace(page, text, start, paragraph, cache)]


def claim_rows(page: Page, text: str) -> list[tuple[int, str]]:
    """``(offset, row)`` for every data row of the page's claim tables.

    A generated table's first two lines are its header and its ``| --- |`` divider, by
    construction -- :func:`eval.scripts.agreement.md_table` writes them and nothing else writes
    between the markers -- so the rows are everything after them. A page that lost them lost its
    block, which :func:`check_blocks` says first and more clearly.
    """
    out: list[tuple[int, str]] = []
    for one in blocks_in(page, text):
        if one.name not in CLAIMS_BLOCKS:
            continue
        offset = one.start
        for i, line in enumerate(text[one.start : one.end].splitlines(keepends=True)):
            here, offset = offset, offset + len(line)
            if i >= 2 and line.startswith("|"):
                out.append((here, line))
    return out


def check_claims(page: Page, text: str) -> list[Problem]:
    """Every number in a claim row is in a file that row names, at the precision it is printed to.

    #120's acceptance, which is #107's rule applied to a table: the claims page is a row per
    claim, and the file and key path are in the row rather than in a paragraph around it. The
    numbers get there by being formatted from those files (:func:`claim_values`), so what this
    catches is a hand-edited cell, a claim whose wording moved without its link, and a source
    that has stopped carrying the value the claim quotes from it.
    """
    cache: dict[tuple[str, str], set[float] | None] = {}
    return [problem for start, row in claim_rows(page, text)
            for problem in trace(page, text, start, row, cache, where="row")]


def check(page: Page, text: str, facts: Facts, bodies: Mapping[str, str]) -> list[Problem]:
    """Everything ``--check`` checks on one page, in the order a reader would notice it."""
    return sorted(
        [*check_blocks(page, text, bodies), *check_tables(page, text),
         *check_prose(page, text), *check_claims(page, text), *check_hex(page, text, facts)],
        key=lambda one: one.line,
    )


def render_check(page: Page, text: str, facts: Facts, bodies: Mapping[str, str],
                 out: TextIO) -> int:
    """``--check`` for one page: what does not trace, one per line, or a count of what does."""
    problems = check(page, text, facts, bodies)
    out.writelines(f"{problem}\n" for problem in problems)
    prose = paragraphs(page, text)
    numbers = sum(len(NUMBER_RE.findall(mask(one))) for _, one in prose)
    links = sum(len(SRC_RE.findall(one)) for _, one in prose)
    if problems:
        out.write(f"\n{plural(len(problems), 'problem')} in {rel_to_repo(page.path)}.\n")
        return 1
    scope = "" if page.upto is None else f" above `{page.upto}`"
    rows = claim_rows(page, text)
    claimed = ""
    if rows:
        claimed = (f", every number in {plural(len(rows), 'claim row')} read back through the "
                   "row's own links")
    out.write(f"{rel_to_repo(page.path)}{scope}: {plural(len(page.blocks), 'generated block')} "
              f"current, {numbers} prose numbers carry {links} source links{claimed}, and every "
              "input matches the sha256 its sidecar records.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite every page's generated blocks, or with ``--check`` verify every number traces.

    ``--check`` writes nothing and exits 1 on a failure, which is what CI runs; ``--inputs``
    lists the artifacts behind the pages and ``--dry-run`` prints them instead of writing them.
    Every mode covers every page, because a diagram that is current in one file and stale in the
    other is the drift this script exists to prevent.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.final_report",
        description="Rewrite the generated blocks in README.md, report/README.md and "
                    "report/day7_claims.md, or check that every number in them traces to a "
                    "committed file.",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the committed pages and write nothing; exit 1 if it fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rewritten pages instead of writing them")
    parser.add_argument("--inputs", action="store_true",
                        help="list the artifacts the pages are built from; write nothing")
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
        written = [(page, page.path.read_text(encoding="utf-8")) for page in pages(src)]
        if args.check:
            return max(render_check(page, text, facts, bodies, sys.stdout)
                       for page, text in written)
        spliced = [(page, text, splice(page, text, bodies)) for page, text in written]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        for page, _, body in spliced:
            sys.stdout.write(f"<!-- {rel_to_repo(page.path)} -->\n{body}")
        return 0
    for page, text, body in spliced:
        if body == text:
            print(f"{rel_to_repo(page.path)} is already current")
            continue
        page.path.write_text(body, encoding="utf-8")
        print(f"rewrote {plural(len(page.blocks), 'block')} in {rel_to_repo(page.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
