"""Day 5's acceptance, in one report: three confidence signals, the labels behind them, the gate.

    uv run python -m eval.scripts.day5_report --check     # what exists, and what makes the rest
    uv run python -m eval.scripts.day5_report             # write report/day5_signals_{key}.md
    uv run python -m eval.scripts.day5_report --dry-run   # print it, write nothing

Six sections in #93's order -- setup, labels, RQ2, RQ3, wiring, limitations -- and every number
in them is read out of a committed artifact: both halves' signals runs (#87), their judge runs and
the review log behind the signals CSVs (#88), the test half's risk-coverage numbers and figure
(#89), the calib fit (#90), its one reading on test (#91), and ``config.yaml``'s evidence gate
(#92). It calls no model, resamples nothing and fits nothing: every interval in it was computed by
the script that owns it, and this one only formats it.

**It refuses to write unless every hash agrees.** Each run file must be the file its sidecar
describes (sha256), each downstream file must name the upstream one it was built from by sha256,
every one of them must be under this config's keys (config hash -> signals -> correctness ->
conformal), and every signals and judge run must have been collected on the committed dataset.
#91's reading is also recomputed from the test CSV and the fit -- a pure function, no resample --
and must equal the committed file. A report built from one stale file is the one kind of error
nobody can see by reading it, so there is no ``--force``.

**It refuses to write while the human pass is incomplete.** A signals CSV whose sidecar counts a
bucket with fewer reviewed rows than rows, or a row with no judge verdict, is not a set of labels.

**Named after the conformal key, not the config hash.** The signals, correctness and conformal
sections are all left out of ``cfg.hash`` (:data:`config.AFTER_THE_RUNS`), so under the Day 3/4
naming a changed verbalized prompt would overwrite this report under the same name.
:func:`config.conformal_key` is the leaf of the key chain -- it hashes the correctness key, which
hashes the signals key, which hashes the config hash -- so it moves when anything this report
reads does.

**The RQ2 and RQ3 tables are the owning scripts' own renderings** (:func:`eval.scripts.
risk_coverage.summary`, :func:`eval.scripts.conformal_eval.summary`), for the reason Day 3's report
reuses #60's: two implementations of one table are two reports that can disagree. What this file
adds around them is what those tables cannot say on their own -- which intervals are degenerate,
which paired differences compare different rows -- and each of those sentences is computed.

**No clock anywhere in the body.** The review log carries timestamps; none are printed, so a
re-run over unchanged inputs rewrites a byte-identical file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from agent.confidence import calls_per_answer, pairs
from agent.confidence import settings as signal_settings
from agent.graph import node_names
from agent.nodes.check_evidence import check_method
from agent.nodes.confidence_gate import POST_ANSWER, conformal_gate
from config import (
    REPORT_DIR,
    Config,
    config_hash,
    conformal_key,
    correctness_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval import conformal
from eval.schema import BUCKETS, SPLITS
from eval.scripts import conformal_eval, risk_coverage
from eval.scripts.agreement import md_table
from eval.scripts.judge import judge_path
from eval.scripts.review import DATASET_PATH
from eval.scripts.review_signals import csv_path
from eval.scripts.risk_coverage import SIGNALS, Table, load_table
from eval.scripts.signals import RUNS_DIR, run_path
from eval.scripts.split import META_SUFFIX, read_sidecar
from eval.validate import plural

#: The report's stem; :func:`report_path` adds the conformal key and the extension.
STEM = "day5_signals"

#: The half every reported number is on. calib appears only as the half the fit was made on.
REPORTED = "test"

#: The review log #88's CSVs were built from; both sidecars pin its sha256.
REVIEW_LOG = "correctness_log.jsonl"


# --- the inputs ---------------------------------------------------------------------------------


class Sources(NamedTuple):
    """Where the report reads from, rooted so a test can point it at a copy."""

    runs_dir: Path = RUNS_DIR
    dataset: Path = DATASET_PATH
    figures_dir: Path = risk_coverage.FIGURES_DIR


class Input(NamedTuple):
    """One artifact the report needs, and the command that produces it when it is missing."""

    name: str
    path: Path
    how: str


def inputs(cfg: Config, src: Sources) -> list[Input]:
    """Every artifact, in the order the sections that read them appear."""
    runs = src.runs_dir
    found = [Input("dataset", src.dataset, "uv run python -m eval.scripts.review --promote")]
    for half in SPLITS:
        signals = run_path(half, cfg, runs)
        judged = judge_path(half, cfg, runs)
        table = csv_path(half, cfg, runs)
        found += [
            Input(f"{half} signals run", signals,
                  f"uv run --env-file .env python -m eval.scripts.signals --split {half}"),
            Input(f"{half} signals sidecar", signals.with_suffix(META_SUFFIX),
                  f"uv run --env-file .env python -m eval.scripts.signals --split {half}"),
            Input(f"{half} judge run", judged,
                  f"uv run --env-file .env python -m eval.scripts.judge --split {half}"),
            Input(f"{half} judge sidecar", judged.with_suffix(META_SUFFIX),
                  f"uv run --env-file .env python -m eval.scripts.judge --split {half}"),
            Input(f"{half} signals CSV", table,
                  f"uv run python -m eval.scripts.review_signals --split {half} --write-csv"),
            Input(f"{half} CSV sidecar", table.with_name(table.name + META_SUFFIX),
                  f"uv run python -m eval.scripts.review_signals --split {half} --write-csv"),
        ]
    return found + [
        Input("review log", runs / REVIEW_LOG,
              "uv run python -m eval.scripts.review_signals --split test --reviewer NAME"),
        Input("risk-coverage numbers", risk_coverage.result_path(REPORTED, cfg, runs),
              "uv run --extra analysis python -m eval.scripts.risk_coverage"),
        Input("risk-coverage figure", risk_coverage.figure_path(REPORTED, cfg, src.figures_dir),
              "uv run --extra analysis python -m eval.scripts.risk_coverage"),
        Input("conformal fit", conformal.result_path(cfg, runs),
              "uv run --extra analysis python -m eval.conformal"),
        Input("conformal test reading", conformal_eval.result_path(cfg, runs),
              "uv run --extra analysis python -m eval.scripts.conformal_eval"),
    ]


def missing(cfg: Config, src: Sources) -> list[Input]:
    """The inputs that are not there. Empty is the condition for writing anything."""
    return [one for one in inputs(cfg, src) if not one.path.exists()]


def report_path(cfg: Config, report_dir: Path = REPORT_DIR) -> Path:
    """``report/day5_signals_{conformal_key}.md``. See the module docstring for why that key."""
    return Path(report_dir) / f"{STEM}_{conformal_key(cfg)}.md"


def render_check(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--check``: the inputs, one per line, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    out.write(f"report {rel_to_repo(report_path(cfg))}  conformal key {conformal_key(cfg)}\n\n")
    for one in found:
        out.write(f"  {'ok     ' if one.path.exists() else 'MISSING'}  {one.name:<{width}}  "
                  f"{rel_to_repo(one.path)}\n")
        if not one.path.exists():
            out.write(f"  {'':7}  {'':<{width}}  {one.how}\n")
    gaps = missing(cfg, src)
    if not gaps:
        out.write("\nEvery input is present; the report can be written if their hashes agree.\n")
        return 0
    out.write(f"\n{plural(len(gaps), 'input')} missing, so the report is not written.\n")
    return 1


# --- reading and checking them ------------------------------------------------------------------


class Half(NamedTuple):
    """One half's signals run, judge run and labelled CSV, each with its sidecar."""

    split: str
    run_meta: dict[str, Any]
    judge_meta: dict[str, Any]
    table: Table
    #: The ``dense_top1`` column, which :class:`Table` leaves out (a diagnostic, not a signal).
    dense: list[float | None]


class Facts(NamedTuple):
    """Everything the report prints, read once and checked before a word is rendered."""

    cfg: Config
    src: Sources
    dataset_sha256: str
    halves: dict[str, Half]
    rc: dict[str, Any]
    fit: dict[str, Any]
    reading: dict[str, Any]
    #: ``(name, repo path, sha256)`` for the header's input table.
    digests: list[tuple[str, str, str]]
    #: The live config with ``evidence.method: conformal``: the gate #92 built and left dormant.
    switched: Config
    gate: Any


def _json(path: Path) -> dict[str, Any]:
    data = read_sidecar(path)
    if data is None:
        raise ValueError(f"no file at {rel_to_repo(path)}")
    return data


def _dense(table: Table) -> list[float | None]:
    with table.path.open(encoding="utf-8", newline="") as f:
        return [None if line["dense_top1"] == "" else float(line["dense_top1"])
                for line in csv.DictReader(f)]


def switched_config(cfg: Config) -> Config:
    """``cfg`` under ``evidence.method: conformal``, everything else as it is."""
    plain = cfg.to_dict()
    plain["evidence"] = {**plain["evidence"], "method": "conformal"}
    return Config(plain)


def read_facts(cfg: Config, src: Sources) -> Facts:
    """Read every input and check every hash that ties them together, or raise ValueError."""
    problems: list[str] = []
    runs = src.runs_dir
    dataset_sha = file_sha256(src.dataset)
    digests = [("dataset", rel_to_repo(src.dataset), dataset_sha)]

    log = runs / REVIEW_LOG
    log_sha = file_sha256(log)
    halves: dict[str, Half] = {}
    for half in SPLITS:
        run = run_path(half, cfg, runs)
        run_meta = _json(run.with_suffix(META_SUFFIX))
        run_sha = file_sha256(run)
        if run_sha != run_meta.get("sha256"):
            problems.append(f"{rel_to_repo(run)} is not the file its sidecar describes")
        if run_meta.get("signals_key") != signals_key(cfg):
            problems.append(f"{rel_to_repo(run)} is signals {run_meta.get('signals_key')}, "
                            f"not {signals_key(cfg)}")

        judged = judge_path(half, cfg, runs)
        judge_meta = _json(judged.with_suffix(META_SUFFIX))
        judge_sha = file_sha256(judged)
        if judge_sha != judge_meta.get("sha256"):
            problems.append(f"{rel_to_repo(judged)} is not the file its sidecar describes")
        if judge_meta.get("correctness_key") != correctness_key(cfg):
            problems.append(f"{rel_to_repo(judged)} is correctness "
                            f"{judge_meta.get('correctness_key')}, not {correctness_key(cfg)}")
        if judge_meta.get("signals_run_sha256") != run_sha:
            problems.append(f"{rel_to_repo(judged)} grades another {half} signals run")

        for name, meta in ((rel_to_repo(run), run_meta), (rel_to_repo(judged), judge_meta)):
            if meta.get("dataset_sha256") != dataset_sha:
                problems.append(f"{name} was made on another dataset than "
                                f"{rel_to_repo(src.dataset)}")

        table = load_table(half, cfg, runs)  # sha256 and signals key, or ValueError
        meta = table.meta
        if meta.get("signals_run_sha256") != run_sha:
            problems.append(f"{rel_to_repo(table.path)} was not built from {rel_to_repo(run)}")
        if meta.get("judge_run_sha256") != judge_sha:
            problems.append(f"{rel_to_repo(table.path)} was not built from {rel_to_repo(judged)}")
        if meta.get("review_log_sha256") != log_sha:
            problems.append(f"{rel_to_repo(table.path)} was not built from the review log on disk")
        if meta.get("n_rows") != run_meta.get("n_lines"):
            problems.append(f"{rel_to_repo(table.path)} has {meta.get('n_rows')} rows, the "
                            f"signals run {run_meta.get('n_lines')}")
        for bucket, counts in sorted(meta.get("by_bucket", {}).items()):
            if counts["reviewed"] != counts["rows"] or counts["no_judge_verdict"]:
                problems.append(f"the {half} review is incomplete: {bucket} has "
                                f"{counts['reviewed']} of {counts['rows']} rows reviewed and "
                                f"{counts['no_judge_verdict']} without a judge verdict")
        halves[half] = Half(half, run_meta, judge_meta, table, _dense(table))
        digests += [(f"{half} signals run", rel_to_repo(run), run_sha),
                    (f"{half} judge run", rel_to_repo(judged), judge_sha),
                    (f"{half} signals CSV", rel_to_repo(table.path), meta["sha256"])]
    digests.append(("review log", rel_to_repo(log), log_sha))

    test_csv, calib_csv = halves["test"].table.meta["sha256"], halves["calib"].table.meta["sha256"]
    rc_path = risk_coverage.result_path(REPORTED, cfg, runs)
    rc = _json(rc_path)
    figure = risk_coverage.figure_path(REPORTED, cfg, src.figures_dir)
    if rc.get("csv_sha256") != test_csv or rc.get("signals_key") != signals_key(cfg):
        problems.append(f"{rel_to_repo(rc_path)} was not computed from the test CSV on disk")
    if file_sha256(figure) != rc.get("figure_sha256"):
        problems.append(f"{rel_to_repo(figure)} is not the figure {rel_to_repo(rc_path)} drew")

    fit_path = conformal.result_path(cfg, runs)
    fit = _json(fit_path)
    reading_path = conformal_eval.result_path(cfg, runs)
    reading = _json(reading_path)
    for name, record in ((rel_to_repo(fit_path), fit), (rel_to_repo(reading_path), reading)):
        for field, want in (("config_hash", cfg.hash), ("signals_key", signals_key(cfg)),
                            ("correctness_key", correctness_key(cfg)),
                            ("conformal_key", conformal_key(cfg))):
            if record.get(field) != want:
                problems.append(f"{name} has {field} {record.get(field)}, this config {want}")
    if fit.get("csv_sha256") != calib_csv:
        problems.append(f"{rel_to_repo(fit_path)} was not fitted on the calib CSV on disk")
    if reading.get("csv_sha256") != test_csv:
        problems.append(f"{rel_to_repo(reading_path)} did not read the test CSV on disk")
    if reading.get("fit_calib_csv_sha256") != calib_csv:
        problems.append(f"{rel_to_repo(reading_path)} applied a fit made on another calib CSV")
    if reading.get("dataset_sha256") != dataset_sha:
        problems.append(f"{rel_to_repo(reading_path)} checked another dataset")
    level = float(cfg["eval"]["bootstrap"]["ci"])
    again = conformal_eval.evaluate(halves[REPORTED].table, fit, level, halves[REPORTED].dense)
    if any(reading.get(key) != value for key, value in again.items()):
        problems.append(f"{rel_to_repo(reading_path)} is not what #91's evaluate gives on the "
                        "test CSV and the fit on disk")
    digests += [("risk-coverage numbers", rel_to_repo(rc_path), file_sha256(rc_path)),
                ("risk-coverage figure", rel_to_repo(figure), rc["figure_sha256"]),
                ("conformal fit", rel_to_repo(fit_path), file_sha256(fit_path)),
                ("conformal test reading", rel_to_repo(reading_path), file_sha256(reading_path))]

    switched = switched_config(cfg)
    try:
        # #92's own check: the pinned sha, alpha, fit_on, threshold and the whole key chain.
        gate = conformal_gate(switched)
    except (ValueError, FileNotFoundError, TypeError) as error:
        problems.append(f"the dormant conformal gate does not build: {error}")
        gate = None

    if problems:
        raise ValueError("refusing to write the report:\n  - " + "\n  - ".join(problems))
    return Facts(cfg, src, dataset_sha, halves, rc, fit, reading, digests, switched, gate)


# --- formatting ---------------------------------------------------------------------------------


def pct(x: float) -> str:
    """A rate as a percentage to one decimal place."""
    return f"{100 * x:.1f}%"


def signal_list(names: Sequence[str]) -> str:
    """Names joined for prose, each in backticks."""
    return ", ".join(f"`{name}`" for name in names)


def bucket_counts(half: Half) -> dict[str, dict[str, int]]:
    """Per bucket: rows in the half, rows answered (and so labelled), and rows labelled correct."""
    answered = Counter(half.table.buckets)
    skipped = Counter(row["bucket"] for row in half.run_meta["skipped_rows"])
    correct = Counter(b for b, c in zip(half.table.buckets, half.table.correct, strict=True) if c)
    return {b: {"rows": answered[b] + skipped[b], "answered": answered[b], "correct": correct[b]}
            for b in BUCKETS}


def tie_groups(points: Sequence[Mapping[str, Any]]) -> list[tuple[float, int]]:
    """``(value, rows at exactly that value)``, strictest first, from #89's cumulative sweep."""
    out, before = [], 0
    for point in points:
        out.append((point["threshold"], point["answered"] - before))
        before = point["answered"]
    return out


# --- the sections -------------------------------------------------------------------------------


def render_header(facts: Facts) -> str:
    """Title, the four keys, the headline AUROC, and the digest of every input.

    The chosen signal and the highest-AUROC signal are both named, because on the test half they
    need not be the same one and reporting only the winner would be picking it after the fact.
    """
    cfg = facts.cfg
    test = facts.halves[REPORTED]
    reading = facts.reading
    chosen = reading["chosen"]
    rc = facts.rc["signals"]
    best = max(SIGNALS, key=lambda name: rc[name]["auroc"]["point"])
    lines = [
        "# Day 5: confidence signals and conformal gating",
        "",
        (
            f"Config hash `{cfg.hash}` · signals `{signals_key(cfg)}` · correctness "
            f"`{correctness_key(cfg)}` · conformal `{conformal_key(cfg)}` · full hash "
            f"`{cfg.full_hash}` · dataset sha256 `{facts.dataset_sha256[:12]}`"
        ),
        "",
        (
            f"On the test half ({len(test.table.ids)} answered rows), `{chosen}`, the signal "
            f"chosen on calib, ranks correct above incorrect answers with AUROC "
            f"{rc[chosen]['auroc']['point']:.3f} [{rc[chosen]['auroc']['lo']:.3f}, "
            f"{rc[chosen]['auroc']['hi']:.3f}]. The highest test AUROC is `{best}`'s "
            f"({rc[best]['auroc']['point']:.3f}, on {rc[best]['n']} non-null rows)."
        ),
        conformal_headline(facts),
        "",
        "## Inputs",
        "",
        (
            "Every file below matched its sidecar, the file downstream of it, this config's keys "
            "and the dataset before a word of this report was written."
        ),
        "",
        md_table(["input", "path", "sha256"],
                 [[name, f"`{path}`", f"`{sha[:12]}`"] for name, path, sha in facts.digests]),
    ]
    return "\n".join(lines)


def conformal_headline(facts: Facts) -> str:
    """One sentence comparing the conformal gate, a fixed 0.7, and answering everything.

    A fit that accepted no threshold is said as exactly that -- the gate answers nothing -- rather
    than reported as an error rate over an empty set.
    """
    reading = facts.reading
    chosen = reading["chosen"]
    block = reading["signals"][chosen]
    fixed = block["fixed"]
    if block["conformal"]["answers_nothing"]:
        head = (
            f"The calib fit accepted no threshold for any signal at alpha = "
            f"{reading['alpha']:.2f}, so the conformal gate answers nothing on test"
            if all(s["conformal"]["answers_nothing"] for s in reading["signals"].values())
            else f"The calib fit accepted no threshold for `{chosen}` at alpha = "
                 f"{reading['alpha']:.2f}, so its conformal gate answers nothing on test"
        )
    else:
        c = block["conformal"]
        head = (f"The conformal gate on `{chosen}` answers {c['answered']} of {c['n_non_null']} "
                f"test rows at realized error {c['realized_error']:.3f}")
    return (
        f"{head}; a fixed 0.7 answers {fixed['answered']} of {fixed['n_non_null']} at "
        f"{fixed['realized_error']:.3f}, against "
        f"{reading['answer_everything']['realized_error']:.3f} for answering everything."
    )


def render_setup(facts: Facts) -> str:
    """Rows and answered rows per bucket and half, the nulls, and what the two runs cost.

    Raises ``ValueError`` if the calib and test runs disagree on model, retrieval mode, sampling
    or prompt digests: the halves are only comparable when the run behind them was the same one.
    """
    cfg = facts.cfg
    calib, test = facts.halves["calib"], facts.halves["test"]
    counts = {half: bucket_counts(facts.halves[half]) for half in SPLITS}
    rows = []
    for bucket in BUCKETS:
        cells = [bucket]
        for half in SPLITS:
            c = counts[half][bucket]
            cells += [str(c["rows"]), str(c["answered"])]
        rows.append(cells)
    rows.append(["**total**", *(
        str(sum(counts[half][b][k] for b in BUCKETS))
        for half in SPLITS for k in ("rows", "answered"))])
    table = md_table(
        ["bucket", *(f"{half} {k}" for half in SPLITS for k in ("rows", "answered"))], rows)

    skipped = "; ".join(
        f"{half}: " + ", ".join(f"{n} {reason}" for reason, n in
                                sorted(facts.halves[half].run_meta["skipped"].items()))
        for half in SPLITS)
    refused = "; ".join(
        f"{half}: " + (", ".join(f"{n} {b}" for b, n in sorted(
            facts.halves[half].run_meta["gate_would_refuse"].items())) or "none")
        for half in SPLITS)

    nulls = []
    for half in SPLITS:
        t = facts.halves[half].table
        parts = [f"`{name}` {sum(v is None for v in t.signals[name])}" for name in SIGNALS]
        tools = sum(facts.halves[half].run_meta["tool_rows"].values())
        nulls.append(f"{half}: " + ", ".join(parts) + f" (of {len(t.ids)}; {tools} tool rows)")
    unparsed = ", ".join(
        f"{half} {facts.halves[half].run_meta['verbalized_unparsed']} verbalized / "
        f"{facts.halves[half].run_meta['judge_unparsed']} judge"
        for half in SPLITS)

    meta = test.run_meta
    for field in ("model", "retrieval_mode", "samples", "sample_temperature", "prompts_sha256"):
        if calib.run_meta[field] != meta[field]:
            raise ValueError(f"the two halves' signals runs differ in {field}")
    s = signal_settings(cfg)
    base = 2  # route_intent and answer
    per_row = base + sum(calls_per_answer(name, s) for name in POST_ANSWER)
    answered = sum(len(facts.halves[half].table.ids) for half in SPLITS)
    prompts = ", ".join(f"{name} `{sha[:12]}`" for name, sha in sorted(
        meta["prompts_sha256"].items()))

    return "\n".join([
        "## 1. Setup",
        "",
        table,
        "",
        (
            f"*Rows* is the half of the frozen dataset; *answered* is what the signals run "
            f"collected and #88 labelled. Skipped before an answer existed: {skipped}. "
            "`no_evidence` rows have nothing for the answer node to cite, and `out_of_scope` "
            "rows never reach dispatch."
        ),
        "",
        (
            f"**The evidence gate was bypassed** (#87); the run's sidecar says: "
            f"\"{meta['gate']}\". Rows the Day 2 `top_score` gate would have refused, and that "
            f"were answered anyway: {refused}."
        ),
        "",
        (
            f"**Nulls per signal.** {'; '.join(nulls)}. `retrieval_score` is null on every tool "
            "row (an exact catalog lookup has no retrieval score), never 1.0. Unparsed replies: "
            f"{unparsed}."
        ),
        "",
        (
            f"**Model and sampling.** `{meta['model']}` for every call, retrieval "
            f"`{meta['retrieval_mode']}`; self-consistency draws {meta['samples']} samples at "
            f"temperature {meta['sample_temperature']} and judges all "
            f"{len(pairs(meta['samples']))} pairs. Prompt sha256: {prompts}."
        ),
        "",
        (
            f"**Cost.** Each answered row took {per_row} chat calls (route, answer, "
            f"{calls_per_answer('verbalized', s)} verbalized, "
            f"{s.samples} samples, {len(pairs(s.samples))} pair judgements) and one embedding "
            f"call, and #88's judge one more: {answered} answered rows across both halves are "
            f"{answered * (per_row + 1)} chat calls. That counts the committed runs only, not "
            "the router calls on skipped rows or any run these replaced. The runs did not record "
            "token counts, so there is no dollar figure here."
        ),
    ])


def render_labels(facts: Facts) -> str:
    """The judge-versus-human confusion per half, who reviewed, and correct/answered per bucket."""
    rows = []
    for half in SPLITS:
        meta = facts.halves[half].table.meta
        m = meta["judge_x_human"]
        n = meta["n_rows"]
        agree = m["correct"]["correct"] + m["incorrect"]["incorrect"]
        overturned = sum(c["overturned"] for c in meta["by_bucket"].values())
        reviewers = ", ".join(f"{who} ({k})" for who, k in sorted(meta["reviewers"].items()))
        rows.append([
            half, str(n), str(m["correct"]["correct"]), str(m["correct"]["incorrect"]),
            str(m["incorrect"]["correct"]), str(m["incorrect"]["incorrect"]),
            str(sum(m["unparsed"].values())), f"{agree} / {n} ({pct(agree / n)})",
            str(overturned), str(meta["verdicts_carried_over"]), reviewers,
        ])
    per_bucket = []
    counts = {half: bucket_counts(facts.halves[half]) for half in SPLITS}
    for bucket in BUCKETS:
        per_bucket.append([bucket, *(
            f"{counts[half][bucket]['correct']} / {counts[half][bucket]['answered']}"
            for half in SPLITS)])
    per_bucket.append(["**total**", *(
        f"{sum(facts.halves[half].table.correct)} / {len(facts.halves[half].table.ids)}"
        for half in SPLITS)])
    return "\n".join([
        "## 2. Labels",
        "",
        (
            "`correct` is a person's verdict on every row; the LLM judge's call was shown first "
            "(#88). Judge × human, with the human verdict second:"
        ),
        "",
        md_table(["half", "n", "judge ✓ human ✓", "judge ✓ human ✗", "judge ✗ human ✓",
                  "judge ✗ human ✗", "judge unparsed", "agreement", "overturned",
                  "carried over", "reviewer (rows)"], rows),
        "",
        (
            "*Carried over* counts verdicts matched to an earlier signals run on (id, "
            "answer_sha256): the answer text is byte-identical, so the person's earlier verdict "
            "is about the same answer."
        ),
        "",
        "Correct / answered per bucket:",
        "",
        md_table(["bucket", *SPLITS], per_bucket),
    ])


def render_rq2(facts: Facts) -> str:
    """RQ2: the risk-coverage figure, #89's own table verbatim, and the notes that qualify it."""
    rc = facts.rc
    figure = Path(rc["figure"])
    link = figure.relative_to("report").as_posix() if figure.parts[0] == "report" else rc["figure"]
    notes = rq2_notes(facts)
    return "\n".join([
        "## 3. RQ2: which signal ranks answers best",
        "",
        f"![Risk-coverage curves, test half]({link})",
        "",
        (
            "Written by `eval/scripts/risk_coverage.py` (#89) from the test CSV; its table, as it "
            "prints it:"
        ),
        "",
        risk_coverage.summary(rc),
        "",
        *(f"- {note}" for note in notes),
    ])


def rq2_notes(facts: Facts) -> list[str]:
    """What #89's table cannot say about itself: degenerate intervals, unpaired rows, calib."""
    rc = facts.rc
    test = facts.halves[REPORTED].table
    out = []
    for name in SIGNALS:
        block = rc["signals"][name]
        a = block["auroc"]
        if block["n"] < rc["n"]:
            ys = [c for v, c in zip(test.signals[name], test.correct, strict=True) if v is not None]
            out.append(
                f"`{name}` is ranked on its {block['n']} non-null rows of {rc['n']} "
                f"({sum(ys)} correct, {len(ys) - sum(ys)} incorrect); "
                f"{a['skipped_resamples']} of {rc['n_resamples']} resamples drew one class or "
                "none of those rows and were skipped."
            )
        if a["lo"] == a["hi"]:
            out.append(
                f"`{name}`'s interval has zero width because every resample that could be scored "
                f"gave {a['point']:.3f}: the rows are perfectly separated, which is what a "
                f"sample of {block['n']} can do. It is not a measured certainty."
            )
    for pair in rc["differences"]:
        a, b = pair.split(" - ")
        na, nb = rc["signals"][a]["n"], rc["signals"][b]["n"]
        if na != nb:
            out.append(
                f"`{pair}` shares the resampled questions but not the rows: each AUROC ranks "
                f"its own non-null rows ({na} vs {nb}), so *excludes 0* here does not say one "
                "signal orders the same answers better than the other."
            )
    fit = facts.fit
    calib = ", ".join(f"`{name}` {fit['signals'][name]['auroc']:.3f}" for name in SIGNALS
                      if fit["signals"][name]["auroc"] is not None)
    order_calib = sorted(SIGNALS, key=lambda s: -(fit["signals"][s]["auroc"] or -1))
    order_test = sorted(SIGNALS, key=lambda s: -rc["signals"][s]["auroc"]["point"])
    out.append(
        f"Calib AUROC, the one #90 chose on: {calib}. The test ordering "
        + ("is the same." if order_calib == order_test
           else f"differs: {signal_list(order_test)} on test against "
                f"{signal_list(order_calib)} on calib.")
        + f" The choice ({fit['chosen']['rule']}) was made before the test labels were read."
    )
    return out


def render_rq3(facts: Facts) -> str:
    """RQ3: #91's tables, why the calib fit landed where it did, and what 0.7 means per scale."""
    reading, fit = facts.reading, facts.fit
    lines = [
        "## 4. RQ3: conformal against a fixed 0.7",
        "",
        (
            "Written by `eval/scripts/conformal_eval.py` (#91), which applied the calib fit to the "
            "test half once; its tables, as it prints them:"
        ),
        "",
        conformal_eval.summary(reading),
        "",
        "**Why the conformal threshold is where it is.** A threshold is accepted on calib when "
        f"(k + 1) / (n + 1) <= alpha = {fit['alpha']:.2f} for the rows it answers, so nothing is "
        f"accepted on fewer than {fit['min_answered_to_accept']} answered rows. The lowest value "
        "of (k + 1) / (n + 1) any threshold reached on calib: "
        + ", ".join(f"`{name}` {fit['signals'][name]['lowest_risk_hat_any_t']:.3f}"
                    for name in SIGNALS)
        + ".",
        "",
        "**What 0.7 means on each scale.**",
        "",
    ]
    for name in SIGNALS:
        b = reading["signals"][name]
        f = b["fixed"]
        lo, hi = b["test_range"]
        tail = (" -- every non-null row, the same rows as answering everything"
                if f["answered"] == f["n_non_null"] else "")
        lines.append(
            f"- `{name}`: {conformal_eval.MEANING[name]}. Its test values run {lo:g} to {hi:g}, "
            f"so the threshold answers {f['answered']} of {f['n_non_null']}{tail}."
        )
    return "\n".join(lines)


def render_wiring(facts: Facts) -> str:
    """Which gate the agent actually runs, and what switching to the conformal one would cost.

    The live method, not the fitted one, is what this section leads with: the conformal gate is
    built and pinned but dormant, and a report that described it as running would be wrong.
    """
    cfg, switched, gate = facts.cfg, facts.switched, facts.gate
    live = check_method(cfg)
    nodes_live = node_names(cfg)
    added = [n for n in node_names(switched) if n not in nodes_live]
    where = ("after `answer`" if gate.signal in POST_ANSWER
             else "in `check_evidence`, before `answer`")
    extra = calls_per_answer(gate.signal, signal_settings(cfg))
    t = "+inf (answer nothing)" if gate.threshold == float("inf") else f"{gate.threshold:g}"
    return "\n".join([
        "## 5. Wiring",
        "",
        (
            f"**The agent runs `evidence.method: {live}`.** Its {len(nodes_live)} nodes: "
            f"{signal_list(nodes_live)}. The "
            f"conformal gate (#92) is built and configured but dormant: `evidence.conformal` "
            f"names `{gate.source}` by sha256, signal `{gate.signal}`, threshold {t}, alpha "
            f"{gate.alpha:.2f}, fitted on {gate.fit_on} ({gate.n} rows)."
        ),
        "",
        (
            f"Switched to `conformal`, the config hash moves from `{cfg.hash}` to "
            f"`{config_hash(switched)}`, the graph "
            + (f"gains {signal_list(added)}, " if added else "keeps its nodes, ")
            + f"the comparison happens {where}, and it adds {plural(extra, 'chat call')} per "
            "answered query. The "
            "switched gate was built while writing this report, so its pinned sha256, alpha, "
            "split, threshold and key chain all still match the fit."
        ),
        "",
        (
            "It is not switched on because the fit's threshold would hand every question that "
            "reaches evidence to the refusal exit."
            if gate.threshold == float("inf") else
            "It is not switched on; that is a config change, and it moves the hash."
        ),
    ])


def render_limitations(facts: Facts) -> str:
    """What these numbers cannot support: tie structure, empty buckets, and the interval's width."""
    test = facts.halves[REPORTED]
    reading, rc = facts.reading, facts.rc
    every = reading["answer_everything"]
    lo, hi = every["ci"]
    ci = round(reading["ci_level"] * 100)
    sc = tie_groups(rc["signals"]["self_consistency"]["points"])
    vb = tie_groups(rc["signals"]["verbalized"]["points"])
    top_vb = vb[0]
    zero = {half: [b for b, c in bucket_counts(facts.halves[half]).items()
                   if c["answered"] and not c["correct"]] for half in SPLITS}
    calib_carry = facts.halves["calib"].table.meta["verdicts_carried_over"]
    reviewers = sorted({who for half in SPLITS
                        for who in facts.halves[half].table.meta["reviewers"]})
    overturned = sum(c["overturned"] for half in SPLITS
                     for c in facts.halves[half].table.meta["by_bucket"].values())
    labelled = sum(len(facts.halves[half].table.ids) for half in SPLITS)
    bullets = [
        (
            f"**Test n.** {len(test.table.ids)} answered rows of the half's "
            f"{test.run_meta['n_rows']}. Answering everything is wrong on {every['wrong']} of "
            f"them, {every['realized_error']:.3f} with a {ci}% Clopper-Pearson interval of "
            f"[{lo:.3f}, {hi:.3f}]: a width of {hi - lo:.3f}, which no difference between two "
            "gates on this half can be read inside of."
        ),
        (
            "**Live samples do not reproduce.** The self-consistency samples are drawn at "
            f"temperature {test.run_meta['sample_temperature']} and the dense side of a live run "
            "jitters, so a rerun of #87 is a new draw, not a check of this one. The committed run "
            "files are the sample, and nothing downstream re-queries."
        ),
        (
            f"**The conformal guarantee is marginal.** Error <= alpha = {reading['alpha']:.2f} "
            "holds in expectation over calib draws, not on this calib half and not on this test "
            f"half. With calib n = {facts.fit['n']} the scan cannot accept a threshold that "
            f"answers fewer than {facts.fit['min_answered_to_accept']} rows, whatever the signal."
        ),
        (
            f"**`self_consistency` is heavily tied.** It takes {len(sc)} values on test: "
            + ", ".join(f"{v:.3f} on {k} rows" for v, k in sc)
            + ". Tied rows are answered or skipped together, so its curve has "
            f"{plural(len(sc), 'point')} and no threshold reaches the coverages between them. "
            f"`verbalized` has its own mass point: {top_vb[1]} of {len(test.table.ids)} test rows "
            f"say {top_vb[0]:g}."
        ),
        (
            f"**One reviewer, and the judge was never overturned.** {labelled} labels, by "
            f"{', '.join(reviewers)}, with the judge's call shown first, and {overturned} "
            f"overturned; {calib_carry} calib verdicts were carried over from an earlier run's "
            "identical answers. There is no second annotator on Day 5's labels, so the labels are "
            "the judge's as accepted by one person, and agreement between two people is unmeasured."
        ),
    ]
    if any(zero.values()):
        bullets.append(
            "**Buckets with no correct answer:** "
            + "; ".join(f"{half} {signal_list(names)}" for half, names in zero.items() if names)
            + ". A signal sees those rows only as wrong answers to push below the right ones; "
            "there is no correct answer in them for it to find, and a fit made on such a half "
            "learns nothing about the bucket except how much of it to skip."
        )
    null_rs = rc["signals"]["retrieval_score"]["n_null"]
    bullets.append(
        f"**`retrieval_score` is null on {null_rs} of {rc['n']} test rows** (the tool rows), so "
        "its curve and AUROC describe the retrieval rows only, and the gate on it would pass tool "
        "rows without a comparison."
    )
    bullets.append(
        "**The gate was bypassed to collect the signals**, so every label is about an answer the "
        "Day 2 gate might not have allowed. The numbers describe the answer node's output, not "
        "the deployed agent's replies."
    )
    return "\n".join(["## 6. Limitations", "", *(f"- {b}" for b in bullets)])


REPRODUCTION = """\
## Reproduction

```bash
uv run python -m eval.scripts.day5_report
```

This file is a function of the files in the Inputs table and of `config.yaml`, which names it.
There is no timestamp in it, so a re-run over unchanged inputs rewrites it byte for byte. The
generator refuses to write if any sha256 in that chain has moved.
"""


def render_report(facts: Facts) -> str:
    """Every section joined, in the order the report reads."""
    sections = [
        render_header(facts),
        render_setup(facts),
        render_labels(facts),
        render_rq2(facts),
        render_rq3(facts),
        render_wiring(facts),
        render_limitations(facts),
        REPRODUCTION.rstrip(),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the Day 5 report, or with ``--check`` list its inputs and with ``--dry-run`` print it.

    The file is named by the conformal key, so a refitted threshold writes a new report rather
    than overwriting the one its numbers no longer belong to.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.day5_report",
        description="Write report/day5_signals_{conformal_key}.md from Day 5's committed runs.",
    )
    parser.add_argument("--check", action="store_true",
                        help="list the artifacts the report is built from; write nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report instead of writing it")
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    src = Sources()
    if args.check:
        return render_check(cfg, src, sys.stdout)
    if missing(cfg, src):
        render_check(cfg, src, sys.stderr)
        return 1
    try:
        body = render_report(read_facts(cfg, src))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.write(body)
        return 0
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {rel_to_repo(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
