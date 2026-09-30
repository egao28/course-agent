"""Day 3's acceptance, in one report: the set, who made it, who checked it, where it is cut.

    uv run python -m eval.scripts.day3_report --check     # what exists, and what makes the rest
    uv run python -m eval.scripts.day3_report             # write report/day3_evalset_{hash}.md
    uv run python -m eval.scripts.day3_report --dry-run   # print it, write nothing

Six sections in #62's order -- composition, provenance, verification, agreement, split,
limitations -- and every number in them is read out of a committed artifact: the frozen dataset
and its sidecar (#61), the draft files and theirs (#56 / #57 / #58), the review log (#59), and
#60's sample beside the sheet a second annotator returned. The report is a function of those
files and of ``config.yaml``, and of nothing else.

**It refuses to write rather than print a section that says "not measured yet".** A file called
``day3_evalset_{hash}.md`` is the claim that Day 3 happened; one with an empty Verification
section is that claim made about a dataset nobody read. ``--check`` lists every input, says which
are missing, and names the command that produces each -- which is the useful half of this tool
until the pass has been run (see **Status** in README).

**No clock anywhere in the body.** #59's log is the one artifact in the repo that carries
timestamps, deliberately, and this report prints none of them: a re-run over unchanged inputs
rewrites a byte-identical file, so a diff under ``report/`` is the dataset or the log having
moved. That is also why the fatigue line here is #59's -- accept rate over the first half of the
judgements against the second, which is an ordering and not a duration.

**The test half is not read.** Nothing here is a metric: the counting tables read the whole file
through :func:`eval.validate.parse_items`, which is the entry point #61 names for the three
readers that legitimately want both halves, and ``load_split`` is deliberately not imported --
``tests/test_day3_report.py`` asserts that it stays that way. Day 4's numbers are Day 4's.

**The agreement section is #60's own rendering, one heading level down** (:func:`demote`). #62
asks for exactly what ``report/day3_agreement_{hash}.md`` already prints -- the per-field table
with its interval, the confusion matrix, the itemised disagreements -- and a second
implementation of those tables is two reports that can disagree about a number a reader will
compare across them.

**Limitations is a section and is computed.** The five things wrong with this dataset are known
in advance, but their sizes are not: what share of rows a second person actually read, how wide a
per-bucket interval on the test half really is, what the question/chunk overlap looks like after
#56 rewrote its worst decile. A limitation written as fixed prose goes stale the moment the
numbers move, so each one here is rendered from the run's own figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import Config, load_config, rel_to_repo, report_path
from eval.schema import BUCKETS, RULES, SPLITS, TARGET_TOTAL, EvalItem
from eval.scripts import (
    agreement,
    build_unanswerable,
    draft_items,
    recheck,
    review,
    split,
    triage,
)
from eval.scripts.agreement import md_table
from eval.scripts.draft_items import quantile
from eval.validate import corpus_doc_ids, plural

#: The report's stem; :func:`config.report_path` adds the config hash and the extension.
STEM = "day3_evalset"

#: A cell with nothing in it -- the em dash #60's tables use, so the two reports read alike.
MISSING = "—"

#: How many rows of a ranked list (edit reasons, subjects) the report prints before stopping.
#: The full distribution is in the artifacts; a report is what somebody reads.
TOP_N = 10


# --- the inputs -----------------------------------------------------------------------------------


class Sources(NamedTuple):
    """Every file the report is built from, rooted so a test can point them elsewhere.

    A record rather than module constants for the reason #60's :class:`~eval.scripts.agreement.
    Packet` is one: a test that had to monkeypatch six globals would stop covering the seventh
    the day one was added, and one of these files is a person's completed work.
    """

    dataset: Path
    drafts: Path
    log: Path
    packet: agreement.Packet
    #: #59's pass, model-assisted or not. Optional, and read only so that the report can *say*
    #: the pass was assisted -- see :func:`limitation_bullets`. A report that quietly left this
    #: out would be describing a different methodology from the one that ran.
    triage: Path = triage.TRIAGE_PATH
    #: #59's pass read a second time over a sample (`eval.scripts.recheck`). Optional, and the
    #: only thing that turns "confirmed" into a number: without it an accept is an accept
    #: whether it took two seconds or two minutes, and this report cannot tell the reader which.
    recheck: Path = recheck.RECHECK_PATH

    @property
    def sidecar(self) -> Path:
        """#61's digest record, named off the dataset rather than stored beside it here."""
        return split.meta_path(self.dataset)


def sources(
    dataset: Path | None = None,
    drafts: Path | None = None,
    log: Path | None = None,
    packet: agreement.Packet | None = None,
    advisory: Path | None = None,
    reread: Path | None = None,
) -> Sources:
    """The committed locations, unless told otherwise."""
    return Sources(
        dataset=review.DATASET_PATH if dataset is None else Path(dataset),
        drafts=draft_items.DRAFTS_DIR if drafts is None else Path(drafts),
        log=review.REVIEW_LOG_PATH if log is None else Path(log),
        packet=agreement.packet() if packet is None else packet,
        triage=triage.TRIAGE_PATH if advisory is None else Path(advisory),
        recheck=recheck.RECHECK_PATH if reread is None else Path(reread),
    )


class Input(NamedTuple):
    """One artifact the report needs, and the command that produces it when it is missing."""

    name: str
    path: Path
    #: What to run to get it. Printed by ``--check``, so it is a command and not a description.
    how: str
    #: ``False`` for an artifact whose absence is reportable rather than fatal -- there is one,
    #: the annotator's questions file, and "they asked nothing" is a finding about
    #: ``GUIDELINES.md`` rather than a hole in the report.
    required: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()


def inputs(src: Sources) -> list[Input]:
    """Every artifact, in the order the sections that read them appear."""
    return [
        Input(
            "drafts",
            src.drafts,
            "uv run --env-file .env python -m eval.scripts.draft_items --bucket fact",
        ),
        Input("review log", src.log, "uv run python -m eval.scripts.review"),
        Input("dataset", src.dataset, "uv run python -m eval.scripts.review --promote"),
        Input("split sidecar", src.sidecar, "uv run python -m eval.scripts.split"),
        Input("agreement sample", src.packet.sample, "uv run python -m eval.scripts.agreement"),
        Input("agreement sidecar", src.packet.meta, "uv run python -m eval.scripts.agreement"),
        Input(
            "returned sheet",
            src.packet.filled,
            "hand over eval/datasets/agreement_instructions.md; the sheet comes back filled in",
        ),
        Input(
            "annotator's questions",
            src.packet.questions,
            "uv run python -m eval.scripts.agreement",
            required=False,
        ),
        Input(
            "machine triage",
            src.triage,
            "uv run --env-file .env python -m eval.scripts.triage",
            required=False,
        ),
        Input(
            "sampled re-read",
            src.recheck,
            "uv run python -m eval.scripts.recheck",
            required=False,
        ),
    ]


def missing(src: Sources) -> list[Input]:
    """The required inputs that are not there. Empty is the condition for writing anything."""
    return [one for one in inputs(src) if one.required and not one.exists]


def drafted_note(src: Sources) -> str:
    """How much of the 120-row plan has been drafted at all -- the gate on everything else.

    Lines rather than parsed rows: this runs before anything is validated, and a draft file with
    one unparsable line in it is still a file whose rows exist. #59's reader is what holds them
    to the schema, and it says so per line.
    """
    if not src.drafts.exists():
        return f"no drafts at {rel_to_repo(src.drafts)}: none of the {TARGET_TOTAL} rows exist yet"
    files = review.find_drafts(src.drafts)
    rows = sum(
        1
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return (
        f"{plural(len(files), 'draft file')} in {rel_to_repo(src.drafts)}, {rows} of "
        f"{TARGET_TOTAL} rows drafted"
    )


def render_check(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--check``: the inputs, one per line, and what to run for each that is absent."""
    found = inputs(src)
    width = max(len(one.name) for one in found)
    out.write(f"report {rel_to_repo(report_path(STEM, cfg))}  config {cfg.hash}\n\n")
    for one in found:
        mark = "ok     " if one.exists else ("MISSING" if one.required else "absent ")
        out.write(f"  {mark}  {one.name:<{width}}  {rel_to_repo(one.path)}\n")
        if not one.exists:
            out.write(f"  {'':7}  {'':<{width}}  {one.how}\n")
    out.write("\n" + drafted_note(src) + "\n")
    gaps = missing(src)
    if not gaps:
        out.write("\nEvery input is present; the report can be written.\n")
        return 0
    out.write(
        f"\n{plural(len(gaps), 'input')} missing, so the report is not written. Day 3's "
        "acceptance is that every section is populated from a committed artifact, and a section "
        "filled in from anywhere else would be the one thing this report exists to rule out.\n"
    )
    return 1


# --- reading them ---------------------------------------------------------------------------------


class DraftFile(NamedTuple):
    """One draft file, its sidecar, and the rows it holds -- the Provenance section's unit."""

    path: Path
    meta: Mapping[str, Any]
    drafts: Sequence[review.Candidate]

    @property
    def bucket(self) -> str:
        """What the sidecar says it is; the rows agree, and #59's reader is what checks that."""
        return str(self.meta.get("bucket", MISSING))

    @property
    def source(self) -> str:
        """How the rows were made: what ``provenance.source`` says, or a model having drafted."""
        recorded = {str(c.provenance["source"]) for c in self.drafts if c.provenance.get("source")}
        return ", ".join(sorted(recorded)) if recorded else "drafted by model"

    def recorded(self, key: str) -> list[str]:
        """The distinct values of one provenance key over these rows, sorted, as strings."""
        return sorted({str(c.provenance[key]) for c in self.drafts if c.provenance.get(key)})


class Facts(NamedTuple):
    """Every artifact, read once. The renderers below are functions of this and nothing else."""

    cfg: Config
    src: Sources
    #: The frozen dataset, in file order, validated on the way in by #61's reader.
    rows: Sequence[EvalItem]
    #: #61's sidecar: the digest, the seed, the per-bucket halves and Day 5's ceiling.
    sidecar: Mapping[str, Any]
    #: What ``--verify`` makes of the committed split, re-derived from the seed.
    verification: split.Verification
    files: Sequence[DraftFile]
    candidates: Sequence[review.Candidate]
    entries: Sequence[Mapping[str, Any]]
    #: The log as one verdict per row, replayed once: #59's own reading of its own file.
    verdicts: Mapping[str, review.Verdict]
    #: Every drafted row's provenance by id, deleted rows included -- the composition tables
    #: read a row's shape and depth out of the file it was drafted in, not out of ``v1.jsonl``,
    #: which carries the six annotated fields and the split and nothing else.
    provenance: Mapping[str, Mapping[str, Any]]
    #: #60's sample, keyed by id: the first annotator's labels as they were when the sheet went
    #: out, which is what the agreement is scored against -- not the dataset as it stands now.
    sample: Mapping[str, EvalItem]
    sample_meta: Mapping[str, Any]
    judged: Sequence[agreement.Filled]
    #: What a model suggested per row, or empty when the pass was not assisted.
    advisory: Mapping[str, triage.Triage]
    #: The sampled re-read of the pass, and what it overturned. ``None`` when none was drawn.
    reread_sample: recheck.Sample | None
    reread: recheck.Result | None

    def of(self, bucket: str) -> list[EvalItem]:
        return [row for row in self.rows if row.bucket == bucket]


def read_facts(cfg: Config, src: Sources) -> Facts:
    """Read every input, or raise a :class:`ValueError` naming the ones that are not there.

    The dataset goes through #61's :func:`eval.scripts.split.read_rows`, so a file that does not
    validate stops the report instead of being counted: every table below is a statement about
    rows Day 4 will score, and a report over a dataset with a dangling gold id describes
    something nobody can run.
    """
    gaps = missing(src)
    if gaps:
        names = ", ".join(rel_to_repo(one.path) for one in gaps)
        raise ValueError(
            f"{plural(len(gaps), 'input')} missing ({names}); run "
            "`python -m eval.scripts.day3_report --check` for what produces each"
        )
    known = corpus_doc_ids()
    rows = split.read_rows(src.dataset, known)
    seed = cfg["eval"]["split_seed"]
    files = []
    for path in review.find_drafts(src.drafts):
        drafts, _ = review.load_candidates([path])
        files.append(DraftFile(path, read_json(draft_items.meta_path(path)), drafts))
    candidates = [candidate for file in files for candidate in file.drafts]
    sample = {item.id: item for item in agreement.load_dataset(src.packet.sample)}
    judged, problems = agreement.read_sheet(src.packet.filled, tuple(sample))
    if problems:
        shown = "; ".join(f"{p.id or 'sample'} {p.column or ''}: {p.message}" for p in problems[:5])
        raise ValueError(
            f"{rel_to_repo(src.packet.filled)} has {plural(len(problems), 'problem')} ({shown}); "
            "run `python -m eval.scripts.agreement --score`, which reports all of them, first"
        )
    entries = review.read_entries(src.log)
    drawn = recheck.read_sample(src.recheck)
    return Facts(
        cfg=cfg,
        src=src,
        rows=rows,
        sidecar=read_json(src.sidecar),
        verification=split.verify(src.dataset, seed, known),
        files=files,
        candidates=candidates,
        entries=entries,
        verdicts=review.replay(entries),
        provenance={c.id: c.provenance for file in files for c in file.drafts},
        sample=sample,
        sample_meta=read_json(src.packet.meta),
        judged=judged,
        advisory=triage.read_triage(src.triage),
        reread_sample=drawn,
        reread=None if drawn is None else recheck.result(drawn.ids, entries),
    )


def read_json(path: Path) -> dict[str, Any]:
    """One sidecar, or an empty mapping when there is none.

    Tolerant because the sidecars this reads are two different kinds of input. #61's is
    :func:`inputs`' to insist on -- the digest is what makes an edit to the eval set a reviewable
    event -- and by the time anything is rendered it is known to be there. A *draft* file's is
    #56's resume record, which #57's and #58's tables do not need to have written; its absence
    costs the Provenance section a seed and a batch count, and costs the report nothing else.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --- small shared renderings ------------------------------------------------------------------


def code(value: Any) -> str:
    """One value as inline code, or the em dash when there is none."""
    return MISSING if value is None or value == "" else f"`{value}`"


def joined(values: Sequence[str]) -> str:
    """A list of values as inline code, comma separated, or the em dash for none."""
    return ", ".join(f"`{value}`" for value in values) if values else MISSING


def demote(section: str) -> str:
    """One of #60's rendered sections, every heading one level deeper. See the module docstring.

    Only a run of hashes followed by a space is a heading: ``#59's log`` at the start of a line
    is a sentence, and three of #60's notes begin one. Split on newlines rather than
    ``splitlines()``, so a section that ended in one still does.
    """
    lines = []
    for line in section.split("\n"):
        stripped = line.lstrip("#")
        is_heading = line.startswith("#") and stripped.startswith(" ")
        lines.append("#" + line if is_heading else line)
    return "\n".join(lines)


def distribution(values: Sequence[float], fmt: str = "{:.0f}") -> str:
    """``n / min / p50 / p90 / max`` on one line -- four numbers, because nine is a table."""
    if not values:
        return MISSING
    parts = (
        ("n", str(len(values))),
        ("min", fmt.format(min(values))),
        ("p50", fmt.format(quantile(values, 0.5))),
        ("p90", fmt.format(quantile(values, 0.9))),
        ("max", fmt.format(max(values))),
    )
    return " ".join(f"{name} {value}" for name, value in parts)


def corpus_fact(facts: Facts, key: str) -> int | None:
    """What the drafting sidecars recorded about the corpus: its chunks, or its subjects.

    Read from a sidecar rather than by loading the corpus, so this report needs the artifacts it
    is about and not the 3 MB of records they were made from. #56 writes it; #57's and #58's
    tables are not drafted from a batch and do not, hence the ``None``.
    """
    for file in facts.files:
        found = file.meta.get("corpus", {}).get(key)
        if isinstance(found, int):
            return found
    return None


def subject_of(doc_id: str) -> str:
    """``2026-sp:MATH-241`` -> ``MATH``. The id format is ``{term}:{SUBJ}-{NUM}``."""
    return doc_id.split(":", 1)[-1].rsplit("-", 1)[0]


def headline(facts: Facts) -> str:
    """The sentence #62 names as this report's headline, in one place so it cannot disagree.

    Drafted counts the rows the three builders wrote, including the ones review deleted --
    otherwise "N drafted, M reviewed" would be two names for the same number and the delete
    count would have nowhere to come from.

    **"reviewed by hand" is claimed only where it is true.** #62 writes the headline with that
    phrase in it, and it is the right phrase for an unassisted pass. Where the pass was
    model-assisted (:mod:`eval.scripts.triage`) the phrase becomes "judged", and what the
    reviewer actually did -- confirmed a machine reading, with some part of it re-read
    independently -- is the sentence after it. A headline that said "by hand" over a log of
    confirmations would be the one sentence in this report a reader could catch it out on, and
    the numbers to catch it with are two sections below.
    """
    verdicts = facts.verdicts
    counts = Counter(verdict.verdict for verdict in verdicts.values())
    did = "judged" if facts.advisory else "reviewed by hand"
    lead = (
        f"**{len(facts.candidates)} rows drafted, {len(verdicts)} {did}, "
        f"{counts['edit']} edited, {counts['delete']} deleted** — leaving "
        f"{len(facts.rows)} in `{rel_to_repo(facts.src.dataset)}`, of which "
        f"{len(facts.judged)} were read again by a second annotator."
    )
    return lead if not facts.advisory else lead + " " + assisted_sentence(facts)


def assisted_sentence(facts: Facts) -> str:
    """What the reviewer did, where "reviewed by hand" would overstate it.

    One sentence, and it carries the two numbers that qualify every count above it: how much of
    the pass agreed with the machine, and how much of it was read a second time properly. Both
    are in the Verification section in full; a reader who stops at the headline should still not
    come away with the wrong claim.
    """
    pairs = [
        (facts.advisory[item_id].suggested, verdict.verdict)
        for item_id, verdict in facts.verdicts.items()
        if item_id in facts.advisory
    ]
    agreed = sum(1 for suggested, verdict in pairs if suggested == verdict)
    made = (
        f"That pass was model-assisted: it agreed with the machine on {agreed} of {len(pairs)} "
        f"rows"
        if pairs
        else "That pass was model-assisted"
    )
    if facts.reread is None or not facts.reread.reread:
        return (
            f"{made}, and **no part of it has been re-read independently**, so how much of it "
            "is a reading and how much a confirmation is not established here."
        )
    found = facts.reread
    lo, hi = found.interval
    return (
        f"{made}; {found.reread} of them were read a second time and "
        f"{plural(len(found.changed), 'verdict')} "
        f"{'was' if len(found.changed) == 1 else 'were'} overturned "
        f"({found.rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}])."
    )


# --- the header -------------------------------------------------------------------------------


def render_header(facts: Facts) -> str:
    """What was measured, over which files, at which digests."""
    sidecar = facts.sidecar
    totals = sidecar.get("totals", {})
    rows = [
        ["config hash", code(facts.cfg.hash)],
        ["catalog term", code(facts.cfg["data"]["term"])],
        [
            "dataset",
            (
                f"`{rel_to_repo(facts.src.dataset)}`, {plural(len(facts.rows), 'row')}, "
                f"sha256 `{str(sidecar.get('sha256'))[:16]}…`"
            ),
        ],
        [
            "split",
            f"seed `{sidecar.get('seed')}`, "
            + ", ".join(f"{totals.get(half, 0)} {half}" for half in SPLITS),
        ],
        [
            "annotation spec",
            (
                "`eval/GUIDELINES.md`, sha256 "
                f"`{str(facts.sample_meta.get('guidelines_sha256'))[:16]}…`"
            ),
        ],
        [
            "review log",
            f"`{rel_to_repo(facts.src.log)}`, {plural(len(facts.entries), 'line')}",
        ],
        [
            "second annotator",
            f"`{rel_to_repo(facts.src.packet.filled)}`, {plural(len(facts.judged), 'row')}",
        ],
        [
            "drafts",
            ", ".join(f"`{rel_to_repo(file.path)}`" for file in facts.files) or MISSING,
        ],
    ]
    return "\n".join(
        [
            "# Day 3 — the evaluation set: what it is, who made it, who checked it",
            "",
            headline(facts),
            "",
            md_table(["", ""], rows),
            "",
            "Produced by:",
            "",
            "```bash",
            "uv run python -m eval.scripts.day3_report",
            "```",
        ]
    )


# --- composition --------------------------------------------------------------------------------


def render_composition(facts: Facts) -> str:
    """Section 1: the counts as they are, the two distributions, and the two bucket breakdowns."""
    parts = [
        "## Composition",
        "",
        (
            "Counts as they are, not as they were planned. `target` is "
            "`eval.schema.RULES`; a bucket below it is verification having deleted rows, which "
            "is what verification is for."
        ),
        "",
        composition_table(facts),
        "",
        question_lengths(facts),
    ]
    for section in (multihop_depth(facts), fact_subjects(facts), unanswerable_shapes(facts)):
        if section:
            parts += ["", section]
    return "\n".join(parts)


def composition_table(facts: Facts) -> str:
    """Per bucket: planned, final, and the halves it was cut into."""
    halves = facts.sidecar.get("buckets", {})
    rows = []
    for bucket in BUCKETS:
        rule = RULES[bucket]
        got = len(facts.of(bucket))
        cut = halves.get(bucket, {})
        rows.append(
            [
                f"`{bucket}`",
                str(rule.target_n),
                f"**{got}**" if got != rule.target_n else str(got),
                str(cut.get("calib", MISSING)),
                str(cut.get("test", MISSING)),
                str(rule.should_abstain).lower(),
                rule.gloss,
            ]
        )
    totals = facts.sidecar.get("totals", {})
    rows.append(
        [
            "**total**",
            str(sum(RULES[b].target_n for b in BUCKETS)),
            f"**{len(facts.rows)}**",
            str(totals.get("calib", MISSING)),
            str(totals.get("test", MISSING)),
            "",
            "",
        ]
    )
    return md_table(
        ["bucket", "target", "final", "calib", "test", "abstain", "what it tests"], rows
    )


def question_lengths(facts: Facts) -> str:
    """The length distribution, in words, per bucket.

    Words and not characters: what a length says here is how much of a question a retriever gets
    to match on, and #57's rows are short because a refusal question has no record to quote.
    """
    rows = [
        [
            f"`{bucket}`",
            distribution([len(row.question.split()) for row in facts.of(bucket)]),
        ]
        for bucket in BUCKETS
    ]
    rows.append(["**all**", distribution([len(row.question.split()) for row in facts.rows])])
    return "\n".join(
        ["**Question length**, in words.", "", md_table(["bucket", "words"], rows)]
    )


def multihop_depth(facts: Facts) -> str:
    """#58's per-row depth, as a histogram over the rows that survived review.

    Off the rows rather than off #58's sidecar: the sidecar's histogram is over the rows that
    file was written with, and this report is about the ones in the dataset.
    """
    provenance = facts.provenance
    depths = [
        provenance[row.id]["depth"]
        for row in facts.of("multihop")
        if provenance.get(row.id, {}).get("depth") is not None
    ]
    if not depths:
        return ""
    counts = Counter(depths)
    rows = [[str(depth), str(counts[depth])] for depth in sorted(counts)]
    return "\n".join(
        [
            (
                "**Multi-hop depth** — the longest prerequisite chain each row's walk covers, "
                "cycle-safe (`provenance.depth`, #58)."
            ),
            "",
            md_table(["depth", "rows"], rows),
            "",
            (
                f"{distribution([float(d) for d in depths])}. Each row also records `levels` "
                "and `closure`; the three disagree on a branching graph, which is #58's finding "
                "and not an error here."
            ),
        ]
    )


def fact_subjects(facts: Facts) -> str:
    """Subject coverage for the ``fact`` bucket, read off the gold ids rather than the drafts.

    Off the ids because that is what the row cites, and a row edited at review cites what it
    cites now; ``provenance.subjects`` is the batch the question was drafted *from*, which is a
    different claim.
    """
    rows = facts.of("fact")
    subjects = Counter(subject_of(one) for row in rows for one in row.gold_doc_ids)
    if not subjects:
        return ""
    corpus = corpus_fact(facts, "subjects")
    share = "" if corpus is None else f" of the {corpus} in the catalog"
    top = ", ".join(f"`{name}` {n}" for name, n in subjects.most_common(TOP_N))
    once = sum(1 for n in subjects.values() if n == 1)
    return (
        f"**`fact` subject coverage** — {plural(len(rows), 'row')} cite "
        f"{plural(len(subjects), 'subject')}{share}, {once} of them once. "
        f"Most cited: {top}."
    )


def unanswerable_shapes(facts: Facts) -> str:
    """#57's four failure shapes, against the table sizes that file is built to."""
    provenance = facts.provenance
    rows = facts.of("unanswerable")
    counts = Counter(
        str(provenance[row.id]["shape"])
        for row in rows
        if provenance.get(row.id, {}).get("shape")
    )
    if not counts:
        return ""
    table = [
        [
            f"`{name}`",
            str(shape.n),
            str(counts.get(name, 0)),
            shape.gloss,
        ]
        for name, shape in build_unanswerable.SHAPES.items()
    ]
    other = sorted(set(counts) - set(build_unanswerable.SHAPES))
    table += [[f"`{name}`", MISSING, str(counts[name]), "not one of #57's four"] for name in other]
    above = sum(
        1
        for row in rows
        if provenance.get(row.id, {}).get("above_gate")
    )
    return "\n".join(
        [
            (
                "**`unanswerable` failure shapes** — four different claims, and only one of "
                "them is about the retriever (#57)."
            ),
            "",
            md_table(["shape", "table", "in `v1`", "what makes it unanswerable"], table),
            "",
            (
                f"{above} of {len(rows)} score above the evidence gate on their own question "
                "(`provenance.above_gate`): the gate is not what will refuse them, so Day 4's "
                "abstention on this bucket tests the answer node and not the threshold."
            ),
        ]
    )


# --- provenance -----------------------------------------------------------------------------------


def render_provenance(facts: Facts) -> str:
    """Section 2: which model drafted which rows, under which prompt, from which seed."""
    kept = {row.id for row in facts.rows}
    rows = []
    for file in facts.files:
        drafted = len(file.drafts)
        survived = sum(1 for candidate in file.drafts if candidate.id in kept)
        batches = file.meta.get("batches")
        rows.append(
            [
                f"`{rel_to_repo(file.path)}`",
                f"`{file.bucket}`",
                file.source,
                joined(file.recorded("model")),
                joined(file.recorded("prompt_version")),
                code(file.meta.get("seed")),
                str(len(batches)) if isinstance(batches, list) else MISSING,
                f"{survived}/{drafted}",
            ]
        )
    headers = [
        "file", "bucket", "source", "model", "prompt", "seed", "batches", "kept",
    ]
    models = Counter(
        str(facts.provenance[row.id]["model"])
        for row in facts.rows
        if facts.provenance.get(row.id, {}).get("model")
    )
    written = (
        ", ".join(f"`{name}` {n}" for name, n in sorted(models.items()))
        if models
        else "no row records a drafting model"
    )
    return "\n".join(
        [
            "## Provenance",
            "",
            (
                "Every row carries the record of how it was made, and `kept` is how many of a "
                "file's rows survived the verification pass below."
            ),
            "",
            md_table(headers, rows),
            "",
            (
                f"Rows in `{rel_to_repo(facts.src.dataset)}` by drafting model: {written}. A row "
                "with no model was constructed or walked out of the prerequisite graph rather "
                "than drafted — see `source`."
            ),
        ]
    )


# --- verification ---------------------------------------------------------------------------------


def render_verification(facts: Facts) -> str:
    """Section 3: what a person did to each bucket, and the reasons they gave."""
    counts = review.tally(facts.candidates, facts.verdicts)
    columns = ("drafted", "accept", "edit", "delete", "unreviewed")
    rows = []
    totals: Counter[str] = Counter()
    for bucket in BUCKETS:
        totals.update(counts[bucket])
        rows.append([f"`{bucket}`", *(str(counts[bucket][column]) for column in columns)])
    rows.append(["**total**", *(f"**{totals[column]}**" for column in columns)])
    parts = [
        "## Verification",
        "",
        headline(facts),
        "",
        md_table(["bucket", *columns], rows),
        "",
        (
            f"From `{rel_to_repo(facts.src.log)}`, {plural(len(facts.entries), 'line')} — one "
            "per changed field, so an edit touching three fields is three lines sharing a "
            "timestamp. A bucket's edit rate is how much Day 4 should trust the pipeline that "
            "drafted it, which is why it is here rather than summarised."
        ),
    ]
    for title, verdict in (("Edit reasons", "edit"), ("Deletes", "delete")):
        reasons = Counter(
            one.reason for one in facts.verdicts.values() if one.verdict == verdict and one.reason
        )
        if reasons:
            parts += [
                "",
                f"**{title}**",
                "",
                md_table(
                    ["reason", "rows"],
                    [[f"`{reason}`", str(n)] for reason, n in reasons.most_common(TOP_N)],
                ),
            ]
    judgements = review.events(facts.entries)
    if judgements:
        parts += [
            "",
            (
                f"**Did judgement drift?** {review.fatigue(judgements)}. This is an ordering, "
                "not a duration: the log carries the clock and this report carries none of it, "
                "so a re-run rewrites this file byte for byte."
            ),
        ]
    assisted = render_assistance(facts)
    if assisted:
        parts += ["", assisted]
    return "\n".join(parts)


def render_assistance(facts: Facts) -> str:
    """How often the reviewer agreed with the machine, when the pass was model-assisted.

    The number the anchoring caveat in :func:`limitation_bullets` needs, and the reason this
    report can disclose the assistance with something better than a disclaimer: a reviewer who
    accepted every suggestion produced a pass worth exactly what the model is worth, and one who
    overrode a third of them did not. The direction of the bias is not measurable here -- what an
    unassisted pass over these rows would have decided is not knowable after the fact -- so the
    agreement rate is an upper bound on the assistance's influence and is labelled as one.
    """
    if not facts.advisory:
        return ""
    pairs = [
        (facts.advisory[item_id].suggested, verdict.verdict)
        for item_id, verdict in facts.verdicts.items()
        if item_id in facts.advisory
    ]
    if not pairs:
        return ""
    agreed = sum(1 for suggested, verdict in pairs if suggested == verdict)
    rows = [
        [
            f"**{suggested}**",
            *(
                str(sum(1 for a, b in pairs if a == suggested and b == verdict))
                for verdict in review.VERDICT_NAMES
            ),
        ]
        for suggested in review.VERDICT_NAMES
    ]
    models = sorted({one.model for one in facts.advisory.values()})
    return "\n".join(
        [
            (
                f"**The pass was model-assisted.** {joined(models)} read "
                f"{plural(len(facts.advisory), 'row')} against `eval/GUIDELINES.md` first "
                f"(`{rel_to_repo(facts.src.triage)}`), and the reviewer saw its suggestion under "
                "each row's chunk text before typing a verdict. They agreed on "
                f"{agreed} of {len(pairs)} rows ({agreed / len(pairs):.0%})."
            ),
            "",
            md_table(["suggested ↓ / judged →", *review.VERDICT_NAMES], rows),
            "",
            (
                "Read the agreement rate as an **upper bound** on how much the suggestions "
                "moved the pass, not as a measurement of it: what an unassisted reader would "
                "have decided about these rows is not recoverable afterwards. The off-diagonal "
                "cells are the rows where a person overrode a machine."
            ),
            "",
            render_reread(facts),
        ]
    )


def render_reread(facts: Facts) -> str:
    """The sampled second reading: the only number here that separates reading from confirming.

    A pass with no re-read gets the sentence that says so, rather than silence. The difference
    between "every row was read" and "every row was confirmed" cannot be recovered from a log
    of one verdict per row, so a report that left this out would leave the reader to assume the
    stronger of the two.
    """
    found, drawn = facts.reread, facts.reread_sample
    if drawn is None or found is None or not found.reread:
        return (
            "**No part of the pass has been re-read.** An accept is an accept in the log whether "
            "it took two seconds or two minutes, so nothing here distinguishes a row that was "
            "read from a row that was confirmed. `python -m eval.scripts.recheck` draws a "
            "stratified sample for a second reading and reports what it overturns; until it has "
            "run, read every count above as *confirmed against the machine reading*, not as "
            "independently checked."
        )
    lo, hi = found.interval
    rows = [
        [
            f"`{one.bucket}`",
            f"`{one.verdict}`",
            str(one.held),
            str(len(one.ids)),
        ]
        for one in drawn.strata
        if one.held
    ]
    body = [
        (
            f"**{found.reread} of the {len(facts.verdicts)} judged rows were judged twice**, "
            f"drawn stratified over bucket and verdict at seed `{drawn.seed}` "
            f"(`{rel_to_repo(facts.src.recheck)}`), including every delete a model asked for. "
            f"{plural(len(found.changed), 'verdict')} "
            f"{'was' if len(found.changed) == 1 else 'were'} overturned: "
            f"**{found.rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}]**."
        ),
        "",
        md_table(["bucket", "verdict", "judged", "re-read"], rows),
    ]
    if found.changed:
        body += [
            "",
            md_table(
                ["id", "first pass", "on re-reading"],
                [[f"`{one.id}`", f"`{one.first}`", f"`{one.second}`"] for one in found.changed],
            ),
        ]
    else:
        body += [
            "",
            (
                f"Nothing overturned. That bounds the pass's error rate at **{hi:.0%}** and is "
                "not the same claim as *the pass was right*; it is what {n} rows support."
            ).format(n=found.reread),
        ]
    if not found.done:
        body += [
            "",
            (
                f"{drawn.ids and found.asked - found.reread} of the sampled rows are still "
                "unread, so the interval above will move."
            ),
        ]
    return "\n".join(body)


# --- agreement ------------------------------------------------------------------------------------


def render_agreement(facts: Facts) -> str:
    """Section 4: #60's numbers, rendered by #60, one heading level down. See :func:`demote`."""
    judged, sample = facts.judged, facts.sample
    scores = [
        agreement.score_field(agreement.pairs_for(field, judged, sample), field)
        for field in agreement.FIELDS
    ]
    scores.append(
        agreement.score_field(
            agreement.derived_pairs(judged, sample),
            agreement.DERIVED,
            "`should_abstain` (derived)",
        )
    )
    cited = agreement.cited_pairs(judged, sample)
    if cited:
        scores.append(
            agreement.score_field(
                cited, "citations_support", "`citations_support` (cited rows only)"
            )
        )
    found = agreement.disagreements(judged, sample)
    questions = facts.src.packet.questions
    asked = (
        f"Questions the annotator asked while working are in "
        f"`{rel_to_repo(questions)}`; they are evidence that `eval/GUIDELINES.md` is incomplete, "
        "and any amendment made in answer to one came **after** this measurement — the "
        "guidelines digest in the header is the version the sheet was filled in against."
        if questions.exists()
        else f"No `{rel_to_repo(questions)}`: the annotator recorded no questions."
    )
    return "\n".join(
        [
            "## Agreement",
            "",
            (
                f"{plural(len(judged), 'row')} of {len(facts.rows)} "
                f"({len(judged) / len(facts.rows):.1%}) were judged a second time, blind, "
                "against `eval/GUIDELINES.md` and nothing else. The tables below are #60's own, "
                f"as they appear in `{rel_to_repo(report_path(agreement.STEM, facts.cfg))}` — "
                "one rendering, so the two reports cannot disagree about a number."
            ),
            "",
            *(
                [demote(one), ""]
                if (one := agreement.render_provenance(facts.src.packet))
                else []
            ),
            demote(agreement.render_numbers(scores, len(judged))),
            "",
            demote(agreement.render_confusion(agreement.pairs_for("bucket", judged, sample))),
            "",
            demote(agreement.render_disagreements(found, len(judged))),
            "",
            asked,
        ]
    )


# --- split ------------------------------------------------------------------------------------


def render_split(facts: Facts) -> str:
    """Section 5: the seed, the halves, the digest, and what sixty calibration rows cost."""
    sidecar = facts.sidecar
    conformal = sidecar.get("conformal", {})
    n_calib = conformal.get("n_calib", sidecar.get("totals", {}).get("calib", 0))
    alpha = conformal.get("alpha_floor")
    ceiling = conformal.get("coverage_ceiling")
    verified = (
        "re-derives from the seed, and the digest matches"
        if facts.verification.ok
        else "**does not re-derive**: " + "; ".join(facts.verification.problems)
    )
    rows = [
        ["seed", f"`{sidecar.get('seed')}` (`eval.split_seed`)"],
        ["sha256", f"`{sidecar.get('sha256')}`"],
        ["sidecar", f"`{rel_to_repo(facts.src.sidecar)}`"],
        ["check", f"`python -m eval.scripts.split --verify` {verified}"],
    ]
    limit = (
        MISSING
        if alpha is None or ceiling is None
        else (
            f"That bounds α below at 1/{n_calib + 1} = {alpha:.4f}, so **{ceiling:.1%} coverage "
            f"is the ceiling** and {out_of_reach(ceiling):.0%} is out of reach at any threshold."
        )
    )
    return "\n".join(
        [
            "## Split",
            "",
            (
                "Stratified 50/50 within each bucket, derived from the seed over each bucket's "
                "sorted ids and frozen into the `split` field of every row (#61). The per-bucket "
                "halves are in the Composition table above; this is the provenance of them."
            ),
            "",
            md_table(["", ""], rows),
            "",
            (
                f"**What {n_calib} calibration rows cost Day 5.** Split conformal prediction "
                "takes the `ceil((n+1)(1-α))`-th smallest calibration score, and that index has "
                f"to exist. {limit} The lever is the split ratio and the time to move it is "
                "before anything is measured, which is why the number is here."
            ),
            "",
            (
                "**No number in this report is computed on either half.** Everything above is a "
                "count over the whole file; Day 4 reports on `test`, once, and Day 5 calibrates "
                "on `calib`."
            ),
        ]
    )


# --- limitations ------------------------------------------------------------------------------


def render_limitations(facts: Facts) -> str:
    """Section 6: the five things wrong with this dataset, each sized from this run's numbers.

    A section and not a footnote, and computed rather than fixed prose: a limitation whose
    numbers have gone stale is worse than none, because it reads as though somebody checked.
    """
    return "\n".join(
        [
            "## Limitations",
            "",
            (
                "Written before anyone asks, and sized from the artifacts above rather than "
                "from the plan."
            ),
            "",
            *[f"{bullet}\n" for bullet in limitation_bullets(facts)],
        ]
    ).rstrip()


def limitation_bullets(facts: Facts) -> list[str]:
    """The six bullets, in #62's order. One function so the section is a list and not a wall."""
    n = len(facts.rows)
    second = len(facts.judged)
    single = n - second
    models = sorted(
        {
            str(facts.provenance[row.id]["model"])
            for row in facts.rows
            if facts.provenance.get(row.id, {}).get("model")
        }
    )
    under_test = facts.cfg["llm"]["model"]
    shared = [name for name in models if name == under_test]
    overlaps = [
        float(facts.provenance[row.id]["overlap"])
        for row in facts.rows
        if facts.provenance.get(row.id, {}).get("overlap") is not None
    ]
    abstain = Counter(row.bucket for row in facts.rows if row.should_abstain)
    term = facts.cfg["data"]["term"]
    corpus = corpus_fact(facts, "chunks")
    catalog = MISSING if corpus is None else f"{corpus:,} courses"
    return [
        (
            f"- **One annotator on {single} of {n} rows.** {second / n:.1%} were "
            f"double-annotated. The κ on that {second / n:.1%} is the only evidence there is "
            f"about the other {single / n:.1%}, and at n={second} it is an estimate with a wide "
            "interval — the Wilson bounds in the Agreement section, not the point estimate, are "
            "what that sample supports."
        ),
        (
            "- **Drafted and evaluated by the same model family.** "
            + (
                f"{joined(shared)} both wrote questions here and is the model under test "
                f"(`llm.model`), so a blind spot it has is a blind spot in the questions and in "
                "the answers, and the two cannot be told apart. Human verification does not "
                "remove the correlation: checking that an answer is *right* does not detect that "
                "the question was *easy for this model*."
                if shared
                else f"The rows were drafted by {joined(models) if models else 'no model'} and "
                f"the model under test is `{under_test}`; they differ, which weakens this "
                "limitation without removing it — the two are the same family and share "
                "training data."
            )
        ),
        (
            "- **Questions derived from chunks share vocabulary with them, which flatters "
            "BM25.** #56 measures the Jaccard overlap between each drafted question and the "
            "record it came from, and rewrites the worst decile. The residual over the rows in "
            f"this set: {distribution(overlaps, '{:.3f}') if overlaps else MISSING}. That is the "
            "distribution to read, not a claim that the leakage was solved; #57's and #58's rows "
            "are not drafted from a record and carry no overlap at all."
        ),
        (
            f"- **{facts.sidecar.get('totals', {}).get('test', 0)} test items.** A per-bucket "
            "accuracy on this set is a wide interval whatever it comes out at — "
            f"{test_intervals(facts)}, as 95% Wilson half-widths at the worst case p=0.5 (a "
            "result near 0 or 1 is tighter). Day 4 must report intervals and not bare "
            "percentages, and this report says so first."
        ),
        (
            f"- **One term, one institution.** `{term}`, {catalog}. Nothing measured on this set "
            "generalises past that, and the `unanswerable` bucket depends on the scope boundary "
            "being exactly this one: `out_of_term` rows are real courses that this export does "
            "not carry, and they stop being unanswerable the moment a second term is indexed."
        ),
        (
            "- **`should_abstain` is one bit where two buckets want different behaviour.** "
            + ", ".join(f"{abstain[bucket]} `{bucket}`" for bucket in BUCKETS if abstain[bucket])
            + " rows carry it. An `ambiguous` row should come back with the clarifying question "
            "in its `gold_answer`; an `unanswerable` row should come back with a refusal and no "
            "question at all. The bit cannot tell those apart, so Day 4's abstention metrics "
            "must condition on `bucket` (#54)."
        ),
        *assistance_bullet(facts),
        *sheet_bullet(facts),
    ]


def sheet_bullet(facts: Facts) -> list[str]:
    """The limitation that the agreement sheet was not the annotator's own typing, if it was not.

    Separate from the assistance bullet because it is a different failure: that one is about a
    machine having read the rows first, this one is about the record of a person's reading being
    second-hand. A reader who discounts one should not thereby discount the other.
    """
    note = facts.src.packet.provenance
    if not note.exists():
        return []
    missing = [row for row in facts.judged if not row.rule]
    return [
        (
            "- **The second annotator's sheet is not their own typing.** How it came back is in "
            f"`{rel_to_repo(note)}` and is quoted above the numbers in the Agreement section. "
            "The arithmetic cannot tell a transcribed sheet from a typed one, so the "
            "distinction survives only because it is written down. What it costs concretely: "
            f"{len(missing)} of the {len(facts.judged)} judged rows cite no guideline rule, so a "
            "disagreement cannot be traced to the line of `eval/GUIDELINES.md` it came from — "
            "which is the main thing #60 draws this sample for — and the two-pass order that "
            "keeps `bucket` from being judged after the citations are opened is unverifiable "
            "here. The labels are a real second reader's; the record of them is not first-hand."
        ),
    ]


def reread_clause(facts: Facts) -> str:
    """The sentence in the limitation that says what the re-read bought, or that there was none."""
    found = facts.reread
    if found is None or not found.reread:
        return (
            "**No part of the pass has been re-read independently**, so nothing here "
            "distinguishes a row that was read from a row that was confirmed."
        )
    lo, hi = found.interval
    return (
        f"{found.reread} of the rows were judged a second time and "
        f"{plural(len(found.changed), 'verdict')} "
        f"{'was' if len(found.changed) == 1 else 'were'} overturned, which puts the pass's "
        f"error rate at {found.rate:.0%} with a 95% interval of [{lo:.0%}, {hi:.0%}] — the "
        f"other {len(facts.verdicts) - found.reread} rows are covered by that interval and by "
        "nothing else."
    )


def assistance_bullet(facts: Facts) -> list[str]:
    """The seventh limitation, and only when there is one: the pass was model-assisted.

    A list rather than a string so that an unassisted pass produces no bullet at all -- a
    limitations section carrying "this was not model-assisted" would be a section answering a
    question nobody asked, and the bullet has to be absent exactly when the caveat does not
    apply.
    """
    if not facts.advisory:
        return []
    models = sorted({one.model for one in facts.advisory.values()})
    digests = sorted({one.guidelines_sha256[:16] for one in facts.advisory.values() if
                      one.guidelines_sha256})
    against = f", against guidelines `{digests[0]}…`" if len(digests) == 1 else ""
    return [
        (
            "- **The verification pass was model-assisted, and a suggestion anchors a "
            f"reviewer.** {joined(models)} read {plural(len(facts.advisory), 'row')} first and "
            "the reviewer saw its accept / edit / delete and its one-sentence reason under each "
            f"row before judging it{against}. A person shown a suggestion agrees with it more "
            "often than one shown nothing, so the accept rate in the Verification section is "
            "not the accept rate of an unassisted pass, and the agreement rate reported there "
            f"bounds the effect rather than measuring it. {reread_clause(facts)} No suggestion "
            "reached the review log: they are a separate append-only file "
            f"(`{rel_to_repo(facts.src.triage)}`) whose rows carry neither a `verdict` nor a "
            "`ts`, which is what makes the two unmistakable for each other."
        ),
    ]


def test_intervals(facts: Facts) -> str:
    """Each bucket's test half and the widest interval an accuracy on it can have.

    The widest, because the point is the size of the set and not the result: at p = 0.5 the
    Wilson interval is at its widest, so this is the best case for a reader wondering how much a
    per-bucket percentage from Day 4 can be trusted.
    """
    halves = facts.sidecar.get("buckets", {})
    parts = []
    for bucket in BUCKETS:
        n = halves.get(bucket, {}).get("test", 0)
        parts.append(f"`{bucket}` {n} rows {interval_width(n)}")
    return ", ".join(parts)


def out_of_reach(coverage: float) -> float:
    """The first whole percent above the coverage ceiling -- #61's :func:`split.ceiling_line`.

    The next percent *above* the ceiling, not the ceiling rounded up: at n=99 the latter would
    claim 100% was the first thing out of reach when 99% already is.
    """
    return (math.floor(round(coverage * 100, 6)) + 1) / 100


def interval_width(n: int) -> str:
    """Half the width of a 95% Wilson interval at p = 0.5 over ``n`` rows, in points."""
    if n < 1:
        return MISSING
    lo, hi = agreement.wilson(n // 2, n)
    return f"±{100 * (hi - lo) / 2:.0f} points"


# --- the report -----------------------------------------------------------------------------------


#: The last section. Says what the file is a function of, so that a reader who finds a diff under
#: ``report/`` knows it means an artifact moved and never that time passed.
REPRODUCTION = """\
## Reproduction

```bash
uv run python -m eval.scripts.day3_report
```

This file is a function of the dataset, its sidecar, the draft files and theirs, the review log
and the returned sheet — and of `config.yaml`, which names it. There is no timestamp and no duration
anywhere in it, so a re-run over unchanged inputs rewrites it byte for byte and a diff under
`report/` is one of those artifacts having changed. The digests in the header are what pins that:
the dataset's is #61's sidecar, and the guidelines' is the version the second annotator was
handed.
"""


def render_report(facts: Facts) -> str:
    """The whole report: what it is, the four bodies of evidence, then what is wrong with it."""
    sections = [
        render_header(facts),
        render_composition(facts),
        render_provenance(facts),
        render_verification(facts),
        render_agreement(facts),
        render_split(facts),
        render_limitations(facts),
        REPRODUCTION.rstrip(),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def summary(facts: Facts, path: Path) -> str:
    """What the terminal gets: the headline, stripped of its markdown, and where the file went."""
    counts = Counter(verdict.verdict for verdict in facts.verdicts.values())
    return "\n".join(
        [
            f"config hash {facts.cfg.hash}",
            f"drafted                {len(facts.candidates)}",
            (
                f"reviewed               {len(facts.verdicts)}  ({counts['accept']} accept, "
                f"{counts['edit']} edit, {counts['delete']} delete)"
            ),
            f"dataset                {len(facts.rows)} rows  "
            + ", ".join(
                f"{facts.sidecar.get('totals', {}).get(half, 0)} {half}" for half in SPLITS
            ),
            f"second annotator       {len(facts.judged)} rows",
            f"wrote                  {rel_to_repo(path)}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.day3_report",
        description="Write report/day3_evalset_{hash}.md: Day 3's dataset, its provenance, the "
        "verification pass over it, the second annotator's numbers, the frozen split, and the "
        "limitations of all of it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="list the artifacts the report is built from and what produces each; write nothing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the report instead of writing it"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"the frozen dataset (default: {rel_to_repo(review.DATASET_PATH)})",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"the review log (default: {rel_to_repo(review.REVIEW_LOG_PATH)})",
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    src = sources(dataset=args.dataset, log=args.log)
    if args.check:
        return render_check(cfg, src, sys.stdout)
    gaps = missing(src)
    if gaps:
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
