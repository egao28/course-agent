"""One cause per failed test row, by a fixed rule order, confirmed by a person (#105).

    uv run python -m eval.scripts.attribute                              # the proposals, no writes
    uv run python -m eval.scripts.attribute --review --reviewer NAME     # resumable
    uv run python -m eval.scripts.attribute --stats
    uv run python -m eval.scripts.attribute --table   # refuses until every row has a final class

"Failed" means end-to-end incorrect per #104, which covers both false releases and false blocks.
The failures are not counted here a second time: :func:`eval.scripts.e2e_metrics.correct` decides,
and :func:`check_against_metrics` then holds the set against the committed
``eval/runs/e2e_metrics_{config_hash}.json`` and refuses if the two disagree by one row. #104's
headline reading is the verdict-backed one, so that is the one attributed (:data:`READING`); the
printout names the rows the exit-only reading would add.

**The four classes, and the order they are checked in.** A row can fit more than one class, so the
first rule that fires wins. Without a fixed order, the table would just reflect the annotator's
taste. The four are :data:`RULES`, and this is :func:`rules_text`, word for word -- the report
prints the same function rather than retyping them, and a test holds this paragraph to it::

    1. Intent misjudged: the predicted `intent` is not in `gold_intents`.
    2. Retrieval failure: the row has `gold_doc_ids`, and none of them is in `evidence_doc_ids`.
    3. Evidence misread: the gold evidence was in the prompt, and the answer is still wrong (a
       wrong fact, a dropped requirement, a wrong citation).
    4. Overconfident: answered when it should have declined or asked (a false release), with the
       intent right and nothing to misread.

For hybrid rows, #78's log says where dense and bm25 ranked the gold, so a rule-2 miss can be
traced to one source; a tool row's miss has no ranks behind it, and the review screen says which
kind it is looking at.

Anything left over -- e.g. a false block where the gold was retrieved but ``top_score`` fell under
the threshold -- goes in a printed **other** row with its ids. It is never folded into one of the
four. Rule 2 does not apply to unanswerable/ambiguous rows, since they have no gold ids.

**Rules 1 and 2 are mechanical; 3 and 4 are proposals.** The first two read two fields and a set
membership, and :func:`propose` marks them ``mechanical``. The last two are where a person is
actually needed -- whether an answer misread evidence it was given, or answered a question it
should have declined, is a reading of the text. A person confirms or overrides **every** row, and
an override needs a one-line reason. Overriding a mechanical class is allowed and counted
separately: it means the rule's inputs are being disputed, which is worth seeing in the log rather
than forbidding.

**The two alphabets.** ``gold_doc_ids`` are chunk ids (``2026-sp:ENGL-250``) and a tool's
``evidence_doc_ids`` are course codes (``ENGL 250``) -- #103 records ``evidence_source`` so rule 2
does not compare the two as strings. :func:`same_course` compares two chunk ids whole, term and
all, and falls back to the shared code matcher (:func:`agent.tools.courses.normalize_code`) only
when one side carries no term. A program id from ``check_requirement`` reads as no course and so
matches no gold id, which is the right answer for it.

**Nothing here writes a class a person did not accept.** :func:`write_rows` is called only from
the review loop, one row at a time; :func:`table` refuses to print until every failure has a
``final``. A distribution table with a machine-filled cell in it is not one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import retrieval
from agent.tools.courses import normalize_code
from config import Config, file_sha256, load_config, rel_to_repo
from eval import intents as gold_intents
from eval.schema import BUCKETS, EvalItem
from eval.scripts import e2e_metrics
from eval.scripts.draft_items import write_json
from eval.scripts.e2e import RUNS_DIR, run_path
from eval.scripts.e2e_labels import csv_path
from eval.scripts.review import DATASET_PATH, Ask, indent, reader
from eval.scripts.split import META_SUFFIX, load_split

#: The half every final number is reported on, and the only one with gold intents (#102, #104).
SPLIT = e2e_metrics.SPLIT

#: Which of #104's two readings of an abstaining row that answered decides what "failed" means.
#: The verdict-backed one is #104's headline; :func:`exit_only_extra` names what the other adds.
READING = "verdict_backed"

#: The four causes, in the order :func:`propose` checks them. ``other`` is not one of them.
CLASSES: tuple[str, ...] = (
    "intent_misjudged",
    "retrieval_failure",
    "evidence_misread",
    "overconfident",
)

#: The printed row for anything the four do not cover. Never folded into one of them.
OTHER = "other"

#: What a ``final`` may be.
FINALS: tuple[str, ...] = (*CLASSES, OTHER)

#: Every key a line of the attribution file carries. The issue's five, plus four that say where
#: the proposal came from and what the reviewer did with it: ``bucket`` (the table is split by
#: it), ``rule`` (the number of the rule the proposal came from -- 1 or 2 decided it, 3 or 4
#: proposed it, and ``null`` is the ``other`` row where none fired), ``action`` (derived from
#: proposed against final, so an override can be counted from this file alone) and ``ts``.
KEYS: tuple[str, ...] = (
    "id",
    "bucket",
    "proposed",
    "rule",
    "final",
    "reason",
    "reviewer",
    "action",
    "ts",
)

RULE_LINE = "-" * 96


class Rule(NamedTuple):
    """One cause: its number in the order, the name the file records, and the issue's sentence."""

    number: int
    name: str
    text: str
    mechanical: bool


#: The rules, as data. The docstring above quotes these sentences and a test checks that it still
#: does; #106 renders them with :func:`rules_text` rather than retyping them.
RULES: tuple[Rule, ...] = (
    Rule(
        1,
        "intent_misjudged",
        "Intent misjudged: the predicted `intent` is not in `gold_intents`.",
        mechanical=True,
    ),
    Rule(
        2,
        "retrieval_failure",
        (
            "Retrieval failure: the row has `gold_doc_ids`, and none of them is in "
            "`evidence_doc_ids`."
        ),
        mechanical=True,
    ),
    Rule(
        3,
        "evidence_misread",
        (
            "Evidence misread: the gold evidence was in the prompt, and the answer is still wrong "
            "(a wrong fact, a dropped requirement, a wrong citation)."
        ),
        mechanical=False,
    ),
    Rule(
        4,
        "overconfident",
        (
            "Overconfident: answered when it should have declined or asked (a false release), "
            "with the intent right and nothing to misread."
        ),
        mechanical=False,
    ),
)

RULE_BY_NAME = {rule.name: rule for rule in RULES}


def rules_text() -> str:
    """The four rules and their order, numbered -- what the report prints word for word."""
    return "\n".join(f"{rule.number}. {rule.text}" for rule in RULES)


def now() -> str:
    """The timestamp a decision carries. Seam for the tests, which must not race a clock."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- the two alphabets ----------------------------------------------------------------------------


def course_key(doc_id: str) -> str | None:
    """``2026-sp:ENGL-250`` and ``ENGL 250`` both read as ``ENGL 250``; anything else is ``None``.

    The term is dropped and the chunk id's hyphen becomes the space the code matcher expects, so
    the comparison runs through :func:`agent.tools.courses.normalize_code` -- the same matcher the
    BM25 index and the dispatch node read codes with -- rather than a third spelling rule here.
    A program id (``gened``, a major name) is not a code and comes back ``None``.
    """
    bare = doc_id.split(":", 1)[-1]
    return normalize_code(bare.replace("-", " "))


def same_course(gold: str, evidence: str) -> bool:
    """Do a gold id and an evidence id name the same course?

    Two chunk ids are compared whole, term and all: dropping the term would let a gold id from one
    term match evidence from another, and this corpus being single-term today is not a reason to
    write a comparison that stops being right when it is not. The code matcher is the fallback for
    the mixed case, which is the one rule 2 actually meets -- a tool row's evidence is a course
    code with no term in it.
    """
    if ":" in gold and ":" in evidence:
        return gold == evidence
    key = course_key(gold)
    return key is not None and key == course_key(evidence)


def gold_in_evidence(item: EvalItem, line: Mapping[str, Any]) -> list[str]:
    """The gold ids this row's evidence carried, in dataset order. Empty is what rule 2 reads."""
    evidence = line["evidence_doc_ids"] or []
    return [g for g in item.gold_doc_ids if any(same_course(g, e) for e in evidence)]


# --- one failed row -------------------------------------------------------------------------------


class Failure(NamedTuple):
    """One end-to-end failure: #104's joined row, the run line it came from, and its gold."""

    row: e2e_metrics.Row
    line: Mapping[str, Any]
    item: EvalItem

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def bucket(self) -> str:
        return self.row.bucket


class Proposal(NamedTuple):
    """What the rules make of a row before anyone reads it."""

    cls: str
    rule: int | None
    basis: str

    @property
    def mechanical(self) -> bool:
        """Did a rule decide this, or is it a proposal? Only rules 1 and 2 decide."""
        return self.cls in RULE_BY_NAME and RULE_BY_NAME[self.cls].mechanical


def propose(failure: Failure) -> Proposal:
    """The first rule that fires, in order, or :data:`OTHER` with what stopped each of them.

    The order is the module docstring's, and this is the only place it is written down as code.
    Rules 1 and 2 decide; 3 and 4 propose. ``basis`` is the fields the decision was made from, so
    the reviewer is shown what fired rather than only which class it produced.
    """
    row, item, line = failure.row, failure.item, failure.line
    if not e2e_metrics.intent_hit(row):
        return Proposal(
            "intent_misjudged",
            1,
            f"router said {row.intent}, gold is {'/'.join(sorted(row.gold_intents))}",
        )
    found = gold_in_evidence(item, line)
    if item.gold_doc_ids and not found:
        gold = ", ".join(item.gold_doc_ids)
        evidence = line["evidence_doc_ids"] or []
        return Proposal(
            "retrieval_failure",
            2,
            (
                f"no evidence at all -- nothing retrieved and no tool result -- so none of {gold}"
                if not evidence
                else f"none of {gold} in {evidence} (from {line['evidence_source']})"
            ),
        )
    if e2e_metrics.answered(row) and item.gold_doc_ids:
        unknown = line["citations_unknown"] or []
        cited = f"; cited {', '.join(unknown)}, which this run cannot show" if unknown else ""
        return Proposal(
            "evidence_misread",
            3,
            f"answered with {', '.join(found)} in the prompt and the verdict is incorrect{cited}",
        )
    if e2e_metrics.false_release(row, READING):
        return Proposal(
            "overconfident",
            4,
            f"{row.bucket} row answered through {row.exit}, no gold ids to misread",
        )
    return Proposal(
        OTHER,
        None,
        (
            f"intent in gold, {len(gold_in_evidence(item, line))} of {len(item.gold_doc_ids)} gold "
            f"ids in the evidence, exit {row.exit}, gate_ok {line['gate_ok']}, "
            f"top_score {line['top_score']}"
        ),
    )


def action_of(proposed: str, final: str, mechanical: bool) -> str:
    """What the reviewer did. An override of a mechanical class is its own word, not a flavour."""
    if proposed == final:
        return "confirmed"
    return "overrode_rule" if mechanical else "overrode_proposal"


#: What :func:`action_of` can return, for the counters and for the tests.
ACTIONS: tuple[str, ...] = ("confirmed", "overrode_proposal", "overrode_rule")


# --- the inputs -----------------------------------------------------------------------------------


class Joined(NamedTuple):
    """Every test row of #103's run, joined and checked once. Both readings are cut from it."""

    rows: list[e2e_metrics.Row]
    lines: dict[str, dict[str, Any]]
    items: dict[str, EvalItem]


def load_joined(
    cfg: Config,
    runs_dir: Path = RUNS_DIR,
    dataset: Path = DATASET_PATH,
    sidecar: Path | None = None,
    guidelines: Path | None = None,
) -> Joined:
    """#103's run, #103's verdicts and #102's intents, joined by #104's own reader.

    Every check on those three files is :mod:`eval.scripts.e2e_metrics`'s, imported: a run that is
    not what its sidecar describes, a CSV that labels another copy of it, or a gold-intent file
    that does not pass ``python -m eval.intents`` is refused before a single row is attributed.
    """
    sidecar = gold_intents.SIDECAR_PATH if sidecar is None else Path(sidecar)
    guidelines = gold_intents.GUIDELINES_PATH if guidelines is None else Path(guidelines)
    lines, _ = e2e_metrics.read_run(cfg, runs_dir)
    verdicts, _ = e2e_metrics.read_verdicts(cfg, runs_dir, run_path(SPLIT, cfg, runs_dir))
    labels, _ = e2e_metrics.read_intents(sidecar, Path(dataset), guidelines)
    items = {item.id: item for item in load_split(SPLIT, dataset)}
    rows = e2e_metrics.join(
        lines, {i: item.should_abstain for i, item in items.items()}, labels, verdicts
    )
    return Joined(rows, {line["id"]: line for line in lines}, items)


def failures_of(joined: Joined, reading: str = READING) -> list[Failure]:
    """The rows #104 counts as end-to-end incorrect, in run order."""
    return [
        Failure(row, joined.lines[row.id], joined.items[row.id])
        for row in joined.rows
        if not e2e_metrics.correct(row, reading)
    ]


def exit_only_extra(joined: Joined) -> list[str]:
    """Rows the exit-only reading would add to the failures. Reported, never attributed.

    #104 reports correctness two ways because there is no clarify exit. This file attributes one
    of them, and a reader is owed the other one's ids rather than a footnote that it differs.
    """
    attributed = {f.id for f in failures_of(joined)}
    return [f.id for f in failures_of(joined, "exit_only") if f.id not in attributed]


def check_against_metrics(
    failures: Sequence[Failure], cfg: Config, runs_dir: Path = RUNS_DIR
) -> dict[str, Any]:
    """Hold the failure set against #104's committed JSON: the issue's acceptance, as a check.

    The two are computed from the same run by the same function, so a difference means one of the
    inputs moved under the other -- which is exactly the state in which a distribution table would
    be quietly wrong about what it is a distribution over.
    """
    path = e2e_metrics.result_path(cfg, runs_dir)
    if not path.exists():
        raise ValueError(
            f"no end-to-end metrics at {rel_to_repo(path)}: run "
            "`python -m eval.scripts.e2e_metrics` first -- the failures are its rows, and this "
            "file checks its count against them"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    block = result["metrics"]["answer_correctness"][READING]
    expected = [i for i in block["denominator_ids"] if i not in set(block["numerator_ids"])]
    found = [f.id for f in failures]
    if sorted(expected) != sorted(found):
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise ValueError(
            f"{rel_to_repo(path)} and this run disagree about which rows failed: "
            f"{len(missing)} in the metrics and not here ({missing[:5]}), {len(extra)} here and "
            f"not in the metrics ({extra[:5]}). Re-run eval.scripts.e2e_metrics"
        )
    return {
        "metrics": rel_to_repo(path),
        "metrics_sha256": file_sha256(path),
        "n_rows": block["n"],
        "n_correct": block["k"],
        "n_failures": block["n"] - block["k"],
    }


# --- the file -------------------------------------------------------------------------------------


def attribution_path(cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/attribution_{split}_{config_hash}.jsonl`` -- one per end-to-end run."""
    return Path(runs_dir) / f"attribution_{SPLIT}_{cfg.hash}.jsonl"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The attribution file as written, refusing a line that is not a person's decision."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for number, text in enumerate(f, start=1):
            if not text.strip():
                continue
            row = json.loads(text)
            missing = [key for key in KEYS if key not in row]
            if missing:
                raise ValueError(
                    f"{rel_to_repo(path)}:{number} is missing {', '.join(missing)}; an "
                    f"attribution line carries all {len(KEYS)} keys"
                )
            if row["final"] not in FINALS or not row["reviewer"]:
                raise ValueError(
                    f"{rel_to_repo(path)}:{number} has no known class or no reviewer; every line "
                    "is a person's decision"
                )
            if row["final"] != row["proposed"] and not (row["reason"] or "").strip():
                raise ValueError(
                    f"{rel_to_repo(path)}:{number} overrides {row['proposed']} with "
                    f"{row['final']} and gives no reason; an override carries one"
                )
            rows.append(row)
    return rows


def current(
    path: Path, failures: Sequence[Failure], proposals: Mapping[str, Proposal]
) -> dict[str, dict[str, Any]]:
    """The decisions that still stand: same rows, same proposals. Anything else is refused.

    A stored line whose proposal has changed was made about a different reading of the run, and
    keeping it would put a verdict from one state of the tree into a table about another.
    """
    stored = read_rows(path)
    known = {f.id for f in failures}
    problems = [f"{row['id']} is not a failure of this run" for row in stored if
                row["id"] not in known]  # fmt: skip
    problems += [
        f"{row['id']} was reviewed against a proposal of {row['proposed']}, which is now "
        f"{proposals[row['id']].cls}"
        for row in stored
        if row["id"] in known and row["proposed"] != proposals[row["id"]].cls
    ]
    if problems:
        raise ValueError(
            f"{rel_to_repo(path)} does not describe this run: "
            + "; ".join(problems[:5])
            + (" ..." if len(problems) > 5 else "")
            + ". Delete it and review again, or restore the run it was made on"
        )
    return {row["id"]: row for row in stored}


def write_rows(
    path: Path,
    decided: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Failure],
    meta: Mapping[str, Any],
) -> Path:
    """The file, in failure order, replaced atomically and fsynced before the next row is shown.

    Rewritten rather than appended to: it is a table of one line per failed row, not a log, and
    twenty lines are cheaper to rewrite than a second file would be to keep in step. The same
    tmp-then-replace the CSV writers use, so a closed terminal loses at most the row on screen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [decided[f.id] for f in failures if f.id in decided]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in body:
            f.write(json.dumps({key: row[key] for key in KEYS}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    write_json(
        path.with_suffix(META_SUFFIX),
        {
            **meta,
            "attribution": rel_to_repo(path),
            "sha256": file_sha256(path),
            "n_attributed": len(body),
            "complete": len(body) == len(failures),
            "reviewers": dict(sorted(Counter(row["reviewer"] for row in body).items())),
            "actions": {a: sum(row["action"] == a for row in body) for a in ACTIONS},
            "by_class": {c: sum(row["final"] == c for row in body) for c in FINALS},
        },
    )
    return path


def sources(cfg: Config, runs_dir: Path, dataset: Path, sidecar: Path) -> dict[str, Any]:
    """Every file the attribution was made from, by path and digest."""
    run = run_path(SPLIT, cfg, runs_dir)
    csv = csv_path(SPLIT, cfg, runs_dir)
    return {
        "split": SPLIT,
        "config_hash": cfg.hash,
        "reading": READING,
        "run": rel_to_repo(run),
        "run_sha256": file_sha256(run),
        "csv": rel_to_repo(csv),
        "csv_sha256": file_sha256(csv),
        "intents": rel_to_repo(Path(sidecar)),
        "intents_sha256": file_sha256(Path(sidecar)),
        "dataset": rel_to_repo(Path(dataset)),
        "dataset_sha256": file_sha256(Path(dataset)),
        "rules": [rule.text for rule in RULES],
    }


# --- the screen -----------------------------------------------------------------------------------


def render_failure(
    failure: Failure, proposal: Proposal, position: int, total: int, seen: Mapping[str, Any] | None
) -> str:
    """One row as the reviewer sees it: what was asked, what came back, and what fired."""
    row, line, item = failure.row, failure.line, failure.item
    verdict = {True: "correct", False: "incorrect", None: "not labelled (it never answered)"}
    gold_label = "clarifying question" if row.bucket == "ambiguous" else "reference"
    lines = [
        RULE_LINE,
        f"[{position}/{total}] {failure.id}   bucket: {row.bucket}   exit: {row.exit}",
        "",
        "question:",
        indent(line["question"]),
        "",
        f"gold ({gold_label}):",
        indent(item.gold_answer or "(empty)"),
        "",
        "system answer:",
        indent(line["answer"] or "(none)"),
    ]
    if line["draft"]:
        lines += [
            "",
            f" !  the citation check discarded this draft (unknown: {line['citations_unknown']}):",
            indent(line["draft"]),
        ]
    lines += [
        "",
        f"intent: {row.intent}   gold intents: {'/'.join(sorted(row.gold_intents))}",
        f"gold ids: {list(item.gold_doc_ids) or '(none)'}",
        (
            f"evidence: {line['evidence_doc_ids'] or '(none)'} from {line['evidence_source']}"
            f"   top_score: {line['top_score']}   gate_ok: {line['gate_ok']}"
        ),
        f"human verdict on the answer: {verdict[row.verdict]}",
    ]
    if proposal.cls == "retrieval_failure" and line["evidence_source"] in retrieval.MODES:
        lines.append(
            " .  a retrieval-sourced miss: #78's hybrid_failures log has the dense and bm25 "
            "ranks of the gold for this row. A tool row's miss has no ranks behind it"
        )
    where = f"rule {proposal.rule}" if proposal.rule else "no rule fired"
    lines += [
        "",
        f"proposed: {proposal.cls}  ({where}"
        + (", mechanical" if proposal.mechanical else ", a proposal")
        + ")",
        indent(proposal.basis),
    ]
    if seen is not None:
        lines.append(
            f"your earlier call: {seen['final']} ({seen['action']}, {seen['reviewer']})"
            + (f" -- {seen['reason']}" if seen["reason"] else "")
        )
    return "\n".join(lines)


def prompt_for(proposal: Proposal) -> str:
    keys = "  ".join(f"[{rule.number}] {rule.name}" for rule in RULES)
    return f"class?  [c]onfirm {proposal.cls}  {keys}  [o]ther  [q]uit > "


def choice_of(answer: str) -> str | None:
    """A keypress as a class name. ``c`` is resolved by the caller, which knows the proposal."""
    choice = answer.strip().lower()
    if choice in {str(rule.number) for rule in RULES}:
        return RULES[int(choice) - 1].name
    if choice in {"o", "other"}:
        return OTHER
    return None


def ask_class(
    proposal: Proposal, out: TextIO, ask: Ask
) -> tuple[str, str | None] | None:
    """``(final, reason)``, or ``None`` to stop. An override is not taken without a reason."""
    while True:
        answer = ask(prompt_for(proposal))
        if answer is None:
            return None
        text = answer.strip().lower()
        if text in {"q", "quit"}:
            return None
        if text in {"c", "confirm"}:
            return proposal.cls, None
        final = choice_of(answer)
        if final is None:
            out.write("c, 1-4, o or q.\n")
            continue
        if final == proposal.cls:
            return final, None
        while True:
            reason = ask(f"overriding {proposal.cls} with {final} -- one line on why > ")
            if reason is None:
                return None
            if reason.strip():
                return final, reason.strip()
            out.write("an override needs a reason: the rule order is what it is arguing with.\n")


def review(
    failures: Sequence[Failure],
    proposals: Mapping[str, Proposal],
    decided: dict[str, dict[str, Any]],
    reviewer: str,
    path: Path,
    meta: Mapping[str, Any],
    out: TextIO,
    ask: Ask,
    redo: bool = False,
) -> Counter[str]:
    """Walk the failures, writing the file before the next row. Returns this sitting's actions."""
    queue = [f for f in failures if redo or f.id not in decided]
    made: Counter[str] = Counter()
    for position, failure in enumerate(queue, start=1):
        proposal = proposals[failure.id]
        out.write(
            render_failure(failure, proposal, position, len(queue), decided.get(failure.id))
            + "\n\n"
        )
        answer = ask_class(proposal, out, ask)
        if answer is None:
            break
        final, reason = answer
        decided[failure.id] = {
            "id": failure.id,
            "bucket": failure.bucket,
            "proposed": proposal.cls,
            "rule": proposal.rule,
            "final": final,
            "reason": reason,
            "reviewer": reviewer,
            "action": action_of(proposal.cls, final, proposal.mechanical),
            "ts": now(),
        }
        write_rows(path, decided, failures, meta)
        made[decided[failure.id]["action"]] += 1
    return made


# --- the table ------------------------------------------------------------------------------------


def distribution(
    failures: Sequence[Failure], decided: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Per class: the count, the share of the failures, the split per bucket, and the ids."""
    buckets = [b for b in BUCKETS if any(f.bucket == b for f in failures)]
    total = len(failures)
    rows: dict[str, Any] = {}
    for cls in FINALS:
        ids = [f.id for f in failures if decided[f.id]["final"] == cls]
        chosen = set(ids)
        rows[cls] = {
            "n": len(ids),
            "share": len(ids) / total if total else None,
            "by_bucket": {b: sum(1 for f in failures if f.bucket == b and f.id in chosen)
                          for b in buckets},
            "ids": ids,
        }
    return {"n_failures": total, "buckets": buckets, "classes": rows}


def render_table(result: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    """The distribution table #106 quotes: four classes, an ``other`` row, counts and shares."""
    buckets = result["buckets"]
    head = " | ".join(buckets)
    out = [
        (
            f"{result['n_failures']} failed rows of {sources['n_rows']} on the "
            f"{sources['split']} half ({sources['n_rows']} - {sources['n_correct']} correct, "
            f"#104, {sources['reading']})"
        ),
        "",
        f"| class | n | share | {head} |",
        "|---|---|---|" + "---|" * len(buckets),
    ]
    for cls in FINALS:
        block = result["classes"][cls]
        cells = " | ".join(str(block["by_bucket"][b]) for b in buckets)
        share = "-" if block["share"] is None else f"{block['share']:.2f}"
        out.append(f"| {cls} | {block['n']} | {share} | {cells} |")
    out += ["", "the rows behind each class:"]
    for cls in FINALS:
        ids = result["classes"][cls]["ids"]
        out.append(f"  {cls}: {', '.join(ids) if ids else '(none)'}")
    other = result["classes"][OTHER]
    if other["n"]:
        out += [
            "",
            (
                f"other ({other['n']}): {', '.join(other['ids'])} -- printed, never folded into "
                "one of the four"
            ),
        ]
    return "\n".join(out)


def table(
    failures: Sequence[Failure], decided: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The distribution, or a refusal naming the rows no person has attributed."""
    missing = [f.id for f in failures if f.id not in decided]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(
            f"refusing to print a table: {len(missing)} of {len(failures)} failures have no "
            f"confirmed class ({shown}). `final` comes only from a person -- finish the pass with "
            "--review --reviewer NAME"
        )
    return distribution(failures, decided)


# --- saying where the pass stands -------------------------------------------------------------


def render_proposals(
    failures: Sequence[Failure],
    proposals: Mapping[str, Proposal],
    decided: Mapping[str, Mapping[str, Any]],
    extra: Sequence[str] = (),
) -> str:
    """What the rules propose, and what a person has done with it so far. Writes nothing."""
    counts = Counter(proposals[f.id].cls for f in failures)
    out = [
        f"{len(failures)} failed rows, {len(decided)} attributed by a person",
        "",
        "the rules, in the order they are checked:",
        indent(rules_text()),
        "",
        "proposed:",
    ]
    for cls in FINALS:
        rule = RULE_BY_NAME.get(cls)
        how = "mechanical" if rule and rule.mechanical else "proposed" if rule else "no rule fired"
        out.append(f"  {cls:<18} {counts.get(cls, 0):>3}   ({how})")
    out += ["", f"{'id':<10}{'bucket':<14}{'proposed':<20}{'rule':<6}final"]
    for failure in failures:
        proposal = proposals[failure.id]
        row = decided.get(failure.id)
        final = "-" if row is None else f"{row['final']} ({row['action']}, {row['reviewer']})"
        out.append(
            f"{failure.id:<10}{failure.bucket:<14}{proposal.cls:<20}"
            f"{proposal.rule or '-'!s:<6}{final}"
        )
    if extra:
        out += [
            "",
            (
                f"#104's exit-only reading would add {len(extra)} more failure(s) "
                f"({', '.join(extra)}); this file attributes the {READING} reading, which is the "
                "headline one"
            ),
        ]
    return "\n".join(out)


# --- the command ----------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    input_fn: Callable[[str], str] = input,
    runs_dir: Path = RUNS_DIR,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", action="store_true", help="attribute the rows, one by one")
    parser.add_argument("--reviewer", help="your name, recorded on every row you decide")
    parser.add_argument("--redo", action="store_true", help="show rows already decided, too")
    parser.add_argument("--stats", action="store_true", help="what the rules propose, and progress")
    parser.add_argument("--table", action="store_true", help="the distribution table, once it can")
    parser.add_argument("--runs-dir", type=Path, default=runs_dir, help="where the runs live")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="the frozen dataset")
    parser.add_argument("--intents", type=Path, default=gold_intents.SIDECAR_PATH,
                        help="#102's gold intents for the test half")  # fmt: skip
    parser.add_argument("--guidelines", type=Path, default=gold_intents.GUIDELINES_PATH,
                        help="the rules the intents were labelled under; its digest is checked")
    args = parser.parse_args(argv)
    cfg = load_config()
    try:
        joined = load_joined(cfg, args.runs_dir, args.dataset, args.intents, args.guidelines)
        failures = failures_of(joined)
        checked = check_against_metrics(failures, cfg, args.runs_dir)
        proposals = {f.id: propose(f) for f in failures}
        path = attribution_path(cfg, args.runs_dir)
        decided = current(path, failures, proposals)
        meta = {
            **sources(cfg, args.runs_dir, args.dataset, args.intents),
            **checked,
        }
        if args.table:
            print(render_table(table(failures, decided), meta), file=out)
            return 0
        if args.review:
            if not args.reviewer:
                raise ValueError("--review needs --reviewer NAME: every class carries who chose it")
            made = review(
                failures, proposals, decided, args.reviewer, path, meta,
                out, reader(out, input_fn), args.redo,
            )  # fmt: skip
            print(f"\n{sum(made.values())} row(s) this sitting: {dict(made)}", file=out)
            print(f"{len(decided)} of {len(failures)} attributed -> {rel_to_repo(path)}", file=out)
            return 0
        print(render_proposals(failures, proposals, decided, exit_only_extra(joined)), file=out)
        if not args.stats:
            print(
                "\nnothing was written. --review --reviewer NAME to attribute them, "
                "--table once every row has a class.",
                file=out,
            )
        return 0
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
