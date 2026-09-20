"""Day 2's acceptance, as a command: ten handwritten queries, three exits, one report.

    uv run python -m eval.scripts.day2_smoke --fake-model      # what can run today
    uv run --env-file .env python -m eval.scripts.day2_smoke   # once the account has credits

Every query in ``eval/datasets/day2_queries.jsonl`` goes through :func:`agent.graph.run` -- the
real dispatch table, the real tools, the real retriever, the real gate -- and the run is written
up in ``report/day2_smoke_{hash}.md``, one section per query, with a count of the three exits at
the top. The milestone's acceptance is that all three appear with a non-zero count, so this script
*checks* that rather than only tabulating it: a run that never refused, or never answered, exits
non-zero with the report still written, because the report is the evidence.

**``expected_intent`` and ``expected_exit`` are what a row should do, not what it does today.** A
disagreement is recorded as one -- in the table, and again in a section with the numbers that
caused it -- and never repaired by editing the dataset. Two of the ten already say something worth
keeping: one expects an answer only a dense index can produce, and one was written when ``majors``
was empty and now refuses one step earlier than the issue predicted.

**``--fake-model`` is the honest version of "run it with no credits".** It forces each row's label
from ``expected_intent`` and puts :class:`llm.StandInChat` behind the answer node, exactly as
``python -m agent --intent ... --fake-model`` does -- the overrides come from the CLI's own
:func:`agent.__main__.build_overrides`, so a row here and that command are the same run. The
report says so in its header, in a section of its own, and beside every answer: the wording of an
answer, and whether a model stays inside its evidence, are *not* measured by it.

**The report carries no timestamp and no latency**, so nothing in it moves unless the run did:
under ``--fake-model`` a re-run rewrites a byte-identical file, and a live re-run differs exactly
as much as the model answered differently (``temperature: 0`` is not determinism -- two
consecutive live runs of the ten queries disagreed on one row, once answering it and once
drafting an answer that cited nothing and was discarded). A diff under ``report/`` is therefore a
change in behaviour, never a change in the clock. What the run was is in the name (the config
hash) and in the header.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import retrieval
from agent.__main__ import build_overrides
from agent.graph import run as run_through_graph
from agent.nodes import Node
from agent.prompts import ANSWER_PROMPT_VERSION, ROUTER_PROMPT_VERSION
from agent.state import EXITS, INTENTS, AgentState, Exit, Intent
from agent.trace import evidence_rows, gate_note, gathered, intent_note, signals
from config import (
    REPO_ROOT,
    Config,
    bm25_path,
    chroma_path,
    collection_name,
    load_config,
    rel_to_repo,
    report_path,
)
from ingest.chunk import chunk_key

#: Where the ten queries live. A path rather than a list in this module: the set is an artifact of
#: its own, Day 4's ablation set will be a different one, and a dataset in a file can be diffed.
DATASET = REPO_ROOT / "eval" / "datasets" / "day2_queries.jsonl"

#: The report's stem; :func:`config.report_path` adds the config hash and the extension.
STEM = "day2_smoke"

#: Every field a dataset row must have, and nothing else. An unexpected key is refused rather than
#: ignored -- a row carrying ``expected_answer`` would be a column nothing reads.
FIELDS = ("query", "expected_intent", "expected_exit", "note")

#: How a table cell says "there is no such number for this row".
MISSING = "—"

#: How much of an evidence block's first line the report's tables show. Wider than the terminal
#: trace's, since a markdown table wraps instead of running off the edge.
LINE_WIDTH = 64


class Query(NamedTuple):
    """One dataset row: the question, what it should do, and why it is in the set."""

    query: str
    expected_intent: Intent
    expected_exit: Exit
    note: str


class Run(NamedTuple):
    """One query's run: the row it came from, and the final state the graph returned."""

    n: int
    row: Query
    state: AgentState

    @property
    def intent_ok(self) -> bool:
        return self.state.get("intent") == self.row.expected_intent

    @property
    def exit_ok(self) -> bool:
        return self.state.get("exit") == self.row.expected_exit


# --- the dataset ----------------------------------------------------------------------------------


def load_queries(path: str | Path = DATASET) -> list[Query]:
    """The dataset as :class:`Query` rows, with every field checked and the line named.

    Coverage -- all five intents, all three exits -- is *not* checked here. This function has to
    load a two-row dataset for a test as happily as the ten-row one, and what the shipped set has
    to cover is a property of that set: ``tests/test_day2_smoke.py`` asserts it.
    """
    path = Path(path)
    rows: list[Query] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{rel_to_repo(path)} line {n}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where} is not JSON: {error}") from None
        rows.append(check_row(raw, where))
    if not rows:
        raise ValueError(f"{rel_to_repo(path)} has no queries in it")
    return rows


def check_row(raw: Any, where: str) -> Query:
    """One row, validated into a :class:`Query`, or a sentence naming the line and the field."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{where} is a {type(raw).__name__}; every row is a JSON object")
    missing = [field for field in FIELDS if field not in raw]
    unknown = [field for field in raw if field not in FIELDS]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"is missing {', '.join(missing)}")
        if unknown:
            parts.append(f"carries {', '.join(map(repr, unknown))}, which nothing reads")
        raise ValueError(f"{where} {' and '.join(parts)}; a row is {{{', '.join(FIELDS)}}}")
    for field in FIELDS:
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"{where} has {field}={raw[field]!r}; every field is a non-empty str")
    if raw["expected_intent"] not in INTENTS:
        raise ValueError(
            f"{where} expects intent {raw['expected_intent']!r}; the five labels are "
            f"{', '.join(INTENTS)} (agent/state.py)"
        )
    if raw["expected_exit"] not in EXITS:
        raise ValueError(
            f"{where} expects exit {raw['expected_exit']!r}; the three exits are "
            f"{', '.join(EXITS)} (agent/state.py)"
        )
    return Query(**{field: raw[field].strip() for field in FIELDS})  # type: ignore[arg-type]


# --- running them ---------------------------------------------------------------------------------


def run_queries(
    cfg: Config,
    rows: Sequence[Query],
    fake_model: bool = False,
    nodes: Mapping[str, Node] | None = None,
) -> list[Run]:
    """Every row through the graph, in order, in one process.

    ``nodes`` is merged over the flags' own overrides and is the test seam: it is how this runs
    with no BM25 pickle on disk, the way :func:`agent.nodes.dispatch.dispatch_node` takes a fake
    ``retrieve``. One process for all ten, so the catalog index, the retriever and the compiled
    graph are each built once -- which is why the CLI is one query per invocation and this is not.
    """
    runs = []
    for n, row in enumerate(rows, start=1):
        forced = row.expected_intent if fake_model else None
        overrides = {**build_overrides(cfg, forced, fake_model), **(nodes or {})}
        runs.append(Run(n, row, run_through_graph(row.query, cfg=cfg, nodes=overrides)))
    return runs


def exit_counts(exits: Sequence[Any]) -> dict[str, int]:
    """How many runs took each of the three exits -- every exit named, including the zeroes.

    Zeroes included because a missing row is the interesting one: Day 2's acceptance is that none
    of the three is zero, and a table that omitted them would leave that to the reader to notice.
    """
    return {name: sum(1 for taken in exits if taken == name) for name in EXITS}


# --- markdown -------------------------------------------------------------------------------------


def missing_exits(runs: Sequence[Run]) -> list[str]:
    """The exits this run never took. Day 2's acceptance is that the list is empty.

    One function rather than the same comprehension in the report and in :func:`main`, so the
    sentence the report prints and the status code the shell sees can never disagree about
    whether the milestone was met.
    """
    taken = exit_counts([run.state.get("exit") for run in runs])
    return [name for name, count in taken.items() if not count]


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub markdown table. Cells are written as given -- the caller does its own quoting."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def quote(text: str) -> str:
    """``text`` as a markdown blockquote, blank lines included, so an answer keeps its shape."""
    return "\n".join(f"> {line}".rstrip() for line in text.splitlines())


def code(value: Any) -> str:
    """One value as inline code, or the em dash when there is none."""
    return MISSING if value is None or value == "" else f"`{value}`"


def agreement(expected: str, actual: Any) -> str:
    """One cell for an expected/actual pair: the value once when they agree, an arrow when not."""
    return code(expected) if actual == expected else f"{code(expected)} → {code(actual)}"


def mark(ok: bool, note: str = "") -> str:
    """A cell that says whether this row did what was expected, with whatever else it has to say.

    Marked in the per-query tables as well as in the summary one: a reader who opened the report
    at a section should not have to scroll back up to find out that the row disagreed.
    """
    return note if ok else " — ".join(part for part in ("✗", note) if part)


def command(fake_model: bool, query: str | None = None, intent: str | None = None) -> str:
    """The command that reproduces this report, or one row of it, copy-pasteably."""
    if query is None:
        if fake_model:
            return "uv run python -m eval.scripts.day2_smoke --fake-model"
        return "uv run --env-file .env python -m eval.scripts.day2_smoke"
    if fake_model:
        return f'uv run python -m agent "{query}" --intent {intent} --fake-model'
    return f'uv run --env-file .env python -m agent "{query}"'


# --- the report -----------------------------------------------------------------------------------


def index_line(cfg: Config) -> str:
    """Which index (or indexes) the retrieval rows of this run read, as repo-relative paths."""
    sources = retrieval.MODE_SOURCES[cfg["agent"]["retrieval_mode"]]
    key = chunk_key(cfg)
    parts = []
    if "bm25" in sources:
        parts.append(f"`{rel_to_repo(bm25_path(cfg, key))}`")
    if "dense" in sources:
        parts.append(f"`{collection_name(cfg, key)}` in `{rel_to_repo(chroma_path())}`")
    return ", ".join(parts)


def render_header(cfg: Config, rows: Sequence[Query], dataset: Path, fake_model: bool) -> str:
    """The title, the command that produced the file, and what the run was."""
    mode = cfg["agent"]["retrieval_mode"]
    method, threshold = cfg["evidence"]["method"], cfg["evidence"]["thresholds"][mode]
    gate = f"`{method}`, {mode} threshold `{threshold}`"
    model = "a stand-in, not a model" if fake_model else f"`{cfg['llm']['model']}`"
    facts = [
        ("config hash", f"`{cfg.hash}`"),
        ("catalog term", f"`{cfg['data']['term']}`"),
        ("retrieval mode", f"`{mode}` (`agent.retrieval_mode`)"),
        ("index", index_line(cfg)),
        ("evidence gate", gate),
        ("prompts", f"router `{ROUTER_PROMPT_VERSION}`, answer `{ANSWER_PROMPT_VERSION}`"),
        ("answers written by", model),
        ("queries", f"`{rel_to_repo(dataset)}`, {len(rows)} rows"),
    ]
    return "\n".join(
        [
            "# Day 2 smoke: ten queries, one of three exits each",
            "",
            "One message in, one of three exits out, once per query, from the command line.",
            "Produced by:",
            "",
            "```bash",
            command(fake_model),
            "```",
            "",
            md_table(("", ""), facts),
        ]
    )


def render_caveats(cfg: Config, fake_model: bool) -> str:
    """What this run does and does not measure -- the section the report is least allowed to skip.

    Written from the run's own settings rather than kept as one fixed paragraph: the file is meant
    to be regenerated once there are credits, and a caveat left behind after the thing it warned
    about was fixed would be worse than none.
    """
    parts = ["## What is and is not measured here", ""]
    if fake_model:
        parts += [
            (
                "**No model was called anywhere in this run.** The OpenAI account behind this "
                "repo has no credits (every call returns 429 `insufficient_quota`), so:"
            ),
            "",
            (
                "- **Nothing was classified.** Each row's intent is forced from its "
                "`expected_intent` (`--intent`, #35), so the intent columns below agree by "
                "construction and say nothing about the router. `intent_raw` is empty on every "
                "row, which is the marker that says so: no other path through that node leaves "
                "it empty."
            ),
            (
                "- **The answers are canned.** `llm.StandInChat` replies with the first line of "
                "the first evidence block and cites that block's `doc_id`. Everything around the "
                "words is real -- the prompt #38 assembled, the citation parse, the check against "
                "`docs`, the exit the node claimed -- but the *wording* of an answer, and whether "
                "a model stays inside the evidence it was given, are not measured here at all."
            ),
        ]
    else:
        parts += [
            (
                f"**This run called `{cfg['llm']['model']}`**: once per query to classify, and "
                "once more for every query that got past the gate. The intent columns below are "
                "the router's own work, and every answer is the model's, kept only because its "
                "citations checked out."
            ),
            "",
            (
                "- **A live run is not reproducible, and `temperature: 0` does not make it one.** "
                "Nothing else in this file can move a line -- there is no clock in it and the "
                "renderer is a function of the run -- so a diff under `report/` after a live "
                "re-run is the model having answered differently, which it does. Read a single "
                "row as one sample and not as the behaviour of the system."
            ),
        ]
    if cfg["agent"]["retrieval_mode"] == "bm25":
        parts += [
            (
                "- **Only the lexical half of the pipeline ran.** `agent.retrieval_mode` is "
                "`bm25`, so every retrieval row here was answered by BM25 alone and the hybrid "
                "path is plumbing-verified, not result-verified by this report. Which mode runs "
                "is a config value, so switching it is a new hash and a new report rather than a "
                "footnote to this one."
            ),
        ]
    parts += [
        "",
        (
            "What the run does establish: the graph, the dispatch table, the three tools, the "
            "evidence gate, the citation check and the two fixed refusals hold together over ten "
            "real questions, and all three exits are reachable from the command line."
        ),
    ]
    return "\n".join(parts)


def render_exits(runs: Sequence[Run]) -> str:
    """The count table Day 2's acceptance is read off, expected beside actual."""
    expected = exit_counts([run.row.expected_exit for run in runs])
    actual = exit_counts([run.state.get("exit") for run in runs])
    rows = [(f"`{name}`", str(expected[name]), str(actual[name])) for name in EXITS]
    missing = missing_exits(runs)
    verdict = (
        f"**{', '.join(missing)} never fired.** Day 2's acceptance is that all three do."
        if missing
        else "**All three exits fired.** That is Day 2's acceptance, and this is the count."
    )
    return "\n".join(["## Exits", "", md_table(("exit", "expected", "actual"), rows), "", verdict])


def render_agreement(runs: Sequence[Run]) -> str:
    """Every row's expected label and exit beside what actually happened."""
    rows = [
        (
            str(run.n),
            run.row.query,
            agreement(run.row.expected_intent, run.state.get("intent")),
            agreement(run.row.expected_exit, run.state.get("exit")),
            "" if run.exit_ok and run.intent_ok else "✗",
        )
        for run in runs
    ]
    headers = ("#", "query", "intent (expected → actual)", "exit (expected → actual)", "")
    return "\n".join(["## Expected against actual", "", md_table(headers, rows)])


def render_disagreements(runs: Sequence[Run]) -> str:
    """The rows that did something else, each with the measurement that explains it.

    A section of its own rather than a mark in a table: the acceptance asks for disagreements to
    be called out, and the useful half is *why*, which is the gate's two numbers or the tool error
    the run recorded.
    """
    off = [run for run in runs if not (run.intent_ok and run.exit_ok)]
    if not off:
        return "## Disagreements\n\nNone: every row took the intent and the exit it expects."
    parts = [
        "## Disagreements",
        "",
        (
            f"{len(off)} of {len(runs)} rows did something other than what the dataset expects. "
            "They are listed here rather than re-labelled in `day2_queries.jsonl`: the "
            "expectation is what the row should do, and the gap is the finding."
        ),
        "",
    ]
    for run in off:
        parts += [
            f"**{run.n}. {run.row.query}**",
            "",
            (
                f"- expected {code(run.row.expected_exit)} by way of "
                f"{code(run.row.expected_intent)}, got {code(run.state.get('exit'))} by way of "
                f"{code(run.state.get('intent'))}"
            ),
            f"- evidence: {gathered(run.state)}; gate: {gate_note(run.state)}",
            f"- why the row is in the set: {run.row.note}",
            "",
        ]
    return "\n".join(parts).rstrip()


def answer_provenance(marks: Mapping[str, Any], fake_model: bool) -> str:
    """One sentence on where the text below came from, and what became of its citations."""
    verified = marks.get("citations_verified")
    unknown = marks.get("citations_unknown")
    written = "the stand-in" if fake_model else "the model"
    if unknown:
        return (
            f"{written} drafted one and it was discarded: it cited "
            f"{', '.join(code(one) for one in unknown)}, which this run never retrieved. What "
            "follows is the hand-off, minted in `agent/nodes/exits.py`."
        )
    if verified is not None and not verified:
        return (
            f"{written} drafted one and it was discarded: it cited nothing at all. What follows "
            "is the hand-off, minted in `agent/nodes/exits.py`."
        )
    if verified:
        cited = ", ".join(code(one) for one in verified)
        return f"written by {written}, citing {cited} — each checked against the evidence above."
    return "fixed text: neither refusal exit calls a model."


def render_run(run: Run, fake_model: bool) -> str:
    """One query, whole: what was expected, what was gathered, and what came out."""
    state = run.state
    parts = [
        f"### {run.n}. {run.row.query}",
        "",
        f"*{run.row.note}*",
        "",
        "```bash",
        command(fake_model, run.row.query, run.row.expected_intent),
        "```",
        "",
        md_table(
            ("", "expected", "actual", ""),
            [
                (
                    "intent",
                    code(run.row.expected_intent),
                    code(state.get("intent")),
                    mark(run.intent_ok, intent_note(state)),
                ),
                (
                    "exit",
                    code(run.row.expected_exit),
                    code(state.get("exit")),
                    mark(run.exit_ok),
                ),
            ],
        ),
        "",
        f"**Evidence** — {gathered(state)}. Gate: {gate_note(state)}.",
        "",
    ]
    rows = evidence_rows(state, LINE_WIDTH)
    if rows:
        parts += [
            md_table(
                ("#", "score", "doc_id", "first line of the block"),
                [(n, score, code(doc_id), line) for n, score, doc_id, line in rows],
            ),
            "",
        ]
    parts += [
        f"**Answer** — {answer_provenance(signals(state), fake_model)}",
        "",
        quote(state["answer"]),
    ]
    return "\n".join(parts)


def render_report(
    cfg: Config, runs: Sequence[Run], dataset: Path = DATASET, fake_model: bool = False
) -> str:
    """The whole report: what the run was, then the numbers, then every query end to end."""
    sections = [
        render_header(cfg, [run.row for run in runs], dataset, fake_model),
        render_caveats(cfg, fake_model),
        render_exits(runs),
        render_agreement(runs),
        render_disagreements(runs),
        (
            f"## The {len(runs)} runs\n\nOne section per query, each reproducible on its own with "
            "the command under it."
        ),
        *[render_run(run, fake_model) for run in runs],
    ]
    return "\n\n".join(sections).rstrip() + "\n"


# --- CLI ------------------------------------------------------------------------------------------


def summary(cfg: Config, runs: Sequence[Run], path: Path, fake_model: bool) -> str:
    """What the terminal gets: the counts, the agreement, and where the report went."""
    actual = exit_counts([run.state.get("exit") for run in runs])
    exits = ", ".join(f"{name} {count}" for name, count in actual.items())
    forced = " (forced, so not the router's)" if fake_model else ""
    model = "a stand-in -- no model was called" if fake_model else cfg["llm"]["model"]
    intents = sum(run.intent_ok for run in runs)
    exits_ok = sum(run.exit_ok for run in runs)
    size = path.stat().st_size / 1024
    return "\n".join(
        [
            f"config hash {cfg.hash}  (mode {cfg['agent']['retrieval_mode']})",
            f"queries                {len(runs)}  ({rel_to_repo(DATASET)})",
            f"answers written by     {model}",
            f"exits                  {exits}",
            (
                f"agreement              intent {intents}/{len(runs)}{forced}, "
                f"exit {exits_ok}/{len(runs)}"
            ),
            f"wrote                  {rel_to_repo(path)}  ({size:.0f} KB)",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.day2_smoke",
        description="Run Day 2's ten queries end to end and write report/day2_smoke_{hash}.md.",
    )
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="force each row's expected_intent and answer with a stand-in: no model is called, "
        "and the report says so on every row",
    )
    parser.add_argument(
        "--dataset", default=None, help=f"queries (default: {rel_to_repo(DATASET)})"
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    dataset = Path(args.dataset) if args.dataset else DATASET
    runs = run_queries(cfg, load_queries(dataset), fake_model=args.fake_model)

    path = report_path(STEM, cfg)
    path.write_text(render_report(cfg, runs, dataset, args.fake_model), encoding="utf-8")
    print(summary(cfg, runs, path, args.fake_model))

    missing = missing_exits(runs)
    if missing:
        print(
            f"\nFAILED: {', '.join(missing)} never fired. Day 2's acceptance is that all three "
            "exits do, over this set; the report is written either way.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
