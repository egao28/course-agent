"""Day 6's end-to-end run: the system that ships, over one half, with the gate **on**, once.

    uv run --env-file .env python -m eval.scripts.e2e --split test

One line per row of the half goes to ``eval/runs/e2e_{split}_{config_hash}.jsonl``, with a sidecar
beside it. Named after ``cfg.hash`` and nothing narrower: this run is the live configuration, not a
variant, and the file is what #104's metrics, #105's attribution and #106's report read.

**Every row gets a line, including the rows that never produce an answer.** #87's signal run kept
only answered rows, because a refused row has no answer to take a signal from; here a refusal *is*
the measurement. The three exits (``agent.state.EXITS``) are what #104 counts false blocks and
false releases from, so a row that left through ``out_of_scope`` or ``refuse_handoff`` is a line
like any other, and the sidecar's exit counts add up to the size of the half.

**The gate is on.** That is the whole point of this run and the reason #87's cannot stand in for
it: that one replaced ``check_evidence`` with a bypass. Here the real node runs, under
``evidence.method`` as ``config.yaml`` has it -- ``top_score`` today, against the per-source
thresholds the sidecar records. #92's conformal gate is built but dormant (its fit accepted no
threshold, so it would refuse every answer, and #91 already measured that it answers 0 of 42), and
this run does not add it as a second row.

**What each field is.**

* ``answer`` -- what the user would have seen: the answer node's text on the ``answer`` exit, and
  otherwise the fixed hand-off or out-of-scope message ``agent/nodes/exits.py`` mints. It is never
  null, because a run that reached an exit with nothing to show is refused upstream
  (``agent.graph.check_final``).
* ``draft`` -- the answer node's text when it is *not* what shipped, i.e. when the citation check
  discarded it (``agent/nodes/answer.py``), and ``null`` otherwise. The draft is the only thing in
  this run that a rerun could not recover, and #105 attributes a wrong citation from it.
* ``citations`` / ``citations_unknown`` -- the doc_ids the answer cited and could show, and the
  ones it cited that were never in ``docs``. Both are ``null`` where the answer node never ran.
* ``evidence_doc_ids`` -- **every** block the answer node was given, in order, not the cited ones.
  #105 needs the whole list to tell a retrieval failure ("the gold was never there") from an
  evidence misread ("it was there and the answer is still wrong").
* ``evidence_source`` -- what produced those ids: a retrieval mode, or the tool that answered. The
  two id spaces differ -- retrieval returns ``{term}:{SUBJ}-{NUM}`` chunk ids and a tool returns a
  course code or a program name (``agent/nodes/dispatch.py``) -- so a comparison against
  ``gold_doc_ids`` that ignores this field is comparing two alphabets.
* ``top_score`` -- the number the gate compared, ``signals['retrieval_top_score']``: ``null`` on a
  tool row, which passes without a threshold, and on a row that never reached the gate.
* ``gate_ok`` -- the gate's verdict, ``null`` where it never ran (an ``out_of_scope`` row). A
  ``refuse_handoff`` row with ``gate_ok`` true is one the citation check discarded, not one the
  gate refused; that is how the two blocks are told apart downstream.
* ``error`` -- a chat call that failed after its retries, or a completion with no answer in it
  (``llm.py``'s three refusals). The row keeps its line, with ``exit`` null, rather than shrinking
  the half; the sidecar counts them and the printout says so. A run with an errored row is an
  infrastructure failure and not a result: the numbers behind it were never read, so deleting the
  file and running again is the fix. Deleting it for any other reason is not.

**Run once.** If the run file exists the run is refused, as ``eval/scripts/conformal_eval.py``
refuses a second reading of the test half. Nothing in the agent or the config changes after this
run: a failure it turns up is #105's subject and goes in the report, not into a fix and a rerun.
Live dense calls are not byte-reproducible anyway (#75, ``eval/scripts/ablation.py``) and neither
is a chat model at temperature 0 in practice, so the committed file is the sample, not a fixture a
rerun would reproduce.

**Correctness is not here.** This file carries what the system did, and nothing that says whether
it was right. The labels are ``eval/scripts/e2e_labels.py``: #88's judge first, then a person's
verdict on every answered row.

Each row costs one router call, plus one answer call when the gate lets it through. No embedding
call is made while ``agent.retrieval_mode`` is ``bm25``; the sidecar records the embedding timeout
anyway, because the mode is one word in the config and the timeouts are what a stalled run is
diagnosed against.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import openai

from agent.graph import run as run_agent
from agent.nodes.answer import answer_node
from agent.nodes.check_evidence import check_method, thresholds
from agent.nodes.route_intent import route_intent_node
from agent.prompts import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    ROUTER_PROMPT_VERSION,
    ROUTER_SYSTEM_PROMPT,
)
from agent.state import EXITS, Evidence
from config import (
    Config,
    bm25_key,
    dense_key,
    file_sha256,
    load_config,
    rel_to_repo,
)
from embedding import OpenAIEmbedder
from eval.schema import BUCKETS, SPLITS, EvalItem, Split
from eval.scripts.ablation import write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.signals import RUNS_DIR, Recorder, sha256_text
from eval.scripts.split import META_SUFFIX, load_split, meta_path, read_sidecar
from ingest.chunk import chunk_key
from llm import Chat, OpenAIChat, check_llm_config, make_chat

#: What a run file line carries, in order. Declared, so every line is the same shape and a field
#: nothing measured is a null cell rather than an absent key (``agent/state.py`` on the same rule).
KEYS: tuple[str, ...] = (
    "id",
    "bucket",
    "split",
    "question",
    "intent",
    "exit",
    "answer",
    "draft",
    "citations",
    "citations_unknown",
    "evidence_doc_ids",
    "evidence_source",
    "top_score",
    "gate_ok",
    "error",
)


def run_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/e2e_{split}_{config_hash}.jsonl`` -- one end-to-end run per half, per config."""
    return Path(runs_dir) / f"e2e_{split}_{cfg.hash}.jsonl"


def timeouts() -> dict[str, float]:
    """The per-request timeouts this run's clients use, read off their own defaults.

    ``llm.make_chat`` and ``embedding.make_embedder`` pass no timeout, so the constructors' values
    are the live ones. Read rather than typed here: a number copied into a sidecar is a number that
    can disagree with the client that produced the run.
    """
    found: dict[str, float] = {}
    for name, func in (("chat", OpenAIChat.__init__), ("embedding", OpenAIEmbedder.__init__)):
        parameter = inspect.signature(func).parameters.get("timeout")
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise ValueError(f"the {name} client no longer takes a default `timeout`")
        found[name] = float(parameter.default)
    return found


# --- one row --------------------------------------------------------------------------------------


def source_of(docs: Sequence[Evidence]) -> str | None:
    """What produced this row's evidence: one source name, or ``None`` for no evidence.

    #36 produces one source per run -- a retrieval mode or a single tool -- so the joined form is
    for a mixture that should not happen, recorded rather than raised: this run has one pass at the
    half and a row lost to an assertion about its provenance is worse than a row that reports it.
    """
    sources = sorted({block.source for block in docs})
    return "+".join(sources) or None


def line_of(row: EvalItem, state: Mapping[str, Any], draft: str | None) -> dict[str, Any]:
    """One finished run as its line: what the system did, and nothing about whether it was right."""
    signals: Mapping[str, Any] = state.get("signals") or {}
    docs: Sequence[Evidence] = state.get("docs") or []
    answer = state.get("answer")
    return {
        "id": row.id,
        "bucket": row.bucket,
        "split": row.split,
        "question": row.question,
        "intent": state.get("intent"),
        "exit": state.get("exit"),
        "answer": answer,
        "draft": draft if draft is not None and draft != answer else None,
        "citations": signals.get("citations_verified"),
        "citations_unknown": signals.get("citations_unknown"),
        "evidence_doc_ids": [block.doc_id for block in docs],
        "evidence_source": source_of(docs),
        "top_score": signals.get("retrieval_top_score"),
        "gate_ok": state.get("evidence_ok"),
        "error": None,
    }


def failed(row: EvalItem, error: str) -> dict[str, Any]:
    """A row whose run did not finish: a line with its reason and no exit to count."""
    return {
        **dict.fromkeys(KEYS),
        "id": row.id,
        "bucket": row.bucket,
        "split": row.split,
        "question": row.question,
        "evidence_doc_ids": [],
        "error": error,
    }


def collect(row: EvalItem, cfg: Config, chat: Chat) -> dict[str, Any]:
    """One question through the whole graph, gate included, as a line.

    Only the two model-calling nodes are injected, and with the same clients the live CLI would
    build: the router's, and a :class:`~eval.scripts.signals.Recorder` around the answer node's, so
    a draft the citation check discards survives the state that no longer carries it. Every other
    node -- dispatch, the gate, both exits -- is the one ``config.yaml`` selects.
    """
    recorder = Recorder(chat)
    nodes = {
        "route_intent": route_intent_node(cfg, chat=chat),
        "answer": answer_node(cfg, chat=recorder),
    }
    try:
        state = run_agent(row.question, cfg, nodes=nodes)
    except (RuntimeError, openai.APIError) as error:
        return failed(row, f"{error}")
    draft = None if recorder.last is None else recorder.last.reply
    return line_of(row, state, draft)


# --- the run --------------------------------------------------------------------------------------


def by_bucket(lines: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Bucket counts in ``BUCKETS`` order, leaving out the buckets with nothing in them."""
    counts = Counter(line["bucket"] for line in lines)
    return {bucket: counts[bucket] for bucket in BUCKETS if counts[bucket]}


def where(lines: Sequence[Mapping[str, Any]], **fields: Any) -> list[Mapping[str, Any]]:
    """The run lines whose every named field equals the value given for it."""
    return [line for line in lines if all(line[key] == value for key, value in fields.items())]


def tally(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts the sidecar and the printout share. ``exits`` plus ``errors`` is every row."""
    errors = [line for line in lines if line["error"] is not None]
    refused = where(lines, exit="refuse_handoff", gate_ok=False)
    return {
        "exits": {name: len(where(lines, exit=name)) for name in EXITS},
        "exits_by_bucket": {name: by_bucket(where(lines, exit=name)) for name in EXITS},
        "errors": len(errors),
        "by_bucket": by_bucket(lines),
        # A hand-off that the gate refused, and one it let through and the citation check then
        # discarded: the same exit, and two different failures for #105 to attribute.
        "gate_refused": by_bucket(refused),
        # The gate refuses two different things, and #105 attributes them differently: a row it
        # had nothing to judge (an empty retrieval, or a tool that produced nothing) and a row
        # whose best block scored under its source's threshold. A scored block that is not a tool
        # result is the only way to reach the comparison, so the two are the whole of `refused`.
        "gate_refused_reasons": {
            "no_evidence": sum(1 for line in refused if line["top_score"] is None),
            "under_threshold": sum(1 for line in refused if line["top_score"] is not None),
        },
        "citation_discarded": by_bucket([line for line in lines if line["draft"] is not None]),
        "tool_rows": by_bucket(
            [line for line in lines if line["top_score"] is None and line["evidence_doc_ids"]]
        ),
        "error_rows": [{"id": line["id"], "error": line["error"]} for line in errors],
    }


def sidecar(
    path: Path,
    dataset: Path,
    split: Split,
    rows: Sequence[EvalItem],
    lines: Sequence[Mapping[str, Any]],
    cfg: Config,
    model: str,
) -> dict[str, Any]:
    """What the run was: the dataset and config it read, the clients it used, how it ended."""
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
        "model": model,
        "embedding_model": cfg["embedding"]["model"],
        "timeouts": timeouts(),
        "retrieval_mode": cfg["agent"]["retrieval_mode"],
        "gate": (
            "on: the live check_evidence node, as config.yaml has it -- not #87's bypass. The "
            "conformal gate (#92) is dormant at this config and was not run as a second row"
        ),
        "evidence": {"method": check_method(cfg), "thresholds": thresholds(cfg)},
        "prompts_sha256": {
            f"router_{ROUTER_PROMPT_VERSION}": sha256_text(ROUTER_SYSTEM_PROMPT),
            f"answer_{ANSWER_PROMPT_VERSION}": sha256_text(ANSWER_SYSTEM_PROMPT),
        },
        "n_rows": len(rows),
        "n_lines": len(lines),
        **tally(lines),
        "ids": [line["id"] for line in lines],
    }


def summary(lines: Sequence[Mapping[str, Any]]) -> str:
    """The counts, as lines a person reads: a check that the run ran, not a result."""
    counts = tally(lines)

    def fmt(by: Mapping[str, int]) -> str:
        return ", ".join(f"{n} {name}" for name, n in by.items()) or "none"

    exits = counts["exits"]
    adds_up = sum(exits.values()) + counts["errors"]
    out = [
        "exits: "
        + " + ".join(f"{n} {name}" for name, n in exits.items())
        + (f" + {counts['errors']} error" if counts["errors"] else "")
        + f" = {adds_up} of {len(lines)} rows",
    ]
    for name in EXITS:
        out.append(f"  {name:<15} {exits[name]:>3}  ({fmt(counts['exits_by_bucket'][name])})")
    discarded, tools = counts["citation_discarded"], counts["tool_rows"]
    out += [
        f"gate refused: {sum(counts['gate_refused'].values())} ({fmt(counts['gate_refused'])})"
        + "; of those {no_evidence} had no evidence to judge and {under_threshold} scored under "
        "their threshold".format(**counts["gate_refused_reasons"]),
        f"citation check discarded a draft: {sum(discarded.values())} ({fmt(discarded)})",
        f"tool rows, top_score null: {sum(tools.values())} ({fmt(tools)})",
    ]
    if counts["errors"]:
        out += [
            f"{counts['errors']} row(s) failed to run: "
            + ", ".join(f"{row['id']} ({row['error']})" for row in counts["error_rows"]),
            (
                "a run with a failed row is not the final run: its numbers were never read, so "
                "delete the file and run again."
            ),
        ]
    return "\n".join(out)


def collect_split(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
    chat: Chat | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run one half end to end, write the run file and its sidecar, and say how it ended."""
    path = run_path(split, cfg, runs_dir)
    if path.exists():
        raise ValueError(
            f"{rel_to_repo(path)} exists: this half has been run end to end for this config. "
            "Delete it only if the run itself failed, never to get another set of answers"
        )
    if chat is None:
        check_llm_config(cfg)
        chat = make_chat(cfg)
    rows = load_split(split, dataset, known)
    print(
        f"{split}: {len(rows)} rows through the whole graph, gate on "
        f"(evidence.method {check_method(cfg)}), config {cfg.hash}",
        file=out,
    )
    lines: list[dict[str, Any]] = []
    for n, row in enumerate(rows, start=1):
        line = collect(row, cfg, chat)
        lines.append(line)
        note = line["error"] or f"{line['intent']} -> {line['exit']}"
        print(f"  [{n:>3}/{len(rows)}] {row.id} {row.bucket:<12} {note}", file=out, flush=True)
        if progress is not None:
            progress(note)

    written = write_lines(path, [{key: line[key] for key in KEYS} for line in lines])
    write_json(
        written.with_suffix(META_SUFFIX),
        sidecar(written, Path(dataset), split, rows, lines, cfg, chat.model),
    )
    print(f"\n{summary(lines)}\n", file=out)
    print(
        f"wrote {rel_to_repo(written)} ({len(lines)} lines, config {cfg.hash}). "
        "Labels next: eval.scripts.e2e_labels --judge, then --review.",
        file=out,
    )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run one half end to end and write the run file; a refusal exits 1 with its reason."""
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
        collect_split(load_config(), args.split, sys.stdout, args.dataset, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
