"""RQ2's run: every row of one half through the Day 2 agent, with three confidence signals each.

    uv run --env-file .env python -m eval.scripts.signals --split calib   # develop against this
    uv run --env-file .env python -m eval.scripts.signals --split test    # read once, for Day 5

One line per answered row goes to ``eval/runs/signals_{split}_{signals_key}.jsonl``, and a sidecar
beside it records what the run was and every row it did *not* answer, with the reason. Correctness
is #88's job: this file carries the answer, the signals and the raw text they were read from, and
nothing that says whether the answer was right.

**The three signals.**

* ``retrieval_score`` -- the top-1 score of the retrieval the agent actually ran
  (``agent.retrieval_mode``, bm25 today), **not** an RRF score: hybrid top-1 is near-constant
  (see ``agent/nodes/check_evidence.py``, ``UNGATEABLE``) and ranks nothing. A **tool row** (an
  exact catalog lookup, ``score is None``) has no retrieval score and gets ``null`` -- never 1.0,
  which would put every lookup at the confident end of #89's curves. Tool rows are counted by
  bucket.
* ``dense_top1`` -- the best cosine ``retrieve(q, "dense", k=1)`` gives the *question*, on every
  answered row, tool rows included. It is not the agent's evidence on a tool row, and it is not a
  stand-in for ``retrieval_score``: it asks how well the question matches anything in the corpus,
  which is defined whatever dispatch did. On the first calib run 36 of 46 answered rows were tool
  rows, so ``retrieval_score`` alone would leave #89 a retrieval signal on 10 rows.
* ``verbalized`` -- a follow-up call after the answer is written (``signals.verbalized_prompt``).
  The prompt asks for a short check of the answer against the evidence and then a last line
  ``Confidence: <integer>``. The integer on that line, in [0, 100], is the signal. A reply whose
  last line is anything else is ``null``, with the reply kept in ``verbalized_raw`` and counted. It
  is never coerced to 50 or 0. The first prompt asked for a bare integer and got 100 on 45 of 46
  calib rows. That run is in this file's history.
* ``self_consistency`` -- ``signals.samples`` answers at ``signals.sample_temperature``, drawn with
  exactly the answer node's prompt and separate from its temperature-0 answer. A judge
  (``signals.judge_prompt``, temperature 0) rules on every pair, and the score is agreeing pairs /
  pairs: with 3 samples, one of {0, 1/3, 2/3, 1}. The samples and the per-pair verdicts are kept,
  not just the ratio. A verdict that is not "yes" or "no" is ``null``, and so is the row's score.

**The gate is bypassed.** Day 2's gate would send some rows to the hand-off exit before anything is
generated, and a refused row has no answer to take a signal from. So ``check_evidence`` is
replaced (a ``nodes=`` override -- the graph already takes injected nodes) by one that lets every
row with evidence through, and records what the real gate *would* have said in ``gate_ok``. Rows
with no evidence cannot be answered at all -- the answer node refuses an empty ``docs`` -- and
``out_of_scope`` rows never reach dispatch: both are counted, listed in the sidecar with their
reason, and skipped. The sidecar says the gate was bypassed.

**What ``answer`` is.** The answer node's temperature-0 draft, recorded as written -- including a
draft its citation check then discarded for the hand-off exit, which ``exit`` and
``citations_unknown`` say. The signals are about the model's answer, and a discarded draft is
still one; whether a discarded row counts as a decline is for #88 to label, not for this run to
drop.

**Nothing here reproduces.** The samples are random by construction and the dense side of a live
run jitters (see ``eval/scripts/ablation.py``). The committed run file is the sample: nothing
downstream re-queries, and a rerun is a different draw, not a check of this one.

The verbalized and self-consistency calls live in :mod:`agent.confidence`, which #92's post-answer
gate calls too, so the live gate and this run read the same signal the same way.

Each answered row costs 2 + samples + pairs chat calls (router, answer, verbalized, 3 samples, 3
judgments = 9 with the shipped settings) and one embedding call.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import openai

from agent.confidence import (  # noqa: F401  (re-exported: #88 and the tests read them here)
    CONFIDENCE_RE,
    PLACEHOLDERS,
    VERDICTS,
    Settings,
    agreement,
    fields_of,
    pairs,
    parse_verbalized,
    parse_verdict,
    self_consistency,
    settings,
    verbalized,
)
from agent.graph import run as run_agent
from agent.nodes import Node
from agent.nodes.answer import answer_node
from agent.nodes.check_evidence import judge, thresholds
from agent.nodes.route_intent import route_intent_node
from agent.prompts import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    ROUTER_PROMPT_VERSION,
    ROUTER_SYSTEM_PROMPT,
)
from agent.state import AgentState, Evidence
from config import (
    REPO_ROOT,
    Config,
    bm25_key,
    dense_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval.schema import BUCKETS, SPLITS, EvalItem, Split
from eval.scripts.ablation import write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import META_SUFFIX, load_split, meta_path, read_sidecar
from ingest.chunk import chunk_key
from llm import Chat, check_llm_config, make_chat
from retrieval import retrieve

#: Where run files go. Committed: they are what Day 5's figures and thresholds are computed from.
RUNS_DIR = REPO_ROOT / "eval" / "runs"


def run_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/signals_{split}_{signals_key}.jsonl`` -- one file per half, per setting."""
    return Path(runs_dir) / f"signals_{split}_{signals_key(cfg)}.jsonl"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the agent, with the gate bypassed and the answer draft kept ---------------------------------


class Draft(NamedTuple):
    """What the answer node sent and what came back: the prompt the samples must reuse."""

    prompt: str
    system: str | None
    reply: str


class Recorder:
    """A :class:`llm.Chat` that remembers the last call it passed through.

    The answer node discards a draft whose citations fail and hands over instead, so the final
    state does not always carry what the model wrote. Wrapping its client is how the draft, and
    the exact prompt behind it, survive either way -- without a second code path that could
    assemble the prompt differently from the node.
    """

    def __init__(self, chat: Chat) -> None:
        self.chat = chat
        self.model = chat.model
        self.last: Draft | None = None

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float | None = None
    ) -> str:
        reply = self.chat.complete(prompt, system=system, temperature=temperature)
        self.last = Draft(prompt, system, reply)
        return reply


def bypassed_gate(cfg: Config) -> tuple[Node, dict[str, bool | None]]:
    """``check_evidence`` that lets every row with evidence through, and what it overrode.

    The Day 2 verdict is recorded rather than lost -- ``gate_ok`` is what #92 compares a
    conformal gate against. An empty ``docs`` still fails: the answer node refuses to run on
    nothing, and there would be no answer to take a signal from anyway.
    """
    table = thresholds(cfg)
    verdicts: dict[str, bool | None] = {}

    def node(state: AgentState) -> dict[str, Any]:
        docs = state.get("docs") or []
        signals: Mapping[str, Any] = state.get("signals") or {}
        try:
            verdict = judge(docs, signals.get("tool_error"), table)
            verdicts["gate_ok"] = verdict.ok
        except ValueError:
            # An ungateable source (hybrid): Day 2 raises rather than returning a verdict.
            verdicts["gate_ok"] = None
        return {"evidence_ok": bool(docs)}

    node.__name__ = "check_evidence"
    return node, verdicts


def retrieval_signal(docs: Sequence[Evidence]) -> tuple[float | None, str]:
    """The top-1 score of the retrieval that ran and its source; ``(None, tool)`` for a lookup."""
    tool = next((block.source for block in docs if block.score is None), None)
    if tool is not None:
        return None, tool
    sources = sorted({block.source for block in docs})
    if len(sources) != 1:
        raise ValueError(f"docs mixes {len(sources)} sources ({', '.join(sources)})")
    return max(float(block.score) for block in docs), sources[0]  # type: ignore[arg-type]


def live_dense_top1(query: str, cfg: Config) -> float | None:
    """The best cosine ``retrieve(q, "dense", k=1)`` returns, or ``None`` for no hit."""
    result = retrieve(query, "dense", k=1, cfg=cfg)
    return result.hits[0].score if result.hits else None


# --- one row ------------------------------------------------------------------------------------


class Outcome(NamedTuple):
    """A row's line, or the reason it has none."""

    line: dict[str, Any] | None
    skipped: str | None = None


def collect(
    row: EvalItem,
    cfg: Config,
    chat: Chat,
    config: Settings,
    dense_top1: Callable[[str, Config], float | None] = live_dense_top1,
) -> Outcome:
    """One row through the agent, then the verbalized follow-up and the samples."""
    recorder = Recorder(chat)
    gate, gate_verdict = bypassed_gate(cfg)
    nodes = {
        "route_intent": route_intent_node(cfg, chat=chat),
        "check_evidence": gate,
        "answer": answer_node(cfg, chat=recorder),
    }
    try:
        state = run_agent(row.question, cfg, nodes=nodes)
    except (RuntimeError, openai.APIError) as error:
        # llm.py's three "no answer in this completion" refusals, or a request that failed after
        # its retries: the row is lost, and says why, rather than losing the whole half.
        return Outcome(None, f"error: {error}")
    signals = state["signals"]
    if state["intent"] == "out_of_scope":
        return Outcome(None, "out_of_scope")
    docs = state["docs"]
    if not docs:
        return Outcome(None, f"no_evidence: {signals.get('tool_error') or 'empty retrieval'}")
    draft = recorder.last
    if draft is None:  # pragma: no cover - the gate let docs through, so the answer node ran
        raise RuntimeError(f"{row.id}: the answer node never called its client")

    score, source = retrieval_signal(docs)
    try:
        cosine = dense_top1(row.question, cfg)
        rated = verbalized(chat, config, row.question, docs, draft.reply)
        sampled = self_consistency(chat, config, row.question, draft.prompt, draft.system)
    except (RuntimeError, openai.APIError) as error:
        return Outcome(None, f"error: {error}")
    return Outcome(
        {
            "id": row.id,
            "bucket": row.bucket,
            "split": row.split,
            "question": row.question,
            "intent": state["intent"],
            "exit": state["exit"],
            "answer": draft.reply,
            "citations_unknown": signals.get("citations_unknown"),
            "gate_ok": gate_verdict.get("gate_ok"),
            "retrieval_score": score,
            "retrieval_source": source,
            "dense_top1": cosine,
            "verbalized": rated.value,
            "verbalized_raw": rated.raw,
            "samples": sampled.samples,
            "pair_agree": sampled.pair_agree,
            "judge_raw": sampled.judge_raw,
            "self_consistency": sampled.value,
        }
    )


# --- the run ------------------------------------------------------------------------------------


def tally(lines: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]) -> dict:
    """The counts the sidecar and the printout share, by bucket where a bucket matters."""

    def by_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts = Counter(row["bucket"] for row in rows)
        return {bucket: counts[bucket] for bucket in BUCKETS if counts[bucket]}

    return {
        "answered": by_bucket(lines),
        "tool_rows": by_bucket([line for line in lines if line["retrieval_score"] is None]),
        "verbalized_unparsed": sum(1 for line in lines if line["verbalized"] is None),
        "judge_unparsed": sum(1 for line in lines if None in line["pair_agree"]),
        "gate_would_refuse": by_bucket([line for line in lines if line["gate_ok"] is False]),
        "citation_discarded": by_bucket([line for line in lines if line["exit"] != "answer"]),
        "skipped": dict(sorted(Counter(row["reason"].split(":")[0] for row in skipped).items())),
    }


def sidecar(
    path: Path,
    dataset: Path,
    split: Split,
    rows: Sequence[EvalItem],
    lines: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    cfg: Config,
    model: str,
) -> dict[str, Any]:
    """What the run was, including that the gate was bypassed and which rows it did not answer."""
    config = settings(cfg)
    key = chunk_key(cfg)
    dataset_meta = read_sidecar(meta_path(dataset)) or {}
    return {
        "run": rel_to_repo(path),
        "sha256": file_sha256(path),
        "config_hash": cfg.hash,
        "signals_key": signals_key(cfg),
        "dense_key": dense_key(cfg, key),
        "bm25_key": bm25_key(cfg, key),
        "dataset": rel_to_repo(dataset),
        "dataset_sha256": dataset_meta.get("sha256"),
        "split": split,
        "model": model,
        "retrieval_mode": cfg["agent"]["retrieval_mode"],
        "gate": (
            "bypassed: every row with evidence was answered; gate_ok is what the Day 2 "
            "top_score gate would have said"
        ),
        "samples": config.samples,
        "sample_temperature": config.sample_temperature,
        "prompts_sha256": {
            f"router_{ROUTER_PROMPT_VERSION}": sha256_text(ROUTER_SYSTEM_PROMPT),
            f"answer_{ANSWER_PROMPT_VERSION}": sha256_text(ANSWER_SYSTEM_PROMPT),
            "verbalized": sha256_text(config.verbalized_prompt),
            "judge": sha256_text(config.judge_prompt),
        },
        "n_rows": len(rows),
        "n_lines": len(lines),
        **tally(lines, skipped),
        "ids": [line["id"] for line in lines],
        "skipped_rows": list(skipped),
    }


def summary(lines: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, Any]]) -> str:
    """The counts, as lines a person reads -- a check that the run ran, not a result."""
    counts = tally(lines, skipped)

    def fmt(by: Mapping[str, int]) -> str:
        return ", ".join(f"{n} {name}" for name, n in by.items()) or "none"

    tools, refused = counts["tool_rows"], counts["gate_would_refuse"]
    discarded = counts["citation_discarded"]
    return "\n".join(
        [
            f"answered {len(lines)} ({fmt(counts['answered'])})",
            f"skipped {len(skipped)} ({fmt(counts['skipped'])})",
            f"tool rows, retrieval_score null: {sum(tools.values())} ({fmt(tools)})",
            f"verbalized unparsed (null): {counts['verbalized_unparsed']}",
            f"unparsed judge verdict (self_consistency null): {counts['judge_unparsed']}",
            f"Day 2 gate would have refused: {sum(refused.values())} ({fmt(refused)})",
            f"draft discarded by the citation check: {sum(discarded.values())} ({fmt(discarded)})",
        ]
    )


def collect_split(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
    chat: Chat | None = None,
    dense_top1: Callable[[str, Config], float | None] = live_dense_top1,
) -> Path:
    """Run one half, write the run file and its sidecar, and say what was answered and skipped."""
    config = settings(cfg)
    if chat is None:
        check_llm_config(cfg)
        chat = make_chat(cfg)
    rows = load_split(split, dataset, known)
    print(
        f"{split}: {len(rows)} rows through the agent (gate bypassed), "
        f"{config.samples} samples at T={config.sample_temperature}",
        file=out,
    )
    lines: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for n, row in enumerate(rows, start=1):
        outcome = collect(row, cfg, chat, config, dense_top1)
        if outcome.line is not None:
            lines.append(outcome.line)
            line = outcome.line
            note = f"verbalized {line['verbalized']}, sc {line['self_consistency']}"
        else:
            skipped.append({"id": row.id, "bucket": row.bucket, "reason": outcome.skipped})
            note = f"skipped: {outcome.skipped}"
        print(f"  [{n:>3}/{len(rows)}] {row.id} {row.bucket:<12} {note}", file=out, flush=True)

    path = write_lines(run_path(split, cfg, runs_dir), lines)
    write_json(
        path.with_suffix(META_SUFFIX),
        sidecar(path, Path(dataset), split, rows, lines, skipped, cfg, chat.model),
    )
    print(f"\n{summary(lines, skipped)}\n", file=out)
    print(
        f"wrote {rel_to_repo(path)} ({len(lines)} lines, signals key {signals_key(cfg)}, "
        f"config {cfg.hash})",
        file=out,
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
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
