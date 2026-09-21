"""The CLI: one message in, one of three exits out, and the whole trace of how it got there.

    uv run --env-file .env python -m agent "What do I need before CS 374?"
    uv run python -m agent "What do I need before CS 374?" --intent prereq_chain --fake-model

The first form is the agent as it is meant to run: the router (#35) classifies, and the answer
node (#38) writes the answer. Both call a model, so both need ``OPENAI_API_KEY`` -- and the
account behind this repo has none, so the second form is what runs today. It is also what Day 2's
smoke report is produced with, and both flags are loud on purpose: a run with no model in it must
not read like a run with one.

``--intent`` skips the classification and nothing else: the dispatch table, the gate, the tools
and the retriever all run exactly as they would have. #35 records ``intent_raw = None`` for a
forced label -- no other path through the router leaves it empty -- so the trace can say "no model
was asked" because the run says so, not because this module passed a flag along.

``--fake-model`` puts :class:`llm.StandInChat` behind the answer node: the prompt is assembled and
sent, the reply is canned (the first line of the first evidence block, citing that block's id),
and the citation check runs against it for real. It requires ``--intent``, because a stand-in
cannot classify -- a faked label would be a routing decision nobody made, printed beside the four
that a dispatch table acted on.

**It prints a trace, not an answer.** Which label was chosen, which row of the dispatch table ran,
what the gate compared and which exit was taken are the things that go wrong; the answer alone
shows none of them, and on the two refusal exits it is fixed text that is the same whatever
happened. :mod:`agent.trace` renders it, and #39's report renders the same facts per query.

One query per process, like ``python -m retrieval``: the caches inside :mod:`retrieval` and
:mod:`agent.graph` are per-process and buy nothing for a single run, and ten queries in a row is
Day 2's smoke run (``python -m eval.scripts.day2_smoke``), which pays the index load once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from time import perf_counter

from agent.graph import run
from agent.nodes import Node
from agent.nodes.answer import answer_node
from agent.nodes.route_intent import route_intent_node
from agent.prompts import ANSWER_PROMPT_VERSION, ROUTER_PROMPT_VERSION
from agent.state import INTENTS
from agent.trace import format_trace
from config import Config, load_config


def build_overrides(cfg: Config, intent: str | None, fake_model: bool) -> dict[str, Node]:
    """The nodes the flags replace -- ``{}`` when neither was given, which is the real agent.

    Built here rather than inside :func:`agent.graph.run` because the two flags are a CLI
    affordance and not a config setting: nothing about a stand-in belongs in ``config.yaml``,
    where it would be hashed into the name of a report it invalidates.
    """
    overrides: dict[str, Node] = {}
    if intent is not None:
        overrides["route_intent"] = route_intent_node(cfg, forced=intent)  # type: ignore[arg-type]
    if fake_model:
        from llm import StandInChat

        overrides["answer"] = answer_node(cfg, chat=StandInChat())
    return overrides


def duration(ms: float) -> str:
    """Milliseconds as the unit a human would have said: ``84 ms``, ``1.4 s``."""
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="Ask the course agent one question and print the whole run.",
    )
    parser.add_argument("query", nargs="+", help="the question, as a person would type it")
    parser.add_argument(
        "--intent",
        choices=INTENTS,
        default=None,
        help="skip the classifier and force this label (#35); everything downstream still runs",
    )
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="answer with a stand-in instead of a model: canned text, real citation check. "
        "Requires --intent, since a stand-in cannot classify",
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)
    if args.fake_model and args.intent is None:
        parser.error(
            "--fake-model needs --intent: the stand-in answers from the evidence in the prompt "
            "and has no label to give the router, and a faked classification would be a routing "
            f"decision nobody made. Pass one of: {', '.join(INTENTS)}"
        )

    cfg = load_config(args.config) if args.config else load_config()
    query = " ".join(args.query)
    started = perf_counter()
    state = run(query, cfg=cfg, nodes=build_overrides(cfg, args.intent, args.fake_model))
    wall_ms = (perf_counter() - started) * 1000

    model = "stand-in" if args.fake_model else cfg["llm"]["model"]
    print(format_trace(state, model=model))
    print(
        f"(config {cfg.hash}, prompts router {ROUTER_PROMPT_VERSION} / answer "
        f"{ANSWER_PROMPT_VERSION}, {duration(wall_ms)} including any index load)"
    )


if __name__ == "__main__":
    sys.exit(main())
