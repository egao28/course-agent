"""Node 1: which of the five questions is this? One label out, and nothing downstream guesses again.

    route_intent ──► out_of_scope          (the label the router itself can end a run with)
         └────────► dispatch               (the other four: #36 decides where the evidence is)

The node asks the model for one label, and then **does not trust the answer**. The completion is
compared against the five literals in :data:`agent.state.INTENTS` and nothing else is accepted: a
sentence, a label with a period on the end, an invented sixth category all become ``out_of_scope``
with ``signals["intent_unparsed"] = True`` and the raw string kept in ``signals["intent_raw"]``.
Free text must never reach #36's dispatch table -- it indexes a table by this value -- and a
classification that could not be read must be *countable* afterwards rather than absorbed.

The prompt itself is :data:`agent.prompts.ROUTER_SYSTEM_PROMPT`, version
:data:`agent.prompts.ROUTER_PROMPT_VERSION`: five glosses, two examples per label and four rules,
assembled there from the label list so that it cannot teach a vocabulary this node would refuse.

**The strictness has a price and the price is recorded.** ``Course_fact`` and ``course_fact.`` are
both in-scope classifications that this node refuses, and a refused in-scope query reaches the
person as "outside what I can answer". The alternative -- lowercasing, stripping punctuation,
matching a label inside a sentence -- buys those back and hides how often the model ignored the
format, which is the number Day 5 needs in order to decide whether the prompt or the parser is
what should change. So the parser stays exact, ``intent_raw`` keeps what actually came back, and
the decision is left to data that does not exist yet rather than to a guess that does.

**An API failure is not a classification.** A 429, a timeout or a cut-off completion propagates
out of the node. Turning those into ``out_of_scope`` would put a billing problem in #39's report
as a refusal the reader would read as a routing decision, and would mix infrastructure failures
into the ``intent_unparsed`` count.

**``intent_raw is None`` means no model was asked.** ``route_intent_node(cfg, forced=label)``
returns a node that skips the call entirely and writes the label it was given -- what #39's
``--intent`` flag is built from, so the rest of the chain can be exercised end to end before a
single token is spent::

    nodes = {"route_intent": route_intent_node(cfg, forced="prereq_chain")}
    run("What do I need before CS 374?", cfg=cfg, nodes=nodes)

That invariant is what keeps such a run honest in the report: a row whose ``intent_raw`` is empty
had no model behind its label, and no other path through this node leaves it empty.

**The client is made on the first query, and ``llm`` is imported where it is needed** rather than
at the top of this module -- the arrangement :func:`retrieval._build_dense` uses, for the same
reason: ``llm`` imports ``embedding``, which imports ``openai``, and a module-level import would
put the SDK in the import graph of ``agent.graph`` itself. The key is read when a client is
*constructed*, so nothing here needs one until a query actually arrives -- and a node built over
an injected fake, or with a forced label, constructs none and imports nothing.

The one eager step is validating the ``llm`` config section, which happens when the node is built:
it reads no key and costs nothing, and a config naming a provider this repo has no client for
should not survive until the first of #39's ten queries. An injected client skips that check, the
way :func:`llm.make_chat` skips the key check for the same reason -- the section is what
``make_chat`` reads, and a fake replaces it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.nodes import Node
from agent.prompts import ROUTER_SYSTEM_PROMPT, router_user_prompt
from agent.state import INTENTS, AgentState, Intent
from config import Config

if TYPE_CHECKING:  # `llm` pulls in openai; see the module docstring
    from llm import Chat


def parse_intent(raw: Any) -> Intent | None:
    """One of the five labels, or ``None`` for anything else the model said.

    Exact equality against :data:`~agent.state.INTENTS`, after whitespace only --
    :meth:`llm.OpenAIChat.complete` already strips, and a fake in a test may not. No lowercasing
    and no punctuation stripping: see the module docstring on why the near misses are counted
    rather than repaired.
    """
    if not isinstance(raw, str):
        return None
    label = raw.strip()
    return label if label in INTENTS else None  # type: ignore[return-value]


def check_forced(label: Any) -> Intent:
    """Refuse a forced label that is not one of the five, naming them.

    The value comes from a CLI flag (#39), so the message is written for a person who typed it.
    """
    if label not in INTENTS:
        raise ValueError(
            f"cannot force intent {label!r}; the five labels are {', '.join(INTENTS)} -- a forced "
            "label bypasses the classifier, it does not extend it"
        )
    return label  # type: ignore[return-value]


def route_intent_node(cfg: Config, chat: Chat | None = None, forced: Intent | None = None) -> Node:
    """Build the router: the query in, one of the five labels out.

    ``chat`` replaces the client for the lifetime of the node -- anything with a ``complete``
    method is a :class:`llm.Chat`, which is how the routing logic is tested with no network.
    ``forced`` skips the classification altogether (see the module docstring); the two are
    mutually exclusive, because a node that was handed both would silently ignore one of them.
    """
    if forced is not None:
        if chat is not None:
            raise ValueError(
                "route_intent_node got both `chat` and `forced`; a forced label makes no model "
                "call, so one of the two would have been ignored"
            )
        label = check_forced(forced)

        def forced_node(state: AgentState) -> dict[str, Any]:
            # intent_raw stays None: nothing was asked, so there is no raw completion to report.
            return {"intent": label, "signals": {"intent_raw": None, "intent_unparsed": False}}

        forced_node.__name__ = "route_intent"
        return forced_node

    if chat is None:
        from llm import check_llm_config

        check_llm_config(cfg)
    built: list[Chat] = []

    def chat_client() -> Chat:
        """The client, made once per node and only if a query actually reaches it."""
        if chat is not None:
            return chat
        if not built:
            from llm import make_chat

            built.append(make_chat(cfg))
        return built[0]

    def node(state: AgentState) -> dict[str, Any]:
        raw = chat_client().complete(
            router_user_prompt(state["query"]), system=ROUTER_SYSTEM_PROMPT
        )
        label = parse_intent(raw)
        if label is not None:
            return {"intent": label, "signals": {"intent_raw": raw, "intent_unparsed": False}}
        # repr() only for a client that broke the `complete() -> str` contract: `intent_raw is
        # None` has to go on meaning "no model was asked", which is what a forced label writes.
        return {
            "intent": "out_of_scope",
            "signals": {
                "intent_raw": raw if isinstance(raw, str) else repr(raw),
                "intent_unparsed": True,
            },
        }

    node.__name__ = "route_intent"
    return node
