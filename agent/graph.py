"""The graph itself: one message in, one of three exits out.

::

    route_intent ──► out_of_scope ──────────────────────────────► END
         │
         └────────► dispatch ──► check_evidence ──► answer ─────► END
                                      │              │
                                      └──────────────┴► refuse_handoff ──► END

**Under a post-answer conformal signal there is a seventh node** (#92). ``evidence.method:
conformal`` with ``verbalized`` or ``self_consistency`` reads its signal off the written answer, so
the graph gains :mod:`agent.nodes.confidence_gate` between ``answer`` and END::

    ... check_evidence ──► answer ──► confidence_gate ──► END
                                             └──────────► refuse_handoff

Under ``top_score``, or conformal on ``retrieval_score``, the node is not built and not wired: the
comparison happens in ``check_evidence``, and a pass-through node would put a step in every trace
that did nothing. :func:`node_names` is the one place that decides.

Six nodes, three exits, and no cycle: a query is classified, evidence is gathered for it, the
evidence is judged sufficient or not, and one of three things is said. ``run(query)`` is the whole
public surface -- the CLI (#39) and every test call that, not the compiled graph object, so the
compilation, the config plumbing and the final-state check have one home::

    from agent.graph import run
    state = run("What do I need before CS 374?")
    state["exit"]          # 'answer' | 'refuse_handoff' | 'out_of_scope'

**The answer node can land on the hand-off exit.** #38 verifies citations *after* generation and
discards an answer that cited a ``doc_id`` it was never given, so there is an edge from ``answer``
to ``refuse_handoff`` that the issue's three-exit sketch does not draw. It is still three exits:
the discarded answer arrives at the same node, and the hand-off message is minted in exactly one
place (:mod:`agent.nodes.exits`) however the run got there. The alternative -- #38 writing its own
refusal text -- is two refusal messages that drift.

**Every node is wrapped in :func:`~agent.state.check_update`**, so a node that writes another
node's key, invents a signal, or returns a key the state does not have fails at the node that did
it. Nothing in LangGraph enforces that: an update with an unknown key is silently dropped, and one
that overwrites another node's key simply wins. See :mod:`agent.state`.

**The routers read one key each and refuse to guess.** ``route_after_intent`` needs ``intent``,
``route_after_gate`` needs ``evidence_ok``, ``route_after_answer`` needs ``exit`` -- and a node
that did not set its own key raises there rather than defaulting to a branch. A missing
classification silently routed to ``out_of_scope`` would look exactly like a correct refusal in
#39's report.

Nodes are built from the config once per compiled graph and the graph is cached per config hash,
the way :func:`retrieval.get_retriever` caches retrievers: #39 runs ten queries in one process, and
the router (#35) and the answer node (#38) each open a chat client on their first query, while
dispatch (#36) reaches a retriever and a 2.9 MB catalog index through caches of their own. A graph
built with injected nodes is never cached -- a fake must not outlive the test that made it.

All six nodes are real as of #38. Until then the answer node was a stub that raised rather than
returning a plausible answer, so that no report could be produced with no model behind it; the
pattern is in this file's history if a later day needs it again.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import Node, NodeBuilder
from agent.nodes.answer import answer_node
from agent.nodes.check_evidence import check_evidence_node
from agent.nodes.confidence_gate import confidence_gate_node, post_answer
from agent.nodes.dispatch import dispatch_node
from agent.nodes.exits import out_of_scope_node, refuse_handoff_node
from agent.nodes.route_intent import route_intent_node
from agent.state import (
    EXITS,
    INTENTS,
    AgentState,
    check_exit,
    check_intent,
    check_update,
    initial_state,
)
from config import Config, load_config

#: How to build each node -- the six real as of #38, and #92's confidence gate, which only a
#: post-answer conformal config builds (:func:`node_names`). A test passes already-built nodes to
#: :func:`build_graph` instead of patching this.
NODE_BUILDERS: dict[str, NodeBuilder] = {
    "route_intent": route_intent_node,
    "dispatch": dispatch_node,
    "check_evidence": check_evidence_node,
    "answer": answer_node,
    "confidence_gate": confidence_gate_node,
    "out_of_scope": out_of_scope_node,
    "refuse_handoff": refuse_handoff_node,
}

#: The nodes only some configs put in the graph, and what puts them there.
CONDITIONAL: dict[str, str] = {
    "confidence_gate": "evidence.method: conformal with a post-answer signal "
    "(verbalized or self_consistency)",
}


def node_names(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """The nodes this config's graph has, in registry order."""
    return tuple(name for name in NODE_BUILDERS if name != "confidence_gate" or post_answer(cfg))


# --- building the nodes ---------------------------------------------------------------------------


def build_nodes(cfg: Config, nodes: Mapping[str, Node] | None = None) -> dict[str, Node]:
    """The graph's nodes (:func:`node_names`), built from ``cfg``, with ``nodes`` used as given.

    ``nodes`` is a partial override: naming one leaves the other five at their registered builder,
    which is what lets a test drive one exit with a single fake, and lets the two nodes that call a
    model run against a fake client while the account has no credits. An unknown name is refused
    rather than ignored -- a typo'd override would otherwise be a fake that never ran and a test
    that passed anyway.

    An overridden node is never *built*: the router and the answer node validate the ``llm`` config
    section, the dispatcher its ``agent`` one and the gate its ``evidence`` one, so building one
    only to throw it away would make an injected fake depend on a config section it does not use.
    """
    unknown = sorted(set(nodes or ()) - set(NODE_BUILDERS))
    if unknown:
        known = ", ".join(NODE_BUILDERS)
        raise KeyError(
            f"no node(s) named {', '.join(map(repr, unknown))} in the graph; its nodes are: {known}"
        )
    names = node_names(cfg)
    absent = sorted(set(nodes or ()) - set(names))
    if absent:
        raise KeyError(
            f"node(s) {', '.join(map(repr, absent))} are not in this config's graph; "
            + "; ".join(f"{name} is built only under {CONDITIONAL[name]}" for name in absent)
        )
    given = dict(nodes or {})
    return {name: given[name] if name in given else NODE_BUILDERS[name](cfg) for name in names}


def guard(name: str, node: Node) -> Node:
    """``node``, with its update checked against what ``name`` is allowed to write.

    The wrapper is what makes :data:`~agent.state.NODE_WRITES` a rule instead of a comment. It
    costs one dict scan per node per query, against a class of bug -- a dropped key, a stolen key,
    an undeclared signal -- that otherwise surfaces as a wrong answer several nodes downstream.
    """

    def guarded(state: AgentState) -> dict[str, Any]:
        return check_update(name, node(state))

    guarded.__name__ = name
    return guarded


# --- routing --------------------------------------------------------------------------------------


def route_after_intent(state: AgentState) -> str:
    """``out_of_scope`` straight to its exit; everything else to dispatch.

    The only intent the graph itself acts on. The other four differ in *where the evidence comes
    from*, which is #36's dispatch table and not a branch here -- four parallel edges into four
    near-identical nodes would put the same table in two places.
    """
    intent = state.get("intent")
    if intent is None:
        raise ValueError(
            "the route_intent node did not set `intent`; every query gets one of "
            f"{', '.join(INTENTS)} -- an unparseable classification is 'out_of_scope' (#35)"
        )
    return "out_of_scope" if check_intent(intent, "route_intent") == "out_of_scope" else "dispatch"


def route_after_gate(state: AgentState) -> str:
    """Enough evidence to answer from, or hand it to a person."""
    evidence_ok = state.get("evidence_ok")
    if evidence_ok is None:
        raise ValueError(
            "the check_evidence node did not set `evidence_ok`; the gate returns True or False "
            "for every run, including one with empty docs (#37)"
        )
    return "answer" if evidence_ok else "refuse_handoff"


def route_after_answer(state: AgentState) -> str:
    """Done -- unless #38's citation check discarded the answer, which hands it over instead.

    The answer node claims its own exit, so this reads that claim rather than re-deriving it from
    ``signals``: the node that decided the answer was unusable is the node that knows.
    """
    exit_ = state.get("exit")
    if exit_ is None:
        raise ValueError(
            "the answer node did not set `exit`; it writes 'answer', or 'refuse_handoff' when its "
            "citation check rejected the draft (#38)"
        )
    check_exit(exit_, "answer")
    if exit_ == "refuse_handoff":
        return "refuse_handoff"
    if exit_ == "out_of_scope":
        raise ValueError(
            "the answer node set exit to 'out_of_scope'; that exit is decided by the router, "
            "before any evidence is gathered"
        )
    return END


def route_after_confidence(state: AgentState) -> str:
    """Show the answer, or hand it over: #92's verdict on the written draft."""
    confident = state.get("confident")
    if confident is None:
        raise ValueError(
            "the confidence_gate node did not set `confident`; it returns True or False for "
            "every answer it sees, False when the signal could not be read (#92)"
        )
    return END if confident else "refuse_handoff"


# --- the graph ------------------------------------------------------------------------------------


def build_graph(
    cfg: Config | None = None, nodes: Mapping[str, Node] | None = None
) -> CompiledStateGraph:
    """Wire the nodes and compile.

    ``cfg`` defaults to the live ``config.yaml``; ``nodes`` overrides individual nodes by name
    (see :func:`build_nodes`). Compiling is cheap and does nothing at the edges -- no index is
    opened, no key is read -- so this is safe to call in a test that never invokes the result.
    """
    config = cfg if isinstance(cfg, Config) else load_config() if cfg is None else Config(cfg)
    built = build_nodes(config, nodes)

    graph = StateGraph(AgentState)
    for name, node in built.items():
        graph.add_node(name, guard(name, node))

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        route_after_intent,
        {"out_of_scope": "out_of_scope", "dispatch": "dispatch"},
    )
    graph.add_edge("dispatch", "check_evidence")
    graph.add_conditional_edges(
        "check_evidence",
        route_after_gate,
        {"answer": "answer", "refuse_handoff": "refuse_handoff"},
    )
    # route_after_answer says "done" as END; with a confidence gate, done means "check it first".
    after_answer = "confidence_gate" if "confidence_gate" in built else END
    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {"refuse_handoff": "refuse_handoff", END: after_answer},
    )
    if "confidence_gate" in built:
        graph.add_conditional_edges(
            "confidence_gate",
            route_after_confidence,
            {"refuse_handoff": "refuse_handoff", END: END},
        )
    graph.add_edge("out_of_scope", END)
    graph.add_edge("refuse_handoff", END)
    return graph.compile()


#: Compiled graphs, keyed by config hash. Module-level, so it is per process.
_GRAPHS: dict[str, CompiledStateGraph] = {}


def get_graph(cfg: Config) -> CompiledStateGraph:
    """The compiled graph for ``cfg``, built on first use and kept for the rest of the process."""
    graph = _GRAPHS.get(cfg.hash)
    if graph is None:
        graph = build_graph(cfg)
        _GRAPHS[cfg.hash] = graph
    return graph


def clear_cache() -> None:
    """Drop every cached graph, so the next ``run`` rebuilds its nodes.

    For a process that changed a node builder after running a query -- a test, or a notebook
    swapping in a real router -- where the cached graph still holds the old one.
    """
    _GRAPHS.clear()


def check_final(state: Mapping[str, Any]) -> AgentState:
    """Refuse a finished run that cannot say how it ended.

    Both halves of the milestone's acceptance, checked once rather than in #39's report: the exit
    is one of the three and is readable off the state, and the answer is text a person can read.
    A run that reached END with ``exit`` unset means an exit node did not run, which is a wiring
    bug in this module and not something for a report to tabulate as a missing row.
    """
    exit_ = state.get("exit")
    if exit_ is None:
        raise ValueError(
            "the run finished with no `exit` set; every path through the graph ends in one of "
            f"{', '.join(EXITS)} (see agent/graph.py)"
        )
    check_exit(exit_, "the run")
    answer = state.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"the run exited by {exit_!r} with no answer text to show")
    return state  # type: ignore[return-value]


def run(
    query: str,
    cfg: Mapping[str, Any] | None = None,
    nodes: Mapping[str, Node] | None = None,
) -> AgentState:
    """One query through the graph; the final state, including which exit it took.

    ``cfg`` defaults to the live ``config.yaml`` and a plain mapping is frozen into a
    :class:`~config.Config`, the idiom :func:`retrieval.retrieve` uses -- so a caller's later edit
    to their own dict cannot change what a cached graph was built from.

    ``nodes`` replaces individual nodes for the duration of one call and bypasses the graph cache
    entirely: a fake chat client behind the router and the answer node is how the three exits are
    exercised with no credits, and ``route_intent_node(cfg, forced=...)`` is how #39's ``--intent``
    skips the classification without skipping anything else.
    """
    state = initial_state(query)
    config = cfg if isinstance(cfg, Config) else load_config() if cfg is None else Config(cfg)
    graph = build_graph(config, nodes) if nodes else get_graph(config)
    return check_final(graph.invoke(state))
