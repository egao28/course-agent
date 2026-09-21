"""The three nodes #36-#38 fill in, standing in until they do.

Each one raises, and that is the decision worth explaining. A stub that returned something
plausible -- ``intent="course_fact"``, an ``answer`` string, ``evidence_ok=True`` -- would let the
graph run end to end today, and #39's smoke report would be produced from it: ten queries, three
exits, a table of numbers, and not one model call or catalog lookup behind any of it. A report that
reads as a run is worse than no report, and the point of the whole Day 2 milestone is that the
trace is honest.

So ``build_graph()`` compiles with these in place -- the wiring, the routing, the two fixed exits
and now the router are all testable without them -- and ``run()`` raises the moment it reaches one,
naming the issue that fills it in. The three exits are proven by injecting nodes (see
``tests/test_graph.py``), which is also how #36-#38 will each be tested on their own before any of
them can run together.
"""

from __future__ import annotations

from typing import NoReturn

from agent.nodes import Node, NodeBuilder
from agent.state import NODE_WRITES, AgentState
from config import Config

#: The node name, the issue that implements it, and what it is for -- so a traceback names the
#: work rather than the placeholder. An entry disappears from here as its issue lands.
PENDING: dict[str, tuple[int, str]] = {
    "dispatch": (36, "gather evidence -- retrieve(), or one of the three tools"),
    "check_evidence": (37, "decide whether the evidence is enough to answer from"),
    "answer": (38, "write the answer from the evidence and verify its citations"),
}


def stub(name: str) -> Node:
    """A node that refuses to pretend, naming its issue, its job and the key it owes the state."""
    if name not in PENDING:
        known = ", ".join(PENDING)
        raise KeyError(f"no pending node named {name!r}; the stubs are: {known}")

    def node(state: AgentState) -> NoReturn:
        issue, job = PENDING[name]
        keys = ", ".join(sorted(NODE_WRITES[name]))
        raise NotImplementedError(
            f"the {name!r} node is a stub: it has to {job} and write {keys} (#{issue}). "
            f"Pass a node for it -- build_graph(nodes={{{name!r}: ...}}) -- or implement #{issue}"
        )

    node.__name__ = f"stub_{name}"
    return node


def stub_builder(name: str) -> NodeBuilder:
    """The :data:`~agent.nodes.NodeBuilder` form of :func:`stub`, for the graph's registry."""

    def build(cfg: Config) -> Node:
        return stub(name)

    build.__name__ = f"build_stub_{name}"
    return build
