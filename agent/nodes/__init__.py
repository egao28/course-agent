"""The graph's nodes: one module each, one state key each.

A node is a plain callable from :class:`~agent.state.AgentState` to a mapping of the keys it
wrote, and it is *built* from the config rather than reading it at call time::

    def route_intent(cfg: Config, chat: Chat | None = None) -> Node:
        ...
        return node

That is the arrangement :data:`retrieval.BUILDERS` already uses for the retrievers, and it is here
for the same two reasons. A node that needs something the config cannot hold -- a chat client
(#35, #38), a retriever (#36) -- takes it as an ordinary argument, so a test injects a fake with a
keyword and not through a framework dictionary, which is what makes the whole graph runnable with
no OpenAI credits. And the built node is a one-argument function, which is the only thing LangGraph
has to know about it. A node whose whole behaviour *is* config (#37's thresholds) takes nothing
extra: a number handed in past the config would be a tuning decision no report could be traced
back to.

``agent.graph`` holds the registry (:data:`~agent.graph.NODE_BUILDERS`), wires the six nodes
together and wraps each one in :func:`~agent.state.check_update`. The router is
:mod:`agent.nodes.route_intent`, the dispatcher :mod:`agent.nodes.dispatch`, the evidence gate
:mod:`agent.nodes.check_evidence` and the two fixed exits :mod:`agent.nodes.exits`; the answer
node is a stub in :mod:`agent.nodes.stubs` until #38 replaces it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent.state import AgentState
from config import Config

#: What LangGraph runs: state in, the keys this node wrote out. Anything else it needs was closed
#: over when it was built.
Node = Callable[[AgentState], Mapping[str, Any]]

#: How the graph gets a node: from the config, once per compiled graph. Extra dependencies (a chat
#: client, a fake tool) are keyword arguments on the concrete builder, defaulted to the real thing.
NodeBuilder = Callable[[Config], Node]
