"""The agent layer: one graph over the retrieval layer's ``retrieve(query, mode)``.

Day 2's half of the repo. :mod:`agent.graph` compiles the six nodes in :mod:`agent.nodes` into the
three-exit graph, :mod:`agent.tools` holds the three catalog tools the dispatcher can call instead
of retrieving, and :mod:`agent.state` is the one record shape all of them read and write.

The dependency arrow points one way: this package imports :mod:`retrieval` and :mod:`config`, and
neither imports it back. :mod:`agent.mcp_server` re-exports the three tools over MCP -- an external
surface, not the graph's own transport -- and ``python -m agent`` (:mod:`agent.__main__`) answers a
single question from the command line.
"""
