"""The three catalog tools the dispatcher calls when a question names its own answer.

:mod:`agent.tools.courses` looks one course code up, :mod:`agent.tools.prereqs` walks the
prerequisite closure behind one, and :mod:`agent.tools.requirements` audits a transcript against a
program's rules. Each one reads ``data/`` directly rather than going through an index: the question
"how many credits is CS 374?" names the row it wants, so a search over 4,502 chunks is the wrong
instrument for it.

Where the *data* ends all three return a value rather than raising -- a not-found record, an
empty chain, a refusal -- and keep raising for a malformed *argument*, so the dispatcher turns
whatever comes back into the single :class:`~agent.state.Evidence` shape the answer node reads.

:mod:`agent.mcp_server` publishes the same three functions over MCP, so the schemas they declare
are the external contract too.
"""
