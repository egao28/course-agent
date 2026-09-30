"""The three course tools as an MCP server -- the external surface, not the graph's transport.

    uv run python -m agent.mcp_server            # serve stdio
    uv run python -m agent.mcp_server --list     # print what a client would see, then exit

``FastMCP("course-tools")`` with one ``@mcp.tool`` per function: :func:`get_course` (#29),
:func:`get_prereq_chain` (#30) and :func:`check_requirement` (#31). Any MCP client -- Claude
Desktop, another agent, a shell script -- gets the course catalog, the prerequisite graph and the
requirement audit without this repo's graph, its config, or its indexes.

**The tool logic lives in plain functions. This module registers them. The graph imports the
functions directly.** A tool call from a LangGraph node (#36) is an ordinary Python call: no
subprocess, no stdio framing, no event loop threaded through the graph, and no second failure mode
when the transport hiccups. Sending the graph to its own tools over MCP would buy nothing -- the
same three functions answer either way -- and would cost a process boundary on every node plus an
async layer the graph would then have to carry. Revisit only if a tool ends up somewhere the graph
cannot import.

The consequence worth keeping is that the Pydantic models are the contract for both callers. A
field added to ``CourseDetail`` for the graph shows up in this server's output schema with no edit
here, and the *input* schemas are built out of the query models' own ``FieldInfo`` objects
(``Annotated[str, CourseQuery.model_fields["course_id"]]``), so a description, a default and a
constraint like ``max_depth >= 1`` each have exactly one source. Re-typing them here is the drift
this borrows its way out of; ``tests/test_mcp_server.py`` compares every served schema against the
model it came from, which is what fails when a field is added to a model and not to a wrapper.

Three things are deliberately not here. **No auth, no HTTP mount, no identity middleware** --
``bloom-mcp`` needs all three because it is deployed behind a gateway; this is a local stdio server
over a published course catalog, and every tool is read-only (which is what the ``readOnlyHint``
annotation tells a client). **No ``OPENAI_API_KEY``** -- these three tools read committed JSON and
never embed, classify or generate, so the server starts with no key, no ``.env`` and no index built;
nothing it imports pulls in ``openai`` or ``chromadb``, and a test pins that in a subprocess.
**No second set of descriptions**: the docstring under each ``@mcp.tool`` *is* the tool description
a client reads, and it is what decides whether a model picks the right tool, so those three
docstrings are written for a model rather than for a maintainer -- the reasoning behind each tool
stays in ``agent/tools/``.

One note on concurrency: FastMCP runs a sync tool in a worker thread, so two calls arriving together
can both find the tools' module-level caches empty and both build one. The builds are pure reads of
committed files and the second assignment wins, so the cost is a duplicated 2.9 MB parse, never a
wrong answer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from fastmcp import Client, FastMCP
from mcp_types import Tool

from agent.tools import courses, prereqs, requirements
from agent.tools.courses import CourseDetail, CourseQuery
from agent.tools.prereqs import PrereqChain, PrereqQuery
from agent.tools.requirements import RequirementQuery, RequirementStatus

#: Read-only and closed-world: every tool answers out of committed JSON, calls nothing over the
#: network, and changes nothing. A client that shows a confirmation prompt before a tool call can
#: use these to skip it.
READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

mcp = FastMCP("course-tools")


# --- the tools ------------------------------------------------------------------------------------
#
# Each wrapper is the MCP registration of the function below it and nothing more: it takes that
# function's arguments, annotated with the query model's own fields, and returns what the function
# returns. The decorator replaces the module-level name with a FunctionTool, which is why these
# call `courses.get_course` rather than a bare name -- and why nothing but FastMCP calls them.


@mcp.tool(annotations=READ_ONLY)
def get_course(
    course_id: Annotated[str, CourseQuery.model_fields["course_id"]],
) -> CourseDetail:
    """Look up one course in the University of Illinois course catalog by its code.

    Answers a question that names a course outright -- "how many credits is CS 374?", "what is
    MATH 231 about?", "does ECE 220 have prerequisites?" -- with that course's catalog record: its
    title, credit hours, the full description, its Degree Attributes line, the prerequisite
    sentence verbatim, and how many sections are scheduled.

    The whole argument has to be one course code. "CS 374 or ECE 374" names two and "what is cs 374
    about" is a question; pull the code out first. Use the retriever, not this tool, for a question
    that describes a course instead of naming it ("machine learning courses").

    A code this catalog does not carry comes back with found=false and a message naming the term
    that was searched -- an answer, not an error. The catalog covers one term, so a real course can
    legitimately be missing from it.
    """
    return courses.get_course(course_id)


@mcp.tool(annotations=READ_ONLY)
def get_prereq_chain(
    course_id: Annotated[str, PrereqQuery.model_fields["course_id"]],
    max_depth: Annotated[int | None, PrereqQuery.model_fields["max_depth"]],
) -> PrereqChain:
    """Walk everything required before a course: the whole prerequisite chain, layered.

    Answers "what do I need before SE 495?" or "how far back does CS 374 go?". levels[0] is the
    immediate prerequisites, levels[1] what those require, and so on; each course appears exactly
    once, at its shortest distance from the one asked about. edges holds the (prerequisite,
    dependent) pairs among the courses reported.

    Quote raw rather than asserting the code list. A catalog sentence is often a choice -- "One of
    MATH 220, MATH 221, MATH 234" -- and the chain flattens it, so the chain says which courses are
    involved while the sentence says how many of them a student actually needs. Some chains also
    carry a cycle (the catalog has a few, and cycle_detected reports them), and some sentences add a
    condition no code can express, such as consent of the instructor.

    For the course's own description or credit hours, call get_course instead.
    """
    return prereqs.get_prereq_chain(course_id, max_depth)


@mcp.tool(annotations=READ_ONLY)
def check_requirement(
    major: Annotated[str, RequirementQuery.model_fields["major"]],
    taken_courses: Annotated[list[str], RequirementQuery.model_fields["taken_courses"]],
) -> RequirementStatus:
    """Check completed courses against a program's requirements, and report what is still missing.

    Answers "what Gen Eds do I still need?" and "am I on track for the CS degree?" for the programs
    this corpus carries requirements for: 'gened' (the campus General Education categories) and the
    degrees named in the major field's description. Every requirement comes back in exactly one of
    three lists -- satisfied, unsatisfied, or not_checked -- and unsatisfied requirements name the
    courses that would supply them.

    Any other program returns supported=false with every list empty and names the programs that are
    available. Do not fill that gap from memory: a degree audit that sounds right and is not is
    checkable only by the advisor the student was trying not to bother. Say the requirements are not
    in this data and hand the question over.

    not_checked is the honest list and belongs in the answer: hours, GPA, placement, and
    advisor-approved course lists are in the catalog but not in this data, so they are reported with
    the catalog's own sentence and never counted as satisfied. Codes this one-term catalog does not
    offer come back in unknown_courses -- they were not ignored, they could not be checked.
    """
    return requirements.check_requirement(major, taken_courses)


# --- what a client sees ---------------------------------------------------------------------------


async def list_tools() -> list[Tool]:
    """Ask a real client what this server serves.

    Through ``Client(mcp)`` -- in-memory, no subprocess, no port -- rather than by reading the
    registry, because a schema that cannot be generated or serialized fails here and not in
    somebody's client. It is the same check ``bloom-mcp/list_tools.py`` makes over HTTP, and it is
    what ``--list`` prints and the test suite asserts on.
    """
    async with Client(mcp) as client:
        return await client.list_tools()


def type_label(schema: Mapping[str, Any]) -> str:
    """An argument's type as a sentence: ``array of string``, or ``integer or null`` for a union."""
    if "anyOf" in schema:
        return " or ".join(type_label(option) for option in schema["anyOf"])
    kind = schema.get("type", "any")
    if kind == "array":
        return f"array of {type_label(schema.get('items', {}))}"
    return str(kind)


def format_tool(tool: Tool) -> str:
    """One tool as a client reads it: description, arguments with their schemas, result fields."""
    lines = [tool.name]
    body = (tool.description or "(no description)").splitlines()
    lines += [f"  {line}" if line else "" for line in body]
    schema = tool.input_schema
    required = set(schema.get("required", []))
    lines.append("  arguments:")
    for name, field in schema.get("properties", {}).items():
        need = "required" if name in required else "optional"
        lines.append(f"    {name} ({type_label(field)}, {need}) -- {field.get('description', '')}")
    fields = ", ".join((tool.output_schema or {}).get("properties", {}))
    lines.append(f"  returns: {fields}")
    return "\n".join(lines)


def format_listing(tools: Sequence[Tool]) -> str:
    """The whole surface, in the order the server registered it."""
    header = f"{mcp.name} serves {len(tools)} tool{'' if len(tools) == 1 else 's'}"
    return "\n\n".join([header, *(format_tool(tool) for tool in tools)])


def main(argv: Sequence[str] | None = None) -> None:
    """Serve the three tools over stdio, or with ``--list`` print their schemas and exit."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.mcp_server",
        description="Serve the three course tools over MCP (stdio), or list what a client sees.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the tools and their schemas and exit, instead of serving",
    )
    args = parser.parse_args(argv)
    if args.list:
        print(format_listing(asyncio.run(list_tools())))
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())
