"""Tests for agent/mcp_server.py: what a client is served, and that it is the same answer.

Every test here goes through a real ``fastmcp.Client`` over the in-memory transport rather than
calling the decorated functions, because the point of the module is the round trip: a schema that
cannot be generated, an argument that will not validate, or a model that will not serialize fails
in a client and nowhere else. That is what ``bloom-mcp/list_tools.py`` checks by hand over HTTP,
run here on every commit instead.

The suite reads the live corpus on purpose, like #29/#30/#31's acceptance sections do: the claim is
that MCP and Python return the *same* object, which is only worth asserting about real records.
Nothing here writes, and ``fresh_caches`` keeps a corpus another test module redirected at a fixture
from answering these.

Two kinds of test, and they fail for different reasons. The schema tests compare each served schema
against the Pydantic model it was built from -- they fail when a field is added to a model and not
to its wrapper, which is the drift the borrowed ``FieldInfo`` annotations exist to prevent. The
round-trip tests compare a call over MCP against the same call in Python -- they fail when the
server starts answering something other than the function it wraps.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp_types import Tool
from pydantic import BaseModel

from agent import mcp_server as ms
from agent.tools import courses as tc
from agent.tools import prereqs as tp
from agent.tools import requirements as tr

REPO_ROOT = Path(__file__).parent.parent

#: The schema each tool's arguments are built from, and the model its result is.
QUERY_MODELS: dict[str, type[BaseModel]] = {
    "get_course": tc.CourseQuery,
    "get_prereq_chain": tp.PrereqQuery,
    "check_requirement": tr.RequirementQuery,
}
RESULT_MODELS: dict[str, type[BaseModel]] = {
    "get_course": tc.CourseDetail,
    "get_prereq_chain": tp.PrereqChain,
    "check_requirement": tr.RequirementStatus,
}


@pytest.fixture(autouse=True)
def fresh_caches():
    """No test inherits a fixture corpus another module pointed the tools at."""
    for module in (tc, tp, tr):
        module.clear_cache()
    yield
    for module in (tc, tp, tr):
        module.clear_cache()


@pytest.fixture(scope="module")
def tools() -> dict[str, Tool]:
    """What a client sees, keyed by tool name. Module-scoped: the listing touches no data."""
    return {tool.name: tool for tool in asyncio.run(ms.list_tools())}


def call(name: str, arguments: dict[str, Any]) -> Any:
    """One tool call over the in-memory transport, as a client makes it."""

    async def go():
        async with Client(ms.mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(go())


def without_titles(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A model's properties as a tool schema spells them -- pydantic's ``title`` keys dropped."""
    return {
        name: {key: value for key, value in field.items() if key != "title"}
        for name, field in schema["properties"].items()
    }


# --- the surface ----------------------------------------------------------------------------------


def test_the_server_serves_exactly_the_three_tools(tools):
    assert set(tools) == {"get_course", "get_prereq_chain", "check_requirement"}


def test_the_server_is_named_for_what_it_serves():
    assert ms.mcp.name == "course-tools"


@pytest.mark.parametrize("name", sorted(QUERY_MODELS))
def test_every_tool_is_declared_read_only(tools, name):
    """Nothing here writes, and a client that prompts before a tool call should not prompt."""
    annotations = tools[name].annotations
    assert annotations is not None
    assert annotations.read_only_hint is True
    assert annotations.open_world_hint is False


@pytest.mark.parametrize("name", sorted(QUERY_MODELS))
def test_every_tool_carries_a_description_written_for_a_client(tools, name):
    """The docstring is what a model reads to pick a tool, so it says what the tool answers."""
    description = tools[name].description or ""
    assert len(description.split()) > 30
    assert "#" not in description, "issue references belong in agent/tools/, not in a tool schema"


# --- the schemas ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(QUERY_MODELS))
def test_the_arguments_are_the_query_models_fields(tools, name):
    """Every field of the query model, with its description and constraints, and nothing else.

    This is the drift guard the whole ``Annotated[..., Model.model_fields[...]]`` shape exists for:
    add a field to a query model without adding the argument, or restate a description here, and
    the two schemas stop matching.
    """
    served = tools[name].input_schema
    model = QUERY_MODELS[name].model_json_schema()
    assert without_titles(served) == without_titles(model)
    assert served.get("required", []) == model.get("required", [])


@pytest.mark.parametrize("name", sorted(RESULT_MODELS))
def test_the_result_schema_is_the_result_models(tools, name):
    """Generated from the model, so a field added for the graph is published with no edit here."""
    served = tools[name].output_schema
    assert served is not None
    model = RESULT_MODELS[name].model_json_schema()
    assert list(served["properties"]) == list(model["properties"])
    assert served.get("required", []) == model.get("required", [])


def test_a_constraint_on_the_model_is_enforced_at_the_call(tools):
    """``max_depth`` is ``ge=1`` on PrereqQuery; the schema says so and the call refuses 0."""
    depth = tools["get_prereq_chain"].input_schema["properties"]["max_depth"]
    assert {"minimum": 1, "type": "integer"} in depth["anyOf"]
    with pytest.raises(ToolError, match="greater than or equal to 1"):
        call("get_prereq_chain", {"course_id": "CS 374", "max_depth": 0})


def test_an_argument_no_model_declares_is_refused():
    with pytest.raises(ToolError, match="Unexpected keyword argument"):
        call("get_course", {"course_id": "CS 374", "mode": "hybrid"})


def test_a_missing_required_argument_is_refused():
    with pytest.raises(ToolError, match="Missing required argument"):
        call("get_course", {})


def test_an_optional_argument_defaults_the_way_its_model_does():
    """``taken_courses`` may be omitted, and omitting it is the model's empty list."""
    omitted = call("check_requirement", {"major": "gened"})
    assert omitted.structured_content == tr.check_requirement("gened").model_dump(mode="json")


# --- the same answer as the function --------------------------------------------------------------


def test_get_course_over_mcp_is_the_call_in_python():
    """#33's acceptance: the dict a client gets is the model the graph gets."""
    result = call("get_course", {"course_id": "cs 374"})
    assert result.structured_content == tc.get_course("cs 374").model_dump(mode="json")
    assert result.structured_content["code"] == "CS 374"
    assert result.structured_content["found"] is True


def test_a_course_the_catalog_does_not_have_is_an_answer_not_an_error():
    """``found=False`` reaches the client as a result; a raising tool would be an MCP error."""
    result = call("get_course", {"course_id": "CS 999"})
    assert result.structured_content["found"] is False
    assert "not in the catalog" in result.structured_content["message"]


def test_get_prereq_chain_over_mcp_is_the_call_in_python():
    result = call("get_prereq_chain", {"course_id": "CS 374"})
    assert result.structured_content == tp.get_prereq_chain("CS 374").model_dump(mode="json")
    assert result.structured_content["levels"][0][0]["code"] == "CS 173"


def test_max_depth_reaches_the_walk():
    result = call("get_prereq_chain", {"course_id": "CS 374", "max_depth": 1})
    assert result.structured_content == tp.get_prereq_chain("CS 374", 1).model_dump(mode="json")
    assert len(result.structured_content["levels"]) == 1
    assert result.structured_content["truncated"] is True


def test_check_requirement_over_mcp_is_the_call_in_python():
    taken = ["RHET 105", "MATH 221"]
    result = call("check_requirement", {"major": "gened", "taken_courses": taken})
    assert result.structured_content == tr.check_requirement("gened", taken).model_dump(mode="json")
    assert [c["id"] for c in result.structured_content["satisfied"]] == ["comp1"]
    assert result.structured_content["unknown_courses"] == ["MATH 221"]


def test_a_program_the_data_does_not_carry_is_refused_over_mcp_too():
    """The refusal is a result, so a client gets the same empty lists the graph does."""
    result = call("check_requirement", {"major": "Aerospace Engineering, BS"})
    assert result.structured_content["supported"] is False
    assert result.structured_content["satisfied"] == []
    assert result.structured_content["unsatisfied"] == []
    assert result.structured_content["not_checked"] == []


# --- the listing and the process ------------------------------------------------------------------


def test_the_listing_names_every_tool_with_its_arguments(tools, capsys):
    ms.main(["--list"])
    printed = capsys.readouterr().out
    assert "course-tools serves 3 tools" in printed
    for name in tools:
        assert f"\n{name}\n" in printed
    assert "course_id (string, required)" in printed
    assert "max_depth (integer or null, optional)" in printed
    assert "taken_courses (array of string, optional)" in printed


def test_the_listing_reports_the_result_fields(tools, capsys):
    ms.main(["--list"])
    printed = capsys.readouterr().out
    assert "returns: found, code, name," in printed


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter in the repo root, with no ``OPENAI_API_KEY`` in its environment."""
    env = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


def test_the_server_starts_without_an_openai_key():
    """#33's acceptance: three JSON files, no embedding, no completion, so no key and no index."""
    listed = run_python("-m", "agent.mcp_server", "--list")
    assert listed.returncode == 0, listed.stderr
    assert "course-tools serves 3 tools" in listed.stdout


def test_importing_the_server_pulls_in_neither_openai_nor_chromadb():
    """The tools read committed JSON; dragging the retrieval stack in would be a cost, not a bug."""
    probe = (
        "import sys, agent.mcp_server; print('openai' in sys.modules, 'chromadb' in sys.modules)"
    )
    loaded = run_python("-c", probe)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "False False"
