"""Tests for report/README.md: the study's front page, held to the code and to its own sources.

A hand-written report cannot be unit tested into being *true*. What it can be held to is every
claim it makes that something else in the repo already decides, and this file holds it to three:

* **the architecture diagram is the graph** -- its node ids are exactly
  :data:`agent.graph.NODE_BUILDERS`, and the dashed ones are exactly
  :data:`agent.graph.CONDITIONAL`. #106 asks for that test by name, because a diagram is the one
  part of a report nobody re-reads after the code moves, and a reader who checks the prose against
  the picture would be checking it against a lie.
* **the sections are the ones that were commissioned, in order** -- motivation, RQs, system, setup,
  results, error analysis, limitations.
* **every ``[src]`` link resolves** -- the file is there, and the key path in the link's title is a
  real key in it, a real column in a JSONL row, or a real heading in a Markdown file. This is the
  cheap half of #107's ``--check``: it catches a mistyped key path, which is invisible to a reader
  and fatal to the promise that every number traces to a committed file. #107 adds the expensive
  half -- comparing the *value* at that path against the digits in the prose.

``report/day4_ablation_c73b197d.md`` is in :data:`PENDING`: the Day 4 report is written and
reviewed on ``feat/77-day4-report`` (#86), which is not an ancestor of this branch, so the two
links into it dangle until that PR merges into ``feat/56-drafting-harness``. Delete the constant
then; a link to anything else that is missing is a failure today.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from agent.graph import CONDITIONAL, NODE_BUILDERS
from config import REPO_ROOT

REPORT = REPO_ROOT / "report" / "README.md"

#: The sections #106 commissioned, in the order it commissioned them. Matched as prefixes of an
#: ``## `` heading, so the wording after the number is free.
SECTIONS = (
    "## 1. Motivation",
    "## 2. Research questions",
    "## 3. System and method",
    "## 4. Experimental setup",
    "## 5. Results",
    "## 6. Error analysis",
    "## 7. Limitations",
)

#: Diagram nodes that are not graph nodes: where a query enters and where a reply leaves. The
#: graph's own START/END, which LangGraph supplies and :func:`agent.graph.node_names` does not.
PSEUDO_NODES = frozenset({"query", "reply"})

#: Linked files that do not exist on this branch yet, and the issue that brings each one. A link
#: to a file outside this set must resolve.
PENDING: dict[str, str] = {
    "report/day4_ablation_c73b197d.md": "#77, on feat/77-day4-report (PR #86)",
}

#: ``0.405 [src](../eval/runs/x.json "a.key.path")`` -- a number's source, and where in it to look.
SRC_RE = re.compile(r'\[src\]\((?P<path>[^)\s]+) "(?P<key>[^"]+)"\)')

#: ``[text](path)`` or ``[text](path "title")``, for any local file the report points at.
LINK_RE = re.compile(r"\[[^\]]+\]\((?P<path>(?!https?:)[^)\s#]+)(?: \"[^\"]*\")?\)")

#: ``<!-- BEGIN gen:name -->`` at the start of a line: a block final_report.py (#107) rewrites.
MARKER_RE = re.compile(r"^<!-- (?P<edge>BEGIN|END) gen:(?P<name>[a-z0-9_]+) -->$", re.MULTILINE)

#: ``route_intent["..."]`` or ``query(["..."])`` -- a node declared in the mermaid block.
DECL_RE = re.compile(r"^\s{4}(?P<name>[a-z_][a-z0-9_]*)[\[(]", re.MULTILINE)

#: ``a --> b``, ``a -->|label| b``, ``a -.-> b``, ``a -.->|label| b``.
EDGE_RE = re.compile(
    r"^\s{4}(?P<src>[a-z_][a-z0-9_]*)\s+(?P<arrow>-->|-\.->)(?:\|[^|]*\|)?\s*(?P<dst>[a-z_][a-z0-9_]*)\s*$",
    re.MULTILINE,
)


@pytest.fixture(scope="module")
def text() -> str:
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def blocks(text: str) -> dict[str, str]:
    """``{name: body}`` for each generated block, checking the markers pair up as it goes."""
    out: dict[str, str] = {}
    open_name: str | None = None
    start = 0
    for m in MARKER_RE.finditer(text):
        name = m.group("name")
        if m.group("edge") == "BEGIN":
            assert open_name is None, f"gen:{name} opens inside gen:{open_name}"
            assert name not in out, f"gen:{name} is used twice; a block name names one table"
            open_name, start = name, m.end()
        else:
            assert open_name == name, f"gen:{name} closes, but gen:{open_name} is open"
            out[name], open_name = text[start : m.start()], None
    assert open_name is None, f"gen:{open_name} is never closed"
    return out


# --- the diagram is the graph ----------------------------------------------------------------


def test_diagram_nodes_are_the_graphs_nodes(blocks: dict[str, str]) -> None:
    """Every node in the picture is a node in the registry, and every node is in the picture."""
    declared = set(DECL_RE.findall(blocks["architecture"]))
    assert declared - PSEUDO_NODES == set(NODE_BUILDERS), (
        "the architecture diagram and agent.graph.NODE_BUILDERS have drifted apart; "
        f"only in the diagram: {sorted(declared - PSEUDO_NODES - set(NODE_BUILDERS))}, "
        f"only in the graph: {sorted(set(NODE_BUILDERS) - declared)}"
    )


def test_diagram_edges_reach_only_declared_nodes(blocks: dict[str, str]) -> None:
    """No edge names a node that was never declared -- a typo'd id draws a second, empty box."""
    declared = set(DECL_RE.findall(blocks["architecture"]))
    edges = EDGE_RE.findall(blocks["architecture"])
    assert edges, "the architecture block has no edges; the regex or the diagram changed shape"
    for src, _, dst in edges:
        assert src in declared, f"edge from undeclared node {src!r}"
        assert dst in declared, f"edge to undeclared node {dst!r}"


def test_only_conditional_nodes_are_dashed(blocks: dict[str, str]) -> None:
    """Dashed edges touch the nodes some configs leave out, and nothing else.

    ``confidence_gate`` is dormant under the config this report describes (``evidence.method:
    top_score``). A solid box for it would say the deployed agent has a calibrated gate.
    """
    edges = EDGE_RE.findall(blocks["architecture"])
    for node in CONDITIONAL:
        touching = [(src, arrow, dst) for src, arrow, dst in edges if node in (src, dst)]
        assert touching, f"{node} is in agent.graph.CONDITIONAL but has no edge in the diagram"
        solid = [e for e in touching if e[1] != "-.->"]
        assert not solid, f"{node} is conditional but is drawn with a solid edge: {solid}"
    for src, arrow, dst in edges:
        if arrow == "-.->":
            assert {src, dst} & set(CONDITIONAL), (
                f"{src} -.-> {dst} is dashed, but neither end is conditional in agent.graph; "
                f"conditional: {sorted(CONDITIONAL)}"
            )


# --- the shape of the report -----------------------------------------------------------------


def test_sections_in_the_commissioned_order(text: str) -> None:
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    found = [h for h in headings if any(h.startswith(s) for s in SECTIONS)]
    assert len(found) == len(SECTIONS), (
        f"missing section(s): {set(SECTIONS) - {h[:40] for h in found}}"
    )
    assert [h.split(".")[0] for h in found] == [s.split(".")[0] for s in SECTIONS], (
        f"sections are out of order: {found}"
    )


def test_every_generated_block_is_named_once(blocks: dict[str, str]) -> None:
    """The fixture does the checking; this asserts there is something to check."""
    assert len(blocks) >= 8, f"only {len(blocks)} generated blocks: {sorted(blocks)}"
    for name, body in blocks.items():
        assert body.strip(), f"gen:{name} is empty"


# --- every link, and every key path inside one -----------------------------------------------


def resolve(path: str) -> Path:
    """A link target, as a path from the repo root (links are relative to ``report/``)."""
    return (REPORT.parent / path).resolve()


def walk(data: Any, key_path: str) -> bool:
    """Is ``key_path`` (dotted) a real path into ``data``?"""
    node = data
    for key in key_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


@pytest.fixture(scope="module")
def links(text: str) -> list[str]:
    return sorted({m.group("path") for m in LINK_RE.finditer(text)})


def test_every_local_link_resolves(links: list[str]) -> None:
    missing = [p for p in links if not resolve(p).exists()]
    unexpected = [p for p in missing if str(resolve(p).relative_to(REPO_ROOT)) not in PENDING]
    assert not unexpected, f"link(s) to files that do not exist: {unexpected}"


def test_pending_links_are_still_pending(links: list[str]) -> None:
    """Every entry in PENDING is still linked and still absent -- or the constant is stale."""
    linked = {str(resolve(p).relative_to(REPO_ROOT)) for p in links}
    for path, why in PENDING.items():
        assert path in linked, f"PENDING lists {path} ({why}), but nothing links to it any more"


def test_src_key_paths_are_real(text: str) -> None:
    """The key path in a ``[src]`` link's title exists in the file the link names.

    A number whose source link points at a key that is not there traces to nothing, and reads
    exactly like one that traces to something.
    """
    checked = 0
    for m in SRC_RE.finditer(text):
        path, key = m.group("path"), m.group("key")
        target = resolve(path)
        if str(target.relative_to(REPO_ROOT)) in PENDING:
            continue
        assert target.exists(), f"[src] link to a missing file: {path}"
        if target.suffix == ".json":
            assert walk(json.loads(target.read_text()), key), f"{path} has no key path {key!r}"
        elif target.suffix == ".jsonl":
            first = json.loads(target.read_text().splitlines()[0])
            assert key in first, f"{path} rows have no column {key!r}"
        elif target.suffix == ".md":
            headings = [
                ln.lstrip("# ").strip()
                for ln in target.read_text().splitlines()
                if ln.startswith("#")
            ]
            assert any(h.startswith(key) for h in headings), f"{path} has no heading {key!r}"
        else:
            pytest.fail(f"[src] link to an unhandled file type: {path}")
        checked += 1
    assert checked >= 10, f"only {checked} [src] links checked; the report should have more"
