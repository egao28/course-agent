"""Tests for report/README.md and the generator behind it: the study's front page, held to its
sources.

A hand-written report cannot be unit tested into being *true*. What it can be held to is every
claim it makes that something else in the repo already decides, and this file holds it to four:

* **the architecture diagram is the graph** -- its node ids are exactly
  :data:`agent.graph.NODE_BUILDERS`, and the dashed edges are exactly the ones only a config in
  :data:`agent.graph.CONDITIONAL` wires. #106 asks for that test by name, because a diagram is the
  one part of a report nobody re-reads after the code moves, and a reader who checks the prose
  against the picture would be checking it against a lie.
* **the sections are the ones that were commissioned, in order** -- motivation, RQs, system, setup,
  results, error analysis, limitations.
* **every link resolves** -- the file a link names is there, whatever the link is for.
* **``final_report.py --check`` passes on the committed file** (#107), which is the expensive half:
  every generated block is what the generator writes today, every number in the prose carries a
  ``[src]`` link, and every linked value is in the linked file at the precision it is printed to.
  That check is one function call, so this file's job is to run it on the committed report and then
  to break the report on purpose, one way at a time, and watch it fail. A check that cannot fail is
  not a check.

There is no exemption list for links. ``report/day4_ablation_c73b197d.md`` (#77) was the one link
that did not resolve here, and the merge of ``feat/56-drafting-harness`` brought it, so a link
target that is missing is simply a failure.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from agent.graph import CONDITIONAL, NODE_BUILDERS
from config import REPO_ROOT, load_config
from eval.scripts import final_report as fr
from eval.scripts.review import DATASET_PATH
from eval.scripts.risk_coverage import FIGURES_DIR
from eval.scripts.signals import RUNS_DIR

pytest.importorskip("scipy")

REPORT = fr.REPORT

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

#: ``[text](path)`` or ``[text](path "title")``, for any local file the report points at.
LINK_RE = re.compile(r"\[[^\]]+\]\((?P<path>(?!https?:)[^)\s#]+)(?: \"[^\"]*\")?\)")

#: ``route_intent["..."]`` or ``query(["..."])`` -- a node declared in the mermaid block.
DECL_RE = re.compile(r"^\s{4}(?P<name>[a-z_][a-z0-9_]*)[\[(]", re.MULTILINE)

#: ``a --> b``, ``a -->|label| b``, ``a -.-> b``, ``a -.->|label| b``.
EDGE_RE = re.compile(
    r"^\s{4}(?P<src>[a-z_][a-z0-9_]*)\s+(?P<arrow>-->|-\.->)(?:\|[^|]*\|)?\s*(?P<dst>[a-z_][a-z0-9_]*)\s*$",
    re.MULTILINE,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def text() -> str:
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def facts(cfg):
    """The committed inputs, read and checked once -- the expensive fixture in this file."""
    return fr.read_facts(cfg, fr.Sources())


@pytest.fixture(scope="module")
def bodies(facts):
    return fr.render_blocks(facts)


@pytest.fixture(scope="module")
def blocks(text: str) -> dict[str, str]:
    """``{name: body}`` for each generated block, checking the markers pair up as it goes."""
    return {one.name: text[one.start : one.end] for one in fr.marked(text)}


def problems(text: str, facts, bodies) -> list[str]:
    return [problem.message for problem in fr.check(text, facts, bodies)]


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


def test_the_diagram_says_the_gate_is_dormant(blocks: dict[str, str], cfg) -> None:
    """The dormant node is labelled dormant, because the picture is of the deployed config."""
    for node in CONDITIONAL:
        line = next(ln for ln in blocks["architecture"].splitlines() if ln.strip().startswith(node))
        assert fr.DORMANT.strip(", ") in line, f"{node} is not drawn as dormant: {line!r}"


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
    assert set(blocks) == set(fr.BLOCKS), "the markers and the renderers have drifted apart"
    for name, body in blocks.items():
        assert body.strip(), f"gen:{name} is empty"


def test_every_local_link_resolves(text: str) -> None:
    links = sorted({m.group("path") for m in LINK_RE.finditer(text)})
    missing = [p for p in links if not (REPORT.parent / p).resolve().exists()]
    assert not missing, f"link(s) to files that do not exist: {missing}"


# --- #107: the committed report passes its own check -----------------------------------------


def test_check_passes_on_the_committed_report(text: str, facts, bodies) -> None:
    """The acceptance criterion: every block current, every number sourced, every value found."""
    assert problems(text, facts, bodies) == []


def test_the_generator_is_idempotent(text: str, bodies) -> None:
    """Rewriting over unchanged inputs writes the same bytes, so --check is not a clock."""
    assert fr.splice(text, bodies) == text


# --- ... and fails, one way at a time --------------------------------------------------------


def test_a_typed_in_wrong_number_fails(text: str, facts, bodies) -> None:
    """The expensive half: a prose number that no longer matches the file it cites."""
    broken = text.replace("with AUROC\n0.847 [0.722, 0.953]", "with AUROC\n0.857 [0.722, 0.953]")
    assert broken != text, "the sentence this test edits has been rewritten"
    found = problems(broken, facts, bodies)
    assert any("0.857 is not a value in" in one for one in found), found


def test_a_number_the_file_still_has_passes(text: str, facts, bodies) -> None:
    """The same sentence, rounded the other way at one more decimal: 0.847 is 0.8471, not 0.8470."""
    ok = text.replace("with AUROC\n0.847 [0.722, 0.953]", "with AUROC\n0.8471 [0.722, 0.953]")
    assert ok != text
    assert [one for one in problems(ok, facts, bodies) if "0.8471" in one] == []


def test_a_stale_block_fails(text: str, facts, bodies) -> None:
    broken = text.replace("| `bm25` | 0.870", "| `bm25` | 0.871")
    assert broken != text, "the day4 table this test edits has changed shape"
    assert any("gen:day4_ablation is out of date" in one for one in problems(broken, facts, bodies))


def test_an_unsourced_number_fails(text: str, facts, bodies) -> None:
    broken = text + "\nThe agent answered 7 questions about the weather.\n"
    assert any("7 has no [src] link" in one for one in problems(broken, facts, bodies))


def test_a_number_in_a_table_outside_a_block_fails(text: str, facts, bodies) -> None:
    broken = text + "\n| rows | 41 |\n"
    assert any("a table outside a gen: block" in one for one in problems(broken, facts, bodies))


def test_a_mistyped_key_path_fails(text: str, facts, bodies) -> None:
    broken = text.replace('"metrics.false_block")', '"metrics.false_blocks")')
    assert broken != text
    found = problems(broken, facts, bodies)
    assert any("has no key path 'metrics.false_blocks'" in one for one in found), found


def test_one_bad_path_in_a_multi_key_title_fails(text: str, facts, bodies) -> None:
    """A title naming three paths is not excused by two of them resolving."""
    broken = text.replace('"n, alpha, min_answered_to_accept")', '"n, alpha, min_answered")')
    assert broken != text
    assert any("has no key path 'n, alpha, min_answered'" in one
               for one in problems(broken, facts, bodies))


def test_a_link_to_a_missing_file_fails(text: str, facts, bodies) -> None:
    broken = text.replace("../eval/runs/e2e_metrics_c73b197d.json \"n\")",
                          "../eval/runs/e2e_metrics_nope.json \"n\")")
    assert broken != text
    assert any("[src] link to a missing file" in one for one in problems(broken, facts, bodies))


def test_a_hex_key_this_config_does_not_produce_fails(text: str, facts, bodies) -> None:
    """A stale hash reads exactly like a current one, and names a file nobody has."""
    broken = text.replace("config hash `c73b197d`", "config hash `c73b197e`", 1)
    assert broken != text
    assert any("`c73b197e` is not a key this config produces" in one
               for one in problems(broken, facts, bodies))


def test_an_unclosed_marker_is_refused(text: str) -> None:
    broken = text.replace("<!-- END gen:day6_exits -->", "")
    with pytest.raises(ValueError, match="gen:day6_exits"):
        fr.marked(broken)


def test_a_block_with_no_renderer_is_refused(text: str) -> None:
    broken = text.replace("gen:day6_exits", "gen:day6_exitss")
    with pytest.raises(ValueError, match="no renderer for gen:day6_exitss"):
        fr.marked(broken)


# --- the sha256 chain ------------------------------------------------------------------------


def test_a_moved_input_is_refused(cfg, tmp_path) -> None:
    """Day 6's chain: the metrics file the attribution pass was run against, changed underneath it.

    Day 4's and Day 5's links are their own readers' tests (``test_day4_report``,
    ``test_day5_report``); this one covers the three files #107 adds to the chain.
    """
    runs = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, runs)
    path = runs / f"e2e_metrics_{cfg.hash}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metrics"]["false_block"]["k"] = 9
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = fr.Sources(runs_dir=runs, dataset=DATASET_PATH, figures_dir=FIGURES_DIR)
    with pytest.raises(ValueError, match="was attributed against another"):
        fr.read_facts(cfg, src)


def test_an_unfinished_attribution_pass_is_refused(cfg, tmp_path) -> None:
    runs = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, runs)
    path = runs / f"attribution_test_{cfg.hash}.meta.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["complete"] = False
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = fr.Sources(runs_dir=runs, dataset=DATASET_PATH, figures_dir=FIGURES_DIR)
    with pytest.raises(ValueError, match="unfinished pass"):
        fr.read_facts(cfg, src)


# --- the rules the check is made of ----------------------------------------------------------


def test_a_table_of_counts_stands_for_its_total() -> None:
    """``exits`` is where the rows went and also how many there were."""
    values = fr._numbers(fr._leaves({"answer": 37, "refuse_handoff": 20, "out_of_scope": 6}))
    assert {3.0, 6.0, 20.0, 37.0, 63.0} == values


def test_a_list_of_ids_stands_for_how_many() -> None:
    values = fr._numbers(fr._leaves(["v1-3002", "v1-3012", "v1-3018"]))
    assert values == {3.0}


def test_a_value_is_compared_at_the_precision_it_is_printed_to() -> None:
    assert fr.explains("0.405", {0.40476190476190477})
    assert fr.explains("0.4", {0.40476190476190477})
    assert not fr.explains("0.404", {0.40476190476190477})
    assert fr.explains("−0.004", {-0.0036585365853658}), "a real minus sign is a minus sign"
    assert not fr.explains("0.004", {-0.0036585365853658}), "and the sign is part of the number"
    assert fr.explains("10,000", {10000}), "a thousands separator is not a decimal point"


def test_a_jsonl_column_stands_for_its_values_and_their_counts(cfg) -> None:
    found = fr.candidates(fr.e2e.run_path("test", cfg, RUNS_DIR), "exit")
    assert {37.0, 20.0, 6.0, 63.0} <= found, "the exits and the rows they are counted over"


def test_a_markdown_heading_stands_for_the_numbers_under_it() -> None:
    found = fr.candidates(REPO_ROOT / "README.md", "Data")
    assert {4502.0, 2086.0, 34.0, 13.0} <= found
    assert 111.0 not in found, "the chunk stats are under Chunking, and must be cited there"


def test_the_allowlist_stays_short() -> None:
    """#107 asks for a *small* allowlist. A number that wants an entry usually wants a link."""
    assert len(fr.ALLOWED) <= 12, (
        "the allowlist is growing; every entry is a number the report does not have to source: "
        + "; ".join(why for _, why in fr.ALLOWED)
    )
