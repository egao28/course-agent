"""Tests for the two pages the generator owns -- report/README.md (#106) and the repo's own
README.md above its `## Reference` divider (#117) -- held to their sources.

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

#120 adds a third page, the claims table, and ``tests/test_day7_claims.py`` holds it: its rules
are a table's rather than a page of prose's, and the two files share nothing but the generator.
What is here is that the report carries every block except that page's (:data:`REPORT_BLOCKS`).

#117 adds a second page and one new thing to hold: **its scope**. The repo's README is a front page
down to ``## Reference`` and a 2,600-line component reference below it, and the check applies to the
first part only. Both directions are tested -- an unsourced number above the divider fails, the same
sentence below it does not -- because a scope that quietly widened would make the reference
impossible to edit, and one that quietly narrowed would make the front page unchecked. The diagram
is on both pages and is asserted to be the same bytes, so the four tests above hold the front
page's copy too.
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
FRONT = fr.FRONT


def page_of(path) -> fr.Page:
    """The :class:`final_report.Page` for ``path``, found by path rather than by position."""
    return next(one for one in fr.pages(fr.Sources()) if one.path == path)

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

#: The front page's sections, in the order #117 commissioned them. The name, the diagram, the
#: three RQ conclusions, the caveats, the commands, the map, and then the divider.
FRONT_SECTIONS = (
    "## Architecture",
    "## What was found",
    "## What this is not",
    "## Quickstart",
    "## Where to read what",
    "## Reference",
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
def report_page() -> fr.Page:
    return page_of(REPORT)


@pytest.fixture(scope="module")
def front_page() -> fr.Page:
    return page_of(FRONT)


@pytest.fixture(scope="module")
def text() -> str:
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def front_text() -> str:
    return FRONT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def facts(cfg):
    """The committed inputs, read and checked once -- the expensive fixture in this file."""
    return fr.read_facts(cfg, fr.Sources())


@pytest.fixture(scope="module")
def bodies(facts):
    return fr.render_blocks(facts)


@pytest.fixture(scope="module")
def blocks(report_page, text: str) -> dict[str, str]:
    """``{name: body}`` for each generated block, checking the markers pair up as it goes."""
    return {one.name: text[one.start : one.end] for one in fr.blocks_in(report_page, text)}


@pytest.fixture(scope="module")
def front_blocks(front_page, front_text: str) -> dict[str, str]:
    """The same, for the one block the repo's README carries."""
    return {one.name: front_text[one.start : one.end]
            for one in fr.blocks_in(front_page, front_text)}


def problems(page, text: str, facts, bodies) -> list[str]:
    return [problem.message for problem in fr.check(page, text, facts, bodies)]


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
    assert set(blocks) == set(fr.REPORT_BLOCKS), (
        "the markers and the renderers have drifted apart")
    for name, body in blocks.items():
        assert body.strip(), f"gen:{name} is empty"


def test_every_local_link_resolves(text: str) -> None:
    links = sorted({m.group("path") for m in LINK_RE.finditer(text)})
    missing = [p for p in links if not (REPORT.parent / p).resolve().exists()]
    assert not missing, f"link(s) to files that do not exist: {missing}"


# --- #107: the committed report passes its own check -----------------------------------------


def test_check_passes_on_the_committed_report(report_page, text: str, facts, bodies) -> None:
    """The acceptance criterion: every block current, every number sourced, every value found."""
    assert problems(report_page, text, facts, bodies) == []


def test_the_generator_is_idempotent(report_page, text: str, bodies) -> None:
    """Rewriting over unchanged inputs writes the same bytes, so --check is not a clock."""
    assert fr.splice(report_page, text, bodies) == text


# --- ... and fails, one way at a time --------------------------------------------------------


def test_a_typed_in_wrong_number_fails(report_page, text: str, facts, bodies) -> None:
    """The expensive half: a prose number that no longer matches the file it cites."""
    broken = text.replace("with AUROC\n0.847 [0.722, 0.953]", "with AUROC\n0.857 [0.722, 0.953]")
    assert broken != text, "the sentence this test edits has been rewritten"
    found = problems(report_page, broken, facts, bodies)
    assert any("0.857 is not a value in" in one for one in found), found


def test_a_number_the_file_still_has_passes(report_page, text: str, facts, bodies) -> None:
    """The same sentence, rounded the other way at one more decimal: 0.847 is 0.8471, not 0.8470."""
    ok = text.replace("with AUROC\n0.847 [0.722, 0.953]", "with AUROC\n0.8471 [0.722, 0.953]")
    assert ok != text
    assert [one for one in problems(report_page, ok, facts, bodies) if "0.8471" in one] == []


def test_a_stale_block_fails(report_page, text: str, facts, bodies) -> None:
    broken = text.replace("| `bm25` | 0.870", "| `bm25` | 0.871")
    assert broken != text, "the day4 table this test edits has changed shape"
    assert any("gen:day4_ablation is out of date" in one
               for one in problems(report_page, broken, facts, bodies))


def test_an_unsourced_number_fails(report_page, text: str, facts, bodies) -> None:
    broken = text + "\nThe agent answered 7 questions about the weather.\n"
    assert any("7 has no [src] link" in one
               for one in problems(report_page, broken, facts, bodies))


def test_a_number_in_a_table_outside_a_block_fails(report_page, text: str, facts, bodies) -> None:
    broken = text + "\n| rows | 41 |\n"
    assert any("a table outside a gen: block" in one
               for one in problems(report_page, broken, facts, bodies))


def test_a_mistyped_key_path_fails(report_page, text: str, facts, bodies) -> None:
    broken = text.replace('"metrics.false_block")', '"metrics.false_blocks")')
    assert broken != text
    found = problems(report_page, broken, facts, bodies)
    assert any("has no key path 'metrics.false_blocks'" in one for one in found), found



def test_one_bad_path_in_a_multi_key_title_fails(report_page, text: str, facts, bodies) -> None:
    """A title naming three paths is not excused by two of them resolving."""
    broken = text.replace('"n, alpha, min_answered_to_accept")', '"n, alpha, min_answered")')
    assert broken != text
    assert any("has no key path 'n, alpha, min_answered'" in one
               for one in problems(report_page, broken, facts, bodies))


def test_a_link_to_a_missing_file_fails(report_page, text: str, facts, bodies) -> None:
    broken = text.replace("../eval/runs/e2e_metrics_c73b197d.json \"n\")",
                          "../eval/runs/e2e_metrics_nope.json \"n\")")
    assert broken != text
    assert any("[src] link to a missing file" in one
               for one in problems(report_page, broken, facts, bodies))


def test_a_hex_key_this_config_does_not_produce_fails(
        report_page, text: str, facts, bodies) -> None:
    """A stale hash reads exactly like a current one, and names a file nobody has."""
    broken = text.replace("config hash `c73b197d`", "config hash `c73b197e`", 1)
    assert broken != text
    assert any("`c73b197e` is not a key this config produces" in one
               for one in problems(report_page, broken, facts, bodies))


def test_an_unclosed_marker_is_refused(text: str) -> None:
    broken = text.replace("<!-- END gen:day6_exits -->", "")
    with pytest.raises(ValueError, match="gen:day6_exits"):
        fr.marked(broken, tuple(fr.BLOCKS))


def test_a_block_with_no_renderer_is_refused(text: str) -> None:
    broken = text.replace("gen:day6_exits", "gen:day6_exitss")
    with pytest.raises(ValueError, match="no renderer for gen:day6_exitss"):
        fr.marked(broken, tuple(fr.BLOCKS))


def test_a_block_the_page_does_not_carry_is_refused(front_page, front_text: str) -> None:
    """The repo's README carries the diagram and nothing else. A table spliced into it would be
    written by a generator that no longer knows which file it is writing."""
    broken = front_text.replace("gen:architecture", "gen:day6_exits")
    with pytest.raises(ValueError, match="no renderer for gen:day6_exits on this page"):
        fr.blocks_in(front_page, broken)


# --- #117: the front page, and the scope of the check ----------------------------------------


def test_the_front_pages_diagram_is_the_reports(front_blocks, blocks) -> None:
    """The same bytes in both files.

    This is what "one diagram, one owner" buys, and it is also what makes the four diagram tests
    above cover the front page: they read the report's copy, and this asserts there is only one.
    """
    assert front_blocks["architecture"] == blocks["architecture"]


def test_the_front_page_carries_only_the_diagram(front_blocks, front_page) -> None:
    assert set(front_blocks) == set(front_page.blocks) == {"architecture"}


def test_front_page_sections_in_the_commissioned_order(front_text: str) -> None:
    headings = [line for line in front_text.splitlines() if line.startswith("## ")]
    found = [h for h in headings if h in FRONT_SECTIONS]
    assert found == list(FRONT_SECTIONS), (
        f"the front page's sections are missing or out of order: {found}"
    )


def test_the_front_page_names_all_three_rqs(front_page, front_text: str) -> None:
    """One conclusion per research question, above the divider, each with its own source link."""
    region = front_page.region(front_text)
    for rq in ("RQ1", "RQ2", "RQ3"):
        line = [one for one in region.splitlines() if one.startswith(f"- **{rq} ")]
        assert len(line) == 1, f"expected one {rq} conclusion bullet, found {len(line)}"


def test_every_front_page_link_resolves(front_page, front_text: str) -> None:
    """Every link above the divider names a file in this repo that is there.

    Targets outside the repo are skipped rather than exempted: `almabot-web` is the other half of
    the project, and a clone of this repo does not contain it. In-page anchors never match
    :data:`LINK_RE` in the first place.
    """
    region = front_page.region(front_text)
    links = sorted({m.group("path") for m in LINK_RE.finditer(region)})
    inside = [one for one in links if (REPO_ROOT / one).resolve().is_relative_to(REPO_ROOT)]
    assert inside, "no in-repo links on the front page; LINK_RE or the page changed shape"
    missing = [one for one in inside if not (REPO_ROOT / one).resolve().exists()]
    assert not missing, f"link(s) to files that do not exist: {missing}"


def test_check_passes_on_the_committed_front_page(front_page, front_text, facts, bodies) -> None:
    """#117's acceptance criterion: every number in the front matter traces to a file."""
    assert problems(front_page, front_text, facts, bodies) == []


def test_the_front_page_generator_is_idempotent(front_page, front_text, bodies) -> None:
    assert fr.splice(front_page, front_text, bodies) == front_text


def test_a_rounded_away_interval_on_the_front_page_fails(
        front_page, front_text, facts, bodies) -> None:
    """RQ1 rests on an interval that contains 0. Rounding the bound away is the one edit that
    would turn the finding into its opposite, so it is the one this test makes."""
    broken = front_text.replace("[−0.085, 0.073]", "[−0.085, 0.003]", 1)
    assert broken != front_text, "RQ1's interval has been rewritten"
    found = problems(front_page, broken, facts, bodies)
    assert any("0.003 is not a value in" in one for one in found), found


def test_an_unsourced_number_above_the_divider_fails(
        front_page, front_text, facts, bodies) -> None:
    broken = front_text.replace(
        "\n## Reference\n", "\nThe agent answered 7 questions about the weather.\n\n"
                            "## Reference\n", 1)
    assert broken != front_text
    assert any("7 has no [src] link" in one
               for one in problems(front_page, broken, facts, bodies))


def test_the_reference_body_is_not_checked(front_page, front_text, facts, bodies) -> None:
    """The same sentence below the divider is not a failure.

    The reference is 2,600 lines of unsourced numbers -- token counts, RRF constants, flag tables
    -- and #117 put a front page above it rather than a `[src]` link on every line of it.
    """
    broken = front_text + "\nThe agent answered 7 questions about the weather.\n"
    assert problems(front_page, broken, facts, bodies) == []


def test_a_table_below_the_divider_is_not_checked(front_page, front_text, facts, bodies) -> None:
    """`check_tables` stops at the divider too: the reference's flag tables are hand-written."""
    assert problems(front_page, front_text + "\n| flag | what it does |\n", facts, bodies) == []


#: A hash `check_hex` would reject: `data/chunks.meta.json`'s chunk key, which is a hash of the
#: term, the strategy and the chunk template rather than anything `config.hash` covers. The
#: reference body names it in backticks already, so the mirrored pair below is not hypothetical.
STALE_HASH = "Chunk key `05d00726`.\n"


def test_a_stale_hash_above_the_divider_fails(front_page, front_text, facts, bodies) -> None:
    """`check_hex` applies to the front matter: a hash there is one this config produces."""
    broken = front_text.replace("\n## Reference\n", f"\n{STALE_HASH}\n## Reference\n", 1)
    assert broken != front_text
    assert any("`05d00726` is not a key this config produces" in one
               for one in problems(front_page, broken, facts, bodies))


def test_a_stale_hash_below_the_divider_is_not_checked(
        front_page, front_text, facts, bodies) -> None:
    """...and stops at the divider. The reference names that key, and five others like it, in the
    sections that explain which part of the config each one hashes."""
    assert problems(front_page, front_text + STALE_HASH, facts, bodies) == []


def test_a_report_relative_src_link_on_the_front_page_fails(
        front_page, front_text, facts, bodies) -> None:
    """A `[src]` link is relative to the page carrying it, so the report's own `../eval/runs/...`
    is a broken link here -- which is the mistake copying a sentence across makes."""
    broken = front_text.replace("[src](eval/runs/conformal_83496e82.json",
                                "[src](../eval/runs/conformal_83496e82.json", 1)
    assert broken != front_text
    assert any("[src] link to a missing file" in one
               for one in problems(front_page, broken, facts, bodies))


def test_the_front_page_needs_its_reference_divider(front_page, front_text: str) -> None:
    """Without the divider there is no front page, and the check says so rather than checking
    2,600 lines of reference or silently checking nothing."""
    broken = front_text.replace("\n## Reference\n", "\n## Everything else\n", 1)
    assert broken != front_text
    with pytest.raises(ValueError, match="has no `## Reference` heading"):
        front_page.region(broken)


BLOCK_FLAG_RE = re.compile(r"--[a-z][a-z-]*")
ADD_ARGUMENT_RE = re.compile(r"""add_argument\(\s*["'](--[a-z][a-z-]*)["']""")

#: ``raise ValueError("choose exactly one of --judge, --review, ...")`` -- a module that refuses
#: to do two things in one sitting, and the flags it refuses between. Named from the message
#: rather than listed here, so a module that adopts the convention is covered by adopting it.
ONE_OF_RE = re.compile(r"choose exactly one of ((?:--[a-z-]+(?:, )?)+)")

#: The same, for the flag that names the person: a module that records a verdict refuses to
#: record it anonymously.
REVIEWER_RE = re.compile(r"--reviewer NAME is required")

#: ... and the modes that return before it asks, so they need no name.
REPORTING = frozenset({"--judge", "--stats", "--write-csv"})


def reproduction_commands(text: str) -> list[str]:
    """The live `uv run` commands of `report/README.md`'s Reproduction block, `&&` split apart."""
    block = text.split("## Reproduction")[1].split("```bash")[1].split("```")[0]
    lines = [re.sub(r"\s+#.*$", "", one) for one in block.splitlines() if one.startswith("uv run")]
    return [part.strip() for one in lines for part in one.split("&&") if part.strip()]


def input_commands(cfg) -> list[str]:
    """The `uv run` lines `--inputs` prints beside an artifact that is not there.

    `inputs()` holds the other half of the same claim -- it is what the block is checked for
    completeness against below -- so a command that would not start is as wrong here as there.
    Five entries are deliberately prose instead and are skipped; the test below names them.
    """
    return [one.how for one in fr.inputs(cfg, fr.Sources()) if one.how.startswith("uv run")]


def module_and_args(command: str) -> tuple[str, list[str]] | None:
    """`uv run ... -m mod --flag value` -> `("mod", ["--flag value", ...])`, or `None` for no `-m`.

    A flag *before* `-m` belongs to `uv` -- `--env-file`, `--extra local` -- and is not the
    module's to define. A value is kept with its flag so that `--split calib` and `--split test`
    are two different commands rather than one.
    """
    if " -m " not in command:
        return None
    module, _, rest = command.split(" -m ", 1)[1].partition(" ")
    tokens, args, i = rest.split(), [], 0
    while i < len(tokens):
        if not tokens[i].startswith("--"):
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            args.append(tokens[i])
            i += 1
    return module, args


def would_not_start(command: str) -> list[str]:
    """Every static reason `command` would stop at its module's argument handling, or `[]`.

    Three rules, each one a defect this repo has actually shipped: a flag the module does not
    define (`eval.conformal --split calib`, exit 2, #132); none or several of the modes a module
    takes exactly one of (`e2e_labels --split test`, exit 1); and a reviewing sitting with nobody's
    name on it (`review_signals --split test`, exit 1). All three are read out of the module's own
    source, so a module that changes its mind is what moves the test.
    """
    found = module_and_args(command)
    if found is None:
        return []
    module, args = found
    source = REPO_ROOT / (module.replace(".", "/") + ".py")
    if not source.exists():
        return [f"{module} is not a module in this repo"]
    text = source.read_text(encoding="utf-8")
    flags = {one.split(" ")[0] for one in args}
    defined = set(ADD_ARGUMENT_RE.findall(text))
    wrong = [f"{command}: {module} does not define {flag}"
             for flag in sorted(flags) if flag not in defined and flag != "--help"]
    for listed in ONE_OF_RE.findall(text):
        modes = set(BLOCK_FLAG_RE.findall(listed))
        if len(flags & modes) != 1:
            got = ", ".join(sorted(flags & modes)) or "none"
            wrong.append(f"{command}: {module} takes exactly one of "
                         f"{', '.join(sorted(modes))}, got {got}")
    if REVIEWER_RE.search(text) and not flags & (REPORTING | {"--reviewer"}):
        wrong.append(f"{command}: {module} refuses a verdict with no name on it; "
                     "give it --reviewer NAME")
    return wrong


def test_every_command_in_the_reproduction_block_would_start(text: str) -> None:
    """A command that stops at argparse exits 2 and the block stops with it, so a reader who
    pastes it gets a usage message instead of the next artifact (#132, #136)."""
    commands = [one for one in reproduction_commands(text) if " -m " in one]
    assert commands, "the Reproduction block names no modules; the section changed shape"
    wrong = [one for command in commands for one in would_not_start(command)]
    assert not wrong, "command(s) that would not start:\n  " + "\n  ".join(wrong)


def test_every_command_the_inputs_list_prints_would_start(cfg) -> None:
    """The same rules on `inputs()`, which is where two of them rotted unseen.

    `--inputs` prints these beside a missing artifact as the thing to run, and nothing read them
    back: `agreement --read SHEET` had no such flag, and the review log's line had no
    `--reviewer` (#136).
    """
    commands = input_commands(cfg)
    assert commands, "inputs() names no commands; the list changed shape"
    wrong = [one for command in commands for one in would_not_start(command)]
    assert not wrong, "command(s) that would not start:\n  " + "\n  ".join(wrong)


def test_every_command_the_inputs_list_names_is_in_the_reproduction_block(cfg, text: str) -> None:
    """#136: the block says it is the whole chain, and `inputs()` is what the chain is.

    Matched on the module and its *module* arguments, so the block may be more explicit than
    `inputs()` -- `risk_coverage --split test` where the default is left implicit -- and may put
    `--extra local` before `-m`, which belongs to `uv`. What it may not do is leave one out, which
    is how twelve commands went missing while the lead sentence said otherwise.
    """
    block = [found for found in map(module_and_args, reproduction_commands(text))
             if found is not None]
    absent = []
    for command in input_commands(cfg):
        module, args = module_and_args(command)
        if not any(module == block_module and set(args) <= set(block_args)
                   for block_module, block_args in block):
            absent.append(command)
    assert not absent, ("the Reproduction block claims every command that writes an input; "
                        "these are not in it:\n  " + "\n  ".join(absent))


def test_the_inputs_with_no_command_are_the_ones_that_are_not_run(cfg) -> None:
    """Five `how` strings are sentences rather than lines to paste: the three pages this script
    writes, and the two corpus files, which are committed output of a step a reader cannot run.
    A sixth is a decision about what the Reproduction block carries, so it fails here until it is
    made rather than being skipped quietly by the completeness test above (#136).
    """
    prose = sorted(one.name for one in fr.inputs(cfg, fr.Sources())
                   if not one.how.startswith("uv run"))
    assert prose == ["corpus: courses", "corpus: prerequisites", "the claims page",
                     "the front page", "the report itself"]


def test_the_reproduction_chain_runs_without_the_private_csv(text: str) -> None:
    """`data.source` is not in the repo, so a live `ingest.export_almabot` would stop a reader
    on an absolute path from the machine that made the export. Its three `data/*.json` outputs
    are committed, and the chain starts at `ingest.chunk` (#132).
    """
    block = text.split("## Reproduction")[1].split("```bash")[1].split("```")[0]
    live = [one for one in block.splitlines() if one.startswith("uv run")]
    assert live, "the Reproduction block names no commands; the section changed shape"
    assert not [one for one in live if "export_almabot" in one]
    assert "# uv run python -m ingest.export_almabot" in block
    assert any("ingest.chunk" in one for one in live)


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
