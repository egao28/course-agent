"""Tests for agent/tools/prereqs.py: the walk, the graph guards, and both exits of the tool.

Three corpora, for three jobs. The walk is tested on hand-made graphs, because the shapes that
matter -- a shared prerequisite, a mutual pair, a self-loop, a cycle below the root -- are four
lines each here and would have to be hunted for in the live one. The tool is tested against the 17
courses ``tests/fixtures/catalog_sample.csv`` produces through the real exporter, with both output
paths redirected into ``tmp_path``. The last section reads the live corpus on purpose: #30's
acceptance is stated about real courses, and its numbers (7 cycles, 47-course closure, 9 levels)
are the reason the walk looks the way it does.

``fresh_cache`` is autouse and clears *both* modules: the chain reads the catalog index for every
node's name, so a test that left ``courses``'s cache behind would answer the next one's names from
the wrong corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.tools import courses as tc
from agent.tools import prereqs as tp
from ingest import export_almabot as ex

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"

#: X 300 requires X 210 *and* X 100, and X 210 requires X 100 too -- the CS 374 / CS 173 shape.
SHORTCUT = {"X 300": ("X 100", "X 210"), "X 210": ("X 100",)}
#: Two independent paths down to one course, and no cycle anywhere in it.
DIAMOND = {"X 300": ("X 210", "X 220"), "X 210": ("X 100",), "X 220": ("X 100",)}
#: "SE 310; concurrent registration in SE 312." on both sides of the pair.
PAIR = {"X 311": ("X 310", "X 312"), "X 312": ("X 310", "X 311"), "X 310": ()}


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits the graph -- or the catalog index -- another one built."""
    tp.clear_cache()
    tc.clear_cache()
    yield
    tp.clear_cache()
    tc.clear_cache()


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV).courses


@pytest.fixture
def corpus(tmp_path, courses, monkeypatch) -> dict[str, Path]:
    """Write both exports where the two modules will look for them."""
    paths = {"courses": tmp_path / "courses.json", "prereqs": tmp_path / "prereqs.json"}
    ex.write_json(paths["courses"], courses)
    ex.write_json(paths["prereqs"], ex.build_prereqs(courses))
    monkeypatch.setattr(tc, "COURSES_PATH", paths["courses"])
    monkeypatch.setattr(tp, "PREREQS_PATH", paths["prereqs"])
    return paths


def codes(chain: tp.PrereqChain) -> list[list[str]]:
    """The chain as codes per level, which is what most assertions are about."""
    return [[node.code for node in level] for level in chain.levels]


# --- the walk -------------------------------------------------------------------------------------


def test_a_straight_chain_is_one_course_per_level():
    walk = tp.walk_chain("X 300", {"X 300": ("X 200",), "X 200": ("X 100",)})
    assert walk.levels == [["X 200"], ["X 100"]]
    assert walk.edges == [("X 100", "X 200"), ("X 200", "X 300")]
    assert (walk.cycle_detected, walk.truncated) == (False, False)


def test_a_course_required_twice_is_reported_at_its_shortest_distance():
    # X 100 is an immediate prerequisite and also sits under X 210. Level 0 is the contract #38
    # reads ("you need these first"), so the immediate spelling wins; dag.ts, drawing arrows,
    # would push X 100 a level back instead.
    walk = tp.walk_chain("X 300", SHORTCUT)
    assert walk.levels == [["X 100", "X 210"]]
    assert ("X 100", "X 210") in walk.edges


def test_a_shared_prerequisite_is_not_a_cycle():
    walk = tp.walk_chain("X 300", DIAMOND)
    assert walk.levels == [["X 210", "X 220"], ["X 100"]]
    assert walk.cycle_detected is False


def test_a_mutual_pair_terminates_and_is_flagged():
    walk = tp.walk_chain("X 311", PAIR)
    assert walk.levels == [["X 310", "X 312"]]
    assert walk.cycle_detected is True
    assert walk.truncated is False


def test_a_self_loop_terminates_and_is_flagged():
    walk = tp.walk_chain("X 100", {"X 100": ("X 100",)})
    assert walk.levels == []
    assert walk.edges == [("X 100", "X 100")]
    assert walk.cycle_detected is True


def test_a_cycle_below_the_root_is_flagged():
    # X 400 is in nobody's chain; the loop it reaches is still a loop, and the consumer is told.
    walk = tp.walk_chain("X 400", {"X 400": ("X 311",), **PAIR})
    assert walk.cycle_detected is True
    assert walk.levels == [["X 311"], ["X 310", "X 312"]]


def test_the_queried_course_is_never_a_node_in_its_own_chain():
    # Even when it is its own prerequisite: the loop shows up as an edge, not as a level entry, so
    # no answer can list the course the student asked about among the ones they need first.
    walk = tp.walk_chain("X 311", PAIR)
    assert "X 311" not in [code for level in walk.levels for code in level]
    assert ("X 311", "X 312") in walk.edges


def test_every_edge_endpoint_is_a_course_the_result_reports():
    for root, graph in (("X 300", SHORTCUT), ("X 300", DIAMOND), ("X 311", PAIR)):
        walk = tp.walk_chain(root, graph)
        reported = {code for level in walk.levels for code in level} | {root}
        assert all({a, b} <= reported for a, b in walk.edges)


def test_max_depth_1_returns_the_immediate_prerequisites_and_says_so():
    walk = tp.walk_chain("X 300", DIAMOND, max_depth=1)
    assert walk.levels == [["X 210", "X 220"]]
    assert walk.truncated is True


def test_a_cap_that_drops_nothing_does_not_claim_truncation():
    assert tp.walk_chain("X 300", DIAMOND, max_depth=2).truncated is False
    # The cap lands exactly on the last level, whose one course is already reported.
    assert tp.walk_chain("X 300", SHORTCUT, max_depth=1).truncated is False


def test_a_cap_never_reports_an_edge_to_a_course_it_dropped():
    walk = tp.walk_chain("X 300", DIAMOND, max_depth=1)
    assert "X 100" not in [code for level in walk.levels for code in level]
    assert all("X 100" not in edge for edge in walk.edges)


def test_the_default_walk_goes_to_exhaustion():
    deep = {f"X {100 + i}": (f"X {99 + i}",) for i in range(1, 40)}
    walk = tp.walk_chain("X 139", deep)
    assert len(walk.levels) == 39
    assert walk.truncated is False


# --- the graph ------------------------------------------------------------------------------------


def test_build_graph_normalizes_every_spelling_it_is_given():
    graph = tp.build_graph([{"course": "cs374", "prereqs": ["cs 225", "CS173"], "coreqs": []}])
    assert graph == {"CS 374": ("CS 173", "CS 225")}


def test_build_graph_ignores_corequisites():
    # A corequisite is taken alongside, so it is not a step in the chain -- and the export keeps
    # the two apart precisely because corequisites are mutual often enough to make cycles.
    graph = tp.build_graph([{"course": "CS 233", "prereqs": [], "coreqs": ["CS 173"]}])
    assert graph == {"CS 233": ()}


def test_a_repeated_record_is_refused():
    row = {"course": "CS 374", "prereqs": ["CS 225"], "coreqs": []}
    with pytest.raises(ValueError, match="appears twice"):
        tp.build_graph([row, {**row, "prereqs": ["CS 173"]}])


@pytest.mark.parametrize(
    "row",
    [
        {"course": "COMPSCI 374", "prereqs": [], "coreqs": []},
        {"course": "CS 374", "prereqs": ["CS 22"], "coreqs": []},
    ],
)
def test_a_code_the_shared_matcher_cannot_read_is_refused(row):
    with pytest.raises(ValueError, match="not a course code"):
        tp.build_graph([row])


def test_a_prerequisite_the_catalog_does_not_have_is_refused(courses):
    # The export filters references through the catalog, so this only happens when the two files
    # come from different runs -- and a node with no record has no name to report.
    by_code = tc.build_index(courses).by_code
    tp.check_catalog({"CS 374": ("CS 173",)}, by_code)
    with pytest.raises(ValueError, match="different runs"):
        tp.check_catalog({"CS 374": ("MATH 213",)}, by_code)


def test_a_missing_edge_list_points_at_the_export(tmp_path, monkeypatch):
    # An absent course is an answer; an absent prereqs.json would make every course answer "no
    # prerequisites", which is the one wrong answer this tool can give.
    monkeypatch.setattr(tp, "PREREQS_PATH", tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError, match="export_almabot"):
        tp.load_prereqs()


def test_an_empty_edge_list_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "prereqs.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(tp, "PREREQS_PATH", path)
    with pytest.raises(ValueError, match="no prerequisite records"):
        tp.load_prereqs()


def test_load_graph_reads_the_path_it_is_given_and_does_not_cache(corpus, tmp_path):
    one = tmp_path / "one.json"
    ex.write_json(one, [{"course": "CS 374", "prereqs": ["CS 225"], "coreqs": []}])
    assert tp.load_graph(one) == {"CS 374": ("CS 225",)}
    assert len(tp.prereq_graph()) == 5


def test_the_edge_list_is_read_once(corpus, monkeypatch):
    reads: list = []
    real = tp.load_prereqs
    monkeypatch.setattr(tp, "load_prereqs", lambda path=None: (reads.append(path), real(path))[1])

    for code in ("CS 421", "CS 374", "CS 999"):
        tp.get_prereq_chain(code)
    assert len(reads) == 1

    tp.clear_cache()
    tp.get_prereq_chain("CS 374")
    assert len(reads) == 2


def test_the_cached_graph_is_cross_checked_against_the_catalog(corpus, monkeypatch):
    stale = json.loads(corpus["prereqs"].read_text(encoding="utf-8"))
    stale.append({"course": "MATH 213", "prereqs": ["MATH 220"], "coreqs": []})
    ex.write_json(corpus["prereqs"], stale)
    with pytest.raises(ValueError, match="different runs"):
        tp.get_prereq_chain("CS 374")


# --- the tool -------------------------------------------------------------------------------------


def test_the_chain_layers_the_whole_closure(corpus):
    chain = tp.get_prereq_chain("cs 421")
    assert chain.found is True
    assert chain.course_id == "CS 421"
    assert codes(chain) == [["CS 233", "CS 374"], ["CS 173", "CS 225"]]
    assert (chain.cycle_detected, chain.truncated) == (False, False)
    assert chain.edges == [
        ("CS 173", "CS 374"),
        ("CS 225", "CS 374"),
        ("CS 233", "CS 421"),
        ("CS 374", "CS 421"),
    ]


def test_every_node_carries_its_name_and_its_own_requirement_sentence(corpus):
    # The sentence is the point: "CS374 and one of CS 233, CS 240" is a disjunction, and the two
    # codes under it are alternatives. #38 quotes this rather than asserting the conjunction.
    chain = tp.get_prereq_chain("CS 421")
    assert chain.raw == "CS374 and one of CS 233, CS 240."
    node = chain.levels[0][1]
    assert (node.code, node.name) == ("CS 374", "Introduction to Algorithms & Models")
    assert node.raw == "One of CS 173, MATH 213; CS 225."
    assert chain.levels[1][0].raw == ""  # CS 173 has no requirements block at all


def test_corequisites_are_not_steps_in_the_chain(corpus):
    # STAT 425: "STAT 100. Concurrent enrollment in CS 225 is preferred." CS 225 is not a step.
    chain = tp.get_prereq_chain("STAT 425")
    assert codes(chain) == [["STAT 100"]]
    assert "CS 225" in chain.raw


def test_a_course_with_no_prerequisites_is_an_answer(corpus):
    chain = tp.get_prereq_chain("ECON 102")
    assert (chain.found, chain.levels, chain.edges) == (True, [], [])
    assert (chain.cycle_detected, chain.truncated, chain.raw) == (False, False, "")
    assert chain.message is None


def test_a_requirement_that_names_no_course_in_the_catalog_still_quotes_itself(corpus):
    # PHYS 211: "PHYS 211 placement test or MATH 221." -- MATH 221 is not offered, so the walk is
    # empty, and the sentence is the only honest answer left.
    chain = tp.get_prereq_chain("phys211")
    assert (chain.found, chain.levels) == (True, [])
    assert chain.raw == "PHYS 211 placement test or MATH 221."


def test_max_depth_reaches_the_tool(corpus):
    chain = tp.get_prereq_chain("CS 421", max_depth=1)
    assert codes(chain) == [["CS 233", "CS 374"]]
    assert chain.truncated is True
    assert tp.get_prereq_chain("CS 421", max_depth=2).truncated is False


def test_not_found_is_a_value_and_carries_the_lookup_tools_message(corpus):
    chain = tp.get_prereq_chain("CS 999")
    assert chain.found is False
    assert chain.course_id == "CS 999"
    assert chain.message == tc.get_course("CS 999").message
    assert (chain.levels, chain.edges, chain.raw) == ([], [], None)


def test_a_query_that_is_not_a_code_is_a_value_too(corpus):
    chain = tp.get_prereq_chain("  what do i need before cs 374  ")
    assert chain.found is False
    assert chain.course_id == "what do i need before cs 374"
    assert "not a course code" in chain.message


@pytest.mark.parametrize("junk", ["", "   ", "?", "CS", "999", "CS 3745", "CS 374 or ECE 374"])
def test_no_query_reaches_the_graph_as_an_exception(corpus, junk):
    assert tp.get_prereq_chain(junk).found is False


def test_a_bad_depth_is_refused_before_anything_is_read(corpus):
    for depth in (0, -1):
        with pytest.raises(ValidationError):
            tp.get_prereq_chain("CS 421", max_depth=depth)


# --- the schemas MCP will publish -----------------------------------------------------------------


def test_the_query_schema_names_the_spellings_and_leaves_depth_optional():
    schema = tp.PrereqQuery.model_json_schema()
    assert schema["required"] == ["course_id"]
    assert "CS374" in schema["properties"]["course_id"]["description"]
    assert "truncated" in schema["properties"]["max_depth"]["description"]


def test_a_result_requires_only_the_course_and_whether_it_was_found():
    assert tp.PrereqChain.model_json_schema()["required"] == ["course_id", "found"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: tp.PrereqQuery(course_id="CS 374", depth=5),
        lambda: tp.PrereqNode(code="CS 374", name="...", text="..."),
        lambda: tp.PrereqChain(course_id="CS 374", found=True, layers=[]),
    ],
)
def test_the_models_refuse_a_field_they_do_not_declare(call):
    with pytest.raises(ValidationError):
        call()


# --- the live corpus: #30's acceptance ------------------------------------------------------------


def test_live_cs374_chain_starts_with_its_immediate_prerequisites():
    chain = tp.get_prereq_chain("cs 374")
    assert chain.found is True
    assert codes(chain)[0] == ["CS 173", "CS 225", "MATH 213"]
    assert chain.raw == "One of CS 173, MATH 213; CS 225."
    assert sum(len(level) for level in chain.levels) == 16


def test_live_cs225_terminates_on_the_cycle_it_is_in():
    # CS 225 -> MATH 412 -> CS 374 -> CS 225, all three edges from "one of" alternatives.
    chain = tp.get_prereq_chain("CS 225")
    assert chain.cycle_detected is True
    assert chain.truncated is False
    assert ("CS 225", "CS 374") in chain.edges
    assert "CS 374" in [node.code for level in chain.levels for node in level]


def test_live_se495_is_walked_to_exhaustion():
    chain = tp.get_prereq_chain("SE 495")
    assert chain.truncated is False
    assert sum(len(level) for level in chain.levels) == 27
    assert len(chain.levels) == 5  # shortest-distance layers; the longest walk in it is 13 deep
    assert longest_walk(chain) == 13


def test_live_a_course_with_no_prerequisites_is_found_with_an_empty_chain():
    chain = tp.get_prereq_chain("ECON 102")
    assert (chain.found, chain.levels, chain.cycle_detected) == (True, [], False)


def test_live_the_largest_closure_in_the_catalog_is_47_courses():
    chain = tp.get_prereq_chain("IS 557")
    assert sum(len(level) for level in chain.levels) == 47


def test_live_every_course_in_the_catalog_answers():
    graph = tp.prereq_graph()
    assert len(graph) == 1135
    assert sum(len(prereqs) for prereqs in graph.values()) == 2086

    chains = {code: tp.get_prereq_chain(code) for code in tc.course_index().by_code}
    assert len(chains) == 4502
    assert all(chain.found for chain in chains.values())
    assert not any(chain.truncated for chain in chains.values())
    assert max(len(chain.levels) for chain in chains.values()) == 9
    assert sum(1 for chain in chains.values() if chain.cycle_detected) == 165
    assert sum(1 for chain in chains.values() if not chain.levels) == 3378

    for chain in chains.values():
        reported = {node.code for level in chain.levels for node in level} | {chain.course_id}
        assert all({a, b} <= reported for a, b in chain.edges)


def test_live_the_seven_cycles_are_where_the_measurement_says_they_are():
    for code in ("THEA 375", "THEA 475", "CS 374", "ACCY 301", "CHEM 204", "MCB 581", "SE 311"):
        assert tp.get_prereq_chain(code).cycle_detected is True, code
    # CS 101 -> MATH 241 -> MATH 231 -> MATH 220 is reached twice over and is not a cycle.
    assert tp.get_prereq_chain("CS 101").cycle_detected is False


def test_live_dag_ts_six_level_cap_would_cut_70_of_the_1135_chains():
    # The reason there is no default cap: with one, 70 courses would come back truncated and 113
    # would be missing a prerequisite outright (195 course-slots across them).
    graph = tp.prereq_graph()
    capped = sum(1 for code in graph if tp.walk_chain(code, graph, 6).truncated)
    assert capped == 70
    assert not any(tp.walk_chain(code, graph).truncated for code in graph)


def longest_walk(chain: tp.PrereqChain) -> int:
    """The longest cycle-free walk in a chain's edges -- the depth dag.ts's cap is measured against.

    Not what ``levels`` reports (each course sits at its *shortest* distance), and computed here
    rather than in the tool because no consumer has asked for it: #30's "12 levels deep" for
    SE 495 is this number, and the point of the test is that the edges still carry it.
    """
    down: dict[str, list[str]] = {}
    for prereq, dependent in chain.edges:
        down.setdefault(dependent, []).append(prereq)

    def walk(code: str, path: frozenset[str]) -> int:
        return max(
            (1 + walk(p, path | {code}) for p in down.get(code, ()) if p not in path), default=0
        )

    return walk(chain.course_id, frozenset())
