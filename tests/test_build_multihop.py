"""Tests for eval/scripts/build_multihop.py: the table, the checks over it, and the file.

Four groups. The first holds the table to #58's acceptance -- 25 rows over five question shapes,
every root at depth >= 2 by both measures, a depth distribution that is not the shallow end of the
graph, at least two rows inside one of the seven cycles, every gold answer naming every record it
cites, and every disjunction still a disjunction. The second checks the graph reader against the
numbers #58 quotes. The third mutates one row at a time and asserts the check that should catch it
does, because a check that has never rejected anything is a comment. The fourth covers the run.

The graph checks run against the **real** committed graph, not a synthetic one: what makes a row
here correct is a fact about 1,135 real prerequisite records, and a five-course fixture would test
the loop rather than the bucket. The mutation tests build a
:class:`~eval.scripts.build_multihop.Graph` by hand instead, since there the thing under test is
the message.
"""

from __future__ import annotations

import json

import pytest

from config import load_config
from eval.schema import RULES, EvalItem, normalize_question
from eval.scripts import build_multihop as bm
from eval.scripts.draft_items import ID_BLOCKS, check_drafts, draft_path, parse_drafts
from eval.scripts.draft_items import PROVENANCE_KEYS as DRAFTED_KEYS
from eval.validate import corpus_doc_ids

CFG = load_config()

#: The bucket's own file, as the repository carries it.
COMMITTED = bm.draft_path(bm.STEM, CFG)


@pytest.fixture(scope="module")
def graph() -> bm.Graph:
    return bm.read_graph()


@pytest.fixture(scope="module")
def rows(graph: bm.Graph) -> list[dict]:
    return bm.build_rows(bm.ITEMS, graph, CFG)


def fake_graph() -> bm.Graph:
    """Six courses: a three-level chain, a branch, and a two-course cycle."""
    edges = {
        "AAA 300": ("AAA 200",),
        "AAA 200": ("AAA 100",),
        "BBB 300": ("AAA 200", "BBB 200"),
        "CCC 301": ("CCC 302",),
        "CCC 302": ("CCC 301",),
    }
    raws = {
        "AAA 300": "AAA 200.",
        "AAA 200": "AAA 100 or consent of instructor.",
        "AAA 100": "",
        "BBB 200": "",
        "BBB 300": "AAA 200 and BBB 200.",
        "CCC 301": "Concurrent registration in CCC 302.",
        "CCC 302": "Concurrent registration in CCC 301.",
    }
    by_code = {code: {"code": code, "name": code, "prereqRaw": raw} for code, raw in raws.items()}
    reverse: dict[str, set[str]] = {}
    for code, prereqs in edges.items():
        for ref in prereqs:
            reverse.setdefault(ref, set()).add(code)
    return bm.Graph(
        edges=edges,
        reverse={code: tuple(sorted(deps)) for code, deps in reverse.items()},
        depth=bm.longest_depths(edges),
        levels={code: len(bm.walk_chain(code, edges).levels) for code in edges},
        closure={
            code: sum(len(level) for level in bm.walk_chain(code, edges).levels) for code in edges
        },
        cycles=bm.cycle_members(edges),
        by_code=by_code,
    )


def item(**kwargs) -> bm.Item:
    """A row that passes every check, so a test can break exactly one thing."""
    return bm.Item(
        **{
            "shape": "chain",
            "root": "AAA 300",
            "question": "what do I need before AAA 300, and what does that require?",
            "gold_answer": "AAA 300 requires AAA 200, and AAA 200 requires AAA 100 or consent.",
            "chain": ("AAA 300", "AAA 200", "AAA 100"),
            **kwargs,
        }
    )


def fields(violations) -> list[str]:
    return [v.field for v in violations]


def message(violations) -> str:
    """The first finding's message, or a sentence that fails the assertion readably."""
    found = list(violations)
    return found[0].message if found else "<the check found nothing>"


# --- the table, against #58's acceptance ----------------------------------------------------------


def test_the_bucket_is_twenty_five_rows_over_five_question_shapes():
    assert len(bm.ITEMS) == bm.TOTAL == RULES[bm.BUCKET].target_n == 25
    assert len(bm.SHAPES) == 5
    for name, rule in bm.SHAPES.items():
        assert sum(1 for one in bm.ITEMS if one.shape == name) == rule.n
        assert rule.n >= 1, f"{name} would not be a shape the bucket actually holds"


def test_every_row_is_what_the_schema_says_a_multihop_row_is(rows):
    for row in rows:
        one = EvalItem(**{k: v for k, v in row.items() if k != "provenance"})
        assert one.bucket == bm.BUCKET
        assert one.should_abstain is RULES[bm.BUCKET].should_abstain is False
        assert len(one.gold_doc_ids) >= RULES[bm.BUCKET].min_gold_ids >= 2


def test_no_row_joins_more_records_than_the_guidelines_allow(rows):
    # "Cap the join at four records. More than that is a list question." -- eval/GUIDELINES.md.
    for row in rows:
        assert bm.MIN_CHAIN <= len(row["gold_doc_ids"]) <= bm.MAX_CHAIN == 4


def test_every_gold_id_is_a_real_chunk_and_the_root_comes_first(rows):
    known = corpus_doc_ids()
    for row in rows:
        assert set(row["gold_doc_ids"]) <= known
        assert len(set(row["gold_doc_ids"])) == len(row["gold_doc_ids"])
        root = row["provenance"]["root"].replace(" ", "-")
        assert row["gold_doc_ids"][0].endswith(f":{root}")


def test_the_ids_come_out_of_the_block_fifty_six_reserved(rows):
    assert [row["id"] for row in rows] == [
        f"v1-{ID_BLOCKS[bm.BUCKET] + n:04d}" for n in range(len(rows))
    ]


def test_no_question_is_asked_twice():
    asked = [normalize_question(one.question) for one in bm.ITEMS]
    assert len(set(asked)) == len(asked)


def test_the_table_checks_out_against_the_committed_graph(graph):
    assert bm.check_table(bm.ITEMS, graph) == []


def test_every_root_is_multi_hop_on_both_measures(graph):
    # 738 courses reach depth >= 2 by longest chain and 718 by the tool's levels; a row that
    # relied on the difference would be multi-hop only under the measure that was convenient.
    for one in bm.ITEMS:
        assert graph.depth[one.root] >= bm.MIN_DEPTH
        assert graph.levels[one.root] >= bm.MIN_DEPTH


def test_the_depth_distribution_is_stratified_and_not_the_shallow_end(rows):
    depths = [row["provenance"]["depth"] for row in rows]
    assert sum(1 for d in depths if d >= bm.DEEP_AT) >= bm.MIN_DEEP
    assert sum(1 for d in depths if d == bm.MIN_DEPTH) <= bm.MAX_SHALLOW
    assert max(depths) >= 12, "no row exercises a chain deep enough for a half-working walk"
    assert len(set(depths)) >= 8, "a distribution over three values is not a stratification"


def test_at_least_two_rows_are_inside_one_of_the_seven_cycles(rows, graph):
    cyclic = [row for row in rows if row["provenance"]["cycle_members"]]
    assert len(cyclic) >= bm.MIN_CYCLIC
    assert {c for row in cyclic for c in row["provenance"]["cycle_members"]} <= graph.cycles


def test_the_chains_are_walks_and_not_just_pairs_of_documents(graph):
    for one in bm.ITEMS:
        rule = bm.shape(one.shape)
        for code in one.chain[1:]:
            if rule.reverse:
                assert one.root in graph.edges.get(code, ())
            else:
                assert code in bm.reachable_from(one.root, graph)


def test_every_answer_names_every_record_it_cites():
    # GUIDELINES rule 4: a gold id supports the claim, not the topic.
    for one in bm.ITEMS:
        for code in one.chain:
            assert code in one.gold_answer


def test_a_disjunction_in_the_sentence_survives_into_the_answer(graph):
    for n, one in enumerate(bm.ITEMS, start=1):
        assert list(bm.check_disjunction(n, one, graph)) == []


def test_every_row_whose_chain_disagrees_with_the_catalog_carries_the_verdict(graph):
    for one in bm.ITEMS:
        if bm.disagreements(one, graph):
            assert one.note.strip(), f"{one.root} has a disagreement and no note"


def test_the_log_carries_all_four_kinds_of_disagreement(rows):
    found = {
        row["provenance"]["root"]: row["provenance"]["disagreements"]
        for row in rows
        if row["provenance"]["disagreements"]
    }
    # A requirement dropped, and ME 200 is offered this term, so it is not the term filter.
    assert ["ME 200"] in [one["dropped_offered"] for one in found["AE 434"]]
    # A recommendation dropped, which is the parser being right.
    assert ["PHYS 213", "PHYS 214"] in [one["dropped_offered"] for one in found["ASTR 405"]]
    # A negation read as a mention.
    assert ["JOUR 317", "JOUR 217"] in [one["dropped_absent"] for one in found["JOUR 340"]]
    # The term filter turning a disjunction into a bare requirement -- #58's ABE 228 case.
    assert ["MATH 221"] in [one["dropped_absent"] for one in found["ATMS 202"]]


def test_the_guidelines_boundary_case_that_was_deleted_is_asked_boundedly_instead():
    # M2 deletes "list every course you have to take before MATH 412" because the transitive
    # answer does not terminate. The course is in the bucket, asked a question that does.
    rows = [one for one in bm.ITEMS if one.root == "MATH 412"]
    assert rows
    for one in rows:
        assert "every course" not in one.question.casefold()
        assert len(one.chain) <= bm.MAX_CHAIN


def test_a_reachable_row_records_what_the_tool_would_have_said(rows):
    # The whole point of several of these rows is that the target IS in the closure and the
    # answer is still no. If none of them were, the bucket would not test the distinction.
    reachable = [row for row in rows if row["provenance"]["shape"] == "reachable"]
    assert reachable
    assert all(row["provenance"]["target_in_closure"] is not None for row in reachable)
    assert any(row["provenance"]["target_in_closure"] for row in reachable)


# --- the graph, read twice ------------------------------------------------------------------------


def test_the_reader_reproduces_the_census_the_issue_quotes(graph):
    assert len(graph.edges) == 1135
    assert sum(1 for d in graph.depth.values() if d >= 2) == 738
    assert max(graph.depth.values()) == 13
    assert len(graph.cycles) == 19


def test_the_two_depth_measures_are_recorded_separately_because_they_disagree(graph):
    # The tool layers by shortest distance and #58 counted longest chains; 718 is not 738.
    assert sum(1 for d in graph.levels.values() if d >= 2) == 718
    assert max(graph.levels.values()) == 9
    assert all(graph.levels[c] <= graph.depth[c] for c in graph.edges)


def test_the_cycles_are_the_seven_components_prereqs_documents(graph):
    assert graph.cycles == frozenset(
        {
            "ACCY 301",
            "ACCY 302",
            "CHEM 204",
            "CHEM 205",
            "CS 225",
            "CS 374",
            "MATH 412",
            "MCB 581",
            "MCB 582",
            "SE 311",
            "SE 312",
            "THEA 375",
            "THEA 376",
            "THEA 377",
            "THEA 378",
            "THEA 475",
            "THEA 476",
            "THEA 477",
            "THEA 478",
        }
    )


def test_the_longest_walk_terminates_inside_a_cycle(graph):
    # SE 495 is the deepest course in the graph and its closure contains SE 311/SE 312.
    assert graph.depth["SE 495"] == 13
    assert {"SE 311", "SE 312"} <= bm.reachable_from("SE 495", graph)


def test_a_component_of_one_is_not_a_cycle():
    assert bm.cycle_members({"A 100": ("B 100",), "B 100": ()}) == frozenset()
    assert bm.cycle_members({"A 100": ("B 100",), "B 100": ("A 100",)}) == frozenset(
        {"A 100", "B 100"}
    )


def test_the_export_never_invents_an_edge_the_sentence_does_not_name(graph):
    # `added` is the half of the disagreement scan that should be empty everywhere, and is.
    added = [code for code in graph.edges if (one := bm.disagreement(code, graph)) and one.added]
    assert added == []


def test_a_dropped_code_is_split_by_whether_the_catalog_offers_it(graph):
    # AE 312's sentence names ME 200 (offered) and MATH 412's names MATH 347 (not offered), and
    # only the first is a parser bug -- the scan must not present them as the same finding.
    assert bm.disagreement("AE 312", graph).dropped_offered == ("ME 200",)
    assert bm.disagreement("AE 312", graph).dropped_absent == ()
    assert bm.disagreement("MATH 412", graph).dropped_absent == ("MATH 347",)
    assert bm.disagreement("MATH 412", graph).dropped_offered == ()


def test_a_course_whose_list_matches_its_sentence_has_no_disagreement(graph):
    assert bm.disagreement("FR 104", graph) is None
    assert bm.disagreement("CEE 462", graph) is None


def test_asserted_is_the_courses_whose_requirements_the_answer_characterizes():
    g = fake_graph()
    # AAA 100 is in the chain as the thing required, so nothing is claimed about ITS sentence.
    assert bm.asserted(item(), g) == ("AAA 300", "AAA 200")
    # Reversed, the dependents are the ones whose sentences name the root.
    unlocks = item(shape="unlocks", root="AAA 200", chain=("AAA 200", "AAA 300", "BBB 300"))
    assert bm.asserted(unlocks, g) == ("AAA 300", "BBB 300")


# --- one check at a time --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chain, expected",
    [
        (("AAA 200", "AAA 100"), "does not start with AAA 300"),
        (("AAA 300",), "joins 1 record"),
        (("AAA 300", "AAA 200", "AAA 100", "BBB 200", "BBB 300"), "joins 5 records"),
        (("AAA 300", "AAA 200", "AAA 200"), "names a course twice"),
        (("AAA 300", "ZZZ 100"), "not in the 2026-sp export"),
        (("AAA 300", "BBB 200"), "not in AAA 300's prerequisite closure"),
    ],
)
def test_a_chain_that_is_not_a_walk_out_of_the_root(chain, expected):
    found = list(bm.check_chain(1, item(chain=chain), fake_graph()))
    assert found and expected in found[0].message


def test_an_unlocks_row_whose_course_does_not_list_the_root():
    row = item(shape="unlocks", root="AAA 200", chain=("AAA 200", "AAA 100"))
    found = list(bm.check_chain(1, row, fake_graph()))
    assert found and "does not list AAA 200 as a prerequisite" in found[0].message


def test_a_root_with_no_chain_at_all():
    row = item(root="AAA 100", chain=("AAA 100", "AAA 200"))
    assert "no record in data/prereqs.json" in message(bm.check_depth(1, row, fake_graph()))


def test_a_root_that_is_one_hop_wearing_this_bucket():
    row = item(root="AAA 200", chain=("AAA 200", "AAA 100"))
    found = list(bm.check_depth(1, row, fake_graph()))
    assert found and "one hop wearing this bucket" in found[0].message


def test_a_target_on_a_shape_that_does_not_take_one():
    found = list(bm.check_target(1, item(target="AAA 100"), fake_graph()))
    assert found and "only a reachable row names one" in found[0].message


def test_a_reachable_row_with_no_target_and_one_that_is_not_a_course():
    row = item(shape="reachable", question="is AAA 100 required for AAA 300?")
    assert "names the course it asks about" in message(bm.check_target(1, row, fake_graph()))
    bad = row._replace(target="ZZZ 100")
    assert "not in the 2026-sp export" in message(bm.check_target(1, bad, fake_graph()))


def test_a_reachable_row_whose_target_is_not_reachable_is_allowed():
    # Half these rows answer "no". Requiring the target to be in the closure would delete them.
    row = item(
        shape="reachable",
        target="BBB 200",
        question="is BBB 200 required for AAA 300?",
        gold_answer="No. AAA 300 requires AAA 200, and AAA 200 requires AAA 100 or consent.",
    )
    assert list(bm.check_target(1, row, fake_graph())) == []


def test_held_on_a_shape_that_does_not_take_it_and_missing_where_it_does():
    found = list(bm.check_held(1, item(held=("AAA 100",)), fake_graph()))
    assert found and "only an eligibility row says what the student has" in found[0].message
    bare = item(shape="eligibility")
    assert "records which courses the question grants" in message(
        bm.check_held(1, bare, fake_graph())
    )


@pytest.mark.parametrize(
    "question, expected",
    [
        ("what do I need before that course?", "does not name AAA 300"),
        ("what do I need before AAA 300 and CCC 301?", "names CCC 301"),
    ],
)
def test_a_question_that_is_not_about_the_row(question, expected):
    found = list(bm.check_question(1, item(question=question)))
    assert found and expected in found[0].message


def test_a_question_that_drops_its_target_or_the_courses_it_grants():
    row = item(shape="reachable", target="AAA 100", question="is anything required for AAA 300?")
    assert "does not name its target AAA 100" in message(bm.check_question(1, row))
    held = item(shape="eligibility", held=("AAA 100",), question="can I take AAA 300 now?")
    assert "which it grants in held" in message(bm.check_question(1, held))


def test_an_answer_that_does_not_use_a_record_it_cites():
    found = list(bm.check_answer(1, item(gold_answer="AAA 300 requires AAA 200.")))
    assert found and "does not name AAA 100" in found[0].message


def test_an_answer_that_is_not_written():
    assert "is empty" in message(bm.check_answer(1, item(gold_answer="   ")))


def test_an_answer_that_turns_a_disjunction_into_a_requirement():
    # AAA 200's sentence is "AAA 100 or consent of instructor."
    row = item(gold_answer="AAA 300 requires AAA 200, and AAA 200 requires AAA 100.")
    found = list(bm.check_disjunction(1, row, fake_graph()))
    assert found and "states no alternative" in found[0].message


def test_a_row_built_on_sentences_that_offer_no_choice_needs_no_alternative():
    g = fake_graph()
    row = item(
        root="BBB 300",
        chain=("BBB 300", "BBB 200"),
        question="what do I need before BBB 300?",
        gold_answer="BBB 300 requires AAA 200 and BBB 200.",
    )
    assert list(bm.check_disjunction(1, row, g)) == []


def test_a_disagreement_nobody_judged():
    g = fake_graph()
    # AAA 300's sentence now names a course its exported list drops, which is exactly the shape
    # of AE 312 and ME 200 in the live graph.
    g = g._replace(
        by_code={**g.by_code, "AAA 300": {**g.by_code["AAA 300"], "prereqRaw": "AAA 200, BBB 200."}}
    )
    assert bm.disagreement("AAA 300", g).dropped_offered == ("BBB 200",)
    found = list(bm.check_note(1, item(), g))
    assert found and "not resolved in the tool's favour" in found[0].message


def test_an_unknown_shape_names_the_five_that_exist():
    found = bm.check_table([item(shape="lookup")], fake_graph())
    assert any("the five are" in v.message for v in found)
    with pytest.raises(ValueError, match="the five are"):
        bm.shape("lookup")


def test_a_short_shape_fails_rather_than_warns(graph):
    found = bm.check_table(bm.ITEMS[:-1], graph)
    assert [v.level for v in found] == ["error"] * len(found)
    assert any("not the 4 the shape is built to hold" in v.message for v in found)
    assert any("24 rows in total, not 25" in v.message for v in found)


@pytest.mark.parametrize(
    "depths, cyclic, expected",
    [
        ([2] * 25, 5, "at depth 5 or more"),
        ([2] * 20 + [9] * 5, 5, "at depth 2, more than"),
        ([9] * 25, 1, "inside one of the 7 cycles"),
    ],
)
def test_a_bucket_that_avoids_the_hard_half_of_the_graph(depths, cyclic, expected):
    found = list(bm.check_distribution(depths, cyclic))
    assert found and any(expected in v.message for v in found)


def test_a_valid_distribution_says_nothing():
    assert list(bm.check_distribution([2, 3, 5, 9, 11, 13, 13], 2)) == []


def test_the_table_reports_every_problem_and_not_the_first():
    broken = item(
        root="AAA 200",
        chain=("AAA 300", "ZZZ 100"),
        question="tell me about courses",
        gold_answer="",
        target="AAA 100",
    )
    found = bm.check_table([broken], fake_graph())
    assert len({f for f in fields(found) if f}) >= 5


# --- the run and the file -------------------------------------------------------------------------


def test_a_run_writes_rows_a_sidecar_and_nothing_that_fails_the_draft_checks(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path)
    assert report.table == [] and report.written == []
    assert len(report.rows) == bm.TOTAL
    assert report.path.exists() and report.meta.exists()


def test_the_run_re_reads_what_it_wrote_with_its_own_provenance_keys(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path)
    drafts, violations = parse_drafts(report.path, bm.PROVENANCE_KEYS)
    assert violations == [] and len(drafts) == bm.TOTAL
    assert set(drafts[0].provenance) == set(bm.PROVENANCE_KEYS)


def test_a_graph_built_row_is_not_a_drafted_one_and_the_reader_says_so(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path)
    assert set(bm.PROVENANCE_KEYS) != set(DRAFTED_KEYS)
    assert check_drafts(report.path, corpus_doc_ids(), keys=DRAFTED_KEYS) != []


def test_it_does_not_write_where_the_drafting_harness_writes(tmp_path):
    # Both produce multihop rows and they carry different provenance; sharing a file name would
    # mean whichever ran second silently clobbered the other.
    assert bm.STEM != bm.BUCKET
    assert bm.build(CFG, drafts_dir=tmp_path).path != draft_path(bm.BUCKET, CFG, tmp_path)


def test_a_run_is_idempotent(tmp_path):
    first = bm.build(CFG, drafts_dir=tmp_path).path.read_bytes()
    assert bm.build(CFG, drafts_dir=tmp_path).path.read_bytes() == first


def test_nothing_is_written_when_the_table_does_not_check_out(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path, items=[item()])
    assert report.table and report.rows == []
    assert not report.path.exists()
    assert "does not check out" in bm.summary(CFG, report)


def test_the_sidecar_answers_the_three_questions_the_report_asks(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path)
    meta = json.loads(report.meta.read_text())
    assert meta["rows"] == bm.TOTAL and meta["source"] == bm.SOURCE
    assert sum(meta["shapes"].values()) == bm.TOTAL
    assert meta["depth"]["max"] == 13 and meta["depth"]["at_least_5"] >= bm.MIN_DEEP
    assert meta["graph"]["depth_2_or_more_by_longest_chain"] == 738
    assert meta["graph"]["depth_2_or_more_by_levels"] == 718
    assert len(meta["cyclic_rows"]) >= bm.MIN_CYCLIC
    assert meta["disagreements"]


def test_the_summary_prints_the_agreement_log(tmp_path):
    report = bm.build(CFG, drafts_dir=tmp_path)
    text = bm.summary(CFG, report)
    assert "tool vs raw" in text
    assert "ME 200 (offered this term)" in text
    assert "MATH 347 (not in this term)" in text


def test_the_committed_file_is_the_one_this_table_produces(tmp_path):
    if not COMMITTED.exists():  # pragma: no cover - a fresh clone before the first run
        pytest.skip(f"{COMMITTED} has not been written yet")
    fresh = bm.build(CFG, drafts_dir=tmp_path)
    assert fresh.path.read_text() == COMMITTED.read_text()
