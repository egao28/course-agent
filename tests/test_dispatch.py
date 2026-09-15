"""Tests for agent/nodes/dispatch.py: the four rows, one record shape, and the three tool errors.

Two corpora and one fake, for three jobs. The tool rows run against the 17 courses
``tests/fixtures/catalog_sample.csv`` produces through the real exporter, with all three tools'
paths redirected into ``tmp_path`` -- a row's evidence is the whole point of the row, so a fake
tool would test nothing. The retrieval rows run against a fake ``retrieve``, because ``indexes/``
is gitignored and a checkout has no 6 MB BM25 pickle to open; what those rows have to get right is
that retrieval ran, with the query verbatim and the configured mode, and that its hits keep their
scores. The last section reads the live corpus, where #36's acceptance is stated -- and asserts the
*route* for the retrieval rows rather than the ranking, which is Day 1's business and is pinned in
``tests/test_retrieve.py``.

``fresh_cache`` is autouse and clears all three tool caches plus the graph's: the audit and the
chain both read the catalog index, so a test that left ``courses``'s cache behind would answer the
next one out of the wrong corpus.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent import graph as ag
from agent.nodes import exits
from agent.nodes.dispatch import (
    DISPATCHED,
    GENED,
    MAX_OPTIONS,
    TOOL_ERRORS,
    catalog_subjects,
    category_line,
    check_retrieval_mode,
    code_spans,
    dispatch_node,
    extract_codes,
    extract_program,
    pick_code,
    program_spellings,
    render_chain,
    render_course,
    render_requirements,
)
from agent.state import INTENTS, Evidence, check_update, initial_state
from agent.tools import courses as tc
from agent.tools import prereqs as tp
from agent.tools import requirements as tr
from config import Config, bm25_key, dense_key, load_config
from ingest import export_almabot as ex
from ingest.chunk import chunk_key, render_chunk
from ingest.tokenize import identity_token
from retrieval import MODES, RetrievalResult
from retrieval.types import Hit

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"
TERM = "2026-sp"
LLM = {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}

#: Two hits shaped like the BM25 index's, scores and all, for the retrieval rows.
HITS = [
    Hit(
        doc_id="2026-sp:CS-374",
        score=29.0337,
        rank=1,
        text="CS 374 — Introduction to Algorithms\nCredits: 4",
        metadata={"code": "CS 374"},
    ),
    Hit(
        doc_id="2026-sp:CS-173",
        score=11.5,
        rank=2,
        text="CS 173 — Discrete Structures\nCredits: 3",
        metadata={"code": "CS 173"},
    ),
]


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits the corpus -- or the graph -- another one built."""
    for module in (tr, tp, tc):
        module.clear_cache()
    ag.clear_cache()
    yield
    for module in (tr, tp, tc):
        module.clear_cache()
    ag.clear_cache()


@pytest.fixture
def cfg() -> Config:
    """A synthetic config: the term the exits quote, the section the router validates, the mode."""
    return Config({"data": {"term": TERM}, "llm": LLM, "agent": {"retrieval_mode": "bm25"}})


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV).courses


@pytest.fixture
def corpus(tmp_path, courses, monkeypatch) -> dict[str, Path]:
    """Write the catalog, the prereq graph and the rules where the three tools will look."""
    paths = {
        "courses": tmp_path / "courses.json",
        "prereqs": tmp_path / "prereqs.json",
        "requirements": tmp_path / "requirements.json",
    }
    ex.write_json(paths["courses"], courses)
    ex.write_json(paths["prereqs"], ex.build_prereqs(courses))
    ex.write_json(paths["requirements"], ex.build_requirements(courses, {"data": {"term": TERM}}))
    monkeypatch.setattr(tc, "COURSES_PATH", paths["courses"])
    monkeypatch.setattr(tp, "PREREQS_PATH", paths["prereqs"])
    monkeypatch.setattr(tr, "REQUIREMENTS_PATH", paths["requirements"])
    return paths


class FakeRetrieval:
    """A stand-in for :func:`retrieval.retrieve`: the hits it was made with, and every call."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.hits = HITS if hits is None else hits
        self.calls: list[tuple] = []

    def __call__(self, query: str, mode: str, cfg=None, **kwargs) -> RetrievalResult:
        self.calls.append((query, mode, cfg))
        return RetrievalResult(
            query=query, mode=mode, config_hash="c0ffee00", hits=self.hits, latency_ms=1.0
        )


def dispatch(cfg: Config, **kwargs):
    """The node under test, with a fake retriever unless the test asked for something else."""
    kwargs.setdefault("retrieve", FakeRetrieval())
    return dispatch_node(cfg, **kwargs)


def run_row(cfg: Config, query: str, intent: str, **kwargs) -> dict:
    """One query through the real node at a forced intent; the update it returned."""
    state = initial_state(query)
    state["intent"] = intent  # type: ignore[typeddict-item]
    return dict(dispatch(cfg, **kwargs)(state))


# --- reading a course code out of a question ------------------------------------------------------


@pytest.mark.parametrize("query", ["Is CS 374 hard?", "is cs 374 hard?", "Is CS374 hard?"])
def test_every_spelling_of_a_code_reads_as_the_one_course(corpus, query):
    # The three spellings the BM25 index collapses (#20): a query must not retrieve under one
    # normalization and look up under another.
    assert extract_codes(query) == ["CS 374"]


def test_the_code_a_question_names_is_the_term_the_index_holds_it_under(corpus):
    # Tied to the index rather than to a string: `identity_token` is what ingest/tokenize.py
    # repeats `bm25.code_boost` times for the course that *is* this code.
    assert identity_token(extract_codes("what about cs374")[0]) == "code:cs374"


def test_every_code_is_returned_in_order_and_once(corpus):
    query = "Is CS 374 harder than CS 173, and is CS374 the same course?"
    assert extract_codes(query) == ["CS 374", "CS 173"]


def test_a_sentence_that_fits_the_pattern_is_not_a_course(corpus):
    # "have 100" matches CODE_RE: two-to-four letters, a space, three digits. There is no subject
    # HAVE, which is what the catalog's own vocabulary is filtering for.
    assert code_spans("I have 100 credits") == ["HAVE 100"]
    assert extract_codes("I have 100 credits") == []
    assert extract_codes("Tell me about the top 100 courses") == []


def test_a_code_inside_a_word_is_not_a_code(corpus):
    assert extract_codes("the ABCS374 identifier") == []


def test_the_subject_vocabulary_is_the_corpus_not_a_word_list(corpus):
    subjects = catalog_subjects()
    assert {"CS", "MATH", "STAT", "ANTH"} <= subjects
    assert "HAVE" not in subjects and "TOP" not in subjects
    assert len(subjects) == 11  # the fixture's eleven; the live catalog has 187


def test_a_stray_match_does_not_shadow_the_course_that_is_in_the_catalog(corpus):
    # MATH 213 is a real code the fixture catalog does not offer; CS 374 is the course asked about.
    assert pick_code(["MATH 213", "CS 374"]) == "CS 374"


def test_a_code_nothing_matches_is_still_the_one_looked_up(corpus):
    # So that "not in this catalog" stays a different sentence from "you named no course".
    assert pick_code(["MATH 213"]) == "MATH 213"
    assert pick_code([]) is None


# --- reading a program out of a question ----------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Which Gen Eds do I still need?",
        "What gen ed requirements are left?",
        "How am I doing on general education?",
    ],
)
def test_a_gen_ed_question_resolves_to_the_campus_audit(corpus, query):
    assert extract_program(query) == GENED


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is left for the Statistics major?", "Statistics, BSLAS"),
        ("What is left for the CS degree?", "Computer Science, BS"),
        ("What does Computer Science, BS still require?", "Computer Science, BS"),
    ],
)
def test_a_degree_resolves_through_the_files_own_aliases(corpus, query, expected):
    assert extract_program(query) == expected


def test_a_course_code_is_not_read_as_the_program_it_shares_a_subject_with(corpus):
    # `stat` is one of the Statistics degree's aliases and the subject of STAT 100. A transcript
    # is not a program, so the codes come out of the text before any spelling is looked for.
    assert extract_program("What is left if I have taken STAT 100?") is None
    assert extract_program("What is left for the Statistics major if I have taken STAT 100?") == (
        "Statistics, BSLAS"
    )


def test_gen_ed_wins_when_a_question_names_both(corpus):
    # The degree's own audit reports the campus Gen Eds in not_checked, unevaluated -- it is the
    # one audit that cannot answer this question.
    assert extract_program("Which Gen Eds are left for the Statistics major?") == GENED


def test_a_program_this_corpus_does_not_carry_is_not_resolved(corpus):
    assert extract_program("What is left for the Physics major?") is None
    assert extract_program("What do I still need to graduate?") is None


def test_the_spellings_come_from_the_file_and_include_both_halves(corpus):
    spellings = program_spellings()
    assert spellings["gen ed"] == GENED
    assert spellings["computer science bs"] == "Computer Science, BS"
    assert spellings["statistics"] == "Statistics, BSLAS"
    # The id and the label answer too, not only the hand-written aliases.
    assert spellings["cs bs"] == "Computer Science, BS"


# --- rendering a tool result as evidence ----------------------------------------------------------


def test_a_lookup_block_reads_like_the_indexed_chunk_of_the_same_course(corpus, courses):
    # One vocabulary for #38: the shared fields are the chunk template's, label for label, so an
    # answer built from a lookup and one built from a retrieved chunk are reading the same shape.
    course = next(c for c in courses if c["code"] == "CS 374")
    block = render_course(tc.get_course("CS 374"))
    assert block.splitlines()[:6] == render_chunk(course, "v1").splitlines()
    assert block.splitlines()[6] == f"Sections: {course['sectionCount']}"


def test_a_chain_block_leads_with_the_catalogs_own_sentence(corpus):
    block = render_chain(tp.get_prereq_chain("CS 374"))
    lines = block.splitlines()
    assert lines[0] == "CS 374 — what the catalog requires before it"
    assert lines[1] == "Prerequisite statement: One of CS 173, MATH 213; CS 225."
    assert lines[2] == (
        "Immediately before it: CS 173 — Discrete Structures; CS 225 — Data Structures"
    )


def test_a_chain_block_says_that_a_level_is_not_a_list_of_required_courses(corpus):
    # 744 of the live catalog's 1,135 statements contain a disjunction, and the layers flatten it:
    # CS 374 asks for CS 225 and *one of* CS 173 / MATH 213, and its first level lists both.
    block = render_chain(tp.get_prereq_chain("CS 374"))
    assert 'where a statement says "one of", only one of them is required' in block


def test_the_caveat_is_absent_when_no_level_could_be_misread(corpus):
    # ANTH 390 needs exactly CS 173. Nothing to disambiguate, so nothing to disclaim.
    block = render_chain(tp.get_prereq_chain("ANTH 390"))
    assert "only one of them is required" not in block
    assert "Immediately before it: CS 173 — Discrete Structures" in block


def test_a_course_with_no_prerequisites_says_so_rather_than_printing_nothing(corpus):
    block = render_chain(tp.get_prereq_chain("CS 173"))
    assert "Prerequisite statement: None listed" in block
    assert "Nothing in this catalog is listed as a prerequisite for it." in block


def test_a_cycle_in_the_catalogs_own_data_is_reported_in_the_block(corpus):
    chain = tp.get_prereq_chain("CS 421")
    assert chain.found
    block = render_chain(chain)
    assert ("form a cycle inside this chain" in block) is chain.cycle_detected


def test_a_truncated_walk_says_there_is_more_below_it(corpus):
    block = render_chain(tp.get_prereq_chain("CS 421", max_depth=1))
    assert "Cut off at the depth asked for" in block


def test_an_audit_block_carries_both_lists_one_requirement_to_a_line(corpus):
    block = render_requirements(tr.check_requirement(GENED, ["STAT 100", "MATH 101"]))
    lines = block.splitlines()
    assert lines[0] == "General Education — requirement audit"
    assert "Satisfied (2):" in lines
    assert "- Quantitative Reasoning I (Quantitative Reasoning) -- satisfied by STAT 100" in block
    assert sum(1 for line in lines if line.startswith("- ")) == 13


def test_an_audit_block_keeps_what_a_course_list_cannot_decide(corpus):
    # 8 of the CS degree's 17 groups are hours, GPA and advising rules. An audit that dropped them
    # would read as "you are done" to a student several requirements from done.
    status = tr.check_requirement("Computer Science, BS", ["CS 173", "CS 225"])
    block = render_requirements(status)
    assert f"Not decided by a list of courses ({len(status.not_checked)}):" in block
    assert "- Campus General Education requirements (General Education)" in block
    assert block.splitlines()[-1].startswith("Caveats: ")


def test_a_gen_ed_audit_has_no_unevaluated_section_to_disclaim(corpus):
    block = render_requirements(tr.check_requirement(GENED, ["STAT 100"]))
    assert "Not decided by a list of courses" not in block


def test_an_unknown_course_in_a_transcript_reaches_the_block(corpus):
    block = render_requirements(tr.check_requirement(GENED, ["STAT 100", "MATH 213"]))
    assert "Courses given that this catalog does not offer: MATH 213" in block


def test_a_long_option_list_is_counted_rather_than_printed(corpus):
    status = tr.check_requirement("Computer Science, BS", [])
    long = next(c for c in status.unsatisfied if len(c.options) > MAX_OPTIONS)
    line = category_line(long)
    assert f"and {len(long.options) - MAX_OPTIONS} more" in line
    assert all(option in line for option in long.options[:MAX_OPTIONS])


# --- the mode comes from the config ---------------------------------------------------------------


def test_the_configured_mode_is_what_the_retrieval_rows_ask_for(cfg, corpus):
    fake = FakeRetrieval()
    query = "Which intro machine learning course should I take?"
    run_row(cfg, query, "compare_recommend", retrieve=fake)
    assert fake.calls == [(query, "bm25", cfg)]


def test_the_query_reaches_retrieval_verbatim(cfg, corpus):
    fake = FakeRetrieval()
    query = "  Which intro ML course should I take??  "
    state = initial_state(query)
    state["intent"] = "compare_recommend"
    dispatch(cfg, retrieve=fake)(state)
    assert fake.calls[0][0] == query


def test_an_override_replaces_the_configured_mode(cfg, corpus):
    fake = FakeRetrieval()
    run_row(cfg, "quantum origami", "compare_recommend", mode="hybrid", retrieve=fake)
    assert fake.calls[0][1] == "hybrid"


def test_a_missing_agent_section_is_refused_when_the_node_is_built():
    with pytest.raises(ValueError, match="no `agent` section"):
        dispatch_node(Config({"data": {"term": TERM}}))


def test_a_missing_mode_names_the_key_and_the_file():
    with pytest.raises(ValueError, match="no retrieval_mode"):
        dispatch_node(Config({"data": {"term": TERM}, "agent": {}}))


def test_a_mode_retrieve_does_not_implement_is_refused_before_any_query():
    with pytest.raises(ValueError, match="agent.retrieval_mode to 'keyword'"):
        dispatch_node(Config({"data": {"term": TERM}, "agent": {"retrieval_mode": "keyword"}}))


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_retrieve_implements_is_accepted(mode):
    cfg = Config({"data": {"term": TERM}, "agent": {"retrieval_mode": mode}})
    assert check_retrieval_mode(cfg) == mode
    assert check_retrieval_mode(cfg, "bm25") == "bm25"


def test_building_the_node_opens_no_index_and_asks_no_question(cfg, corpus):
    fake = FakeRetrieval()
    dispatch_node(cfg, retrieve=fake)
    assert fake.calls == []


# --- the four rows --------------------------------------------------------------------------------


def test_a_course_fact_with_a_code_is_a_lookup_not_a_search(cfg, corpus):
    fake = FakeRetrieval()
    update = run_row(cfg, "How many credits is CS 374?", "course_fact", retrieve=fake)
    (block,) = update["docs"]
    assert (block.doc_id, block.source, block.score) == ("CS 374", "get_course", None)
    assert block.text.startswith("CS 374 — Introduction to Algorithms")
    assert fake.calls == []
    assert update["signals"] == {"retrieval_mode": None, "tool_error": None}


def test_a_course_fact_with_no_code_is_a_search(cfg, corpus):
    fake = FakeRetrieval()
    update = run_row(cfg, "What does the algorithms course cover?", "course_fact", retrieve=fake)
    assert [block.doc_id for block in update["docs"]] == ["2026-sp:CS-374", "2026-sp:CS-173"]
    assert len(fake.calls) == 1
    assert update["signals"] == {"retrieval_mode": "bm25", "tool_error": None}


def test_a_prereq_question_with_a_code_walks_the_chain(cfg, corpus):
    update = run_row(cfg, "What do I need before CS 374?", "prereq_chain")
    (block,) = update["docs"]
    assert (block.doc_id, block.source, block.score) == ("CS 374", "get_prereq_chain", None)
    assert "Immediately before it: CS 173" in block.text
    assert update["signals"] == {"retrieval_mode": None, "tool_error": None}


def test_a_prereq_question_with_no_code_refuses_rather_than_guessing(cfg, corpus):
    # #36's third acceptance. Asking a model which course was meant is how an agent ends up
    # answering confidently about the wrong one.
    fake = FakeRetrieval()
    update = run_row(cfg, "What are the prerequisites?", "prereq_chain", retrieve=fake)
    assert update["docs"] == []
    assert update["signals"] == {"retrieval_mode": None, "tool_error": "no_course_code"}
    assert fake.calls == []


def test_a_requirement_question_audits_the_program_it_names(cfg, corpus):
    update = run_row(
        cfg,
        "Which Gen Eds do I still need if I have taken STAT 100 and MATH 101?",
        "degree_requirement",
    )
    (block,) = update["docs"]
    assert block.doc_id == "General Education"
    assert (block.source, block.score) == ("check_requirement", None)
    assert "satisfied by STAT 100" in block.text
    assert update["signals"] == {"retrieval_mode": None, "tool_error": None}


def test_a_requirement_question_reads_the_transcript_out_of_the_query(cfg, corpus):
    update = run_row(cfg, "Which Gen Eds are left if I took MATH 101?", "degree_requirement")
    assert "satisfied by MATH 101" in update["docs"][0].text


def test_a_requirement_question_naming_no_auditable_program_refuses(cfg, corpus):
    update = run_row(cfg, "What is left for the Physics major?", "degree_requirement")
    assert update["docs"] == []
    assert update["signals"] == {"retrieval_mode": None, "tool_error": "no_program"}


def test_a_recommendation_is_always_a_search(cfg, corpus):
    fake = FakeRetrieval()
    query = "Which intro ML course should I take?"
    update = run_row(cfg, query, "compare_recommend", retrieve=fake)
    assert len(update["docs"]) == 2
    assert len(fake.calls) == 1


@pytest.mark.parametrize("intent", ["course_fact", "prereq_chain"])
def test_a_course_this_catalog_does_not_have_is_a_different_refusal(cfg, corpus, intent):
    # MATH 213 is a real code this export does not offer. "Not in the catalog" and "you named no
    # course" are different sentences, and the reader is owed the right one.
    update = run_row(cfg, "What about MATH 213?", intent)
    assert update["docs"] == []
    assert update["signals"]["tool_error"] == "course_not_found"


# --- one record shape -----------------------------------------------------------------------------


ROWS = (
    ("course_fact", "How many credits is CS 374?"),
    ("course_fact", "What does the algorithms course cover?"),
    ("prereq_chain", "What do I need before CS 374?"),
    ("degree_requirement", "Which Gen Eds do I still need if I have taken STAT 100?"),
    ("compare_recommend", "Which intro ML course should I take?"),
)


@pytest.mark.parametrize(("intent", "query"), ROWS)
def test_every_row_returns_the_same_record_shape(cfg, corpus, intent, query):
    # #36's fourth acceptance, asserted on the type: a raw Hit has doc_id, text and score, so
    # everything downstream would work except the one field the gate reads.
    update = run_row(cfg, query, intent)
    assert update["docs"]
    assert all(isinstance(block, Evidence) for block in update["docs"])
    assert all(isinstance(block.text, str) and block.text for block in update["docs"])


@pytest.mark.parametrize(("intent", "query"), ROWS)
def test_every_row_writes_only_what_dispatch_owns(cfg, corpus, intent, query):
    update = run_row(cfg, query, intent)
    assert set(update) == {"docs", "signals"}
    assert check_update("dispatch", update) == update


@pytest.mark.parametrize(("intent", "query"), ROWS)
def test_a_tool_row_carries_no_score_and_a_retrieval_row_keeps_its_own(cfg, corpus, intent, query):
    # The distinction #37 reads: `score is None` is an exact catalog lookup and skips the
    # threshold, and a retrieved score is on its retriever's own scale.
    update = run_row(cfg, query, intent)
    scores = [block.score for block in update["docs"]]
    if update["signals"]["retrieval_mode"] is None:
        assert scores == [None]
    else:
        assert scores == [29.0337, 11.5]


def test_the_rows_are_the_five_labels_minus_the_one_that_never_arrives():
    assert set(DISPATCHED) == set(INTENTS) - {"out_of_scope"}
    assert len(DISPATCHED) == 4


def test_every_tool_error_has_a_sentence_for_the_person_who_gets_it(corpus):
    # The cross-module guard: a label with no entry in exits.REASONS reaches the reader as the
    # label itself, which teaches them nothing.
    assert set(TOOL_ERRORS) <= set(exits.REASONS)


# --- what must never happen -----------------------------------------------------------------------


def test_an_out_of_scope_intent_reaching_this_node_is_a_wiring_bug(cfg, corpus):
    with pytest.raises(ValueError, match="reached with intent 'out_of_scope'"):
        run_row(cfg, "Who is the head football coach?", "out_of_scope")


def test_a_query_with_no_intent_is_refused_rather_than_dispatched(cfg, corpus):
    with pytest.raises(ValueError, match="no `intent` set"):
        dispatch(cfg)(initial_state("How many credits is CS 374?"))


def test_free_text_never_indexes_the_table(cfg, corpus):
    # #35 guarantees one of five labels; this is what happens if that ever breaks.
    with pytest.raises(ValueError, match="not one of the five labels"):
        run_row(cfg, "q", "prerequisites, I think")


def test_importing_the_dispatcher_pulls_in_neither_openai_nor_chromadb():
    # The #22 rule: `retrieve` builds its retrievers lazily, so a run that answers from a tool --
    # or one in a checkout with no dense index and no key -- must not pay for chromadb.
    code = (
        "import agent.nodes.dispatch, sys; "
        "print('openai' in sys.modules, 'chromadb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


# --- the node inside the graph --------------------------------------------------------------------


def test_a_real_lookup_reaches_the_answer_node_as_evidence(cfg, corpus):
    seen: list[list[Evidence]] = []

    def answer(state):
        seen.append(list(state["docs"]))
        return {"answer": "CS 374 is 4 hours [CS 374].", "exit": "answer"}

    state = ag.run(
        "How many credits is CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "check_evidence": lambda s: {"evidence_ok": True},
            "answer": answer,
        },
    )
    assert state["exit"] == "answer"
    assert [block.doc_id for block in seen[0]] == ["CS 374"]
    assert state["signals"]["tool_error"] is None


def test_a_missing_code_ends_the_run_at_the_hand_off_with_its_reason(cfg, corpus):
    state = ag.run(
        "What are the prerequisites?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "prereq_chain"},
            "check_evidence": lambda s: {"evidence_ok": False},
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["docs"] == []
    assert exits.REASONS["no_course_code"][0] in state["answer"]
    assert exits.REASONS["no_course_code"][1] in state["answer"]


def test_the_graph_builds_the_real_dispatcher_for_the_live_config():
    built = ag.build_nodes(load_config())
    assert built["dispatch"].__name__ == "dispatch"


# --- the live corpus ------------------------------------------------------------------------------


def test_the_live_config_carries_the_section_this_node_reads():
    live = load_config()
    assert live["agent"]["retrieval_mode"] in MODES
    assert check_retrieval_mode(live) == live["agent"]["retrieval_mode"]


def test_the_agent_section_renames_no_index():
    # The same acceptance #28 asserted for `llm:`: a section no index is a function of moves the
    # full hash (the reports get new names) and must move none of the three keys an artifact on
    # disk is named after.
    live = load_config()
    without = Config({k: v for k, v in live.to_dict().items() if k != "agent"})
    assert live.hash != without.hash
    assert chunk_key(live) == chunk_key(without)
    assert dense_key(live, chunk_key(live)) == dense_key(without, chunk_key(without))
    assert bm25_key(live, chunk_key(live)) == bm25_key(without, chunk_key(without))


def test_the_live_catalog_has_the_subject_vocabulary_the_filter_leans_on():
    assert len(catalog_subjects()) == 187
    assert {"CS", "IS", "ME"} <= catalog_subjects()
    assert not {"HAVE", "TOP", "ALL", "ONE"} & catalog_subjects()


def test_what_do_i_need_before_cs_374_is_a_chain_not_a_retrieval():
    # #36's first acceptance, on the real corpus.
    cfg = Config({"data": {"term": TERM}, "agent": {"retrieval_mode": "bm25"}})
    fake = FakeRetrieval()
    update = run_row(cfg, "What do I need before CS 374?", "prereq_chain", retrieve=fake)
    (block,) = update["docs"]
    assert block.source == "get_prereq_chain"
    assert fake.calls == []
    assert "Immediately before it: CS 173 — Discrete Structures; CS 225 — Data Structures; " in (
        block.text
    )
    assert "Prerequisite statement: One of CS 173, MATH 213; CS 225." in block.text


def test_which_intro_machine_learning_course_should_i_take_is_a_retrieval_not_a_tool():
    # #36's second acceptance. Which courses come back is Day 1's question and is pinned in
    # tests/test_retrieve.py; this asserts the route, since `indexes/` is gitignored.
    cfg = Config({"data": {"term": TERM}, "agent": {"retrieval_mode": "bm25"}})
    fake = FakeRetrieval()
    query = "Which intro machine learning course should I take?"
    update = run_row(cfg, query, "compare_recommend", retrieve=fake)
    assert fake.calls == [(query, "bm25", cfg)]
    assert {block.source for block in update["docs"]} == {"bm25"}


def test_a_live_course_the_2026_sp_catalog_does_not_offer_refuses_by_name():
    # CS 999 does not exist and CS 428 is not offered this term; both are "not in this catalog",
    # which is the sentence the hand-off gives, and neither is "you named no course".
    cfg = Config({"data": {"term": TERM}, "agent": {"retrieval_mode": "bm25"}})
    for query in ("How many credits is CS 999?", "What do I need before CS 428?"):
        intent = "course_fact" if "credits" in query else "prereq_chain"
        update = run_row(cfg, query, intent)
        assert update["docs"] == []
        assert update["signals"]["tool_error"] == "course_not_found"
