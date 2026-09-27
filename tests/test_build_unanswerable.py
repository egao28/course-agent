"""Tests for eval/scripts/build_unanswerable.py: the table, the checks over it, and the file.

Three groups, and the middle one is the largest. The first holds the table to #57's acceptance --
25 rows, four shapes, empty gold ids, a written refusal, every invented code absent from
``data/chunks.jsonl`` and every real-but-out-of-term course confirmed absent with the corpus's
own evidence that it is real. The second mutates one row at a time and asserts the check that
should catch it does, because a check that has never rejected anything is a comment. The third
covers the run: what is written, that it is idempotent, and that nothing is written when the
table does not check out.

The corpus checks run against the **real** export, not a synthetic one. What makes a row in this
bucket correct is a fact about 4,502 real courses, and a five-course fixture would test the loop
rather than the bucket. The mutation tests build a :class:`~eval.scripts.build_unanswerable.Corpus`
by hand instead, since there the thing under test is the message.
"""

from __future__ import annotations

import json

import pytest

from agent.tools.courses import CourseDetail
from config import bm25_path, load_config, rel_to_repo
from eval.schema import RULES, EvalItem, normalize_question
from eval.scripts import build_unanswerable as bu
from eval.scripts.draft_items import PROVENANCE_KEYS as DRAFTED_KEYS
from eval.scripts.draft_items import check_drafts, parse_drafts
from eval.validate import corpus_doc_ids
from ingest.chunk import CHUNKS_PATH, TEMPLATE_FIELDS, chunk_key, load_chunks

CFG = load_config()

#: The bucket's own file, as the repository carries it.
COMMITTED = bu.draft_path(bu.BUCKET, CFG)

#: Live retrieval needs the BM25 pickle, which is gitignored and rebuildable. The scoring tests
#: skip without it rather than failing on a fresh clone.
HAS_INDEX = bm25_path(CFG, chunk_key(CFG)).exists()


@pytest.fixture(scope="module")
def corpus() -> bu.Corpus:
    return bu.read_corpus()


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """The table as rows, unscored -- the shape checks do not need an index."""
    return bu.build_rows(bu.ITEMS, CFG, scored=False)


def fake_corpus() -> bu.Corpus:
    """Five courses, two of which are named by a description and not offered."""
    return bu.Corpus(
        codes=frozenset({"CS 225", "CS 527", "BADM 205"}),
        named={"CS 428": ("CS 527",), "GGIS 205": ("BADM 205",)},
    )


def errors(found) -> list[str]:
    return [v.message for v in found if v.level == "error"]


# --- the table, against #57's acceptance-----------------------------------------------------------


def test_the_bucket_is_twenty_five_rows_in_four_shapes():
    assert len(bu.ITEMS) == bu.TOTAL == 25
    counts = {name: sum(1 for i in bu.ITEMS if i.shape == name) for name in bu.SHAPES}
    assert counts == {name: rule.n for name, rule in bu.SHAPES.items()}
    assert counts == {"invented": 6, "out_of_term": 7, "field_absent": 6, "not_course": 6}


def test_every_row_records_which_failure_shape_it_is_testing(rows):
    assert [row["provenance"]["shape"] for row in rows] == [item.shape for item in bu.ITEMS]
    assert all(row["provenance"]["shape"] in bu.SHAPES for row in rows)


def test_every_row_abstains_with_no_gold_ids_and_a_written_refusal(rows):
    for row in rows:
        assert row["bucket"] == bu.BUCKET
        assert row["should_abstain"] is True
        assert row["gold_doc_ids"] == []
        assert row["gold_answer"].endswith((".", "?"))
        assert len(row["gold_answer"].split()) >= 5


def test_the_bucket_matches_what_the_schema_says_an_unanswerable_row_is(rows):
    rule = RULES[bu.BUCKET]
    assert rule.target_n == len(rows)
    assert all(row["should_abstain"] is rule.should_abstain for row in rows)
    assert all(len(row["gold_doc_ids"]) <= rule.max_gold_ids for row in rows)


def test_the_ids_come_out_of_the_block_fifty_six_reserved(rows):
    assert [row["id"] for row in rows] == [f"v1-{2000 + n:04d}" for n in range(1, 26)]
    assert len({row["id"] for row in rows}) == 25


def test_no_question_is_asked_twice(rows):
    asked = [normalize_question(row["question"]) for row in rows]
    assert len(set(asked)) == len(asked)


def test_the_table_checks_out_against_the_live_corpus(corpus):
    assert bu.check_table(bu.ITEMS, corpus) == []


def test_every_invented_code_is_absent_from_the_corpus(corpus):
    for item in bu.ITEMS:
        if item.shape != "invented":
            continue
        for code in item.codes:
            assert code not in corpus.codes
            assert code not in corpus.named, f"{code} is named by the catalog, so it is real"


@pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="chunks.jsonl has not been built here")
def test_no_absent_code_reaches_a_chunk():
    """#57 names ``data/chunks.jsonl`` outright: the file the indexes were actually built from."""
    built = {chunk["metadata"]["code"] for chunk in load_chunks(CHUNKS_PATH)}
    for item in bu.ITEMS:
        if bu.SHAPES[item.shape].codes == "present":
            assert all(code in built for code in item.codes)
        else:
            assert all(code not in built for code in item.codes)


def test_every_out_of_term_course_is_real_on_the_corpus_own_evidence(corpus):
    """Confirmed absent, and confirmed real by a course that IS offered naming it."""
    seen = 0
    for item in bu.ITEMS:
        if item.shape != "out_of_term":
            continue
        seen += 1
        for code in item.codes:
            assert code not in corpus.codes
            assert item.named_by in corpus.named[code]
            assert corpus.has(item.named_by)
    assert seen == 7


def test_every_field_absent_row_is_about_a_course_that_is_offered(corpus):
    for item in bu.ITEMS:
        if item.shape != "field_absent":
            continue
        assert item.codes and all(corpus.has(code) for code in item.codes)
        assert item.field not in bu.known_fields()


def test_the_invented_shape_holds_both_kinds_of_invention():
    """#57 asks for invented codes under a real subject AND invented subjects."""
    invented = [item for item in bu.ITEMS if item.shape == "invented"]
    subjects = {code.split()[0] for item in invented for code in item.codes}
    assert "MATH" in subjects and "CS" in subjects, "no real-subject, invented-number row"
    assert any(not item.codes for item in invented), "no invented-subject row"
    assert any("8123" in item.question for item in invented), "no non-three-digit number"


def test_a_refusal_names_every_course_it_is_about():
    """`eval/GUIDELINES.md` rule 1: a claim whose subject is ambiguous cannot be scored."""
    for item in bu.ITEMS:
        for code in item.codes:
            assert code in item.gold_answer


def test_the_spec_worked_cases_are_in_the_set():
    questions = {item.question for item in bu.ITEMS}
    assert "how many credits is CS 428?" in questions  # GUIDELINES U1
    assert "what should I take next semester?" in questions  # GUIDELINES A2
    assert "is there a course about quantum origami?" in questions  # PR #51's negative query


def test_no_row_asks_for_a_section_count():
    """GUIDELINES U2: get_course answers it, so it is a deleted row and not an unanswerable one."""
    assert "section_count" in bu.known_fields()
    assert not any("section" in item.question for item in bu.ITEMS)


# --- what the corpus reader reads------------------------------------------------------------------


def test_the_reader_sees_the_whole_export(corpus):
    assert len(corpus.codes) == 4502
    assert corpus.has("CS 374") and not corpus.has("CS 428")


def test_the_reader_finds_the_courses_the_catalog_names_but_does_not_offer(corpus):
    assert corpus.named["CS 428"] == ("CS 527",)
    assert corpus.named["GGIS 205"] == ("BADM 205",)
    assert "CS 527" in corpus.codes and "CS 428" not in corpus.codes


def test_the_reader_works_off_the_committed_json_not_the_built_chunks(tmp_path):
    """The evidence must be checkable on a fresh clone, where chunks.jsonl does not exist."""
    small = tmp_path / "courses.json"
    small.write_text(
        json.dumps(
            [
                {"code": "CS 527", "desc": "Prerequisite: CS 428."},
                {"code": "CS 225", "desc": "No codes here."},
            ]
        )
    )
    read = bu.read_corpus(small)
    assert read.codes == frozenset({"CS 527", "CS 225"})
    assert read.named == {"CS 428": ("CS 527",)}


def test_known_fields_is_the_template_and_the_tool_together():
    fields = bu.known_fields()
    assert set(TEMPLATE_FIELDS["v1"]) <= fields
    assert set(CourseDetail.model_fields) <= fields
    assert {"prereqRaw", "prereq_raw", "section_count", "attrs"} <= fields
    assert "instructor" not in fields and "textbook" not in fields


# --- one check at a time---------------------------------------------------------------------------


def one(item: bu.Item) -> list[str]:
    """Every error the checks find in a single row, against the hand-built corpus."""
    known = fake_corpus()
    fields = bu.known_fields()
    return [
        *errors(bu.check_codes(1, item, known)),
        *errors(bu.check_named_by(1, item, known)),
        *errors(bu.check_field(1, item, fields)),
        *errors(bu.check_question(1, item)),
        *errors(bu.check_answer(1, item)),
    ]


def test_an_invented_row_whose_course_is_really_offered():
    found = one(
        bu.Item("invented", "how many credits is CS 225?", "No CS 225 here.", codes=("CS 225",))
    )
    assert any("IS in the 2026-sp export" in m for m in found)


def test_an_invented_row_whose_code_the_catalog_talks_about_belongs_in_the_other_shape():
    found = one(
        bu.Item("invented", "how many credits is CS 428?", "No CS 428 here.", codes=("CS 428",))
    )
    assert any("move the row to out_of_term" in m for m in found)


def test_an_out_of_term_row_nothing_vouches_for():
    found = one(
        bu.Item(
            "out_of_term",
            "what is ZZZ 100 about?",
            "No ZZZ 100.",
            codes=("ZZZ 100",),
            named_by="CS 225",
        )
    )
    assert any("cannot vouch for it being a real course" in m for m in found)


def test_an_out_of_term_row_whose_witness_does_not_name_it():
    found = one(
        bu.Item(
            "out_of_term",
            "what is CS 428 about?",
            "No CS 428.",
            codes=("CS 428",),
            named_by="CS 225",
        )
    )
    assert any("does not name CS 428" in m for m in found)


def test_an_out_of_term_row_whose_witness_is_itself_missing():
    found = one(
        bu.Item(
            "out_of_term",
            "what is CS 428 about?",
            "No CS 428.",
            codes=("CS 428",),
            named_by="CS 999",
        )
    )
    assert any("is not in the export either" in m for m in found)


def test_an_out_of_term_row_with_no_witness_at_all():
    found = one(bu.Item("out_of_term", "what is CS 428 about?", "No CS 428.", codes=("CS 428",)))
    assert any("records the course whose description names the missing one" in m for m in found)


def test_a_field_absent_row_about_a_course_that_is_not_there():
    found = one(
        bu.Item(
            "field_absent",
            "who teaches CS 428?",
            "No instructor for CS 428.",
            codes=("CS 428",),
            field="instructor",
        )
    )
    assert any("out_of_term row wearing the wrong shape" in m for m in found)


@pytest.mark.parametrize("field", ["section_count", "desc", "prereqRaw", "attrs", "credits"])
def test_a_field_absent_row_asking_for_something_the_catalog_has(field):
    """``section_count`` is GUIDELINES U2 -- a deleted row, caught here rather than in review."""
    found = one(
        bu.Item(
            "field_absent",
            "who teaches CS 225?",
            "Not for CS 225.",
            codes=("CS 225",),
            field=field,
        )
    )
    assert any("is a field the knowledge base carries" in m for m in found)


def test_a_field_absent_row_that_does_not_say_which_fact_it_wants():
    found = one(
        bu.Item("field_absent", "who teaches CS 225?", "Not for CS 225.", codes=("CS 225",))
    )
    assert any("records WHICH fact it asks for" in m for m in found)


def test_a_not_course_row_that_names_a_course():
    found = one(
        bu.Item(
            "not_course",
            "where do I park for CS 225?",
            "Parking is not here.",
            codes=("CS 225",),
        )
    )
    assert any("not about a course" in m for m in found)


def test_a_field_on_a_shape_that_does_not_take_one():
    item = bu.Item("not_course", "where do I park?", "Parking is not here.", field="parking")
    found = one(item)
    assert any("only a field_absent row carries one" in m for m in found)


def test_a_witness_on_a_shape_that_does_not_take_one():
    item = bu.Item("not_course", "where do I park?", "Parking is not here.", named_by="CS 225")
    found = one(item)
    assert any("only an out_of_term row carries one" in m for m in found)


def test_a_question_that_names_a_course_the_row_did_not_declare():
    found = one(
        bu.Item(
            "invented",
            "how many credits is CS 999 or CS 998?",
            "No CS 999.",
            codes=("CS 999",),
        )
    )
    assert any("which the row does not declare" in m for m in found)


def test_a_question_that_has_lost_the_course_it_declares():
    found = one(bu.Item("invented", "how many credits is it?", "No CS 999.", codes=("CS 999",)))
    assert any("does not name CS 999" in m for m in found)


def test_a_question_that_has_lost_a_code_the_pattern_cannot_see():
    found = one(bu.Item("invented", "how many credits?", "No MATH 8123.", codes=("MATH 8123",)))
    assert any("does not contain 'MATH 8123'" in m for m in found)


def test_a_refusal_that_does_not_name_the_course():
    item = bu.Item("invented", "how many credits is CS 999?", "No such course.", codes=("CS 999",))
    found = one(item)
    assert any("does not name CS 999" in m for m in found)


def test_a_refusal_that_is_not_written():
    found = one(bu.Item("not_course", "where do I park?", "   "))
    assert any("records the refusal that would be correct" in m for m in found)


def test_an_invented_row_may_name_no_course_at_all():
    item = bu.Item("invented", "is there a course about quantum origami?", "There is not.")
    assert one(item) == []


def test_every_other_shape_must_name_one():
    found = one(bu.Item("field_absent", "who teaches it?", "Not recorded.", field="instructor"))
    assert any("every field_absent row names a course" in m for m in found)


def test_an_unknown_shape_names_the_four_that_exist():
    found = bu.check_table([bu.Item("hard", "why?", "Because.")], fake_corpus())
    assert any("the four are" in v.message for v in found)
    with pytest.raises(ValueError, match="unknown shape"):
        bu.shape("hard")


def test_a_short_shape_fails_rather_than_warns():
    """Unlike a bucket count, which :mod:`eval.validate` only warns about: this table is not
    sampled, drafted or verified down -- a shape that is short is an edit that did not finish."""
    found = bu.check_table(bu.ITEMS[:-1], bu.read_corpus())
    assert any("not the 6 the shape is built to hold" in v.message for v in found)
    assert any("24 rows in total, not 25" in v.message for v in found)
    assert all(v.level == "error" for v in found)


def test_the_table_reports_every_problem_and_not_the_first():
    broken = [
        bu.Item("invented", "how many credits is CS 225?", "No CS 225.", codes=("CS 225",)),
        bu.Item("out_of_term", "what is CS 428 about?", "No CS 428.", codes=("CS 428",)),
        bu.Item("field_absent", "who teaches CS 225?", "Not for CS 225.", codes=("CS 225",)),
    ]
    found = bu.check_table(broken, fake_corpus())
    assert len({v.line for v in found if v.line}) == 3
    assert [v.item_id for v in found if v.line] == ["v1-2001", "v1-2002", "v1-2003"]


# --- scoring, which labels nothing-----------------------------------------------------------------


def test_the_gate_comes_from_the_evidence_section_not_from_a_number_here():
    assert bu.gate_for("bm25", CFG) == CFG["evidence"]["thresholds"]["bm25"] == 15.0


def test_a_mode_the_gate_cannot_judge_is_refused_with_the_measurement():
    with pytest.raises(ValueError, match="cannot be gated by top_score"):
        bu.gate_for("hybrid", CFG)


def test_the_mode_defaults_to_the_one_the_agent_runs(monkeypatch):
    seen: list[str] = []

    def note(query, mode, cfg):
        seen.append(mode)
        return bu.Score(1.0, "2026-sp:CS-374")

    monkeypatch.setattr(bu, "score_query", note)
    bu.build_rows(bu.ITEMS[:1], CFG)
    assert set(seen) == {CFG["agent"]["retrieval_mode"]}


def test_scoring_off_leaves_the_keys_there_and_null(rows):
    for row in rows:
        p = row["provenance"]
        assert set(p) == set(bu.PROVENANCE_KEYS)
        assert p["top_score"] is None and p["top_doc_id"] is None
        assert p["code_scores"] is None and p["above_gate"] is None
        assert p["retrieval_mode"] is None, "a row never scored must not claim a mode"


def test_scoring_off_opens_no_index(monkeypatch):
    monkeypatch.setattr(
        bu, "score_query", lambda *a, **k: pytest.fail("--no-retrieval must not query")
    )
    assert bu.build_rows(bu.ITEMS, CFG, scored=False)


def test_above_the_gate_is_the_comparison_and_nothing_else(monkeypatch):
    monkeypatch.setattr(bu, "score_query", lambda q, mode, cfg: bu.Score(15.0, "2026-sp:CS-374"))
    row = bu.build_rows(bu.ITEMS[:1], CFG)[0]
    assert row["provenance"]["above_gate"] is True, "the gate passes a score equal to it"
    monkeypatch.setattr(bu, "score_query", lambda q, mode, cfg: bu.Score(14.9999, "2026-sp:CS-374"))
    assert bu.build_rows(bu.ITEMS[:1], CFG)[0]["provenance"]["above_gate"] is False


def test_a_score_does_not_change_a_row(monkeypatch):
    """GUIDELINES: never run retrieve() to decide a label. The six fields must not depend on it."""
    monkeypatch.setattr(bu, "score_query", lambda q, mode, cfg: bu.Score(99.0, "2026-sp:CS-374"))
    scored = bu.build_rows(bu.ITEMS, CFG)
    plain = bu.build_rows(bu.ITEMS, CFG, scored=False)
    fields = set(EvalItem.model_fields)
    assert [{k: r[k] for k in fields} for r in scored] == [{k: r[k] for k in fields} for r in plain]


# --- the run and the file--------------------------------------------------------------------------


def test_a_run_writes_rows_a_sidecar_and_nothing_that_fails_the_draft_checks(tmp_path):
    report = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    assert report.table == [] and errors(report.written) == []
    assert report.path.name == f"unanswerable_{CFG.hash}.jsonl"
    assert report.meta.exists()
    assert len(report.path.read_text().splitlines()) == 25


def test_the_run_re_reads_what_it_wrote_with_its_own_provenance_keys(tmp_path):
    report = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    parsed, found = parse_drafts(report.path, bu.PROVENANCE_KEYS)
    assert len(parsed) == 25 and found == []
    assert errors(check_drafts(report.path, corpus_doc_ids(), bu.PROVENANCE_KEYS)) == []


def test_a_constructed_row_is_not_a_drafted_one_and_the_reader_says_so(tmp_path):
    """The two provenance shapes are different on purpose; holding one to the other's keys fails."""
    report = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    _, found = parse_drafts(report.path)  # #56's default keys
    assert found and all("provenance" == v.field for v in found)
    assert set(bu.PROVENANCE_KEYS) != set(DRAFTED_KEYS)
    assert "model" not in bu.PROVENANCE_KEYS and "source" not in DRAFTED_KEYS


def test_a_run_is_idempotent(tmp_path):
    first = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    body = first.path.read_bytes(), first.meta.read_bytes()
    second = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    assert (second.path.read_bytes(), second.meta.read_bytes()) == body


def test_nothing_is_written_when_the_table_does_not_check_out(tmp_path):
    extra = bu.Item("invented", "how many credits is CS 225?", "No CS 225.", codes=("CS 225",))
    bad = (*bu.ITEMS, extra)
    report = bu.build(CFG, scored=False, drafts_dir=tmp_path, items=bad)
    assert errors(report.table) and report.rows == []
    assert not report.path.exists() and not report.meta.exists()


def test_the_sidecar_answers_the_two_questions_the_report_asks(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "score_query", lambda q, mode, cfg: bu.Score(20.0, "2026-sp:CS-374"))
    report = bu.build(CFG, drafts_dir=tmp_path)
    meta = json.loads(report.meta.read_text())
    assert meta["shapes"] == {"invented": 6, "out_of_term": 7, "field_absent": 6, "not_course": 6}
    assert meta["rows"] == 25 and meta["gate"] == 15.0 and meta["scored"] is True
    assert len(meta["above_gate"]) == 25, "every row scored 20.0, which is over the gate"
    assert meta["config_hash"] == CFG.hash


def test_the_summary_calls_out_the_rows_the_gate_would_not_refuse(tmp_path, monkeypatch):
    scores = iter([bu.Score(99.0, "2026-sp:CS-374")] + [bu.Score(1.0, "2026-sp:CS-374")] * 200)
    monkeypatch.setattr(bu, "score_query", lambda q, mode, cfg: next(scores))
    report = bu.build(CFG, drafts_dir=tmp_path)
    text = bu.summary(CFG, report)
    assert "above the gate 1 of 25" in text
    assert "v1-2001  invented" in text
    assert "a Day 4 refusal here is the model's and not the evidence check's" in text


def test_the_summary_says_when_nothing_was_scored(tmp_path):
    report = bu.build(CFG, scored=False, drafts_dir=tmp_path)
    assert "not run, so no row has a score" in bu.summary(CFG, report)


def test_main_exits_zero_and_prints_where_it_wrote(tmp_path, capsys):
    assert bu.main(["--no-retrieval", "--drafts-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"config hash {CFG.hash}" in out and "rows                 25" in out


def test_main_exits_non_zero_when_the_table_is_wrong(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bu, "ITEMS", bu.ITEMS[:3])
    assert bu.main(["--no-retrieval", "--drafts-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "does not check out against the corpus" in captured.err
    assert "nothing was written" in captured.out
    assert list(tmp_path.iterdir()) == []


def test_main_refuses_a_mode_the_gate_cannot_judge(tmp_path):
    with pytest.raises(ValueError, match="cannot be gated by top_score"):
        bu.main(["--mode", "hybrid", "--drafts-dir", str(tmp_path)])


def test_the_mode_is_refused_even_on_the_run_that_would_not_use_it(tmp_path):
    """Accepting --mode hybrid under --no-retrieval would only move the failure to the next run."""
    with pytest.raises(ValueError, match="cannot be gated by top_score"):
        bu.main(["--mode", "hybrid", "--no-retrieval", "--drafts-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


# --- the committed bucket--------------------------------------------------------------------------


def test_the_committed_file_is_in_the_tree():
    assert COMMITTED.exists(), (
        f"{rel_to_repo(COMMITTED)} is #57's deliverable; rebuild it with "
        "`uv run python -m eval.scripts.build_unanswerable`"
    )


def test_the_committed_file_is_what_the_table_says(rows):
    """An edit to ITEMS without a re-run fails here, which is the point of committing the file."""
    committed = [json.loads(line) for line in COMMITTED.read_text().splitlines()]
    fields = set(EvalItem.model_fields)
    assert [{k: r[k] for k in fields} for r in committed] == [
        {k: r[k] for k in fields} for r in rows
    ]
    for written, built in zip(committed, rows, strict=True):
        for key in ("shape", "codes", "named_by", "field", "note", "source"):
            assert written["provenance"][key] == built["provenance"][key]


def test_the_committed_file_was_built_under_this_config():
    committed = [json.loads(line) for line in COMMITTED.read_text().splitlines()]
    assert {row["provenance"]["config_hash"] for row in committed} == {CFG.hash}
    assert json.loads(bu.meta_path(COMMITTED).read_text())["config_hash"] == CFG.hash


def test_the_committed_file_passes_its_own_checks():
    assert errors(check_drafts(COMMITTED, corpus_doc_ids(), bu.PROVENANCE_KEYS)) == []


@pytest.mark.skipif(not HAS_INDEX, reason="the BM25 index has not been built here")
def test_the_committed_scores_reproduce():
    """The scores are a function of the committed table and the index the config names."""
    committed = [json.loads(line) for line in COMMITTED.read_text().splitlines()]
    fresh = bu.build_rows(bu.ITEMS, CFG)
    for written, built in zip(committed, fresh, strict=True):
        assert written["provenance"]["top_score"] == built["provenance"]["top_score"]
        assert written["provenance"]["top_doc_id"] == built["provenance"]["top_doc_id"]
        assert written["provenance"]["code_scores"] == built["provenance"]["code_scores"]


@pytest.mark.skipif(not HAS_INDEX, reason="the BM25 index has not been built here")
def test_the_gate_refuses_a_bare_invented_code_and_passes_the_question_it_sits_in():
    """The finding this file exists to record, pinned so a tokenizer change moves it visibly."""
    committed = {
        row["id"]: row["provenance"] for row in map(json.loads, COMMITTED.read_text().splitlines())
    }
    invented = [p for p in committed.values() if p["shape"] == "invented" and p["codes"]]
    assert all(score < p["gate"] for p in invented for score in p["code_scores"].values()), (
        "a bare invented code is refused by the gate"
    )
    assert sum(1 for p in invented if p["above_gate"]) == 3, (
        "three of the five invented-code rows clear the gate once the code is in a sentence"
    )
    assert all(p["above_gate"] for p in committed.values() if p["shape"] == "field_absent"), (
        "a field_absent row names a course that IS in the corpus, so the gate cannot refuse it"
    )
    assert all(p["above_gate"] for p in committed.values() if p["shape"] == "out_of_term"), (
        "the catalog talks about these courses, so the gate cannot refuse them either"
    )
    assert sum(1 for p in committed.values() if p["above_gate"]) == 17
