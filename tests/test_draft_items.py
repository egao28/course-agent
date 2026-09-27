"""Tests for eval/scripts/draft_items.py: the plan, the prompt, the drops, and the resume.

No model is called anywhere here, and no test needs one: every batch goes through a client that
is three lines long, which is the whole point of :func:`eval.scripts.draft_items.make_drafter`
taking one. What is asserted is the machinery around the words -- that the batch plan is a
function of the corpus and the seed alone, that a row citing an id it was not shown is dropped
rather than repaired, that a crashed run resumes without re-rolling the sample or re-paying for
a batch, and that the file on disk validates against :mod:`eval.schema` when the run is over.

Most tests run over a 40-record synthetic corpus (eight subjects, five courses each, four
batches of ten), because what they check is the dealing and not the catalog. The handful that
are *about* the catalog -- that the example ids in the prompt are not real ids, that 4,502
records make 450 batches -- say so and use the real one.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from config import Config
from eval.schema import RULES, normalize_question
from eval.scripts import draft_items as draft
from eval.scripts.draft_items import (
    BATCH_SIZE,
    DRAFT_RULES,
    DRAFTABLE,
    EXAMPLE_BATCHES,
    EXAMPLE_RECORDS,
    EXAMPLE_REPLIES,
    ID_BLOCKS,
    PROVENANCE_KEYS,
    Dropped,
    Kept,
    Settings,
    StandInDrafter,
    build_row,
    check_drafts,
    credit_question,
    draft_path,
    draft_user_prompt,
    flag_top_decile,
    jaccard,
    main,
    make_drafter,
    meta_path,
    parse_completion,
    parse_drafts,
    plan_batches,
    quantile,
    question_overlap,
    render_draft_prompt,
    stratified_order,
)
from ingest.chunk import load_chunks
from ingest.tokenize import tokenize
from llm import OpenAIChat, StandInChat

#: A synthetic config. The values that matter here are ``seed`` (the plan), ``bm25.tokenizer``
#: (the overlap measurement) and the ``llm`` section (which client :func:`make_drafter` builds);
#: ``chunk`` and ``data`` are there because the sidecar records ``chunk_key``.
CFG = Config(
    {
        "seed": 42,
        "data": {"term": "2026-sp"},
        "chunk": {"strategy": "one_course_one_chunk", "template_version": "v1"},
        "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "simple_v1", "code_boost": 2},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800},
    }
)

SUBJECTS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ")
TOKENIZER = tokenize


def make_chunk(subject: str, number: int, credits: int = 3, desc: str | None = None) -> dict:
    """One record in the shape ``ingest.chunk`` renders, with only the fields this module reads."""
    code = f"{subject} {number}"
    text = (
        f"{code} — {subject} Course {number}\n"
        f"Credits: {credits}\n"
        "Term: 2026-sp\n"
        f"Description: {desc or f'A course about {subject.lower()} things, numbered {number}.'}\n"
        "Prerequisites: None listed"
    )
    return {
        "doc_id": f"2026-sp:{subject}-{number}",
        "text": text,
        "metadata": {
            "code": code,
            "subject": subject,
            "number": number,
            "level": number // 100 * 100,
            "credits": float(credits),
            "term": "2026-sp",
        },
    }


#: Ten subjects, four courses each: 40 records, four batches of ten, one subject per slot.
CORPUS = [make_chunk(subject, 100 + 10 * n) for subject in SUBJECTS for n in range(4)]
KNOWN = frozenset(chunk["doc_id"] for chunk in CORPUS)
FIRST = plan_batches(CORPUS, CFG["seed"])[0]
SETTINGS = Settings(bucket="fact", per_batch=3, model="fake-model", config_hash=CFG.hash, seed=42)


class ScriptedChat:
    """A client that replies from a list, in order. An exception in it is raised instead."""

    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


class EchoChat:
    """Drafts one usable row per batch, about that batch's first record.

    Deliberately batch-dependent: a client that replied with the same question every time would
    make every run after the first look like a duplicate-drop, and the resume tests would pass
    for the wrong reason.
    """

    model = "echo-model"

    def __init__(self, ids_per_row: int = 1):
        self.ids_per_row = ids_per_row
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        ids = re.findall(r"\(doc_id: ([^)]+)\)", prompt)[: self.ids_per_row]
        return json.dumps(
            [
                {
                    "question": f"what should i know about {ids[0].split(':')[1].lower()}?",
                    "answer": f"{ids[0]} is in the catalog.",
                    "doc_ids": ids,
                }
            ]
        )


def reply(*rows) -> str:
    """A model reply carrying ``rows``, each ``(question, answer, doc_ids)``."""
    return json.dumps(
        [{"question": q, "answer": a, "doc_ids": list(ids)} for q, a, ids in rows]
    )


def run(chat, bucket="fact", batches=1, tmp_path=None, per_batch=3, fake=False):
    return draft.draft(
        CFG,
        bucket,
        chat=chat,
        batches=batches,
        per_batch=per_batch,
        chunks=CORPUS,
        drafts_dir=tmp_path,
        fake=fake,
    )


# --- the plan -------------------------------------------------------------------------------------


def test_the_same_seed_deals_the_same_batches():
    assert [b.doc_ids for b in plan_batches(CORPUS, 42)] == [
        b.doc_ids for b in plan_batches(CORPUS, 42)
    ]


def test_a_module_global_shuffle_cannot_shift_the_stream():
    # The whole reason the sampler holds its own random.Random: a second script in the same
    # process (or a library at import time) reseeding the global one must not move the eval set.
    import random

    first = [b.doc_ids for b in plan_batches(CORPUS, 42)]
    random.seed(0)
    random.random()
    assert [b.doc_ids for b in plan_batches(CORPUS, 42)] == first


def test_a_different_seed_deals_different_batches():
    assert [b.doc_ids for b in plan_batches(CORPUS, 7)] != [
        b.doc_ids for b in plan_batches(CORPUS, 42)
    ]


def test_asking_for_more_batches_extends_the_same_plan():
    # What makes the resume free: batch 3 is the same ten records whether 3 or 30 were asked for.
    plan = plan_batches(CORPUS, 42)
    assert [b.doc_ids for b in plan[:2]] == [b.doc_ids for b in plan_batches(CORPUS, 42)[:2]]


def test_the_order_holds_every_record_exactly_once():
    order = stratified_order(CORPUS, 42)
    assert len(order) == len(CORPUS)
    assert {chunk["doc_id"] for chunk in order} == KNOWN


def test_a_batch_is_ten_different_subjects():
    # The point of stratifying: ten consecutive records are never two from one department.
    for batch in plan_batches(CORPUS, 42):
        subjects = [chunk["metadata"]["subject"] for chunk in batch.chunks]
        assert len(set(subjects)) == len(subjects) == BATCH_SIZE


def test_the_order_does_not_depend_on_the_file_order():
    # Subjects are sorted before they are shuffled, so re-exporting the corpus in another order
    # is not a new eval set.
    shuffled = list(reversed(CORPUS))
    assert [c["doc_id"] for c in stratified_order(shuffled, 42)] == [
        c["doc_id"] for c in stratified_order(CORPUS, 42)
    ]


def test_batches_are_numbered_from_one():
    assert [batch.n for batch in plan_batches(CORPUS, 42)] == [1, 2, 3, 4]


def test_the_short_tail_is_dropped():
    # 4,502 records make 450 batches of ten and leave two over; a batch of two is not a batch the
    # prompt's wording describes.
    plan = plan_batches(load_chunks(), 42)
    assert len(plan) == 450
    assert all(len(batch.chunks) == BATCH_SIZE for batch in plan)


def test_every_subject_is_reachable_early():
    # Round-robin, not proportional: over the first four batches of the real corpus the largest
    # department gets no more slots than the smallest.
    plan = plan_batches(load_chunks(), 42)[:4]
    subjects = [c["metadata"]["subject"] for batch in plan for c in batch.chunks]
    assert max(subjects.count(s) for s in set(subjects)) == 1


# --- the prompt -----------------------------------------------------------------------------------

#: Bumped together with ``DRAFT_PROMPT_VERSION``. Editing a rule, a task or an example without
#: bumping the version fails here rather than quietly producing a second "v1" -- the arrangement
#: ``tests/test_prompts.py`` has for the agent's two prompts, for the same reason: provenance
#: records the version per row, so two prompts under one version is two sets in one file.
PROMPT_DIGESTS = {
    "fact": "6e27c0603b4826bb",
    "multihop": "53a823149cbda254",
    "ambiguous": "0a083d6365174fc0",
}


@pytest.mark.parametrize("bucket", DRAFTABLE)
def test_the_prompt_is_pinned(bucket):
    digest = hashlib.sha256(render_draft_prompt(bucket).encode("utf-8")).hexdigest()[:16]
    assert digest == PROMPT_DIGESTS[bucket], (
        f"the {bucket} prompt changed; bump DRAFT_PROMPT_VERSION and this digest together"
    )


@pytest.mark.parametrize("bucket", DRAFTABLE)
def test_every_bucket_gets_every_rule(bucket):
    prompt = render_draft_prompt(bucket)
    assert all(rule in prompt for rule in DRAFT_RULES)


@pytest.mark.parametrize("bucket", DRAFTABLE)
def test_the_example_reply_obeys_its_own_bucket(bucket):
    # An example that cites an id its example batch does not show teaches rule 4 by breaking it.
    rows = json.loads(EXAMPLE_REPLIES[bucket])
    shown = set(EXAMPLE_BATCHES[bucket])
    assert rows and all(set(row["doc_ids"]) <= shown for row in rows)
    if bucket == "multihop":
        assert all(len(row["doc_ids"]) >= RULES["multihop"].min_gold_ids for row in rows)
    if bucket == "ambiguous":
        assert all(row["answer"].rstrip().endswith("?") for row in rows)


def test_the_example_ids_are_not_real_ids():
    # SUBJ and OTHR are not among the catalog's 187 subjects, so an example id echoed into a real
    # draft is dropped by the batch check instead of read as a citation.
    known = {chunk["doc_id"] for chunk in load_chunks()}
    assert not set(EXAMPLE_RECORDS) & known


def test_the_unanswerable_bucket_has_no_prompt():
    with pytest.raises(ValueError, match="constructed"):
        render_draft_prompt("unanswerable")


def test_the_user_prompt_shows_every_record_with_its_id():
    prompt = draft_user_prompt(FIRST.chunks, "fact", 3)
    assert all(f"(doc_id: {one})" in prompt for one in FIRST.doc_ids)
    assert "at most 3 fact question(s)" in prompt


def test_the_answer_stand_in_cannot_answer_a_drafting_prompt():
    # llm.StandInChat exists so a Day 2 report can be produced with no credits. A drafting run
    # that could quietly use it would write canned questions into a file the whole of Day 3
    # trusts, so the two prompts keep their doc_ids in shapes that do not match.
    with pytest.raises(ValueError, match="nothing to answer with"):
        StandInChat().complete(draft_user_prompt(FIRST.chunks, "fact", 3))


# --- reading a reply ------------------------------------------------------------------------------


def test_a_plain_array_parses():
    assert parse_completion('[{"question": "q"}]') == [{"question": "q"}]


def test_a_fenced_array_is_unwrapped():
    # Rule 8 forbids the fence and models produce it anyway; throwing the batch away for that
    # costs ten records and a call to make a point at the model.
    assert parse_completion('```json\n[{"question": "q"}]\n```') == [{"question": "q"}]


def test_a_reply_that_is_not_json_says_what_arrived():
    with pytest.raises(ValueError, match="Here are three questions"):
        parse_completion("Here are three questions about these courses!")


def test_a_reply_that_is_not_an_array_is_refused():
    with pytest.raises(TypeError, match="not the array"):
        parse_completion('{"question": "q"}')


# --- one drafted question into a row --------------------------------------------------------------


def build(
    bucket="fact",
    question="how long is aaa 100?",
    answer="AAA 100 is 3 hours.",
    ids=None,
    n=1,
    seen=None,
):
    raw = {
        "question": question,
        "answer": answer,
        "doc_ids": list(FIRST.doc_ids[:1] if ids is None else ids),
    }
    settings = SETTINGS._replace(bucket=bucket)
    return build_row(raw, FIRST, settings, n, KNOWN, {} if seen is None else seen, TOKENIZER)


def test_a_good_row_is_kept_with_its_whole_provenance():
    row = build()
    assert isinstance(row, Kept)
    assert row.item.id == "v1-0001"
    assert set(row.provenance) == set(PROVENANCE_KEYS)
    assert row.provenance["batch_doc_ids"] == list(FIRST.doc_ids)
    assert row.provenance["cited_doc_ids"] == [FIRST.doc_ids[0]]
    assert row.provenance["model"] == "fake-model"
    assert row.provenance["config_hash"] == CFG.hash
    assert row.provenance["seed"] == 42
    assert row.provenance["prompt_version"] == draft.DRAFT_PROMPT_VERSION
    assert row.provenance["flagged"] is False


def test_the_bucket_fixes_should_abstain():
    # The field is redundant with the bucket by design, so the pipeline must never be the place
    # the two disagree: a model is not asked for it at all.
    assert build().item.should_abstain is False
    assert build(bucket="ambiguous", ids=FIRST.doc_ids[:2]).item.should_abstain is True


def test_each_bucket_mints_ids_in_its_own_block():
    # Three draft files are written independently and merged by #59; ids that collide there would
    # be a corruption that still loads.
    assert build(n=1).item.id == "v1-0001"
    assert build(bucket="multihop", ids=FIRST.doc_ids[:2], n=1).item.id == "v1-1001"
    assert build(bucket="ambiguous", ids=FIRST.doc_ids[:2], n=1).item.id == "v1-3001"
    assert len(set(ID_BLOCKS.values())) == len(ID_BLOCKS)


def test_an_id_the_model_invented_is_dropped():
    row = build(ids=["2026-sp:ZZZ-999"])
    assert isinstance(row, Dropped) and row.reason == "dangling"
    assert "not in the corpus" in row.detail


def test_a_real_id_that_was_not_in_the_batch_is_dropped_too():
    # The stricter half of the check, and the one the prompt is actually about: the ten ids were
    # in front of the model, so citing an eleventh is not a lookup failure.
    outside = next(one for one in KNOWN if one not in FIRST.ids)
    row = build(ids=[outside])
    assert isinstance(row, Dropped) and row.reason == "outside-batch"


@pytest.mark.parametrize(
    "raw",
    [
        {"question": "q", "answer": "a"},
        {"question": "q", "answer": "a", "doc_ids": "2026-sp:AAA-100"},
        ["not", "an", "object"],
    ],
)
def test_a_malformed_element_is_dropped_for_shape(raw):
    row = build_row(raw, FIRST, SETTINGS, 1, KNOWN, {}, TOKENIZER)
    assert isinstance(row, Dropped) and row.reason == "shape"


def test_a_blank_question_is_dropped_for_shape():
    row = build(question="   ")
    assert isinstance(row, Dropped) and row.reason == "shape"


def test_a_fact_row_citing_nothing_is_dropped():
    row = build(ids=[])
    assert isinstance(row, Dropped) and row.reason == "bucket"


def test_a_one_id_multihop_row_is_dropped():
    row = build(bucket="multihop", ids=FIRST.doc_ids[:1])
    assert isinstance(row, Dropped) and row.reason == "bucket"


def test_a_multihop_row_citing_one_record_twice_is_dropped():
    # One hop wearing two ids -- the failure that makes the whole bucket meaningless.
    row = build(bucket="multihop", ids=[FIRST.doc_ids[0], FIRST.doc_ids[0]])
    assert isinstance(row, Dropped) and row.reason == "bucket"


def test_an_ambiguous_row_with_one_candidate_is_dropped():
    # If only one record could be meant, the question has a referent and is a fact question.
    row = build(bucket="ambiguous", ids=FIRST.doc_ids[:1])
    assert isinstance(row, Dropped) and row.reason == "bucket"


def test_an_ambiguous_row_keeps_its_candidates_in_provenance():
    # `eval/GUIDELINES.md`'s two ambiguous cases carry no gold ids -- the answer is a clarifying
    # question, not a claim traceable to a record -- but #59 still needs to see what the question
    # was ambiguous between.
    row = build(bucket="ambiguous", ids=FIRST.doc_ids[:2], answer="Which course do you mean?")
    assert isinstance(row, Kept)
    assert row.item.gold_doc_ids == []
    assert row.provenance["cited_doc_ids"] == list(FIRST.doc_ids[:2])
    assert row.provenance["overlap"] is None


def test_the_same_question_twice_is_one_row():
    question = "how long is aaa 100?"
    seen = {normalize_question(question): "v1-0007"}
    row = build(question=question.upper(), seen=seen)
    assert isinstance(row, Dropped) and row.reason == "duplicate"
    assert "v1-0007" in row.detail


def test_provenance_records_the_subject_and_level_of_every_cited_record():
    # Day 4 slices on these: a set that is all 100-level courses is a different measurement.
    row = build(bucket="multihop", ids=FIRST.doc_ids[:2])
    assert len(row.provenance["subjects"]) == 2
    assert all(level % 100 == 0 for level in row.provenance["levels"])


# --- the leak, measured ---------------------------------------------------------------------------


def test_jaccard_is_zero_when_either_side_is_empty():
    assert jaccard([], ["a"]) == 0.0
    assert jaccard(["a"], []) == 0.0


def test_jaccard_is_one_for_the_same_terms():
    assert jaccard(["a", "b"], ["b", "a", "a"]) == 1.0


def test_a_question_copied_out_of_its_record_overlaps_more_than_one_a_student_would_type():
    # The hazard this measurement exists for, on one record: both questions are answerable from
    # it, and only one of them hands BM25 the answer.
    chunk = make_chunk("AAA", 100, desc="Stochastic calculus for jump diffusion processes.")
    copied = "stochastic calculus for jump diffusion processes"
    typed = "which course covers random processes?"
    assert question_overlap(copied, [chunk], TOKENIZER) > question_overlap(
        typed, [chunk], TOKENIZER
    )


def test_the_overlap_of_a_row_is_its_worst_record_not_its_average():
    # A second gold record that shares nothing does not make a leaky question less leaky.
    hot = make_chunk("AAA", 100, desc="Jump diffusion processes and stochastic calculus.")
    cold = make_chunk("BBB", 200, desc="Beekeeping, honey extraction and apiary management.")
    question = "jump diffusion processes"
    assert question_overlap(question, [hot, cold], TOKENIZER) == question_overlap(
        question, [hot], TOKENIZER
    )


def test_a_row_with_no_gold_record_is_not_measured():
    assert question_overlap("anything", [], TOKENIZER) is None


def test_the_quantile_is_a_value_some_row_actually_has():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert quantile(values, 0.9) == 0.9
    assert quantile(values, 0.5) == 0.5
    assert quantile([0.42], 0.9) == 0.42


def test_the_quantile_does_not_depend_on_the_order_given():
    assert quantile([0.3, 0.1, 0.2], 0.5) == quantile([0.1, 0.2, 0.3], 0.5)


def test_the_top_decile_is_flagged_and_ties_go_with_it(tmp_path):
    chat = EchoChat()
    report = run(chat, batches=4, tmp_path=tmp_path)
    assert report.threshold is not None
    rows, _ = parse_drafts(report.path)
    flagged = [row for row in rows if row.provenance["flagged"]]
    assert flagged, "some row is always in the top decile of a non-empty distribution"
    assert all(row.provenance["overlap"] >= report.threshold for row in flagged)
    assert all(
        row.provenance["flagged"] or row.provenance["overlap"] < report.threshold for row in rows
    )


def test_the_flag_moves_as_the_file_grows(tmp_path):
    # Deliberate: the decile only exists relative to a set, and #59 rewrites whatever is in the
    # top decile of the file it is handed.
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"v1-000{n}",
                    "bucket": "fact",
                    "question": f"question {n}",
                    "gold_answer": "an answer",
                    "gold_doc_ids": [FIRST.doc_ids[0]],
                    "should_abstain": False,
                    "provenance": {key: None for key in PROVENANCE_KEYS} | {"overlap": n / 10},
                }
            )
            for n in range(1, 6)
        )
        + "\n",
        encoding="utf-8",
    )
    threshold, flagged = flag_top_decile(path)
    assert (threshold, flagged) == (0.5, 1)


def test_a_credit_question_is_marked_not_dropped():
    # The 1,415 variable-credit courses cannot be identified on this branch (#17 is on main), so
    # the harness marks and #59 checks -- a drop rule that cannot see its data deletes good rows.
    assert credit_question("how many credits is aaa 100?", TOKENIZER)
    assert credit_question("how many hours is it?", TOKENIZER)
    assert not credit_question("what are the prerequisites for aaa 100?", TOKENIZER)


# --- a run ----------------------------------------------------------------------------------------


def test_a_run_writes_rows_a_sidecar_and_a_file_that_validates(tmp_path):
    report = run(EchoChat(), batches=2, tmp_path=tmp_path)
    assert report.kept == report.rows == 2
    assert report.path.exists() and report.sidecar.exists()
    assert [v for v in report.violations if v.level == "error"] == []
    assert check_drafts(report.path, KNOWN) == []


def test_every_written_row_carries_provenance(tmp_path):
    report = run(EchoChat(), batches=2, tmp_path=tmp_path)
    rows, violations = parse_drafts(report.path)
    assert violations == []
    assert all(set(row.provenance) == set(PROVENANCE_KEYS) for row in rows)
    assert [row.provenance["batch"] for row in rows] == [1, 2]


def test_the_sidecar_records_what_each_batch_did(tmp_path):
    report = run(EchoChat(), batches=2, tmp_path=tmp_path)
    meta = json.loads(report.sidecar.read_text(encoding="utf-8"))
    assert [record["batch"] for record in meta["batches"]] == [1, 2]
    assert meta["batches"][0]["kept"] == ["v1-0001"]
    assert meta["seed"] == 42 and meta["batch_size"] == BATCH_SIZE
    assert meta["corpus"]["chunks"] == len(CORPUS)


def test_a_resume_does_not_re_draft_a_completed_batch(tmp_path):
    first = run(EchoChat(), batches=2, tmp_path=tmp_path)
    before = [(row.item.id, row.item.question) for row in parse_drafts(first.path)[0]]

    chat = EchoChat()
    second = run(chat, batches=4, tmp_path=tmp_path)
    assert second.skipped == [1, 2]
    assert chat.calls == 2, "only the two new batches were paid for"
    after = [(row.item.id, row.item.question) for row in parse_drafts(second.path)[0]]
    assert after[:2] == before
    assert [row.item.id for row in parse_drafts(second.path)[0]] == [
        "v1-0001",
        "v1-0002",
        "v1-0003",
        "v1-0004",
    ]


def test_a_batch_that_kept_nothing_is_still_done(tmp_path):
    # The reason the sidecar exists at all: a file of rows cannot say that batch 1 ran and
    # produced nothing, so a resume would pay for it again.
    bad = reply(("who teaches it?", "Nobody knows.", ["2026-sp:ZZZ-999"]))
    first = run(ScriptedChat(bad), batches=1, tmp_path=tmp_path)
    assert first.kept == 0 and first.drops == {"dangling": 1}

    chat = EchoChat()
    second = run(chat, batches=1, tmp_path=tmp_path)
    assert second.skipped == [1] and chat.calls == 0


def test_a_failed_batch_is_not_marked_done(tmp_path):
    first = run(ScriptedChat(RuntimeError("429 insufficient_quota")), batches=1, tmp_path=tmp_path)
    assert len(first.failed) == 1
    assert "insufficient_quota" in first.failed[0].error

    chat = EchoChat()
    second = run(chat, batches=1, tmp_path=tmp_path)
    assert second.skipped == [] and chat.calls == 1 and second.kept == 1


def test_a_failed_batch_does_not_lose_the_rows_before_it(tmp_path):
    good = reply(("what is aaa about?", "AAA 100 is a course.", [FIRST.doc_ids[0]]))
    report = run(ScriptedChat(good, RuntimeError("boom")), batches=2, tmp_path=tmp_path)
    assert report.rows == 1 and len(report.failed) == 1
    assert json.loads(report.sidecar.read_text(encoding="utf-8"))["rows"] == 1


def test_questions_past_the_quota_are_ignored_rather_than_dropped(tmp_path):
    five = reply(
        *[
            (f"question number {n} about aaa?", f"Answer {n}.", [FIRST.doc_ids[0]])
            for n in range(5)
        ]
    )
    report = run(ScriptedChat(five), batches=1, per_batch=2, tmp_path=tmp_path)
    assert report.kept == 2
    assert report.results[0].ignored == 3
    assert report.drops == {}


def test_the_same_question_in_two_batches_is_dropped_the_second_time(tmp_path):
    same = reply(("how long is it?", "AAA 100 is 3 hours.", [FIRST.doc_ids[0]]))
    second = plan_batches(CORPUS, CFG["seed"])[1]
    again = reply(("How long is it?", "BBB 100 is 3 hours.", [second.doc_ids[0]]))
    report = run(ScriptedChat(same, again), batches=2, tmp_path=tmp_path)
    assert report.rows == 1
    assert report.drops == {"duplicate": 1}


def test_two_buckets_in_one_directory_do_not_collide(tmp_path):
    facts = run(EchoChat(), bucket="fact", batches=2, tmp_path=tmp_path)
    hops = run(EchoChat(ids_per_row=2), bucket="multihop", batches=2, tmp_path=tmp_path)
    assert facts.path != hops.path
    fact_ids = [row.item.id for row in parse_drafts(facts.path)[0]]
    hop_ids = [row.item.id for row in parse_drafts(hops.path)[0]]
    assert not set(fact_ids) & set(hop_ids)
    assert hop_ids == ["v1-1001", "v1-1002"]


def test_the_stand_in_writes_somewhere_else_entirely(tmp_path):
    # A canned row in fact_{hash}.jsonl would be indistinguishable from a drafted one by the time
    # anybody looked.
    real = draft_path("fact", CFG, tmp_path)
    fake = draft_path("fact", CFG, tmp_path, fake=True)
    assert real.name.endswith(".jsonl") and fake.name.endswith(".standin.jsonl")
    report = run(StandInDrafter("fact"), batches=1, tmp_path=tmp_path, fake=True)
    assert report.path == fake and not real.exists()
    assert all(row.provenance["model"] == "stand-in" for row in parse_drafts(report.path)[0])


def test_the_sidecar_sits_beside_the_rows():
    assert meta_path(draft_path("fact", CFG)).name.endswith(".meta.json")


def test_a_run_with_a_different_seed_is_refused(tmp_path):
    report = run(EchoChat(), batches=1, tmp_path=tmp_path)
    meta = json.loads(report.sidecar.read_text(encoding="utf-8"))
    report.sidecar.write_text(json.dumps({**meta, "seed": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the same set"):
        run(EchoChat(), batches=1, tmp_path=tmp_path)


def test_rows_without_a_sidecar_are_refused(tmp_path):
    report = run(EchoChat(), batches=1, tmp_path=tmp_path)
    report.sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="which batches have run cannot be known"):
        run(EchoChat(), batches=1, tmp_path=tmp_path)


def test_a_sidecar_without_its_rows_is_refused(tmp_path):
    report = run(EchoChat(), batches=1, tmp_path=tmp_path)
    report.path.unlink()
    with pytest.raises(FileNotFoundError, match="restore it"):
        run(EchoChat(), batches=1, tmp_path=tmp_path)


def test_more_batches_than_the_corpus_holds_is_refused(tmp_path):
    with pytest.raises(ValueError, match="40 records make 4"):
        run(EchoChat(), batches=99, tmp_path=tmp_path)


def test_the_unanswerable_bucket_cannot_be_drafted(tmp_path):
    with pytest.raises(ValueError, match="constructed instead"):
        run(EchoChat(), bucket="unanswerable", batches=1, tmp_path=tmp_path)


# --- reading a draft file back --------------------------------------------------------------------


def write_draft(tmp_path, **changes):
    row = {
        "id": "v1-0001",
        "bucket": "fact",
        "question": "how long is aaa 100?",
        "gold_answer": "AAA 100 is 3 hours.",
        "gold_doc_ids": [FIRST.doc_ids[0]],
        "should_abstain": False,
        "provenance": {key: None for key in PROVENANCE_KEYS},
    }
    path = tmp_path / "drafts.jsonl"
    path.write_text(json.dumps({**row, **changes}) + "\n", encoding="utf-8")
    return path


def test_a_row_without_provenance_is_an_error(tmp_path):
    path = write_draft(tmp_path)
    path.write_text(
        json.dumps({k: v for k, v in json.loads(path.read_text()).items() if k != "provenance"})
        + "\n",
        encoding="utf-8",
    )
    assert any("is missing" in v.message for v in check_drafts(path, KNOWN))


def test_a_short_provenance_is_an_error(tmp_path):
    path = write_draft(tmp_path, provenance={"batch": 1})
    assert any("is missing" in v.message for v in check_drafts(path, KNOWN))


def test_a_provenance_key_nothing_reads_is_an_error(tmp_path):
    provenance = {key: None for key in PROVENANCE_KEYS} | {"vibes": "good"}
    path = write_draft(tmp_path, provenance=provenance)
    assert any("'vibes'" in v.message for v in check_drafts(path, KNOWN))


def test_the_six_dataset_fields_are_checked_too(tmp_path):
    path = write_draft(tmp_path, gold_doc_ids=["2026-sp:AAA 100"])
    messages = [v.message for v in check_drafts(path, KNOWN)]
    assert any("is not in the corpus" in message for message in messages)


def test_a_drafts_file_is_not_measured_against_the_120_row_plan(tmp_path):
    # It is one bucket, mid-run, and short by construction: a warning saying so on every run is
    # a warning nobody reads.
    path = write_draft(tmp_path)
    assert [v for v in check_drafts(path, KNOWN) if v.level == "warning"] == []


# --- which client a run drafts with ---------------------------------------------------------------


def test_the_configured_model_is_the_default():
    chat = make_drafter(CFG, "fact", client=object())
    assert isinstance(chat, OpenAIChat)
    assert (chat.model, chat.temperature, chat.max_tokens) == ("gpt-4o-mini", 0.0, 800)


def test_another_model_costs_the_repo_no_new_config_hash():
    # The one mitigation there is for drafter and agent under test being the same model, and it
    # must not rename every artifact in the repo to try it.
    chat = make_drafter(CFG, "fact", model="a-different-model", client=object())
    assert chat.model == "a-different-model"
    assert CFG.hash == Config(CFG.to_dict()).hash


def test_the_fake_route_is_the_stand_in():
    chat = make_drafter(CFG, "fact", fake=True)
    assert isinstance(chat, StandInDrafter) and chat.model == "stand-in"


def test_a_bad_llm_section_is_a_sentence():
    with pytest.raises(ValueError, match="llm.provider"):
        make_drafter(Config({"llm": {**CFG["llm"], "provider": "anthropic"}}), "fact")


def test_the_openai_route_sends_the_system_prompt_and_the_records(tmp_path):
    from types import SimpleNamespace

    sent: list[dict] = []

    def create(model, messages, temperature, max_tokens):
        sent.append({"model": model, "messages": messages})
        message = SimpleNamespace(content='[]', refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    run(make_drafter(CFG, "fact", client=client), batches=1, tmp_path=tmp_path)
    assert sent[0]["model"] == "gpt-4o-mini"
    assert sent[0]["messages"][0]["content"] == render_draft_prompt("fact")
    assert all(f"(doc_id: {one})" in sent[0]["messages"][1]["content"] for one in FIRST.doc_ids)


# --- the command line -----------------------------------------------------------------------------


def test_the_cli_drafts_into_the_directory_it_is_given(tmp_path):
    status = main(
        ["--bucket", "fact", "--batches", "1", "--fake-model", "--drafts-dir", str(tmp_path)]
    )
    assert status == 0
    written = list(tmp_path.glob("fact_*.standin.jsonl"))
    assert len(written) == 1
    rows, violations = parse_drafts(written[0])
    assert rows and violations == []


def test_the_cli_exits_non_zero_when_a_batch_fails(tmp_path, monkeypatch, capsys):
    failing = ScriptedChat(RuntimeError("429"), RuntimeError("429"))
    monkeypatch.setattr(draft, "make_drafter", lambda *a, **k: failing)
    status = main(["--bucket", "fact", "--batches", "2", "--drafts-dir", str(tmp_path)])
    assert status == 1
    assert "retries them" in capsys.readouterr().err


def test_the_cli_refuses_the_constructed_bucket(capsys):
    with pytest.raises(SystemExit):
        main(["--bucket", "unanswerable", "--batches", "1"])
    assert "unanswerable" in capsys.readouterr().err
