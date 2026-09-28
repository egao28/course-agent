"""Tests for eval/scripts/signals.py: three signals per row, the bypassed gate, the files.

The chat client is a fake that answers by what it is asked -- the router's system prompt, the
answer node's, the verbalized follow-up, a judge prompt -- so the whole agent runs with no network,
through the real graph, the real dispatch table and the real catalog for a tool row. BM25 is a fake
injected through ``retrieval.BUILDERS`` as in ``tests/test_ablation.py``, and ``dense_top1`` is
injected directly. The dataset is a small frozen fixture in ``tmp_path``, never ``v1.jsonl``.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pytest

from agent.prompts import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from config import Config, file_sha256, load_config, signals_key
from eval.schema import RULES, EvalItem, row_json
from eval.scripts import signals as sg
from eval.scripts.draft_items import write_json
from eval.scripts.split import meta_path
from retrieval import BUILDERS, clear_cache
from retrieval.types import Hit

KNOWN = frozenset(f"2026-sp:{code}" for code in ("CS-374", "CS-446", "CS-441", "MATH-213"))

#: (id, bucket, split, question, gold, the router's label, the verbalized reply, judge replies).
ROWS = [
    # A retrieval row: BM25 answers, and its fake scores sit under the 15.0 threshold, so the Day 2
    # gate would have refused it -- the bypass answers it anyway.
    ("v1-0001", "fact", "calib", "what is machine learning about?", ["2026-sp:CS-446"],
     "compare_recommend", "Supported.\nConfidence: 85", ["yes", "yes", "no"]),
    # A tool row: an exact lookup, no retrieval score. An unparseable verbalized reply and an
    # unparseable judge verdict.
    ("v1-0002", "fact", "calib", "how many hours is cs 374?", ["2026-sp:CS-374"],
     "course_fact", "Confidence: 90\nI'd say.", ["Yes.", "maybe", "no"]),
    # Never reaches dispatch.
    ("v1-0003", "unanswerable", "calib", "what is the weather today?", [],
     "out_of_scope", "0", ["yes"] * 3),
    # Reaches dispatch and comes back with nothing: a prereq question naming no course.
    ("v1-0004", "ambiguous", "calib", "what do i need before the course?", [],
     "prereq_chain", "0", ["yes"] * 3),
    # The other half: must not be read by a calib run.
    ("v1-0005", "multihop", "test", "what comes after math 213?",
     ["2026-sp:MATH-213", "2026-sp:CS-374"], "compare_recommend", "50", ["yes"] * 3),
]  # fmt: skip

BY_QUESTION = {row[3]: row for row in ROWS}
DOC_ID_RE = re.compile(r"^doc_id: (.+)$", re.MULTILINE)


class FakeChat:
    """Answers by prompt type, and keeps every call as (kind, prompt, system, temperature)."""

    model = "fake-model"

    def __init__(self, cite: str | None = None) -> None:
        self.calls: list[tuple[str, str, str | None, float | None]] = []
        self.cite = cite
        self.judged: dict[str, int] = {}

    def complete(self, prompt, *, system=None, temperature=None):
        question = next(q for q in BY_QUESTION if q in prompt)
        row = BY_QUESTION[question]
        if system == ROUTER_SYSTEM_PROMPT:
            kind, reply = "router", row[5]
        elif system == ANSWER_SYSTEM_PROMPT:
            doc_id = self.cite or DOC_ID_RE.search(prompt).group(1)
            if temperature is None:
                kind, reply = "answer", f"The answer. [{doc_id}]"
            else:
                n = sum(1 for call in self.calls if call[0] == "sample")
                kind, reply = "sample", f"Sample {n}. [{doc_id}]"
        elif "Confidence: <integer>" in prompt:
            kind, reply = "verbalized", row[6]
        elif "Do A and B agree?" in prompt:
            n = self.judged.get(question, 0)
            self.judged[question] = n + 1
            kind, reply = "judge", row[7][n]
        else:  # pragma: no cover - a prompt this fake does not know
            raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")
        self.calls.append((kind, prompt, system, temperature))
        return reply


class FakeRetriever:
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        ids = ["2026-sp:CS-446", "2026-sp:CS-441"]
        return [Hit(d, 10.0 / r, r, f"text of {d}", {}) for r, d in enumerate(ids, start=1)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    clear_cache()
    monkeypatch.setitem(BUILDERS, "bm25", lambda cfg: FakeRetriever())
    yield
    clear_cache()


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "v1.jsonl"
    items = [
        EvalItem(
            id=id_,
            bucket=bucket,
            question=question,
            gold_answer="an answer",
            gold_doc_ids=gold,
            should_abstain=RULES[bucket].should_abstain,
            split=split,
        )
        for id_, bucket, split, question, gold, *_ in ROWS
    ]
    path.write_text("".join(json.dumps(row_json(i)) + "\n" for i in items), encoding="utf-8")
    write_json(meta_path(path), {"sha256": file_sha256(path)})
    return path


def collect(cfg, dataset, tmp_path, chat=None, split="calib"):
    out = StringIO()
    chat = chat or FakeChat()
    path = sg.collect_split(
        cfg,
        split,
        out,
        dataset,
        tmp_path / "runs",
        known=KNOWN,
        chat=chat,
        dense_top1=lambda query, cfg: 0.61,
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return lines, meta, out.getvalue(), chat


# --- the run --------------------------------------------------------------------------------------


def test_every_row_of_the_half_is_answered_once_or_skipped_with_a_reason(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    assert [line["id"] for line in lines] == ["v1-0001", "v1-0002"]
    assert meta["ids"] == ["v1-0001", "v1-0002"]
    assert meta["skipped_rows"] == [
        {"id": "v1-0003", "bucket": "unanswerable", "reason": "out_of_scope"},
        {"id": "v1-0004", "bucket": "ambiguous", "reason": "no_evidence: no_course_code"},
    ]
    assert meta["skipped"] == {"no_evidence": 1, "out_of_scope": 1}
    assert meta["n_rows"] == 4 and meta["n_lines"] == 2
    assert "v1-0005" not in printed  # the test half is not read by a calib run
    assert "answered 2 (2 fact)" in printed
    assert "skipped 2 (1 no_evidence, 1 out_of_scope)" in printed


def test_a_retrieval_row_carries_its_top1_and_the_gate_it_bypassed(cfg, dataset, tmp_path):
    line = collect(cfg, dataset, tmp_path)[0][0]
    assert line["retrieval_score"] == 10.0  # the max, from the mode that ran
    assert line["retrieval_source"] == "bm25"
    assert line["dense_top1"] == 0.61
    assert line["gate_ok"] is False  # 10.0 < 15.0: Day 2 would have handed this over
    assert line["exit"] == "answer"
    assert line["answer"] == "The answer. [2026-sp:CS-446]"
    assert line["verbalized"] == 85 and line["verbalized_raw"] == "Supported.\nConfidence: 85"
    assert line["pair_agree"] == [True, True, False]
    assert line["self_consistency"] == pytest.approx(2 / 3)


def test_a_tool_row_has_a_null_retrieval_score_never_one(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    line = lines[1]
    assert line["retrieval_score"] is None
    # The question's own dense top-1 is still recorded: it does not depend on what dispatch did.
    assert line["dense_top1"] == 0.61
    assert line["retrieval_source"] == "get_course"
    assert line["gate_ok"] is True
    assert meta["tool_rows"] == {"fact": 1}
    assert "tool rows, retrieval_score null: 1 (1 fact)" in printed


def test_an_unparseable_verbalized_reply_is_null_and_kept_and_counted(cfg, dataset, tmp_path):
    lines, meta, printed, _ = collect(cfg, dataset, tmp_path)
    assert lines[1]["verbalized"] is None
    assert lines[1]["verbalized_raw"] == "Confidence: 90\nI'd say."  # not the last line
    assert meta["verbalized_unparsed"] == 1
    assert "verbalized unparsed (null): 1" in printed


def test_an_unparseable_judge_verdict_nulls_the_score_not_shrinks_it(cfg, dataset, tmp_path):
    lines, meta, _, _ = collect(cfg, dataset, tmp_path)
    assert lines[1]["pair_agree"] == [True, None, False]
    assert lines[1]["judge_raw"] == ["Yes.", "maybe", "no"]
    assert lines[1]["self_consistency"] is None
    assert meta["judge_unparsed"] == 1


def test_samples_reuse_the_answer_prompt_at_the_sample_temperature(cfg, dataset, tmp_path):
    lines, _, _, chat = collect(cfg, dataset, tmp_path)
    answers = [call for call in chat.calls if call[0] == "answer"]
    samples = [call for call in chat.calls if call[0] == "sample"]
    assert len(samples) == 3 * len(answers) == 6
    first = answers[0]
    for kind, prompt, system, temperature in samples[:3]:
        assert (prompt, system) == (first[1], first[2])
        assert temperature == 0.7
    assert first[3] is None  # the answer itself is drawn at the config's temperature
    assert lines[0]["samples"] == ["Sample 0. [2026-sp:CS-446]", "Sample 1. [2026-sp:CS-446]",
                                   "Sample 2. [2026-sp:CS-446]"]  # fmt: skip
    # Each judge sees one pair, in (0, 1), (0, 2), (1, 2) order.
    judged = [call[1] for call in chat.calls if call[0] == "judge"][:3]
    for prompt, (i, j) in zip(judged, [(0, 1), (0, 2), (1, 2)], strict=True):
        assert f"Answer A:\nSample {i}." in prompt and f"Answer B:\nSample {j}." in prompt


def test_the_verbalized_prompt_carries_the_question_the_evidence_and_the_answer(
    cfg, dataset, tmp_path
):
    _, _, _, chat = collect(cfg, dataset, tmp_path)
    prompt = next(call[1] for call in chat.calls if call[0] == "verbalized")
    assert "what is machine learning about?" in prompt
    assert "doc_id: 2026-sp:CS-446" in prompt
    assert "The answer. [2026-sp:CS-446]" in prompt


def test_a_draft_the_citation_check_discarded_is_still_recorded(cfg, dataset, tmp_path):
    lines, meta, _, _ = collect(cfg, dataset, tmp_path, chat=FakeChat(cite="NOPE 999"))
    assert lines[0]["exit"] == "refuse_handoff"
    assert lines[0]["answer"] == "The answer. [NOPE 999]"
    assert lines[0]["citations_unknown"] == ["NOPE 999"]
    assert meta["citation_discarded"] == {"fact": 2}


def test_the_sidecar_says_what_the_run_was(cfg, dataset, tmp_path):
    _, meta, _, _ = collect(cfg, dataset, tmp_path)
    path = tmp_path / "runs" / f"signals_calib_{signals_key(cfg)}.jsonl"
    assert meta["sha256"] == file_sha256(path)
    assert meta["config_hash"] == cfg.hash
    assert meta["signals_key"] == signals_key(cfg)
    assert meta["dataset_sha256"] == file_sha256(dataset)
    assert meta["model"] == "fake-model"
    assert meta["gate"].startswith("bypassed")
    assert meta["samples"] == 3 and meta["sample_temperature"] == 0.7
    assert set(meta["prompts_sha256"]) == {"router_v2", "answer_v1", "verbalized", "judge"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in meta["prompts_sha256"].values())


def test_a_completion_error_skips_the_row_with_its_reason(cfg, dataset, tmp_path):
    class Truncating(FakeChat):
        def complete(self, prompt, *, system=None, temperature=None):
            if temperature is not None:
                raise RuntimeError("chat completion was cut off at the max_tokens ceiling (800)")
            return super().complete(prompt, system=system, temperature=temperature)

    lines, meta, _, _ = collect(cfg, dataset, tmp_path, chat=Truncating())
    assert lines == []
    reasons = [row["reason"] for row in meta["skipped_rows"]]
    assert reasons[:2] == ["error: chat completion was cut off at the max_tokens ceiling (800)"] * 2
    assert meta["skipped"] == {"error": 2, "no_evidence": 1, "out_of_scope": 1}


def test_split_is_required(capsys):
    with pytest.raises(SystemExit):
        sg.main([])
    assert "--split" in capsys.readouterr().err


# --- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "value"),
    [("Confidence: 85", 85), ("It checks out.\nConfidence: 0\n", 0), ("confidence: 100%", 100),
     ("Confidence: 70.", 70), ("Confidence: 101", None), ("Confidence: -5", None),
     ("Confidence: 85.5", None), ("85", None), ("Confidence: 85\nbut maybe less", None),
     ("Says 90 in the evidence.\nConfidence: 40", 40), ("", None), ("Confidence: eighty", None)],
)  # fmt: skip
def test_verbalized_is_the_integer_on_the_last_confidence_line_or_null(reply, value):
    assert sg.parse_verbalized(reply) == value


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [("yes", True), ("Yes.", True), (" NO ", False), ("no.", False), ("yes, mostly", None),
     ("y", None), ("", None)],
)  # fmt: skip
def test_a_verdict_is_yes_or_no_or_null(reply, verdict):
    assert sg.parse_verdict(reply) is verdict


def test_agreement_takes_four_values_for_three_samples():
    assert sg.pairs(3) == [(0, 1), (0, 2), (1, 2)]
    assert {sg.agreement(v) for v in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])} == {
        0.0,
        1 / 3,
        2 / 3,
        1.0,
    }
    assert sg.agreement([True, None, True]) is None


# --- config ---------------------------------------------------------------------------------------


def with_signals(cfg, **changes) -> Config:
    data = cfg.to_dict()
    return Config({**data, "signals": {**data["signals"], **changes}})


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"samples": 1}, "samples"),
        ({"samples": True}, "samples"),
        ({"sample_temperature": 0}, "sample_temperature"),
        ({"verbalized_prompt": "How sure? {answer}"}, "placeholders"),
        ({"judge_prompt": "{question} {a} {b} {c}"}, "placeholders"),
    ],
)
def test_a_signals_section_that_cannot_run_is_refused(cfg, changes, match):
    with pytest.raises(ValueError, match=match):
        sg.settings(with_signals(cfg, **changes))


def test_the_shipped_section_is_valid_and_names_the_run(cfg, tmp_path):
    config = sg.settings(cfg)
    assert config.samples == 3 and config.sample_temperature == 0.7
    assert sg.run_path("test", cfg, tmp_path).name == f"signals_test_{signals_key(cfg)}.jsonl"
