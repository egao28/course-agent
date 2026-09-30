"""Tests for eval/scripts/recheck.py: the draw, and the number the re-read produces.

The thing this file is about is a distinction the review log cannot record on its own -- a row
that was *read* against a row that was *confirmed* -- so the tests are in two halves.

The first is the draw: stratified over (bucket, verdict) because those two fail differently, a
function of the seed and the judged id set and nothing else, and every model-recommended delete
taken whole because a deleted row leaves no trace downstream to be wrong in.

The second is the measurement. #59's log is append-only and replays latest-wins, so a row judged
twice carries both verdicts; :func:`eval.scripts.recheck.overturns` reads the pair off the log,
and the tests drive real verdicts through :func:`eval.scripts.review.append_verdict` rather than
writing log lines by hand. A row still carrying one verdict is *not yet re-read*, which is a
third state and the one a half-finished re-read is in.

Nothing here writes into ``eval/datasets/``.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from config import load_config
from eval.schema import BUCKETS, RULES, EvalItem, item_id
from eval.scripts import recheck as rc
from eval.scripts import review
from eval.scripts.draft_items import ID_BLOCKS, PROVENANCE_KEYS

#: A pass the size of the real one: the buckets #56 drafted, in the proportions it drafted them.
DRAFTED = {"fact": 54, "multihop": 25, "unanswerable": 25, "ambiguous": 48}


def provenance() -> dict[str, Any]:
    out: dict[str, Any] = dict.fromkeys(PROVENANCE_KEYS, None)
    out.update({"flagged": False, "credit_question": False})
    return out


def candidates() -> list[review.Candidate]:
    """One drafted row per id, with ids minted the way #56 mints them."""
    out = []
    for bucket, n in DRAFTED.items():
        for i in range(1, n + 1):
            number = ID_BLOCKS[bucket] + i
            one = EvalItem(
                id=item_id(number),
                bucket=bucket,
                question=f"question {number}?",
                gold_answer=f"answer {number}.",
                gold_doc_ids=["2026-sp:TEST-001"] * (2 if bucket == "multihop" else 1)
                if bucket in {"fact", "multihop"}
                else [],
                should_abstain=RULES[bucket].should_abstain,
            )
            out.append(review.Candidate(Path(f"{bucket}_abc.jsonl"), i, one, provenance()))
    return out


def judge_all(path: Path, rows: list[review.Candidate], deletes: set[str]) -> dict[str, Any]:
    """A first pass over every row: accept, except the ids named."""
    for n, candidate in enumerate(rows):
        ts = f"2026-09-27T22:00:{n % 60:02d}.{n:06d}Z"
        verdict = (
            review.Verdict(candidate.id, "delete", "variable credits", ts)
            if candidate.id in deletes
            else review.Verdict(candidate.id, "accept", None, ts)
        )
        review.append_verdict(path, verdict)
    return review.read_verdicts(path)


@pytest.fixture
def rows() -> list[review.Candidate]:
    return candidates()


@pytest.fixture
def log(tmp_path: Path, rows: list[review.Candidate]):
    """A judged pass with twelve deletes in it, as the real one has."""
    deletes = {item_id(ID_BLOCKS["fact"] + i) for i in range(1, 10)}
    deletes |= {item_id(ID_BLOCKS["ambiguous"] + i) for i in (1, 2)}
    deletes |= {item_id(ID_BLOCKS["multihop"] + 1)}
    path = tmp_path / "review_log.jsonl"
    judge_all(path, rows, deletes)
    return path


@pytest.fixture
def cfg():
    return load_config()


# --- the draw -------------------------------------------------------------------------------------


def test_the_draw_covers_every_bucket_and_both_verdicts(rows, log: Path):
    """Stratified on (bucket, verdict): an accept and a delete are different errors."""
    judged = review.read_verdicts(log)
    sample = rc.draw(rows, judged, seed=42)
    assert len(sample.ids) == rc.SAMPLE_N
    cells = {(one.bucket, one.verdict) for one in sample.strata if one.ids}
    assert {b for b, _ in cells} == set(BUCKETS)
    assert {v for _, v in cells} == {"accept", "delete"}


def test_the_draw_is_a_function_of_the_seed_and_the_id_set(rows, log: Path):
    judged = review.read_verdicts(log)
    assert rc.draw(rows, judged, 42).ids == rc.draw(rows, judged, 42).ids
    assert rc.draw(rows, judged, 43).ids != rc.draw(rows, judged, 42).ids


def test_file_order_does_not_move_the_draw(rows, log: Path):
    """Cells are sorted before the draw, for the reason #60's sampler sorts: otherwise sorting
    the drafts re-draws the sample."""
    judged = review.read_verdicts(log)
    first = rc.draw(rows, judged, 42).ids
    assert rc.draw(list(reversed(rows)), judged, 42).ids == first


def test_unjudged_rows_are_out_of_scope(rows, log: Path):
    """A re-read is of the pass, and a row with no verdict has not been passed over yet."""
    judged = review.read_verdicts(log)
    del judged[rows[0].id]
    sample = rc.draw(rows, judged, 42)
    assert rows[0].id not in sample.ids
    assert sum(one.held for one in sample.strata) == len(judged)


def test_every_model_recommended_delete_is_taken_whole(rows, log: Path):
    """Not sampled: a deleted row leaves nothing in the dataset for Day 4 to be wrong about."""
    judged = review.read_verdicts(log)
    always = [item_id(ID_BLOCKS["ambiguous"] + 1), item_id(ID_BLOCKS["multihop"] + 1)]
    sample = rc.draw(rows, judged, 42, always=always)
    assert set(always) <= set(sample.ids)
    assert len(sample.ids) == rc.SAMPLE_N


def test_a_small_cell_is_not_rounded_out_of_the_sample(rows, log: Path):
    """Largest remainder first, so the one-row `multihop` delete cell is still represented."""
    judged = review.read_verdicts(log)
    sample = rc.draw(rows, judged, 42, n=30)
    held = {(one.bucket, one.verdict): one for one in sample.strata}
    assert held[("multihop", "delete")].held == 1
    assert len(held[("multihop", "delete")].ids) == 1


def test_model_deletes_are_the_ones_the_lookup_did_not_ask_for():
    """Nine of the twelve came from a table; those are not a judgement to second-guess."""
    from eval.scripts.triage import Triage

    judged = {
        "v1-0001": review.Verdict("v1-0001", "delete", "variable credits", "t"),
        "v1-1006": review.Verdict("v1-1006", "delete", "claim not supported", "t"),
        "v1-0002": review.Verdict("v1-0002", "accept", None, "t"),
    }
    advisory = {
        "v1-0001": Triage("v1-0001", "delete", "GUIDELINES rule 2: AAS 465 has a range.", {}, "m"),
        "v1-1006": Triage("v1-1006", "delete", "a claim no record supports.", {}, "m"),
        "v1-0002": Triage("v1-0002", "accept", "fine.", {}, "m"),
    }
    assert rc.model_deletes(judged, advisory) == ["v1-1006"]


# --- what the re-read found -----------------------------------------------------------------------


def test_a_row_judged_once_has_not_been_re_read(rows, log: Path):
    """The third state, and the one a half-finished re-read is in."""
    sample = rc.draw(rows, review.read_verdicts(log), 42)
    found = rc.result(sample.ids, review.read_entries(log))
    assert found.reread == 0
    assert not found.done
    assert found.rate is None


def test_a_second_verdict_is_read_off_the_log(rows, log: Path):
    """No new file: #59's log is append-only and keeps both verdicts. That is why it has a clock."""
    judged = review.read_verdicts(log)
    sample = rc.draw(rows, judged, 42)
    for one in sample.ids[:10]:  # confirming the first verdict, whatever it was
        held = judged[one]
        review.append_verdict(
            log, review.Verdict(one, held.verdict, held.reason, "2026-09-28T09:00:00.000000Z")
        )
    moved = next(one for one in sample.ids[:10] if judged[one].verdict == "accept")
    review.append_verdict(
        log, review.Verdict(moved, "delete", "wrong on re-read", "2026-09-28T09:05:00.0Z")
    )
    found = rc.result(sample.ids, review.read_entries(log))
    assert found.reread == 10
    assert [one.id for one in found.changed] == [moved]
    assert found.rate == 0.1
    assert found.changed[0] == rc.Overturn(moved, "accept", "delete")


def test_the_interval_is_what_the_re_read_supports(rows, log: Path):
    """Nothing overturned is not "the pass was right" -- it is an upper bound, and a wide one."""
    judged = review.read_verdicts(log)
    sample = rc.draw(rows, judged, 42)
    for one in sample.ids:
        held = judged[one]
        review.append_verdict(
            log, review.Verdict(one, held.verdict, held.reason, "2026-09-28T09:00:00.000000Z")
        )
    found = rc.result(sample.ids, review.read_entries(log))
    assert found.done
    assert not found.changed
    lo, hi = found.interval
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.1135, abs=1e-4), hi
    out = StringIO()
    assert rc.render_result(found, out) == 0
    assert "bounds the pass's error rate at 11%" in out.getvalue()


def test_a_delete_overturned_into_an_accept_counts_too(rows, log: Path):
    """Both directions are errors, and this one is the invisible direction."""
    sample = rc.draw(rows, review.read_verdicts(log), 42)
    deleted = next(
        one for one in sample.ids if review.read_verdicts(log)[one].verdict == "delete"
    )
    review.append_verdict(
        log, review.Verdict(deleted, "accept", None, "2026-09-28T09:00:00.000000Z")
    )
    found = rc.result(sample.ids, review.read_entries(log))
    assert [(one.first, one.second) for one in found.changed] == [("delete", "accept")]


# --- the sidecar and the command ------------------------------------------------------------------


def test_the_sidecar_round_trips(cfg, rows, log: Path, tmp_path: Path):
    from eval.scripts.draft_items import write_json

    judged = review.read_verdicts(log)
    sample = rc.draw(rows, judged, 42)
    path = tmp_path / "recheck.meta.json"
    write_json(path, rc.sidecar(sample, cfg, judged))
    back = rc.read_sample(path)
    assert back is not None
    assert back.ids == sample.ids
    assert back.seed == 42
    assert sum(one.held for one in back.strata) == len(judged)


def test_no_sample_is_not_an_error(tmp_path: Path):
    assert rc.read_sample(tmp_path / "nothing.json") is None


def test_the_command_names_every_sampled_id(rows, log: Path):
    """Thirty ids are not typed by hand, so the tool prints the whole command."""
    sample = rc.draw(rows, review.read_verdicts(log), 42)
    line = rc.command(sample.ids)
    assert line.startswith("uv run python -m eval.scripts.review --triage --redo ")
    for one in sample.ids:
        assert one in line


def test_redrawing_over_a_begun_re_read_is_refused(cfg, rows, log: Path, tmp_path: Path):
    """It would report an overturn rate over rows nobody was asked to read."""
    out = tmp_path / "recheck.meta.json"
    assert rc.main(["--log", str(log), "--out", str(out)]) == 0
    assert rc.main(["--log", str(log), "--out", str(out)]) == 1


def test_result_needs_a_sample_first(tmp_path: Path, log: Path):
    assert rc.main(["--result", "--log", str(log), "--out", str(tmp_path / "none.json")]) == 2


# --- review --redo takes the ids ------------------------------------------------------------------


def test_redo_with_ids_shows_only_those(rows, log: Path):
    """Three states: no flag is the pass, bare --redo is everything, ids are a named few."""
    judged = review.read_verdicts(log)
    assert review.wanted(rows, judged, None) == []
    assert len(review.wanted(rows, judged, [])) == len(rows)
    picked = [rows[0].id, rows[5].id]
    assert [c.id for c in review.wanted(rows, judged, picked)] == picked


def test_redo_with_an_unknown_id_shows_nothing(rows, log: Path):
    assert review.wanted(rows, review.read_verdicts(log), ["v1-9999"]) == []
