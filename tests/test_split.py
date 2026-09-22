"""Tests for eval/scripts/split.py: the draw, the freeze, the digest, and the loader.

Five groups.

The first is the derivation, over a **synthetic 120-row dataset** with the bucket counts
:data:`eval.schema.RULES` targets. What it checks is arithmetic and independence: the table in
#61 comes out exactly, the two halves stay within one row of each other, and the assignment moves
with the seed and the id set and with nothing else -- not file order, not the other buckets.

The second is the freeze: what lands in the file, what lands in the sidecar, and the two things
the command refuses to do. Re-freezing a dataset whose id set has moved is the important one; it
is the only operation here that can quietly invalidate a number that has already been reported.

The third is ``--verify``, the check the suite runs. Each of its three failures is provoked
separately -- an unfrozen file, a hand-edited half, an edited row -- and then together, because a
verify that stopped at the first would hide the others on every run until the first was fixed.

The fourth is :func:`eval.scripts.split.load_split`: the argument is required, ``"all"`` is not a
half, and a file whose digest has moved is refused rather than loaded.

The fifth is about the repository as it stands. ``eval/datasets/v1.jsonl`` does not exist -- 70 of
the 120 rows are still undrafted and #59's promote step has never run -- so the checks over the
committed dataset **skip**, in the same way #54's do. They are written now so that the day the
file lands, a split that does not re-derive is a CI failure and not a discovery.

Nothing here writes into ``eval/datasets/``: every test freezes a dataset it built in
``tmp_path``. The one exception is read-only -- the group that reads the committed file if it is
there.
"""

from __future__ import annotations

import json
import math
from io import StringIO
from math import comb
from pathlib import Path

import pytest

from config import file_sha256, load_config
from eval.schema import BUCKETS, RULES, SPLITS, EvalItem, item_id, row_json
from eval.scripts import split as sp
from eval.validate import corpus_doc_ids, validate_file

# --- a dataset --------------------------------------------------------------------------------

#: The shape #54 targets: 50 / 25 / 25 / 20.
COUNTS = {bucket: RULES[bucket].target_n for bucket in BUCKETS}

#: Gold ids per bucket, inside what each rule allows.
IDS = {"fact": 1, "multihop": 2, "unanswerable": 0, "ambiguous": 0}


@pytest.fixture(scope="module")
def known() -> frozenset[str]:
    """The real corpus. The rows below are synthetic; their citations are not, because
    :func:`eval.scripts.split.read_rows` validates before it freezes."""
    return corpus_doc_ids()


@pytest.fixture(scope="module")
def real_ids(known: frozenset[str]) -> list[str]:
    return sorted(known)


def make_rows(real_ids: list[str], counts: dict[str, int] | None = None) -> list[EvalItem]:
    """A dataset in bucket order, ids minted in pipeline order the way #56 mints them."""
    counts = COUNTS if counts is None else counts
    rows: list[EvalItem] = []
    for bucket in BUCKETS:
        for _ in range(counts[bucket]):
            n = len(rows) + 1
            start = (n * 7) % (len(real_ids) - 2)
            rows.append(
                EvalItem(
                    id=item_id(n),
                    bucket=bucket,
                    question=f"synthetic {bucket} question {n}?",
                    gold_answer=f"synthetic answer {n}",
                    gold_doc_ids=real_ids[start : start + IDS[bucket]],
                    should_abstain=RULES[bucket].should_abstain,
                )
            )
    return rows


def write(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row_json(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rows(real_ids: list[str]) -> list[EvalItem]:
    return make_rows(real_ids)


@pytest.fixture
def dataset(tmp_path: Path, rows) -> Path:
    return write(tmp_path / "v1.jsonl", rows)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def seed(cfg) -> int:
    return cfg["eval"]["split_seed"]


def freeze(cfg, path: Path, seed: int, known: frozenset[str], **kw) -> tuple[int, str]:
    out = StringIO()
    code = sp.freeze(cfg, path, seed, out, known=known, **kw)
    return code, out.getvalue()


def halves(path: Path) -> dict[str, str | None]:
    return {
        json.loads(line)["id"]: json.loads(line).get("split")
        for line in path.read_text().splitlines()
    }


# --- the derivation -------------------------------------------------------------------------


def test_the_table_in_the_issue_comes_out_exactly(rows, seed):
    """fact 25/25, multihop 13/12, unanswerable 12/13, ambiguous 10/10 -- #61's table."""
    counts = {
        stratum.bucket: (len(stratum.calib), len(stratum.test))
        for stratum in sp.derive(rows, seed).strata
    }
    assert counts == {
        "fact": (25, 25),
        "multihop": (13, 12),
        "unanswerable": (12, 13),
        "ambiguous": (10, 10),
    }


def test_the_halves_are_sixty_and_sixty(rows, seed):
    assert sp.derive(rows, seed).totals == {"calib": 60, "test": 60}


def test_every_row_is_in_exactly_one_half(rows, seed):
    assignment = sp.derive(rows, seed)
    assert set(assignment.halves) == {row.id for row in rows}
    assert set(assignment.halves.values()) == set(SPLITS)
    for stratum in assignment.strata:
        assert not set(stratum.calib) & set(stratum.test)


def test_the_odd_buckets_alternate_which_half_takes_the_extra_row(rows, seed):
    extras = {stratum.bucket: stratum.extra for stratum in sp.derive(rows, seed).strata}
    assert extras == {
        "fact": None,  # 50 rows, nothing to hand out
        "multihop": "calib",  # first odd bucket in BUCKETS order
        "unanswerable": "test",  # second, and so the other way
        "ambiguous": None,
    }


def test_four_odd_buckets_alternate_all_the_way_down(real_ids, seed):
    """Alternation is over the odd buckets in order, not over all four."""
    odd = make_rows(real_ids, {bucket: 3 for bucket in BUCKETS})
    extras = [stratum.extra for stratum in sp.derive(odd, seed).strata]
    assert extras == ["calib", "test", "calib", "test"]
    assert sp.derive(odd, seed).totals == {"calib": 6, "test": 6}


def test_a_bucket_of_one_row_still_lands_somewhere(real_ids, seed):
    one = make_rows(real_ids, {"fact": 1, "multihop": 0, "unanswerable": 0, "ambiguous": 0})
    assignment = sp.derive(one, seed)
    assert assignment.halves == {"v1-0001": "calib"}
    assert assignment.totals == {"calib": 1, "test": 0}


def test_an_empty_bucket_contributes_nothing(real_ids, seed):
    none = make_rows(real_ids, {bucket: 0 for bucket in BUCKETS})
    assignment = sp.derive(none, seed)
    assert assignment.halves == {}
    assert all(stratum.n == 0 and stratum.extra is None for stratum in assignment.strata)


def test_the_same_seed_gives_the_same_split_and_another_seed_does_not(rows, seed):
    assert sp.derive(rows, seed).halves == sp.derive(rows, seed).halves
    assert sp.derive(rows, seed).halves != sp.derive(rows, seed + 1).halves


def test_the_split_does_not_depend_on_the_order_of_the_file(rows, seed):
    """A later re-sort of v1.jsonl must not re-split the dataset -- the reason derive() sorts."""
    shuffled = list(reversed(rows))
    assert [r.id for r in shuffled] != [r.id for r in rows]
    assert sp.derive(shuffled, seed).halves == sp.derive(rows, seed).halves


def test_deleting_two_fact_rows_leaves_the_other_three_buckets_where_they_were(rows, seed):
    """Per-bucket streams: a deletion re-draws its own bucket and no other -- parity held."""
    before = sp.derive(rows, seed).halves
    fewer = [row for row in rows if row.id not in {"v1-0003", "v1-0004"}]
    after = sp.derive(fewer, seed).halves
    for row in fewer:
        if row.bucket != "fact":
            assert after[row.id] == before[row.id]


def test_deleting_one_row_flips_the_parity_and_so_the_odd_buckets_swap(rows, seed):
    """The one coupling between buckets, asserted rather than left to be discovered.

    fact 50 -> 49 makes *fact* the first odd bucket, so the extra row that went to `calib` in
    `multihop` goes to `test` instead. It is why a dataset whose id set has moved has to be
    re-frozen deliberately (see the refusal in freeze()) and not on the next run.
    """
    before = {s.bucket: s.extra for s in sp.derive(rows, seed).strata}
    fewer = [row for row in rows if row.id != "v1-0003"]
    after = {s.bucket: s.extra for s in sp.derive(fewer, seed).strata}
    assert before == {"fact": None, "multihop": "calib", "unanswerable": "test", "ambiguous": None}
    assert after == {
        "fact": "calib",
        "multihop": "test",
        "unanswerable": "calib",
        "ambiguous": None,
    }
    assert sp.derive(fewer, seed).totals == {"calib": 60, "test": 59}


def test_assign_writes_the_half_onto_the_row_and_changes_nothing_else(rows, seed):
    assignment = sp.derive(rows, seed)
    assigned = sp.assign(rows, assignment)
    assert [row.id for row in assigned] == [row.id for row in rows]  # file order kept
    for before, after in zip(rows, assigned, strict=True):
        assert after.split == assignment.halves[before.id]
        assert row_json(before) == {k: v for k, v in row_json(after).items() if k != "split"}


def test_a_uniform_split_would_land_the_ambiguous_bucket_badly_often():
    """The docstring's two numbers: 81% not 10/10, and 8.5% at 6/14 or worse. Hypergeometric."""

    def p(x: int, N: int = 120, K: int = 20, n: int = 60) -> float:
        return comb(K, x) * comb(N - K, n - x) / comb(N, n)

    assert round(1 - p(10), 2) == 0.81
    assert round(sum(p(x) for x in range(7)) + sum(p(x) for x in range(14, 21)), 3) == 0.085


# --- the freeze -----------------------------------------------------------------------------


def test_a_freeze_writes_a_half_on_every_row_and_keeps_the_file_in_order(dataset, cfg, seed, known):
    code, out = freeze(cfg, dataset, seed, known)
    assert code == 0
    written = halves(dataset)
    assert list(written) == [f"v1-{n:04d}" for n in range(1, 121)]
    assert set(written.values()) == {"calib", "test"}
    assert sorted(written.values()).count("calib") == 60
    assert "98.4% is the ceiling and 99% is out of reach" in out


def test_the_frozen_file_still_validates_and_carries_seven_columns(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    assert [v for v in validate_file(dataset, known) if v.level == "error"] == []
    first = json.loads(dataset.read_text().splitlines()[0])
    assert list(first)[-1] == "split"


def test_the_sidecar_records_the_digest_of_the_file_as_written(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    meta = json.loads(sp.meta_path(dataset).read_text())
    assert meta["sha256"] == file_sha256(dataset)
    assert meta["seed"] == seed and meta["config_hash"] == cfg.hash
    assert meta["rows"] == 120 and meta["totals"] == {"calib": 60, "test": 60}
    assert meta["buckets"]["multihop"] == {"calib": 13, "test": 12}
    assert meta["conformal"] == {
        "n_calib": 60,
        "alpha_floor": 1 / 61,
        "coverage_ceiling": 60 / 61,
    }


def test_freezing_twice_changes_nothing(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    first = dataset.read_bytes(), sp.meta_path(dataset).read_bytes()
    code, _ = freeze(cfg, dataset, seed, known)
    assert code == 0
    assert (dataset.read_bytes(), sp.meta_path(dataset).read_bytes()) == first


def test_a_dry_run_prints_the_table_and_writes_nothing(dataset, cfg, seed, known):
    code, out = freeze(cfg, dataset, seed, known, dry_run=True)
    assert code == 0 and "nothing written" in out
    assert "multihop         25      13     12   (odd row -> calib)" in out
    assert not sp.meta_path(dataset).exists()
    assert all(half is None for half in halves(dataset).values())


def test_a_re_split_that_would_move_rows_is_refused_and_says_which(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    before = halves(dataset)
    code, out = freeze(cfg, dataset, seed + 1, known)
    assert code == 1
    assert "would change halves" in out and "--refreeze" in out
    assert halves(dataset) == before  # nothing written


def test_refreeze_does_it_anyway(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    before = halves(dataset)
    code, _ = freeze(cfg, dataset, seed + 1, known, refreeze=True)
    assert code == 0 and halves(dataset) != before
    assert json.loads(sp.meta_path(dataset).read_text())["seed"] == seed + 1


def test_a_dataset_that_does_not_validate_is_not_frozen(dataset, cfg, seed, known):
    lines = dataset.read_text().splitlines()
    broken = json.loads(lines[0]) | {"gold_doc_ids": ["2026-sp:NOPE 999"]}
    dataset.write_text("\n".join([json.dumps(broken), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not validate"):
        freeze(cfg, dataset, seed, known)
    assert not sp.meta_path(dataset).exists()


def test_a_short_bucket_is_a_warning_and_still_freezes(tmp_path, real_ids, cfg, seed, known):
    """Verification deletes rows; a dataset of 118 is still the dataset."""
    short = make_rows(real_ids, {**COUNTS, "fact": 48})
    path = write(tmp_path / "v1.jsonl", short)
    code, _ = freeze(cfg, path, seed, known)
    assert code == 0
    assert sum(1 for half in halves(path).values() if half == "calib") == 59


def test_a_missing_dataset_says_how_it_is_made(tmp_path, cfg, seed, known):
    with pytest.raises(ValueError, match="review every row"):
        freeze(cfg, tmp_path / "nothing.jsonl", seed, known)


# --- verify ---------------------------------------------------------------------------------


def test_verify_passes_over_what_the_freeze_just_wrote(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    result = sp.verify(dataset, seed, known)
    assert result.ok and result.rows == 120
    out = StringIO()
    assert sp.report_verification(dataset, result, cfg, out) == 0
    assert "ok --" in out.getvalue()


def test_verify_fails_on_an_unfrozen_file(dataset, seed, known):
    result = sp.verify(dataset, seed, known)
    assert not result.ok
    assert any("carry no split" in problem for problem in result.problems)


def test_verify_catches_a_hand_edited_half(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    lines = dataset.read_text().splitlines()
    row = json.loads(lines[0])
    flipped = "test" if row["split"] == "calib" else "calib"
    lines[0] = json.dumps(row | {"split": flipped})
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = sp.verify(dataset, seed, known).problems
    assert any(f"{row['id']} (fact) is {flipped}, derived {row['split']}" == p for p in problems)


def test_verify_catches_an_edited_row_through_the_digest(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    dataset.write_text(
        dataset.read_text().replace('synthetic answer 1"', 'edited answer"', 1), encoding="utf-8"
    )
    problems = sp.verify(dataset, seed, known).problems
    assert any("changed after it was frozen" in problem for problem in problems)


def test_verify_reports_a_missing_sidecar(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    sp.meta_path(dataset).unlink()
    assert any("has no sidecar" in problem for problem in sp.verify(dataset, seed, known).problems)


def test_verify_reports_every_problem_at_once(dataset, cfg, seed, known):
    """Two moved halves and an edited answer come back as three findings, not the first one."""
    freeze(cfg, dataset, seed, known)
    lines = dataset.read_text().splitlines()
    for n in (0, 1):
        row = json.loads(lines[n])
        lines[n] = json.dumps(row | {"split": "test" if row["split"] == "calib" else "calib"})
    lines[2] = json.dumps(json.loads(lines[2]) | {"gold_answer": "edited"})
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = sp.verify(dataset, seed, known)
    assert sum(1 for p in result.problems if "derived" in p) == 2
    assert any("changed after it was frozen" in p for p in result.problems)
    out = StringIO()
    assert sp.report_verification(dataset, result, cfg, out) == 1
    assert "FAILED -- 3 problems" in out.getvalue()


def test_a_half_frozen_file_is_refused_before_anything_reads_a_half(dataset, cfg, seed, known):
    """`eval.validate.check_split` calls it, so the freeze, the verify and the loader all do."""
    freeze(cfg, dataset, seed, known)
    lines = dataset.read_text().splitlines()
    lines[0] = json.dumps({k: v for k, v in json.loads(lines[0]).items() if k != "split"})
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for call in (
        lambda: sp.verify(dataset, seed, known),
        lambda: sp.load_split("calib", dataset, known),
        lambda: freeze(cfg, dataset, seed, known),
    ):
        with pytest.raises(ValueError, match="a dataset is frozen or it is not"):
            call()


# --- the loader -------------------------------------------------------------------------------


def test_load_split_hands_back_one_half_and_the_two_partition_the_file(dataset, cfg, seed, known):
    freeze(cfg, dataset, seed, known)
    calib = sp.load_split("calib", dataset, known)
    test = sp.load_split("test", dataset, known)
    assert len(calib) == len(test) == 60
    assert {row.id for row in calib} & {row.id for row in test} == set()
    assert {row.id for row in calib} | {row.id for row in test} == {
        json.loads(line)["id"] for line in dataset.read_text().splitlines()
    }
    assert {row.split for row in calib} == {"calib"}


def test_the_half_is_a_required_argument():
    """No default: the failure guarded against is code that never chose a half."""
    import inspect

    which = inspect.signature(sp.load_split).parameters["which"]
    assert which.default is inspect.Parameter.empty


@pytest.mark.parametrize("asked", ["all", "Calib", "", None, "both"])
def test_there_is_no_way_to_ask_for_both_halves(dataset, cfg, seed, known, asked):
    freeze(cfg, dataset, seed, known)
    with pytest.raises(ValueError, match="there is no 'all'"):
        sp.load_split(asked, dataset, known)
    assert not hasattr(sp, "load_all")


def test_load_split_refuses_an_unfrozen_dataset(dataset, known):
    with pytest.raises(ValueError, match="is not frozen"):
        sp.load_split("calib", dataset, known)


def test_load_split_refuses_a_dataset_that_has_been_edited_since_the_freeze(
    dataset, cfg, seed, known
):
    freeze(cfg, dataset, seed, known)
    dataset.write_text(
        dataset.read_text().replace('synthetic answer 2"', 'edited"', 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after it was frozen"):
        sp.load_split("test", dataset, known)


# --- what sixty calibration rows can be asked for ---------------------------------------------


def test_the_conformal_floor_is_one_over_n_plus_one():
    assert sp.conformal_limit(60) == (1 / 61, 60 / 61)
    assert round(sp.conformal_limit(60)[0], 4) == 0.0164


def test_the_quantile_index_is_why_the_floor_is_there():
    """ceil((n+1)(1-alpha)) has to be an index into n scores; at alpha just under 1/61 it is not."""
    n = 60
    alpha, _ = sp.conformal_limit(n)
    assert math.ceil((n + 1) * (1 - alpha)) == n
    assert math.ceil((n + 1) * (1 - alpha * 0.99)) == n + 1


def test_the_sentence_names_the_ceiling_and_the_first_target_out_of_reach():
    line = sp.ceiling_line(60)
    assert "alpha >= 1/61 = 0.0164" in line
    assert "98.4% is the ceiling and 99% is out of reach" in line
    # Not "the ceiling rounded up": at n=99 the ceiling is 99.0% and 100% is the unreachable one.
    assert "99.0% is the ceiling and 100% is out of reach" in sp.ceiling_line(99)


def test_a_calibration_set_of_nothing_is_not_a_calibration_set():
    with pytest.raises(ValueError, match="at least one row"):
        sp.conformal_limit(0)


# --- the repository as it stands ----------------------------------------------------------------

#: #62's deliverable, and #59's before it. Absent until 70 more rows are drafted and reviewed.
V1 = sp.DATASET_PATH


@pytest.fixture
def committed() -> Path:
    if not V1.exists():
        pytest.skip(f"{V1.name} is #59's --promote output and is not in the tree yet")
    return V1


def test_the_committed_split_is_the_one_the_seed_derives(committed, seed):
    """The acceptance line #61 asks for: drift fails here, it does not turn up in week three."""
    result = sp.verify(committed, seed)
    assert result.problems == ()


def test_the_committed_digest_is_recorded_beside_the_dataset(committed):
    meta = sp.read_sidecar(sp.meta_path(committed))
    assert meta is not None, "the frozen dataset must carry its sha256 in a committed sidecar"
    assert meta["sha256"] == file_sha256(committed)


def test_every_committed_row_carries_a_half(committed):
    assert all(half in SPLITS for half in halves(committed).values())


def test_the_config_names_a_split_seed(cfg):
    """#61 adds `eval:` to config.yaml; without it the freeze has no provenance to record."""
    assert isinstance(cfg["eval"]["split_seed"], int)
