"""Tests for eval/scripts/bootstrap.py: the paired procedure, the run-file checks, calib-then-test.

The procedure is tested on hand-made vectors, where the right interval is known without running
it. The I/O is tested on run files written into ``tmp_path`` in #75's format. One test reads the
two committed run files, which the committed ``bootstrap_{bootstrap_key}.json`` was computed from;
it re-queries nothing.

``cfg`` is the live config without its ``rerank`` section -- the three-mode procedure #76 shipped.
The stretch row (#79) has tests of its own at the end, under ``live``.
"""

from __future__ import annotations

import json
import shutil
import time
from io import StringIO

import numpy as np
import pytest

from config import Config, bootstrap_key, file_sha256, load_config
from eval.scripts import ablation as ab
from eval.scripts import bootstrap as bs
from eval.scripts import rerank as rr
from eval.scripts.draft_items import write_json
from retrieval import MODES


@pytest.fixture
def live():
    return load_config()


@pytest.fixture
def cfg(live):
    return Config({k: v for k, v in live.to_dict().items() if k != "rerank"})


# --- the procedure ------------------------------------------------------------------------------


def test_identical_vectors_give_a_difference_of_exactly_zero():
    x = np.random.default_rng(0).random(41)
    boot = bs.paired_bootstrap({"a": x, "b": x.copy()}, 10_000, seed=1)
    assert boot.difference("a", "b") == (0.0, 0.0, 0.0)
    assert not boot.difference("a", "b").excludes_zero


def test_a_mode_better_on_every_question_has_an_interval_strictly_above_zero():
    x = np.random.default_rng(0).random(41)
    gap = bs.paired_bootstrap({"a": x + 0.01, "b": x}, 10_000, seed=1).difference("a", "b")
    assert gap.lo > 0 and gap.excludes_zero


def test_the_same_seed_gives_the_same_output():
    scores = {"a": np.arange(41) % 2, "b": np.arange(41) % 3 == 0}
    one, two = (bs.paired_bootstrap(scores, 2_000, seed=7) for _ in range(2))
    assert np.array_equal(one.indices, two.indices)
    assert one.interval("a") == two.interval("a") and one.difference("a", "b") == two.difference(
        "a", "b"
    )
    other = bs.paired_bootstrap(scores, 2_000, seed=8)
    assert not np.array_equal(one.indices, other.indices)


def test_every_mode_is_resampled_with_the_same_indices():
    # b = a + 1 on every question: paired, every resample's difference is 1; drawn separately,
    # it would spread by roughly the whole range of a.
    a = np.arange(10, dtype=float)
    boot = bs.paired_bootstrap({"a": a, "b": a + 1, "c": a[::-1].copy()}, 500, seed=3)
    for mode, v in boot.scores.items():
        assert np.array_equal(boot.means[mode], v[boot.indices].mean(axis=1))
    assert np.allclose(boot.means["b"] - boot.means["a"], 1.0, atol=1e-12)
    assert boot.indices.shape == (500, 10)


def test_a_perfect_mode_has_a_degenerate_interval_not_a_certain_one():
    boot = bs.paired_bootstrap({"a": np.ones(41), "b": np.zeros(41)}, 1_000, seed=0)
    assert boot.interval("a") == (1.0, 1.0, 1.0) and boot.interval("a").degenerate
    assert not bs.paired_bootstrap({"a": np.arange(41) % 2}, 1_000, seed=0).interval("a").degenerate


def test_the_interval_brackets_the_point_and_uses_ci():
    x = np.random.default_rng(5).random(41)
    wide = bs.paired_bootstrap({"a": x}, 10_000, seed=0, ci=0.99).interval("a")
    narrow = bs.paired_bootstrap({"a": x}, 10_000, seed=0, ci=0.5).interval("a")
    assert wide.lo < narrow.lo <= narrow.point <= narrow.hi < wide.hi
    assert narrow.point == pytest.approx(x.mean())


def test_ten_thousand_resamples_on_41_questions_run_well_under_a_second():
    scores = {mode: np.random.default_rng(i).random(41) for i, mode in enumerate(MODES)}
    start = time.perf_counter()
    boot = bs.paired_bootstrap(scores, 10_000, seed=76)
    boot.difference("hybrid", "bm25")
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize(
    ("scores", "ci", "match"),
    [
        ({}, 0.95, "no score vectors"),
        ({"a": np.ones(3), "b": np.ones(4)}, 0.95, "same length"),
        ({"a": np.ones((2, 2))}, 0.95, "1-D"),
        ({"a": np.ones(3)}, 1.0, "ci must be"),
        ({"a": np.ones(0)}, 0.95, "n >= 1"),
    ],
)
def test_bad_input_is_refused(scores, ci, match):
    with pytest.raises(ValueError, match=match):
        bs.paired_bootstrap(scores, 10, seed=0, ci=ci)


def test_strongest_single_ignores_hybrid_and_records_a_tie():
    assert bs.strongest_single({"dense": 0.7, "bm25": 0.9, "hybrid": 1.0}) == ("bm25", False)
    assert bs.strongest_single({"dense": 0.8, "bm25": 0.8}) == ("dense", True)


# --- the run files ------------------------------------------------------------------------------


def write_run(cfg, runs_dir, split, per_mode, **meta_overrides):
    """A run file in #75's format: ``per_mode[mode]`` is one (recall, mrr) pair per row."""
    n = len(next(iter(per_mode.values())))
    ids = [f"v1-{i:04d}" for i in range(1, n + 1)]
    lines = [
        {
            "id": row_id,
            "bucket": "fact",
            "mode": mode,
            "ranked": [],
            "gold": ["x"],
            "recall_at_5": per_mode[mode][i][0],
            "mrr_at_5": per_mode[mode][i][1],
        }
        for i, row_id in enumerate(ids)
        for mode in MODES
    ]
    path = ab.write_lines(ab.run_path(split, cfg, runs_dir), lines)
    meta = {
        "sha256": file_sha256(path),
        "config_hash": cfg.hash,
        "dense_key": "d" * 8,
        "bm25_key": "b" * 8,
        "dataset_sha256": "s" * 64,
        "split": split,
        "k": 5,
        "modes": list(MODES),
        "n": n,
        "buckets": {"fact": n},
        "ids": ids,
        **meta_overrides,
    }
    write_json(path.with_suffix(".meta.json"), meta)
    return path


#: On calib dense beats bm25; on test bm25 beats dense. The choice must follow calib.
CALIB = {
    "dense": [(1.0, 1.0), (1.0, 1.0), (1.0, 0.5)],
    "bm25": [(0.0, 0.0), (1.0, 0.5), (1.0, 0.5)],
    "hybrid": [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
}
TEST = {
    "dense": [(0.0, 0.0), (0.0, 0.0), (1.0, 0.2), (0.0, 0.0)],
    "bm25": [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 0.5)],
    "hybrid": [(1.0, 0.5), (1.0, 0.5), (0.0, 0.0), (1.0, 0.5)],
}


def test_the_strongest_single_is_chosen_on_calib_and_tested_on_test(cfg, tmp_path):
    write_run(cfg, tmp_path, "calib", CALIB)
    write_run(cfg, tmp_path, "test", TEST)
    out = StringIO()
    path = bs.bootstrap(cfg, out, tmp_path)
    result = json.loads(path.read_text())
    for metric in bs.METRICS:
        block = result["metrics"][metric]
        assert block["strongest_single"]["mode"] == "dense"
        assert block["difference"]["subtrahend"] == "dense"
    recall = result["metrics"]["recall_at_5"]
    assert recall["test"]["bm25"]["point"] == 1.0 and recall["test"]["bm25"]["degenerate"]
    assert recall["difference"]["point"] == pytest.approx(0.75 - 0.25)
    assert result["n"] == {"calib": 3, "test": 4}
    assert path.name == f"bootstrap_{bootstrap_key(cfg)}.json"
    assert result["bootstrap_key"] == bootstrap_key(cfg) and result["config_hash"] == cfg.hash
    assert result["full_hash"] == cfg.full_hash
    assert "strongest single on calib = dense" in out.getvalue()
    assert "degenerate" in out.getvalue()


def test_the_output_is_the_same_on_a_second_run(cfg, tmp_path):
    write_run(cfg, tmp_path, "calib", CALIB)
    write_run(cfg, tmp_path, "test", TEST)
    first = bs.bootstrap(cfg, StringIO(), tmp_path).read_bytes()
    assert bs.bootstrap(cfg, StringIO(), tmp_path).read_bytes() == first


def test_a_missing_run_says_how_to_make_it(cfg, tmp_path):
    write_run(cfg, tmp_path, "calib", CALIB)
    with pytest.raises(ValueError, match="--split test"):
        bs.bootstrap(cfg, StringIO(), tmp_path)


def test_an_edited_run_file_is_refused(cfg, tmp_path):
    path = write_run(cfg, tmp_path, "test", TEST)
    path.write_text(path.read_text().replace('"recall_at_5": 0.0', '"recall_at_5": 1.0', 1))
    with pytest.raises(ValueError, match="sha256"):
        bs.load_run("test", cfg, tmp_path)


def test_a_run_from_another_config_is_refused(cfg, tmp_path):
    write_run(cfg, tmp_path, "test", TEST, config_hash="0" * 8)
    with pytest.raises(ValueError, match="was run under 00000000"):
        bs.load_run("test", cfg, tmp_path)


def test_a_missing_line_is_refused_rather_than_shrinking_one_mode(cfg, tmp_path):
    path = write_run(cfg, tmp_path, "test", TEST)
    kept = path.read_text().splitlines(keepends=True)[:-1]
    path.write_text("".join(kept))
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    write_json(meta_path, {**meta, "sha256": file_sha256(path)})
    with pytest.raises(ValueError, match="1 missing"):
        bs.load_run("test", cfg, tmp_path)


def test_halves_run_against_different_indexes_are_refused(cfg, tmp_path):
    write_run(cfg, tmp_path, "calib", CALIB, bm25_key="c" * 8)
    write_run(cfg, tmp_path, "test", TEST)
    with pytest.raises(ValueError, match="different bm25_keys"):
        bs.bootstrap(cfg, StringIO(), tmp_path)


def test_the_committed_result_is_what_the_committed_runs_give(live, tmp_path):
    committed = bs.result_path(live)
    for run in (ab.run_path("calib", live), ab.run_path("test", live), rr.run_path("test", live)):
        shutil.copy(run, tmp_path / run.name)
        shutil.copy(run.with_suffix(".meta.json"), tmp_path / run.with_suffix(".meta.json").name)
    fresh = json.loads(bs.bootstrap(live, StringIO(), tmp_path).read_text())
    fresh["runs"] = json.loads(committed.read_text())["runs"]  # tmp_path is not the repo
    assert fresh == json.loads(committed.read_text())


# --- the stretch row (#79) ------------------------------------------------------------------------


def write_rerank_run(cfg, runs_dir, per_row):
    """A rerank run in #79's format, paired with the test run already in ``runs_dir``."""
    test_run = bs.load_run("test", cfg, runs_dir)
    lines = [
        {"id": i, "mode": rr.MODE, "recall_at_5": r, "mrr_at_5": m}
        for i, (r, m) in zip(test_run.ids, per_row, strict=True)
    ]
    path = ab.write_lines(rr.run_path("test", cfg, runs_dir), lines)
    write_json(
        path.with_suffix(".meta.json"),
        {
            "run": "eval/runs/" + path.name,
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "rerank_key": rr.rerank_key(cfg),
            "rerank": dict(cfg["rerank"]),
            "paired_with": {"sha256": file_sha256(test_run.path)},
            "ids": test_run.ids,
            "k": 5,
            "pool": cfg["rerank"]["pool"],
        },
    )
    return path


RERANK = [(1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)]


def test_the_rerank_row_is_resampled_with_the_three_modes_indices(cfg, live, tmp_path):
    write_run(live, tmp_path, "calib", CALIB)
    write_run(live, tmp_path, "test", TEST)
    three = json.loads(bs.bootstrap(cfg, StringIO(), tmp_path).read_text())
    write_rerank_run(live, tmp_path, RERANK)
    out = StringIO()
    four = json.loads(bs.bootstrap(live, out, tmp_path).read_text())
    for metric in bs.METRICS:
        block = four["metrics"][metric]
        assert {m: block["test"][m] for m in MODES} == three["metrics"][metric]["test"]
        assert block["difference"] == three["metrics"][metric]["difference"]
        gap = block["rerank_difference"]
        assert (gap["minuend"], gap["subtrahend"]) == ("hybrid_rerank", "hybrid")
    mrr = four["metrics"]["mrr_at_5"]["rerank_difference"]
    assert mrr["point"] == pytest.approx(0.75 - 0.375)
    assert four["rerank"]["rerank_key"] == rr.rerank_key(live)
    assert four["rerank"]["model"] == live["rerank"]["model"]
    assert "hybrid_rerank - hybrid" in out.getvalue()


def test_with_a_rerank_section_the_rerank_run_is_required(live, tmp_path):
    write_run(live, tmp_path, "calib", CALIB)
    write_run(live, tmp_path, "test", TEST)
    with pytest.raises(ValueError, match="eval.scripts.rerank --split test"):
        bs.bootstrap(live, StringIO(), tmp_path)
