"""Tests for eval/scripts/risk_coverage.py: the curve, AURC, AUROC and its paired bootstrap.

The procedure is tested on hand-made signals whose answer is known without running it: a perfect
signal, a reversed one, a constant one, ties. The I/O is tested on toy CSVs written into
``tmp_path`` in #88's format with a sidecar like #88's. No committed CSV exists yet, because the
human pass has not been done, so nothing here reads one.
"""

from __future__ import annotations

import csv
import json
from io import StringIO

import numpy as np
import pytest

from config import Config, file_sha256, load_config, signals_key
from eval.scripts import risk_coverage as rc
from eval.scripts.draft_items import write_json
from eval.scripts.review_signals import CSV_COLUMNS, csv_path


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def small_b(cfg):
    """The committed config with B cut to 300, so an end-to-end run takes a second."""
    data = cfg.to_dict()
    data["eval"]["bootstrap"]["n_resamples"] = 300
    return Config(data)


# --- the curve --------------------------------------------------------------------------------


def test_a_perfect_signal_has_zero_risk_until_it_must_answer_the_wrong_rows():
    points = rc.risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert [p.coverage for p in points] == [0.25, 0.5, 0.75, 1.0]
    assert [p.risk for p in points] == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert points[0].threshold == 0.9 and points[-1].answered == 4


def test_tied_values_form_one_step():
    # Four distinct values over eight rows: four points, not eight.
    s = [1, 1, 2 / 3, 2 / 3, 2 / 3, 1 / 3, 0, 0]
    y = [1, 0, 1, 1, 0, 0, 0, 1]
    points = rc.risk_coverage(s, y)
    assert [p.answered for p in points] == [2, 5, 6, 8]
    assert points[0].risk == 0.5  # the two 1s together, whatever their row order
    assert rc.risk_coverage(s[::-1], y[::-1]) == points


def test_null_rows_are_left_out_of_the_denominator():
    points = rc.risk_coverage([None, 0.9, None, 0.1], [0, 1, 0, 0])
    assert [(p.coverage, p.risk, p.answered) for p in points] == [(0.5, 0.0, 1), (1.0, 0.5, 2)]


def test_a_signal_null_everywhere_has_no_curve():
    with pytest.raises(ValueError, match="null on every row"):
        rc.risk_coverage([None, None], [1, 0])


def test_aurc_is_the_area_under_the_step():
    points = rc.risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert rc.aurc(points) == pytest.approx(0.25 * (0 + 0 + 1 / 3 + 0.5))
    # One tied group answers everything at once: the area is the answer-everything risk.
    assert rc.aurc(rc.risk_coverage([0.5] * 4, [1, 0, 0, 0])) == pytest.approx(0.75)


def test_aurc_without_ties_is_the_mean_risk_over_row_coverages():
    rng = np.random.default_rng(3)
    s, y = rng.random(40), rng.integers(0, 2, 40)
    order = np.argsort(-s)
    risks = [1 - y[order[: k + 1]].mean() for k in range(40)]
    assert rc.aurc(rc.risk_coverage(list(s), list(y))) == pytest.approx(np.mean(risks))


# --- AUROC --------------------------------------------------------------------------------------


def test_a_perfect_signal_has_auroc_one_and_a_reversed_one_zero():
    y = [1, 1, 1, 0, 0]
    assert rc.auroc([0.9, 0.8, 0.7, 0.2, 0.1], y) == 1.0
    assert rc.auroc([0.1, 0.2, 0.3, 0.8, 0.9], y) == 0.0


def test_a_constant_signal_is_refused_not_scored_one_half():
    with pytest.raises(ValueError, match="constant"):
        rc.auroc([0.7] * 4, [1, 0, 1, 0])


def test_one_class_is_refused():
    with pytest.raises(ValueError, match="both classes"):
        rc.auroc([0.1, 0.5, 0.9], [1, 1, 1])


def test_ties_count_half():
    # One correct and one incorrect row tied, one of each separated: (1 + 0.5) / 2.
    assert rc.auroc([1, 0.5, 0.5, 0], [1, 1, 0, 0]) == pytest.approx(
        (1 * 1 + 1 * 1 + 0.5 + 1) / 4
    )


def test_auroc_skips_null_rows():
    assert rc.auroc([None, 0.9, 0.1, None], [0, 1, 0, 1]) == 1.0


# --- the paired bootstrap ---------------------------------------------------------------------


def test_all_signals_share_one_draw_and_the_same_seed_repeats_it():
    y = [1, 0] * 10
    signals = {"a": [0.9, 0.1] * 10, "b": [0.1, 0.9] * 10}
    one, two = (rc.bootstrap_auroc(signals, y, 200, seed=5) for _ in range(2))
    assert np.array_equal(one.indices, two.indices)
    assert np.array_equal(one.aurocs["a"], two.aurocs["a"], equal_nan=True)
    # a perfect, b reversed, on every resample that has both classes: paired, a - b is always 1.
    finite = np.isfinite(one.aurocs["a"])
    assert np.all(one.aurocs["a"][finite] == 1.0) and np.all(one.aurocs["b"][finite] == 0.0)
    assert np.array_equal(finite, np.isfinite(one.aurocs["b"]))


def test_a_one_class_resample_is_skipped_and_counted():
    # n = 3 with one incorrect row: a draw misses it with probability (2/3)^3 = 0.30.
    boot = rc.bootstrap_auroc({"a": [0.9, 0.8, 0.1]}, [1, 1, 0], 2_000, seed=1)
    lo, hi, skipped = rc.percentile_interval(boot.aurocs["a"], 0.95)
    assert 0.2 * 2_000 < skipped < 0.4 * 2_000
    assert (lo, hi) == (1.0, 1.0)


def test_every_resample_skipped_is_an_error():
    with pytest.raises(ValueError, match="every resample"):
        rc.percentile_interval(np.full(5, np.nan), 0.95)


# --- the CSV and the command ------------------------------------------------------------------


def write_table(runs_dir, cfg, rows, split="test"):
    """A toy signals CSV and sidecar in #88's format; ``rows`` are (bucket, rs, vb, sc, correct)."""
    path = csv_path(split, cfg, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for i, (bucket, rs, vb, sc, correct) in enumerate(rows):
            cells = {
                "id": f"q{i:03d}",
                "bucket": bucket,
                "split": split,
                "retrieval_score": rs,
                "dense_top1": 0.5,
                "verbalized": vb,
                "self_consistency": sc,
                "judge_correct": correct,
                "correct": correct,
                "reviewer": "toy",
            }
            writer.writerow(["" if cells[c] is None else cells[c] for c in CSV_COLUMNS])
    write_json(
        path.with_name(path.name + ".meta.json"),
        {"sha256": file_sha256(path), "signals_key": signals_key(cfg), "n_rows": len(rows)},
    )
    return path


def toy_rows(n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        correct = int(rng.random() < 0.6)
        rs = None if i % 7 == 0 else round(float(rng.random() + correct), 3)
        vb = None if i % 11 == 0 else int(rng.integers(0, 101))
        sc = [0, 1 / 3, 2 / 3, 1][int(rng.integers(0, 4))]
        rows.append(("fact" if i % 3 else "multihop", rs, vb, sc, correct))
    return rows


def test_a_missing_csv_is_refused_by_name(tmp_path, cfg):
    with pytest.raises(ValueError, match="--write-csv"):
        rc.load_table("test", cfg, tmp_path)


def test_a_csv_that_changed_after_its_sidecar_is_refused(tmp_path, cfg):
    path = write_table(tmp_path, cfg, toy_rows())
    with path.open("a", encoding="utf-8") as f:
        f.write("q999,fact,test,0.1,0.1,1,1,1,1,toy\n")
    with pytest.raises(ValueError, match="sha256"):
        rc.load_table("test", cfg, tmp_path)


def test_a_csv_for_another_signals_key_is_refused(tmp_path, cfg):
    path = write_table(tmp_path, cfg, toy_rows())
    meta = path.with_name(path.name + ".meta.json")
    write_json(meta, {**json.loads(meta.read_text()), "signals_key": "00000000"})
    with pytest.raises(ValueError, match="00000000"):
        rc.load_table("test", cfg, tmp_path)


def test_empty_cells_are_nulls(tmp_path, cfg):
    write_table(tmp_path, cfg, toy_rows())
    table = rc.load_table("test", cfg, tmp_path)
    assert table.signals["retrieval_score"][0] is None and table.signals["verbalized"][0] is None
    assert sum(v is None for v in table.signals["retrieval_score"]) == len(range(0, 30, 7))


def test_the_run_writes_the_figure_and_numbers_and_says_n(tmp_path, small_b):
    write_table(tmp_path, small_b, toy_rows())
    out = StringIO()
    figure, numbers = rc.run("test", small_b, out, tmp_path, tmp_path / "figures")
    result = json.loads(numbers.read_text())
    assert figure.name == f"risk_coverage_test_{signals_key(small_b)}.png"
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert result["n"] == 30 and result["n_resamples"] == 300
    assert result["signals"]["retrieval_score"]["n_null"] == 5
    assert result["signals"]["verbalized"]["n_null"] == 3
    assert len(result["signals"]["self_consistency"]["points"]) <= 4
    assert set(result["differences"]) == {
        "retrieval_score - verbalized",
        "retrieval_score - self_consistency",
        "verbalized - self_consistency",
    }
    text = out.getvalue()
    assert "test n = 30" in text and "| retrieval_score | 25 | 5 |" in text


def test_the_same_csv_and_seed_give_the_same_bytes(tmp_path, small_b):
    write_table(tmp_path, small_b, toy_rows())
    first = [p.read_bytes() for p in rc.run("test", small_b, StringIO(), tmp_path, tmp_path)]
    second = [p.read_bytes() for p in rc.run("test", small_b, StringIO(), tmp_path, tmp_path)]
    assert first == second


def test_the_command_reports_a_missing_csv_and_exits_1(tmp_path, capsys):
    assert rc.main(["--runs-dir", str(tmp_path), "--figures-dir", str(tmp_path)]) == 1
    assert "no test signals CSV" in capsys.readouterr().err
