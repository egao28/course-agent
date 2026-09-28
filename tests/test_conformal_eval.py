"""Tests for eval/scripts/conformal_eval.py: #90's threshold and a fixed 0.7, applied to test once.

The interval and the thresholds are tested on numbers whose answer is known without running the
code. The provenance checks are tested on a toy chain in tmp_path: a dataset, both halves' signals
run sidecars, #88-format CSVs, and a fit made by #90's own ``run``. Then once against the
committed files.
"""

from __future__ import annotations

import json
import math
from io import StringIO

import pytest

from config import file_sha256, load_config
from eval import conformal as cf
from eval.scripts import conformal_eval as ce
from eval.scripts import risk_coverage as rc
from eval.scripts.draft_items import write_json
from eval.scripts.judge import judge_path
from eval.scripts.review import DATASET_PATH
from eval.scripts.signals import RUNS_DIR, run_path
from eval.scripts.split import META_SUFFIX
from tests.test_risk_coverage import toy_rows, write_table


@pytest.fixture
def cfg():
    return load_config()


# --- the numbers --------------------------------------------------------------------------------


def test_clopper_pearson_matches_the_textbook_values():
    lo, hi = ce.clopper_pearson(0, 10, 0.95)
    assert lo == 0.0 and hi == pytest.approx(0.30850, abs=1e-5)  # 1 - 0.025 ** (1/10)
    lo, hi = ce.clopper_pearson(5, 10, 0.95)
    assert (lo, hi) == (pytest.approx(0.18709, abs=1e-5), pytest.approx(0.81291, abs=1e-5))
    assert ce.clopper_pearson(10, 10, 0.95)[1] == 1.0
    assert ce.clopper_pearson(0, 0, 0.95) == (0.0, 1.0)
    with pytest.raises(ValueError, match="0 <= k <= n"):
        ce.clopper_pearson(3, 2, 0.95)


def test_an_infinite_threshold_answers_nothing_and_has_no_error_rate():
    block = ce.apply([90, 80, None], [1, 0, 1], math.inf, 0.95)
    assert block["answers_nothing"] and block["t"] is None
    assert (block["answered"], block["wrong"], block["realized_error"]) == (0, 0, None)
    assert block["ci"] == [0.0, 1.0]
    assert (block["n_non_null"], block["n_null"], block["coverage"]) == (2, 1, 0.0)


def test_a_null_row_is_never_answered():
    block = ce.apply([None, None, 5.0], [0, 0, 1], -math.inf, 0.95)
    assert (block["answered"], block["wrong"], block["coverage"]) == (1, 0, 1.0)


def test_the_fixed_threshold_is_put_on_each_signals_scale():
    at = {name: ce.FIXED * ce.SCALE[name] for name in rc.SIGNALS}
    assert at["verbalized"] == pytest.approx(70)
    v = ce.apply([70, 69, 100], [1, 1, 1], at["verbalized"], 0.95)
    assert v["answered"] == 2
    # 0.7 falls between 2/3 and 1, so only unanimous samples clear it.
    s = ce.apply([1.0, 2 / 3, 1 / 3, 0.0], [1, 1, 1, 1], at["self_consistency"], 0.95)
    assert s["answered"] == 1
    # Raw BM25 scores run far above 0.7, so every non-null row clears it.
    r = ce.apply([7.2, 31.0, None], [1, 0, 1], at["retrieval_score"], 0.95)
    assert (r["answered"], r["n_non_null"]) == (2, 2)


# --- a toy chain --------------------------------------------------------------------------------


def chain(tmp_path, cfg):
    """A dataset, both signals-run sidecars, both CSVs, and #90's fit, all consistent."""
    dataset = tmp_path / "v1.jsonl"
    dataset.write_text('{"id": "q000"}\n', encoding="utf-8")
    for split, seed in (("calib", 1), ("test", 2)):
        run = run_path(split, cfg, tmp_path)
        run.write_text(f"{split}\n", encoding="utf-8")
        write_json(run.with_suffix(META_SUFFIX),
                   {"sha256": file_sha256(run), "dataset_sha256": file_sha256(dataset)})
        path = write_table(tmp_path, cfg, toy_rows(seed=seed), split=split)
        meta = path.with_name(path.name + ".meta.json")
        write_json(meta, {**json.loads(meta.read_text()),
                          "signals_run": str(run), "signals_run_sha256": file_sha256(run),
                          "judge_run": str(judge_path(split, cfg, tmp_path))})
    cf.run(cfg, StringIO(), tmp_path)
    return dataset


def edit_json(path, **changes):
    write_json(path, {**json.loads(path.read_text()), **changes})


def test_the_run_writes_both_methods_on_every_signal_and_says_n(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    out = StringIO()
    path = ce.run(cfg, out, tmp_path, dataset)
    assert path.name == f"conformal_eval_{cf.result_path(cfg).stem.split('_')[1]}.json"
    result = json.loads(path.read_text())
    assert result["split"] == "test" and result["n"] == 30 and result["ci_level"] == 0.95
    assert set(result["signals"]) == set(rc.SIGNALS)
    for name, block in result["signals"].items():
        assert set(block) == {"conformal", "fixed", "test_range"}
        assert block["conformal"]["target_error"] == 0.10 and block["fixed"]["target_error"] is None
        assert block["fixed"]["meaning"] == ce.MEANING[name]
    assert result["signals"]["retrieval_score"]["fixed"]["n_null"] == 5
    assert result["dense_top1"] == {"n": 30, "max": 0.5, "answered_at_0.7": 0}
    assert result["dataset_sha256"] == file_sha256(dataset)
    text = out.getvalue()
    assert "test n = 30" in text and "| conformal (calib-fit t) |" in text
    assert "Appendix" in text and "(no guarantee)" in text


def test_the_test_half_is_read_once(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    ce.run(cfg, StringIO(), tmp_path, dataset)
    with pytest.raises(ValueError, match="has been read"):
        ce.run(cfg, StringIO(), tmp_path, dataset)


def test_a_missing_fit_is_refused_by_name(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    cf.result_path(cfg, tmp_path).unlink()
    with pytest.raises(ValueError, match="python -m eval.conformal"):
        ce.run(cfg, StringIO(), tmp_path, dataset)


def test_a_changed_dataset_is_refused(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    dataset.write_text('{"id": "q001"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="collected on dataset"):
        ce.run(cfg, StringIO(), tmp_path, dataset)
    assert not ce.result_path(cfg, tmp_path).exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conformal_key", "00000000", "conformal_key 00000000"),
        ("correctness_key", "00000000", "correctness_key 00000000"),
        ("signals_key", "00000000", "signals_key 00000000"),
        ("alpha", 0.2, "alpha 0.2"),
        ("split", "test", "made on 'test'"),
        ("csv_sha256", "0" * 64, "not the calib CSV the fit was made on"),
    ],
)
def test_a_fit_that_does_not_match_is_refused(tmp_path, cfg, field, value, message):
    dataset = chain(tmp_path, cfg)
    edit_json(cf.result_path(cfg, tmp_path), **{field: value})
    with pytest.raises(ValueError, match=message):
        ce.run(cfg, StringIO(), tmp_path, dataset)


def test_labels_from_another_judge_run_are_refused(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    csv = rc.load_table("test", cfg, tmp_path).path
    edit_json(csv.with_name(csv.name + ".meta.json"), judge_run="eval/runs/judge_test_0000.jsonl")
    with pytest.raises(ValueError, match="correctness key"):
        ce.run(cfg, StringIO(), tmp_path, dataset)


def test_a_csv_not_built_from_the_signals_run_on_disk_is_refused(tmp_path, cfg):
    dataset = chain(tmp_path, cfg)
    run = run_path("test", cfg, tmp_path)
    edit_json(run.with_suffix(META_SUFFIX), sha256="0" * 64)
    with pytest.raises(ValueError, match="test CSV was not built from the test signals run"):
        ce.run(cfg, StringIO(), tmp_path, dataset)


def test_the_command_reports_a_refusal_and_exits_1(tmp_path, capsys):
    assert ce.main(["--runs-dir", str(tmp_path), "--dataset", str(tmp_path / "none")]) == 1
    assert "no conformal fit" in capsys.readouterr().err


# --- the committed files ------------------------------------------------------------------------


def test_the_committed_result_is_what_the_committed_inputs_give(cfg):
    committed = ce.result_path(cfg)
    if not committed.exists():
        pytest.skip("the test half has not been read yet")
    fit = ce.load_fit(cfg, RUNS_DIR)
    table = rc.load_table("test", cfg, RUNS_DIR)
    fresh = {
        **ce.check_provenance(table, fit, cfg, RUNS_DIR, DATASET_PATH),
        **ce.evaluate(table, fit, float(cfg["eval"]["bootstrap"]["ci"]), ce.read_dense_top1(table)),
    }
    assert fresh == json.loads(committed.read_text())
    # #90 answers nothing on every signal, so on test too, and the main row has no error rate.
    assert all(b["conformal"]["answers_nothing"] for b in fresh["signals"].values())
    assert fresh["chosen"] == "verbalized"
