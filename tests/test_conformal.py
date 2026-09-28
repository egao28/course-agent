"""Tests for eval/conformal.py: the fixed-sequence threshold scan and the calib-only fit.

The rule is tested on hand-made scores whose answer is known without running it: a zig-zag whose
loosest passing threshold sits past a failure, sets too small to pass at all, all-tied scores, and
the exact alpha boundary. The I/O is tested on toy CSVs in #88's format, and once against the
committed calib CSV and the committed result.
"""

from __future__ import annotations

import json
import math
from io import StringIO

import pytest

from config import Config, conformal_key, correctness_key, load_config, signals_key
from eval import conformal as cf
from eval.scripts import risk_coverage as rc
from eval.scripts.signals import RUNS_DIR
from tests.test_risk_coverage import toy_rows, write_table


@pytest.fixture
def cfg():
    return load_config()


# --- select_threshold -----------------------------------------------------------------------------


def zigzag():
    """19 right, 1 wrong, 2 wrong, then 40 right: passes, passes, FAILS, then passes again."""
    scores = [100 - i for i in range(19)] + [81, 80, 79] + [50 - i * 0.5 for i in range(40)]
    correct = [1] * 19 + [0, 0, 0] + [1] * 40
    return scores, correct


def test_the_scan_stops_at_the_first_failure_not_the_global_best():
    scores, correct = zigzag()
    # The loosest t (answer all 62, 3 wrong) has (3+1)/(62+1) = 0.063 and would pass on its own.
    assert cf._acceptable(62, 3, 0.10)
    # But t = 80 (21 rows, 2 wrong: 3/22 = 0.136) fails first, so the scan ends at t = 81.
    assert cf.select_threshold(scores, correct, 0.10) == (81, 20, 1)
    steps = cf.scan(scores, correct, 0.10)
    assert steps[-1] == cf.Step(80, 21, 2, True, False)
    assert [step.tested for step in steps[:8]] == [False] * 8  # n = 1..8 cannot pass on any labels
    assert all(step.acceptable for step in steps[8:-1])
    assert cf.lowest_risk_hat(scores, correct) == pytest.approx(1 / 20)  # t = 82: 19 rows, 0 wrong


def test_candidates_too_small_to_pass_are_skipped_not_failed():
    # The top row alone gives (0+1)/(1+1) = 0.5. Failing it would end every scan of distinct
    # scores at the first row; it is skipped instead, and so are the next seven.
    scores = list(range(100, 0, -1))
    assert cf.select_threshold(scores, [1] * 100, 0.10) == (1, 100, 0)
    # Errors among the skipped rows still count once the scan reaches n = 9.
    assert cf.select_threshold(scores, [0] + [1] * 99, 0.10) == (math.inf, 0, 0)  # 2/10
    steps = cf.scan(scores, [0] + [1] * 99, 0.10)
    assert [s.tested for s in steps] == [False] * 8 + [True]


def test_no_acceptable_t_is_infinity_with_nothing_answered():
    # 8 rows all correct: (0+1)/(8+1) = 0.111 > 0.10, so not even answering all of them passes.
    assert cf.select_threshold([0.5] * 8, [1] * 8, 0.10) == (math.inf, 0, 0)
    assert cf.select_threshold([], [], 0.10) == (math.inf, 0, 0)
    assert cf.scan([], [], 0.10) == []


def test_nine_rows_with_no_errors_is_exactly_the_boundary():
    assert cf.select_threshold([0.5] * 9, [1] * 9, 0.10) == (0.5, 9, 0)
    assert cf.select_threshold([0.5] * 19, [1] * 18 + [0], 0.10) == (0.5, 19, 1)  # 2/20
    assert cf.select_threshold([0.5] * 18, [1] * 17 + [0], 0.10) == (math.inf, 0, 0)  # 2/19


def test_all_ties_are_one_candidate():
    steps = cf.scan([2 / 3] * 30, [1] * 27 + [0] * 3, 0.10)
    assert steps == [cf.Step(2 / 3, 30, 3, True, False)]  # 4/31 = 0.129
    assert cf.select_threshold([2 / 3] * 30, [1] * 28 + [0] * 2, 0.10) == (2 / 3, 30, 2)


def test_tied_rows_enter_together_whatever_their_order():
    # Four values like self_consistency: the 1.0 group is answered as a whole.
    # The wrong row sits first in the 1.0 group; row order must not split the group.
    scores = [1.0] * 19 + [2 / 3] * 5 + [1 / 3] * 3 + [0.0] * 2
    correct = [0] + [1] * 18 + [0] * 5 + [1] * 5
    steps = cf.scan(scores, correct, 0.10)
    assert [(s.n, s.k, s.tested, s.acceptable) for s in steps] == [(19, 1, True, True),
                                                                   (24, 6, True, False)]
    assert cf.select_threshold(scores, correct, 0.10) == (1.0, 19, 1)  # 2/20 = 0.10
    assert cf.select_threshold(scores[::-1], correct[::-1], 0.10) == (1.0, 19, 1)


def test_null_scores_are_left_out():
    scores = [None, 0.9] + [0.8] * 9
    correct = [0, 1] + [1] * 9
    assert cf.select_threshold(scores, correct, 0.10) == (0.8, 10, 0)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        cf.scan([0.5], [1], 0)
    with pytest.raises(ValueError, match="2 scores for 1"):
        cf.scan([0.5, 0.4], [1], 0.1)
    with pytest.raises(ValueError, match="0 or 1"):
        cf.scan([0.5], [2], 0.1)


# --- choose_signal ------------------------------------------------------------------------------


def test_the_highest_auroc_wins():
    assert cf.choose_signal({"retrieval_score": (0.76, 10), "verbalized": (0.79, 46),
                             "self_consistency": (0.58, 46)}) == "verbalized"


def test_a_tie_goes_to_more_rows_then_to_signals_order():
    assert cf.choose_signal({"retrieval_score": (0.8, 10), "verbalized": (0.8, 46)}) == "verbalized"
    same = {"self_consistency": (0.8, 46), "verbalized": (0.8, 46)}
    assert cf.choose_signal(same) == "verbalized"


def test_a_signal_without_an_auroc_cannot_be_chosen():
    assert cf.choose_signal({"retrieval_score": (None, 46), "verbalized": (0.1, 5)}) == "verbalized"
    with pytest.raises(ValueError, match="nothing to choose"):
        cf.choose_signal({"retrieval_score": (None, 46)})


def test_threshold_reads_null_back_as_infinity():
    assert cf.threshold({"t": None}) == math.inf
    assert cf.threshold({"t": 0.5}) == 0.5


# --- config -------------------------------------------------------------------------------------


def test_alpha_moves_only_its_own_key_and_the_full_hash(cfg):
    assert cfg["conformal"]["alpha"] == 0.10
    data = cfg.to_dict()
    edited = Config({**data, "conformal": {"alpha": 0.2}})
    assert edited.hash == cfg.hash == "c73b197d"
    assert signals_key(edited) == signals_key(cfg)
    assert correctness_key(edited) == correctness_key(cfg)
    assert conformal_key(edited) != conformal_key(cfg)
    assert edited.full_hash != cfg.full_hash
    # A new judge prompt relabels the CSV, so it names a new fit too.
    prompts = {**data["correctness"]["judge_prompts"], "fact": "x"}
    relabelled = Config({**data, "correctness": {"judge_prompts": prompts}})
    assert conformal_key(relabelled) != conformal_key(cfg)
    with pytest.raises(ValueError, match="no `conformal` section"):
        conformal_key(Config({k: v for k, v in data.items() if k != "conformal"}))


# --- the fit ------------------------------------------------------------------------------------


def test_the_fit_reads_calib_only(tmp_path, cfg):
    write_table(tmp_path, cfg, toy_rows(), split="test")
    with pytest.raises(ValueError, match="no calib signals CSV"):
        cf.run(cfg, StringIO(), tmp_path)


def test_the_run_writes_every_signal_and_the_choice(tmp_path, cfg):
    write_table(tmp_path, cfg, toy_rows(), split="calib")
    out = StringIO()
    path = cf.run(cfg, out, tmp_path)
    assert path.name == f"conformal_{conformal_key(cfg)}.json"
    result = json.loads(path.read_text())
    assert result["split"] == "calib" and result["n"] == 30 and result["alpha"] == 0.10
    assert result["min_answered_to_accept"] == 9
    assert result["signals"]["retrieval_score"]["n_null"] == 5
    assert result["signals"]["verbalized"]["n_null"] == 3
    for block in result["signals"].values():
        assert block["answers_nothing"] == (block["t"] is None)
        assert block["scan"], "every signal here has non-null rows"
    assert result["chosen"]["signal"] in rc.SIGNALS
    assert "calib n = 30" in out.getvalue()


def test_the_command_reports_a_missing_csv_and_exits_1(tmp_path, capsys):
    assert cf.main(["--runs-dir", str(tmp_path)]) == 1
    assert "no calib signals CSV" in capsys.readouterr().err


def test_the_committed_result_is_what_the_committed_calib_csv_gives(cfg, tmp_path):
    committed = cf.result_path(cfg)
    fresh = cf.fit(rc.load_table("calib", cfg, RUNS_DIR), cfg)
    assert fresh == json.loads(committed.read_text())
    # What the file says today: no signal reaches alpha = 0.10 at any t, so all three answer
    # nothing, and verbalized goes on to #91/#92 on calib AUROC.
    assert all(b["answers_nothing"] for b in fresh["signals"].values())
    assert fresh["chosen"]["signal"] == "verbalized"
