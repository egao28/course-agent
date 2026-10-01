"""Tests for #140's diagnosis: why the fitted gate accepts no threshold.

The script asserts three things about the committed fit, and each is a different kind of claim:

* **It reproduces the fit.** The enumeration here and `eval.conformal`'s scan are two readings of
  one rule, so `read_facts` refuses to write unless the replay is the committed scan step for
  step. These tests break the fit on purpose and watch the refusal happen, because a reproduction
  check that cannot fail is decoration.
* **It computes nothing new.** Every number is a function of two committed files, so the report
  is byte-identical over unchanged inputs and the `test` half is never opened.
* **It separates the three causes.** Saturation, the observed errors and the finite-sample
  correction are each arithmetic with a known shape, and the tests below pin that shape on
  constructed signals rather than on the committed one: a saturated signal that offers one cut, a
  continuous one that offers every cut, a clean signal that only the correction refuses.

The arithmetic is exact. `alpha` reaches the rule as a `Fraction`, because the sweep in section 4
asks it about ratios like `6/29` whose nearest float is below them, and a cut whose risk is
exactly `alpha` must pass rather than fail by a hair.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction

import pytest

from config import load_config
from eval.conformal import scan, select_threshold
from eval.scripts import saturation as sat
from eval.scripts.risk_coverage import RUNS_DIR, SIGNALS

pytest.importorskip("scipy")

ALPHA = Fraction(1, 10)
LEVEL = 0.95


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def facts(cfg):
    """The committed CSV and fit, read and cross-checked once."""
    return sat.read_facts(cfg, sat.Sources())


@pytest.fixture(scope="module")
def report(facts) -> str:
    return sat.render_report(facts)


# --- the three causes, on signals built to isolate one each ------------------------------------


#: Every row on one value: one cut, and it answers everything.
SATURATED = [1.0] * 10
#: A distinct value per row: every cut from 1 row to 10 is available.
CONTINUOUS = [float(i) for i in range(10)]


def test_a_saturated_signal_offers_one_cut() -> None:
    s = sat.saturation("x", SATURATED, [1] * 10, float(ALPHA))
    assert s.levels == 1
    assert s.sizes == (10,)
    assert s.at_top == 10


def test_a_continuous_signal_offers_every_cut() -> None:
    s = sat.saturation("x", CONTINUOUS, [1] * 10, float(ALPHA))
    assert s.sizes == tuple(range(1, 11))
    assert s.unreachable == (), "nothing between the floor and n is missing"


def test_saturation_names_the_sets_the_correction_would_take_but_the_signal_cannot() -> None:
    """The last column of section 1: a set the arithmetic allows and no threshold produces."""
    s = sat.saturation("x", SATURATED, [1] * 10, float(ALPHA))
    assert s.unreachable == tuple(range(9, 10)), "9 is answerable in principle, not by this signal"


def test_the_correction_alone_refuses_a_perfect_small_cut() -> None:
    """Eight perfect answers are refused: 1/9 > 0.10, with no error anywhere in them."""
    cuts = sat.cuts(CONTINUOUS[:8], [1] * 8, float(ALPHA), LEVEL)
    smallest = cuts[-1]
    assert (smallest.n, smallest.k, smallest.raw) == (8, 0, 0.0)
    assert not smallest.testable and not smallest.acceptable
    assert "correction" in smallest.blocker(float(ALPHA))


def test_the_same_cut_one_row_larger_is_accepted() -> None:
    """Nine perfect answers pass: 1/10 ≤ 0.10 exactly, which is why alpha is a Fraction."""
    cuts = sat.cuts(CONTINUOUS[:9], [1] * 9, float(ALPHA), LEVEL)
    assert cuts[-1].n == 9 and cuts[-1].acceptable


def test_observed_errors_are_named_as_the_blocker_when_the_raw_rate_is_over_alpha() -> None:
    cuts = sat.cuts(SATURATED, [1] * 8 + [0] * 2, float(ALPHA), LEVEL)
    assert cuts[0].raw == pytest.approx(0.2)
    assert "observed errors" in cuts[0].blocker(float(ALPHA))


def test_the_decomposition_adds_up(facts) -> None:
    """`risk_hat − α` is exactly `(raw − α) + (risk_hat − raw)`: section 3 is a split, not two
    estimates."""
    for name in SIGNALS:
        for cut in facts.cuts[name]:
            total = cut.risk_hat - facts.alpha
            assert total == pytest.approx((cut.raw - facts.alpha) + (cut.risk_hat - cut.raw))


def test_required_n_and_satisfied_below_are_the_same_rule_read_both_ways() -> None:
    for p in (0.01, 0.05, 0.08):
        n = sat.required_n(p, 0.10)
        assert n is not None
        assert sat.satisfied_below(n, 0.10) >= p
        assert sat.satisfied_below(n - 1, 0.10) < p


def test_no_sample_size_helps_once_the_error_reaches_alpha() -> None:
    assert sat.required_n(0.10, 0.10) is None
    assert sat.required_n(0.25, 0.10) is None


def test_max_k_is_the_most_errors_a_cut_could_carry() -> None:
    assert sat.max_k(28, 0.10) == 1
    assert sat.max_k(9, 0.10) == 0
    assert sat.max_k(8, 0.10) == -1, "a cut that cannot pass even clean"


# --- the replay is the rule --------------------------------------------------------------------


def test_the_replay_is_the_committed_scan(facts) -> None:
    """Step for step, at the committed alpha — the check `read_facts` makes before writing."""
    for name in SIGNALS:
        values = facts.table.signals[name]
        replayed = sat.replay(values, facts.table.correct, Fraction(str(facts.alpha)))
        committed = scan(values, facts.table.correct, facts.alpha)
        assert [tuple(one) for one in replayed] == [tuple(one) for one in committed]


def test_the_replay_accepts_a_ratio_the_float_rule_rejects() -> None:
    """Why alpha is a Fraction: at alpha = 6/29 a cut whose risk is exactly 6/29 must pass."""
    values, correct = [1.0] * 28, [1] * 23 + [0] * 5
    exact = Fraction(6, 29)
    assert sat.chosen_cut(values, correct, exact) is not None
    assert select_threshold(values, correct, float(exact))[0] == float("inf"), (
        "the float nearest 6/29 is below it, so the same cut fails: this is the bug the "
        "Fraction replay exists to avoid")


def test_the_smallest_alpha_is_the_one_the_rule_accepts_at(facts) -> None:
    for name in SIGNALS:
        values = facts.table.signals[name]
        a = sat.smallest_alpha(values, facts.table.correct)
        if a is None:
            continue
        assert sat.chosen_cut(values, facts.table.correct, a) is not None
        below = sorted({Fraction(c.k + 1, c.n + 1)
                        for c in sat.cuts(values, facts.table.correct, 0.5, LEVEL)
                        if Fraction(c.k + 1, c.n + 1) < a})
        assert all(sat.chosen_cut(values, facts.table.correct, one) is None for one in below)


def test_acceptance_is_not_monotone_in_alpha() -> None:
    """The reason section 4 replays instead of bisecting: a looser alpha can accept less.

    The highest-scoring row is the only correct one. At alpha = 1/2 that single row is testable
    and passes. At alpha = 1/5 it is not testable, so the scan skips past it to a cut that fails.
    """
    values, correct = CONTINUOUS[:5], [0, 0, 0, 0, 1]
    assert sat.chosen_cut(values, correct, Fraction(1, 2)) is not None
    assert sat.chosen_cut(values, correct, Fraction(1, 5)) is None


# --- the committed diagnosis -------------------------------------------------------------------


def test_the_chosen_signal_saturates(facts) -> None:
    """The finding, as a test: the fit's signal cannot offer the small set the rule would take."""
    chosen = facts.fit["chosen"]["signal"]
    s = facts.sat[chosen]
    assert s.at_top > sat.min_answered(facts.alpha)
    assert sat.min_answered(facts.alpha) in s.unreachable


def test_the_report_is_the_same_bytes_twice(facts) -> None:
    assert sat.render_report(facts) == sat.render_report(facts)


def test_the_committed_report_is_what_the_generator_writes(facts) -> None:
    """A committed report nothing re-renders is a report that can go stale unnoticed."""
    path = sat.report_path(facts.cfg)
    assert path.exists(), f"run `python -m eval.scripts.saturation` to write {path.name}"
    assert path.read_text(encoding="utf-8") == sat.render_report(facts)


def test_the_report_answers_all_three_questions(report: str) -> None:
    for question in ("Does saturation block it?", "Do the observed errors block it?",
                     "Does the finite-sample correction block it?"):
        assert f"**{question}**" in report


def test_every_rate_in_the_answer_carries_its_interval(report: str) -> None:
    """#140: no bare point estimate. The answer quotes the top cut, so it quotes its interval."""
    answer = report.split("## 6. The answer")[1].split("## Limitations")[0]
    assert "[" in answer and "]" in answer


def test_the_report_does_not_read_the_test_half(facts, report: str) -> None:
    assert "test" not in {one.path.name.split("_")[1] for one in sat.inputs(facts.cfg,
                                                                           facts.src)}
    assert "signals_test" not in report


def test_a_fit_from_another_config_is_refused(cfg, tmp_path) -> None:
    runs = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, runs)
    path = sat.result_path(cfg, runs)
    fit = json.loads(path.read_text(encoding="utf-8"))
    fit["config_hash"] = "deadbeef"
    path.write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config_hash"):
        sat.read_facts(cfg, sat.Sources(runs_dir=runs))


def test_a_fit_whose_scan_this_does_not_reproduce_is_refused(cfg, tmp_path) -> None:
    """The reproduction check, broken on purpose: a recorded step that the rule would not take."""
    runs = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, runs)
    path = sat.result_path(cfg, runs)
    fit = json.loads(path.read_text(encoding="utf-8"))
    chosen = fit["chosen"]["signal"]
    fit["signals"][chosen]["scan"][0]["k"] += 1
    path.write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="replayed scan is not the one the fit recorded"):
        sat.read_facts(cfg, sat.Sources(runs_dir=runs))


def test_a_fit_made_on_another_csv_is_refused(cfg, tmp_path) -> None:
    runs = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, runs)
    path = sat.result_path(cfg, runs)
    fit = json.loads(path.read_text(encoding="utf-8"))
    fit["csv_sha256"] = "0" * 64
    path.write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not made on the calib CSV on disk"):
        sat.read_facts(cfg, sat.Sources(runs_dir=runs))
