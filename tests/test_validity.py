"""Tests for #142: what the committed threshold rule controls, and what a valid one would.

The module makes one statistical claim and one editorial one, and they are tested differently.

* **The statistical claim** is that `binomial_p` and `cp_upper` are two readings of one test —
  `cp_upper(k, n, δ) ≤ α` exactly when `binomial_p(k, n, α) ≤ δ` — and that fixed-sequence testing
  over them is stricter than the committed rule everywhere it matters. Those are checkable over a
  grid, and the grid is what is checked: a duality asserted in prose and never computed is how a
  report acquires a second unverified claim while fixing the first.
* **The editorial claim** is that no prose in the repo still says the committed rule carries a
  guarantee. That is a grep, and it is here because #142 exists precisely because five reports
  said something the module they describe refuses to say.

The committed rule is never changed and never re-fitted: every test below reads the same
committed calibration table #140 reads, and `eval/conformal.py` is imported only to borrow the
arithmetic the two rules share.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import REPO_ROOT, load_config
from eval.scripts import saturation as sat
from eval.scripts import validity as v
from eval.scripts.risk_coverage import SIGNALS

pytest.importorskip("scipy")

ALPHA, DELTA = 0.10, 0.05


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def facts(cfg):
    return sat.read_facts(cfg, sat.Sources())


@pytest.fixture(scope="module")
def report(facts) -> str:
    return v.render_report(facts)


# --- the test, read two ways -------------------------------------------------------------------


def test_the_bound_and_the_p_value_are_one_test() -> None:
    """`cp_upper(k, n, δ) ≤ α` exactly when `binomial_p(k, n, α) ≤ δ`, over a grid."""
    for n in (1, 2, 5, 9, 28, 29, 46, 100):
        for k in range(min(n, 6) + 1):
            by_bound = v.cp_upper(k, n, DELTA) <= ALPHA
            by_p = v.binomial_p(k, n, ALPHA) <= DELTA
            assert by_bound == by_p, (n, k, v.cp_upper(k, n, DELTA), v.binomial_p(k, n, ALPHA))


def test_the_p_value_rises_with_the_errors() -> None:
    """More errors is weaker evidence against "the risk is over budget"."""
    ps = [v.binomial_p(k, 40, ALPHA) for k in range(8)]
    assert ps == sorted(ps)


def test_the_bound_is_above_the_point_estimate() -> None:
    """An upper confidence limit is above the rate it bounds; the plug-in is not a bound."""
    for n, k in ((28, 5), (46, 20), (33, 10), (9, 0)):
        assert v.cp_upper(k, n, DELTA) > (k + 1) / (n + 1)


def test_a_clean_set_of_the_published_size_clears_the_bound() -> None:
    need = v.min_clean(ALPHA, DELTA)
    assert v.cp_upper(0, need, DELTA) <= ALPHA
    assert v.cp_upper(0, need - 1, DELTA) > ALPHA


def test_the_valid_rule_needs_more_rows_than_the_committed_one() -> None:
    """The headline comparison: 9 rows against 29, and the gap is the missing δ."""
    assert sat.min_answered(ALPHA) < v.min_clean(ALPHA, DELTA)


def test_the_committed_rule_accepts_where_the_valid_rule_does_not() -> None:
    """Nine perfect answers: the plug-in is exactly α, the bound is nowhere near it."""
    assert (0 + 1) / (9 + 1) <= ALPHA
    assert v.cp_upper(0, 9, DELTA) > ALPHA


def test_a_large_enough_clean_set_is_rejected_by_the_valid_rule() -> None:
    """The rule can fire: it is strict, not broken."""
    n = v.min_clean(ALPHA, DELTA)
    steps = v.ltt([float(i) for i in range(n)], [1] * n, ALPHA, DELTA)
    assert v.chosen(steps) is not None


# --- on the committed rows ---------------------------------------------------------------------


def test_the_valid_rule_accepts_nothing_for_any_signal(facts) -> None:
    for name in SIGNALS:
        steps = v.ltt(facts.table.signals[name], facts.table.correct, facts.alpha, v.DELTA)
        assert v.chosen(steps) is None, f"{name} would be accepted; the report says none is"


def test_the_chosen_signals_only_cut_is_too_small_even_to_test(facts) -> None:
    """#142's sharpest number: the one cut the signal offers is below the testable floor."""
    name = facts.fit["chosen"]["signal"]
    steps = v.ltt(facts.table.signals[name], facts.table.correct, facts.alpha, v.DELTA)
    assert not steps[0].testable
    assert facts.sat[name].at_top == v.min_clean(facts.alpha, v.DELTA) - 1


def test_the_two_rules_walk_the_same_candidates(facts) -> None:
    """Compared candidate for candidate: the valid rule changes the criterion, not the family."""
    for name in SIGNALS:
        values = facts.table.signals[name]
        mine = [(one.t, one.n, one.k)
                for one in v.ltt(values, facts.table.correct, facts.alpha, v.DELTA)]
        theirs = [(one.t, one.n, one.k) for one in facts.cuts[name]]
        assert mine == theirs[: len(mine)]


def test_the_report_is_the_same_bytes_twice(facts) -> None:
    assert v.render_report(facts) == v.render_report(facts)


def test_the_committed_report_is_what_the_generator_writes(facts) -> None:
    path = v.report_path(facts.cfg)
    assert path.exists(), f"run `python -m eval.scripts.validity` to write {path.name}"
    assert path.read_text(encoding="utf-8") == v.render_report(facts)


def test_the_report_names_the_functional_and_the_theorem(report: str) -> None:
    assert "R(t) = P(wrong | signal ≥ t)" in report
    assert "Learn-then-Test" in report
    assert "family-wise error rate" in report


def test_the_report_says_which_hypothesis_fails(report: str) -> None:
    """#142's acceptance: one sentence on why split conformal's theorem is not the one."""
    section = report.split("## 2. ")[1].split("## 3. ")[0]
    assert "rank" in section and "count of errors" in section


def test_the_report_does_not_change_the_gate(report: str) -> None:
    for promise in ("reported, never fitted", "`conformal.alpha`"):
        assert promise in report


# --- the editorial claim: no prose still says "guarantee" --------------------------------------


#: The sentences #142 retired. Each one asserted that the committed rule carries a guarantee,
#: which is what :mod:`eval.conformal` has always said it does not. Written out rather than
#: grepped for the word, because the word is legitimate wherever the prose says a guarantee is
#: *absent*, or explains what split conformal's own guarantee is — this file's report does both.
RETIRED = (
    "The conformal guarantee is marginal",
    "guarantee is marginal over calibration draws",
    "Its guarantee holds in expectation over calib",
)

#: Everywhere a reader could meet one: the reports, the two front pages, and the eval sources
#: that generate them.
def written() -> list[Path]:
    """Every file whose prose a reader is held to, as paths."""
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "report").glob("*.md")),
            *sorted((REPO_ROOT / "eval").rglob("*.py"))]


def test_no_retired_sentence_survives_anywhere() -> None:
    """The defect #142 exists for: five places asserted what the module refuses to assert."""
    offenders = []
    for path in written():
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(REPO_ROOT)}: {phrase!r}"
                      for phrase in RETIRED if phrase in text]
    assert not offenders, "retired sentences still in the tree:\n  " + "\n  ".join(offenders)


def test_the_report_says_plainly_that_there_is_none() -> None:
    """Deleting the claim is half of it; the other half is saying what is true instead."""
    text = (REPO_ROOT / "report" / "README.md").read_text(encoding="utf-8")
    assert "carries no guarantee, marginal or otherwise" in text


def test_the_module_that_would_have_to_provide_it_still_says_it_does_not() -> None:
    """`eval/conformal.py` was right all along; this pins the sentence the reports drifted from."""
    source = (REPO_ROOT / "eval" / "conformal.py").read_text(encoding="utf-8")
    assert "not a high-probability bound" in source
    assert "nothing here computes one" in source
