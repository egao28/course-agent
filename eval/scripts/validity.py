"""#142: what the abstention gate actually controls, and what would have to change to say so.

    uv run python -m eval.scripts.validity             # write report/day8_validity_{key}.md
    uv run python -m eval.scripts.validity --check     # list the inputs; write nothing
    uv run python -m eval.scripts.validity --dry-run   # print the report instead of writing it

#140 asked why the gate accepts nothing and answered it in arithmetic. This asks the prior
question: by what theorem, over what randomness, does accepting a threshold control anything?

**The short answer is that the committed rule controls nothing with any stated confidence**, and
:mod:`eval.conformal`'s own docstring says as much -- "a finite-sample-corrected risk estimate,
not a high-probability bound... nothing here computes one". What this module adds is the rest of
the sentence: which functional would have to be controlled, which hypothesis of split conformal's
theorem the construction does not meet, and what a rule that does carry a theorem accepts on the
same rows. The reports, which say the gate has a marginal guarantee, are what #142 corrects.

**The target is a selective risk.** The gate answers a row when its signal clears ``t``, so what
would have to be bounded is ``R(t) = P(wrong | signal >= t)`` for a future row exchangeable with
the calibration rows -- an error rate among the answered, not a marginal error rate and not a
prediction set's coverage.

**The ``+1`` is not conformal's ``n + 1``.** Split conformal's correction is a *rank*: the test
point is one of ``n + 1`` exchangeable scores, so the quantile index moves by one. Here the ``+1``
is added to a *count of errors* among a selected subset. The formulas resemble each other and the
objects do not, which is why the theorem does not carry over however the arithmetic is done.

**What a theorem would need** is the Learn-then-Test shape (Angelopoulos et al.), which
:mod:`eval.conformal` names but does not implement: a p-value per candidate that is valid for
``H_t: R(t) > alpha``, a family fixed in advance, and a procedure controlling the family-wise
error rate at some ``delta``. Then the chosen threshold satisfies ``P(R(t) <= alpha) >= 1 - delta``.
:func:`ltt` is that rule, with the binomial-tail p-value and the same fixed-sequence ordering, and
it is **reported, never fitted**: #92's gate, ``conformal.alpha`` and every file named after
``conformal_key`` stay exactly as they are. A second rule in the pipeline would rename every
committed artifact, which is a decision this does not make.

**``delta`` is a constant here, not a config key.** Adding one to the ``conformal`` section would
move ``conformal_key`` and rename every file the study cites. This is a report about a rule, not
a rule.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, TextIO

from config import REPORT_DIR, Config, conformal_key, load_config, rel_to_repo
from eval.conformal import _present
from eval.scripts.agreement import md_table
from eval.scripts.risk_coverage import SIGNALS
from eval.scripts.saturation import (
    Facts,
    Sources,
    inputs,
    min_answered,
    missing,
    num,
    read_facts,
)
from eval.scripts.signals import RUNS_DIR
from eval.validate import plural

#: ``report/day8_validity_{conformal_key}.md``, beside #140's report and named by the same key.
STEM = "day8_validity"

#: The error probability a valid rule would have to name. Not in ``config.yaml``: see the module
#: docstring -- a key in the ``conformal`` section would rename every committed artifact, and
#: nothing here fits a gate. 0.05 is the level every interval in the study already uses.
DELTA = 0.05


def report_path(cfg: Config, report_dir: Path = REPORT_DIR) -> Path:
    """``report/day8_validity_{conformal_key}.md``."""
    return Path(report_dir) / f"{STEM}_{conformal_key(cfg)}.md"


# --- the two rules ------------------------------------------------------------------------------


def binomial_p(k: int, n: int, alpha: float) -> float:
    """A valid p-value for ``H_t: R(t) > alpha`` -- ``P(Bin(n, alpha) <= k)``.

    Rows are drawn independently, so conditional on ``n`` the selected ones are independent draws
    from the law of a row given it clears ``t``, and ``k ~ Bin(n, R(t))``. Under the null the
    error rate is at least ``alpha``, which makes ``k`` stochastically at least ``Bin(n, alpha)``,
    so this tail is conservative rather than exact. Small means the null is implausible: a few
    errors where the budget predicts many.
    """
    from scipy.stats import binom

    return float(binom.cdf(k, n, alpha))


def cp_upper(k: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper limit on ``R(t)`` at confidence ``1 - delta``.

    The same test read as an interval: ``cp_upper(k, n, delta) <= alpha`` exactly when
    ``binomial_p(k, n, alpha) <= delta``, which ``tests/test_validity.py`` checks over a grid. It
    is printed as well as the p-value because a bound is the thing a reader can compare to
    ``alpha`` by eye, and the plug-in statistic it replaces is also a number on that scale.
    """
    from scipy.stats import beta

    return 1.0 if k == n else float(beta.ppf(1 - delta, k + 1, n - k))


def min_clean(alpha: float, delta: float) -> int:
    """Smallest all-correct answer set a valid bound could accept: the first ``n`` with
    ``cp_upper(0, n, delta) <= alpha``."""
    n = 1
    while cp_upper(0, n, delta) > alpha:
        n += 1
    return n


class Step(NamedTuple):
    """One candidate under both rules, so the two can be read off the same row."""

    t: float
    n: int
    k: int
    #: ``(k + 1) / (n + 1)``, the committed rule's statistic.
    plug_in: float
    #: The committed rule's verdict: is the estimate under budget?
    plug_in_accepts: bool
    p: float
    upper: float
    #: Could this ``n`` reject even with no errors? A count, so skipping on it reads no label --
    #: the same argument :mod:`eval.conformal` makes for its own skip.
    testable: bool
    rejected: bool


def ltt(values: Sequence[float | None], correct: Sequence[int], alpha: float,
        delta: float) -> list[Step]:
    """Fixed-sequence testing over the candidates, strictest first, at level ``delta``.

    The ordering, the skip and the stop are :func:`eval.conformal.scan`'s, so the two rules are
    compared candidate for candidate rather than across different families. What changes is the
    criterion: a valid p-value against ``delta`` instead of a point estimate against ``alpha``.

    Fixed-sequence testing controls the family-wise error rate when the order is fixed before the
    labels are seen. It is fixed in *form* here -- strictest first -- but the candidates are the
    observed calibration values, so the family is data-derived. That is a real gap and the report
    says so; on these rows it changes nothing, because no candidate is rejected at any threshold.
    """
    rows = _present(values, correct)
    steps: list[Step] = []
    for t in sorted({s for s, _ in rows}, reverse=True):
        answered = [c for s, c in rows if s >= t]
        n, k = len(answered), len(answered) - sum(answered)
        plug_in = (k + 1) / (n + 1)
        testable = cp_upper(0, n, delta) <= alpha
        p, upper = binomial_p(k, n, alpha), cp_upper(k, n, delta)
        steps.append(Step(t, n, k, plug_in, plug_in <= alpha, p, upper, testable,
                          testable and p <= delta))
        if testable and not steps[-1].rejected:
            break
    return steps


def chosen(steps: Sequence[Step]) -> Step | None:
    """What the valid rule would answer with: the last candidate it rejected the null for."""
    rejected = [step for step in steps if step.rejected]
    return rejected[-1] if rejected else None


# --- saying it --------------------------------------------------------------------------------


def render_header(f: Facts) -> str:
    """The title, the question, and the two files it is a function of."""
    rows = [
        ["half", f"`{f.fit['split']}`, {plural(f.fit['n'], 'row')}"],
        ["α", f"`{f.alpha:.2f}` (`conformal.alpha`), the error budget"],
        ["δ", f"`{DELTA:.2f}`, a constant of this report (see below), not a config key"],
        *[[name, f"`{path}` sha256 `{digest[:12]}`"] for name, path, digest in f.digests],
    ]
    lead = (
        "#140 asked why the gate accepts nothing and answered it in arithmetic. This file asks "
        "the prior question: **by what theorem, over what randomness, does accepting a threshold "
        "control anything?** The denominator is the number of calibration rows the candidate "
        "selects, and the rule then walks across candidates; both are decided by the data. Exact "
        "arithmetic settled how the comparison is computed, not what it licenses."
    )
    short = (
        "The short answer is that the committed rule controls nothing with any stated "
        "confidence, and `eval/conformal.py` already says so — *\"a finite-sample-corrected risk "
        "estimate, not a high-probability bound... nothing here computes one\"*. What is added "
        "here is the rest of the sentence, and the correction of five reports that say otherwise."
    )
    return "\n".join([
        "# Day 8 — what the abstention gate actually controls",
        "", lead, "", short, "",
        "| | |",
        "| --- | --- |",
        *(f"| {a} | {b} |" for a, b in rows),
    ])


def render_target(f: Facts) -> str:
    """Section 1: the functional, the assumption, and what could break it here."""
    target = (
        "The gate answers a row when its signal clears `t`. So the quantity that would have to be "
        "bounded is the **selective risk**"
    )
    formula = "R(t) = P(wrong | signal ≥ t)"
    after = (
        "for a future row exchangeable with the calibration rows — an error rate *among the "
        "answered*. Not a marginal error rate over all rows, and not a prediction set's coverage."
    )
    assumption = (
        "**The assumption.** Rows are drawn independently from one population, and the selection "
        "event is a function of the signal with `t` fixed before the labels are read. Then, "
        "conditional on how many rows clear `t`, the selected ones are independent draws from the "
        "law of a row given it clears `t`, and the error count is binomial in that number. Every "
        "statement below rests on that sentence."
    )
    breaks = (
        "**What could break it here.** The labels are one person's verdict with an LLM judge's "
        "call displayed first, so errors are not independent of each other in the way the "
        "sentence needs. The signals come from one sampled run. The questions were drafted in "
        "four designed buckets rather than sampled from the questions students ask, so "
        "\"exchangeable with a future row\" is a property of a future row drawn *from this eval "
        "set's population*, which is not the deployment population. None of that is fixed by "
        "choosing a better rule."
    )
    return (f"## 1. What would have to be controlled\n\n{target}\n\n> **{formula}**\n\n"
            f"{after}\n\n{assumption}\n\n{breaks}")


def render_not_conformal(f: Facts) -> str:
    """Section 2: which hypothesis of split conformal's theorem the construction does not meet."""
    rank = (
        "Split conformal's correction is a **rank**. Its guarantee is that a prediction set built "
        "at the `⌈(n+1)(1−α)⌉`-th smallest calibration score covers a future label with "
        "probability at least `1 − α`, and the `n + 1` is there because the test point is one of "
        "`n + 1` exchangeable scores: the index moves by one to account for it."
    )
    count = (
        "The `+1` in `(k+1)/(n+1)` is added to a **count of errors** among a selected subset. It "
        "is not a quantile of scores, the subset is chosen by the threshold being tested, and the "
        "target is conditional on that selection rather than marginal. The formulas resemble each "
        "other; the objects do not. So the theorem's conclusion is not available here, however "
        "exactly the arithmetic is done — which is what #140's `Fraction` work did and did not "
        "settle."
    )
    fixed = (
        "This is true even for a single threshold fixed in advance. There, the selected rows are "
        "an honest sample and `k/n` is an honest estimate of `R(t)` — but `(k+1)/(n+1) ≤ α` says "
        "*the estimate is under budget*, not *the risk is under budget with confidence `1 − δ`*. "
        "There is no `δ` in the committed rule, so there is no probability attached to anything "
        "it accepts."
    )
    return (f"## 2. Why split conformal's theorem is not the one\n\n{rank}\n\n{count}\n\n"
            f"{fixed}")


def render_rules(f: Facts) -> str:
    """Section 3: the two criteria side by side, and what each one buys."""
    rows = [
        ["the committed rule (#90)", "`(k+1)/(n+1) ≤ α`", "a point estimate against the budget",
         "nothing with a stated probability"],
        ["a valid rule (Learn-then-Test)", "`P(Bin(n, α) ≤ k) ≤ δ`",
         "a valid p-value against δ", "`P(R(t) ≤ α) ≥ 1 − δ`, over a family fixed in advance"],
    ]
    ltt_said = (
        "**What a theorem needs**, in the Learn-then-Test shape (Angelopoulos et al.) that "
        "`eval/conformal.py` names but does not implement: a p-value per candidate valid for "
        "`H_t: R(t) > α`, a family of candidates fixed before the labels are seen, and a "
        "procedure that controls the family-wise error rate at `δ`. Fixed-sequence testing is "
        "such a procedure, and the committed scan already has its ordering and its stopping rule "
        "— so what is missing is not the procedure but the test it is made of."
    )
    grid = (
        "**One gap this report does not close.** Fixed-sequence testing controls the family-wise "
        "error rate when the family is fixed in advance. The candidates below are the observed "
        "calibration values, so the family is data-derived. `verbalized` and `self_consistency` "
        "have declared scales that would supply a fixed grid; `retrieval_score` is an unbounded "
        "BM25 score and has none. On these rows the distinction changes nothing, because no "
        "candidate is rejected at any threshold — but it would have to be settled before any "
        "number here was called a guarantee."
    )
    return "\n".join(["## 3. The two rules, side by side", "",
                      md_table(["rule", "accepts when", "what it compares", "what it controls"],
                               rows),
                      "", ltt_said, "", grid])


def render_candidates(f: Facts) -> str:
    """Section 4: every candidate under both rules, on the same rows."""
    out = ["## 4. The same candidates, under both rules", "",
           ("The ordering, the skip and the stop are the committed scan's, so the two rules are "
            "compared candidate for candidate rather than across different families. `testable` "
            "is whether a set of that size could be rejected with no errors at all: it reads a "
            "count and no label, which is the argument the committed rule makes for its own "
            "skip."), ""]
    for name in SIGNALS:
        steps = ltt(f.table.signals[name], f.table.correct, f.alpha, DELTA)
        rows = [[f"`{one.t:g}`", str(one.n), str(one.k), num(one.plug_in),
                 "yes" if one.plug_in_accepts else "no", f"{one.p:.4f}", num(one.upper),
                 "yes" if one.testable else "no — too small to reject",
                 "**yes**" if one.rejected else "no"]
                for one in steps]
        verdict = chosen(steps)
        said = f"**`{name}`** — "
        if verdict is not None:
            said += f"the valid rule would answer at `{verdict.t:g}`, {verdict.n} rows."
        elif steps and steps[-1].testable:
            said += (f"the valid rule rejects nothing: the first candidate large enough to test "
                     f"answers {steps[-1].n} rows and fails on its errors, which ends the scan.")
        else:
            said += (f"the valid rule rejects nothing, and never reaches a candidate it could "
                     f"test — all {plural(len(steps), 'candidate')} are too small.")
        out += [said, "",
                md_table(["t", "answered", "wrong", "(k+1)/(n+1)", "committed accepts?",
                          "p-value", f"{round((1 - DELTA) * 100)}% upper bound", "testable?",
                          "valid rule rejects?"], rows), ""]
    return "\n".join(out).rstrip("\n")


def render_sizes(f: Facts) -> str:
    """Section 5: the smallest clean set each rule needs, against what the signal can offer."""
    plug, valid = min_answered(f.alpha), min_clean(f.alpha, DELTA)
    name = f.fit["chosen"]["signal"]
    at_top = f.sat[name].at_top
    rows = [
        ["the committed rule", "`(k+1)/(n+1) ≤ α`", plural(plug, "row")],
        ["a valid rule", f"upper bound ≤ α at δ = {DELTA:.2f}", plural(valid, "row")],
    ]
    said = (
        f"A valid bound needs {plural(valid, 'all-correct row')} before it can clear α at all, "
        f"against {plural(plug, 'row')} for the point estimate. `{name}`'s only selective cut "
        f"answers {at_top} — **one row short**. So under a rule that carries a theorem, the one "
        "threshold this signal can offer is not rejected, not even tested: it is skipped as too "
        "small, and the scan moves on to a looser cut that fails on its errors."
    )
    stronger = (
        "That makes #140's conclusion stronger rather than weaker. The saturation it found is not "
        "an artefact of a lenient criterion — under the stricter and better-founded one, the gap "
        "between what the signal can select and what any rule could accept is wider."
    )
    return "\n".join(["## 5. What each rule needs before it can accept anything", "",
                      md_table(["rule", "criterion", "smallest all-correct set"], rows), "",
                      said, "", stronger])


def render_answer(f: Facts) -> str:
    """Section 6: #142's question, answered."""
    name = f.fit["chosen"]["signal"]
    one = (
        "**What does the committed rule control?** Nothing with a stated probability. It accepts "
        "a threshold when a point estimate of the selective risk is under the budget. That is a "
        "description of the calibration half, not a guarantee about a future row, and no `δ` "
        "appears anywhere in it."
    )
    two = (
        "**Does split conformal's theorem apply?** No. Its `n + 1` is a rank correction for a "
        "test point among exchangeable scores; this `+1` is added to a count of errors in a "
        "selected subset, and the target is conditional on the selection. Both the construction "
        "and the functional are different."
    )
    three = (
        f"**What would?** Learn-then-Test, with a valid p-value per candidate and a `δ`. Run that "
        f"way on these rows it accepts nothing for any signal, and `{name}`'s only selective cut "
        "is not even large enough to test. The honest description of what is committed is an "
        "**abstention gate** fitted by a **threshold rule**, and that is what the reports now say."
    )
    return f"## 6. The answer\n\n{one}\n\n{two}\n\n{three}"


def render_limits(f: Facts) -> str:
    """What this does not establish, and how to re-run it."""
    scope = (
        "**Nothing here changes the gate.** #92's dormant gate, `conformal.alpha` and every file "
        "named after `conformal_key` are untouched. The valid rule is reported, never fitted: "
        "putting a second rule in the pipeline would rename every committed artifact, and that is "
        "a decision this does not make."
    )
    names = (
        "**The identifiers keep their names.** `eval/conformal.py`, `conformal_key` and "
        "`conformal.alpha` still say *conformal* because renaming them would move every file the "
        "study cites. The name is historical; the prose around it is what #142 corrects."
    )
    delta = (
        f"**`δ = {DELTA:.2f}` is a choice, not a result.** It is the level every interval in the "
        "study already uses. A different δ moves the smallest testable set, and the report would "
        "have to be re-read at that δ — which is the point of its being named rather than buried."
    )
    family = (
        "**The family is data-derived.** Section 3 says why that matters and why it changes "
        "nothing on these rows. It would have to be settled before any number here was called a "
        "guarantee."
    )
    assume = (
        "**The assumption in section 1 is not verified.** Independence and exchangeability with a "
        "future row are assumed, and section 1 lists three reasons they are doubtful here. A "
        "valid rule over invalid assumptions is not valid; it is only honest about its arithmetic."
    )
    commands = (
        "```bash\n"
        "uv run python -m eval.scripts.validity            # rewrite this file\n"
        "uv run python -m eval.scripts.validity --check    # list the inputs; write nothing\n"
        "```"
    )
    return (f"## Limitations\n\n{scope}\n\n{names}\n\n{delta}\n\n{family}\n\n{assume}\n\n"
            f"## Reproduction\n\n{commands}\n\nTwo committed files, no model call, no resample, "
            "and the `test` half is not read.")


def render_report(f: Facts) -> str:
    """The whole file, in the order #142 asks its questions."""
    parts = [render_header(f), render_target(f), render_not_conformal(f), render_rules(f),
             render_candidates(f), render_sizes(f), render_answer(f), render_limits(f)]
    return "\n\n".join(parts).rstrip("\n") + "\n"


def summary(f: Facts, path: Path) -> str:
    """One line for the terminal."""
    name = f.fit["chosen"]["signal"]
    return (f"wrote {rel_to_repo(path)}: a valid bound needs "
            f"{plural(min_clean(f.alpha, DELTA), 'all-correct row')}, and `{name}` can offer "
            f"{f.sat[name].at_top}")


def render_inputs(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--check``: the inputs, one per line, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    out.write(f"report {rel_to_repo(report_path(cfg, src.report_dir))}  "
              f"conformal key {conformal_key(cfg)}  delta {DELTA}\n\n")
    for one in found:
        out.write(f"  {'ok     ' if one.path.exists() else 'MISSING'}  {one.name:<{width}}  "
                  f"{rel_to_repo(one.path)}\n")
        if not one.path.exists():
            out.write(f"  {'':7}  {'':<{width}}  {one.how}\n")
    gaps = missing(cfg, src)
    if gaps:
        out.write(f"\n{plural(len(gaps), 'input')} missing, so nothing is written.\n")
        return 1
    out.write("\nEvery input is present.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Write the verification, or with ``--check`` list its inputs and ``--dry-run`` print it."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.validity",
        description="Write report/day8_validity_{conformal_key}.md: what the committed threshold "
                    "rule controls, which theorem would be needed, and what a valid rule accepts "
                    "on the same rows.",
    )
    parser.add_argument("--check", action="store_true",
                        help="list the inputs and what makes each; write nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report instead of writing it")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where the runs are")
    args = parser.parse_args(argv)

    cfg = load_config()
    src = Sources(args.runs_dir)
    if args.check:
        return render_inputs(cfg, src, sys.stdout)
    if missing(cfg, src):
        render_inputs(cfg, src, sys.stderr)
        return 1
    try:
        facts = read_facts(cfg, src)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    body = render_report(facts)
    if args.dry_run:
        sys.stdout.write(body)
        return 0
    path = report_path(cfg, src.report_dir)
    path.write_text(body, encoding="utf-8")
    print(summary(facts, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
