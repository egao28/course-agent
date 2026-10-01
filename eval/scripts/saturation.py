"""#140: why the abstention gate selects no answers — saturation, errors, and the correction.

    uv run python -m eval.scripts.saturation             # write report/day8_saturation_{key}.md
    uv run python -m eval.scripts.saturation --check     # list the inputs; write nothing
    uv run python -m eval.scripts.saturation --dry-run   # print the report instead of writing it

RQ3 reported that no threshold is accepted at ``alpha = 0.10`` and stopped at "not measurable at
this calibration size" — a sentence that names the sample size as the cause without checking that
it is the cause. Three things could each be it, and this script says how much of the
non-acceptance each one owns:

* **Saturation** — which answer-set sizes the signal can produce at all. A scan can only cut where
  the signal changes value, so a signal that puts most of its rows on one value offers a handful
  of coarse cuts and nothing between them.
* **The observed errors** — how wrong the answers in each of those sets are. This is the part that
  is about the model rather than about the arithmetic.
* **The finite-sample correction** — how much of the distance to ``alpha`` the ``+1`` in
  ``(k + 1) / (n + 1)`` contributes, as against the raw error rate. The correction also forbids
  outright any set smaller than ``ceil(1/alpha - 1)`` rows, whatever the labels say.

The three have different fixes — a finer signal, a better answerer, more calibration rows — so a
report that does not separate them cannot recommend one.

**It reads the calib half and nothing else.** The fit is a calib object; a threshold diagnosed on
the rows it will be judged on is a description, not a guarantee, which is :mod:`eval.conformal`'s
own reason for never reading test. Already-committed test readings are quoted where they
corroborate, and nothing on test is computed here.

**It computes nothing new.** No model call, no resampling, no run: every number is a function of
the committed calib CSV and the committed fit. Over unchanged inputs it writes the same bytes.

**Enumerating every cut is a diagnostic, not a rule.** :func:`eval.conformal.scan` is
fixed-sequence on purpose, and :func:`eval.conformal.lowest_risk_hat` already records that
choosing a threshold by the best cut anywhere is a global selection the ``+1`` does not cover.
This script prints every cut because a diagnostic may look where a rule may not, and it fits
nothing: the only threshold it names is the committed fit's, re-derived here and refused if the
two disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import (
    REPORT_DIR,
    Config,
    conformal_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval.conformal import _acceptable, _present, result_path, select_threshold
from eval.scripts.agreement import md_table
from eval.scripts.conformal_eval import clopper_pearson
from eval.scripts.day5_report import pct
from eval.scripts.risk_coverage import SIGNALS, Table, csv_path, load_table
from eval.scripts.signals import RUNS_DIR
from eval.validate import plural

#: ``report/day8_saturation_{conformal_key}.md``. Named by the key the labels and ``alpha`` decide,
#: like Day 5's report, because moving either makes this a different diagnosis.
STEM = "day8_saturation"

#: The half this reads. There is no ``--split``: see the module docstring.
SPLIT = "calib"

#: Error rates to ask "how many rows would it take?" of, in :func:`required_n`. Spanning the
#: observed interval rather than centred on the point estimate, which is the whole question.
HYPOTHESISED = (0.02, 0.05, 0.08, 0.10)


class Sources(NamedTuple):
    """Where the diagnosis reads from, rooted so a test can point it at a copy."""

    runs_dir: Path = RUNS_DIR
    report_dir: Path = REPORT_DIR


class Input(NamedTuple):
    """One artifact this needs, and the command that produces it when it is missing."""

    name: str
    path: Path
    how: str


def inputs(cfg: Config, src: Sources) -> list[Input]:
    """The two committed files every number here is a function of."""
    table = csv_path(SPLIT, cfg, src.runs_dir)
    return [
        Input(f"{SPLIT} signals CSV", table,
              f"uv run python -m eval.scripts.review_signals --split {SPLIT} --write-csv"),
        Input(f"{SPLIT} signals sidecar", table.with_name(table.name + ".meta.json"),
              f"uv run python -m eval.scripts.review_signals --split {SPLIT} --write-csv"),
        Input("conformal fit", result_path(cfg, src.runs_dir), "uv run python -m eval.conformal"),
    ]


def missing(cfg: Config, src: Sources) -> list[Input]:
    """The inputs that are not there. Empty is the condition for writing anything."""
    return [one for one in inputs(cfg, src) if not one.path.exists()]


def report_path(cfg: Config, report_dir: Path = REPORT_DIR) -> Path:
    """``report/day8_saturation_{conformal_key}.md``."""
    return Path(report_dir) / f"{STEM}_{conformal_key(cfg)}.md"


# --- the cuts a signal can offer ------------------------------------------------------------------


class Cut(NamedTuple):
    """One candidate threshold, and every way of reading how far it is from ``alpha``.

    ``risk_hat`` is the acceptance statistic ``(k + 1) / (n + 1)`` and ``raw`` is ``k / n``, the
    error rate without the correction. The difference between them is what the ``+1`` costs at
    this ``n``, and keeping the two apart is this script's reason for existing.
    """

    t: float
    n: int
    k: int
    #: Could ``n`` rows pass with no errors at all? ``False`` means the correction alone forbids
    #: this cut, whatever its labels are, and the scan skips rather than fails it.
    testable: bool
    acceptable: bool
    raw: float
    lo: float
    hi: float
    risk_hat: float

    def blocker(self, alpha: float) -> str:
        """Which of the three things stops this cut, in the order they bind."""
        if self.acceptable:
            return "—"
        if not self.testable:
            return "correction (too few rows to pass with k = 0)"
        if self.raw > alpha:
            return "observed errors (raw rate alone is over α)"
        return "correction (raw rate is under α, the +1 is not)"


def cuts(values: Sequence[float | None], correct: Sequence[int], alpha: float,
         level: float) -> list[Cut]:
    """Every distinct value as a candidate, strictest first — including past where a scan stops.

    :func:`eval.conformal.scan` returns the prefix the fixed-sequence rule actually tests, which
    is one step long for a saturated signal and says nothing about the rest of the curve. This
    enumerates all of them for the diagnosis, using the same ``_present`` and ``_acceptable`` the
    fit uses so that the diagnostic and the gate can never disagree about one arithmetic.
    """
    rows = _present(values, correct)
    out = []
    for t in sorted({s for s, _ in rows}, reverse=True):
        answered = [c for s, c in rows if s >= t]
        n, k = len(answered), len(answered) - sum(answered)
        lo, hi = clopper_pearson(k, n, level)
        out.append(Cut(t, n, k, _acceptable(n, 0, alpha), _acceptable(n, k, alpha),
                       k / n if n else 0.0, lo, hi, (k + 1) / (n + 1)))
    return out


class Sat(NamedTuple):
    """What a signal's resolution is, which is what decides the cuts it can offer."""

    name: str
    n_rows: int
    n_null: int
    levels: int
    #: Rows sitting on the single highest value: the smallest answer set the signal can select.
    at_top: int
    #: Every answer-set size some threshold produces, ascending.
    sizes: tuple[int, ...]
    #: Sizes that could be accepted on count alone but that no threshold produces. Non-empty is
    #: saturation doing the blocking: a set the arithmetic allows and the signal cannot make.
    unreachable: tuple[int, ...]


def saturation(name: str, values: Sequence[float | None], correct: Sequence[int],
               alpha: float) -> Sat:
    """A signal's resolution: how many distinct values, and which answer sets they can cut."""
    rows = _present(values, correct)
    sizes = sorted({cut.n for cut in cuts(values, correct, alpha, 0.95)})
    floor = min_answered(alpha)
    total = len(rows)
    return Sat(name, total, len(values) - total, len({s for s, _ in rows}),
               min(sizes, default=0), tuple(sizes),
               tuple(n for n in range(floor, total + 1) if n not in set(sizes)))


def min_answered(alpha: float) -> int:
    """``ceil(1/alpha - 1)``: the smallest answer set the correction can ever accept."""
    return math.ceil(1 / alpha - 1)


def alpha_floor(n: int) -> Fraction:
    """``1 / (n + 1)``: no alpha below this accepts anything on ``n`` calibration rows."""
    return Fraction(1, n + 1)


class Candidate(NamedTuple):
    """One step of the replayed scan: ``(t, answered, wrong, tested, acceptable)``."""

    t: float
    n: int
    k: int
    tested: bool
    acceptable: bool


def replay(values: Sequence[float | None], correct: Sequence[int],
           alpha: Fraction) -> list[Candidate]:
    """:func:`eval.conformal.scan`, step for step, with exact ``alpha``.

    The sweep below asks the rule about ratios such as ``6/29``, and ``scan`` takes a float: the
    float nearest ``6/29`` is below it, so a cut whose risk is exactly ``alpha`` would be refused
    by a hair. Everything here is :class:`~fractions.Fraction` arithmetic for that reason. The
    order, the skip and the stop are ``scan``'s, and :func:`read_facts` refuses to write a report
    unless this replay reproduces the committed fit at the committed ``alpha``.
    """
    rows = _present(values, correct)
    steps: list[Candidate] = []
    for t in sorted({s for s, _ in rows}, reverse=True):
        answered = [c for s, c in rows if s >= t]
        n, k = len(answered), len(answered) - sum(answered)
        tested = Fraction(1, n + 1) <= alpha
        steps.append(Candidate(t, n, k, tested, tested and Fraction(k + 1, n + 1) <= alpha))
        if tested and not steps[-1].acceptable:
            break
    return steps


def chosen_cut(values: Sequence[float | None], correct: Sequence[int],
               alpha: Fraction) -> Candidate | None:
    """What the rule would answer at this ``alpha``: the last candidate that passed, or None."""
    passed = [step for step in replay(values, correct, alpha) if step.acceptable]
    return passed[-1] if passed else None


def smallest_alpha(values: Sequence[float | None], correct: Sequence[int]) -> Fraction | None:
    """The smallest ``alpha`` at which the fixed-sequence rule accepts any threshold.

    Swept over the only values where acceptability can change — each cut's own
    ``(k + 1) / (n + 1)`` — and replayed through the rule rather than reasoned about. Acceptance
    is *not* monotone in ``alpha``: raising it makes a smaller, stricter cut testable, and that
    cut can fail and end the scan before a larger one that would have passed.
    """
    rows = _present(values, correct)
    if not rows:
        return None
    candidates = sorted({Fraction(cut.k + 1, cut.n + 1)
                         for cut in cuts(values, correct, 0.5, 0.95)})
    return next((a for a in candidates if a < 1 and chosen_cut(values, correct, a)), None)


def satisfied_below(n: int, alpha: float) -> float:
    """The largest true error rate at which ``n`` answered rows already satisfy the correction.

    :func:`required_n` read the other way: ``alpha - (1 - alpha) / n``. Below it the rows in hand
    are enough and only the errors can refuse a cut; between it and ``alpha`` the correction
    would want more rows. Negative where ``n`` is too small to accept anything at all.
    """
    return alpha - (1 - alpha) / n


def required_n(p: float, alpha: float) -> int | None:
    """Answered rows needed before the correction stops binding, if the error rate were ``p``.

    From ``(p*n + 1) / (n + 1) <= alpha``, so ``n >= (1 - alpha) / (alpha - p)``. ``None`` when
    ``p >= alpha``: no number of rows accepts a threshold whose true error is over the budget,
    which is the whole difference between "the sample was small" and "the answers were wrong".

    It is not a power calculation. It treats the error rate as exact and ignores that ``k`` is an
    integer drawn with variance, so it is a floor on what would be needed, not a sample size to
    plan with.
    """
    return None if p >= alpha else math.ceil((1 - alpha) / (alpha - p))


# --- reading the committed files ------------------------------------------------------------------


class Facts(NamedTuple):
    """Everything the report prints, read once and checked against the committed fit."""

    cfg: Config
    src: Sources
    table: Table
    fit: dict[str, Any]
    alpha: float
    level: float
    cuts: dict[str, list[Cut]]
    sat: dict[str, Sat]
    #: ``(name, repo path, sha256)`` for the header's input table.
    digests: list[tuple[str, str, str]]


def read_facts(cfg: Config, src: Sources) -> Facts:
    """Read the CSV and the fit, and refuse unless this script re-derives the committed fit.

    ``load_table`` already checks the CSV against its sidecar. What is added here is that the
    enumeration below reproduces the fit on disk — same per-signal threshold, same answered and
    wrong counts, same null counts. A diagnosis of a fit it cannot reproduce is a diagnosis of
    something else.
    """
    table = load_table(SPLIT, cfg, src.runs_dir)
    fit_path = result_path(cfg, src.runs_dir)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    alpha, level = float(fit["alpha"]), float(cfg["eval"]["bootstrap"]["ci"])

    problems = []
    for field, want in (("config_hash", cfg.hash), ("signals_key", signals_key(cfg)),
                        ("conformal_key", conformal_key(cfg))):
        if fit.get(field) != want:
            problems.append(f"the fit has {field} {fit.get(field)}, this config {want}")
    if fit.get("csv_sha256") != table.meta.get("sha256"):
        problems.append("the fit was not made on the calib CSV on disk")
    if fit.get("n") != len(table.ids):
        problems.append(f"the fit is over {fit.get('n')} rows, the CSV has {len(table.ids)}")

    found: dict[str, list[Cut]] = {}
    sat: dict[str, Sat] = {}
    for name in SIGNALS:
        values = table.signals[name]
        found[name] = cuts(values, table.correct, alpha, level)
        sat[name] = saturation(name, values, table.correct, alpha)
        block = fit["signals"][name]
        steps = replay(values, table.correct, Fraction(str(alpha)))
        committed = [(s["t"], s["n"], s["k"], s["tested"], s["acceptable"]) for s in block["scan"]]
        if [tuple(step) for step in steps] != committed:
            problems.append(f"{name}: the replayed scan is not the one the fit recorded "
                            f"({len(steps)} steps here, {len(committed)} in the file)")
        t, n_answered, k_wrong = select_threshold(values, table.correct, alpha)
        if (None if math.isinf(t) else t) != block["t"]:
            problems.append(f"{name}: re-deriving the fit gives t {t}, the file says {block['t']}")
        if (n_answered, k_wrong) != (block["n_answered"], block["k_wrong"]):
            problems.append(f"{name}: re-deriving gives {n_answered} answered / {k_wrong} wrong, "
                            f"the file says {block['n_answered']} / {block['k_wrong']}")
        if sat[name].n_null != block["n_null"]:
            problems.append(f"{name}: {sat[name].n_null} null rows here, {block['n_null']} in "
                            "the file")
    if problems:
        raise ValueError("refusing to diagnose a fit this does not reproduce:\n  - "
                         + "\n  - ".join(problems))

    digests = [(f"{SPLIT} signals CSV", rel_to_repo(table.path), file_sha256(table.path)),
               ("conformal fit", rel_to_repo(fit_path), file_sha256(fit_path))]
    return Facts(cfg, src, table, fit, alpha, level, found, sat, digests)


# --- saying it --------------------------------------------------------------------------------


def num(x: float) -> str:
    """Three places, and a real minus sign."""
    return f"{x:.3f}".replace("-", "−")


def signed(x: float) -> str:
    """A difference, with its sign kept and never printed as ``−0.000``."""
    return f"{round(x, 3) + 0.0:+.3f}".replace("-", "−")


def interval(cut: Cut) -> str:
    """``0.179 [0.061, 0.369]`` -- a cut's raw error rate and its Clopper-Pearson interval."""
    return f"{num(cut.raw)} [{num(cut.lo)}, {num(cut.hi)}]"


def max_k(n: int, alpha: float) -> int:
    """The most wrong answers a set of ``n`` could carry and still be accepted; -1 if none can."""
    if not _acceptable(n, 0, alpha):
        return -1
    return max(k for k in range(n + 1) if _acceptable(n, k, alpha))


def render_header(f: Facts) -> str:
    """The title, what this is, and the two files every number is a function of."""
    floor = alpha_floor(f.fit["n"])
    rows = [
        ["half", f"`{SPLIT}`, {plural(f.fit['n'], 'row')}"],
        ["α", f"`{f.alpha:.2f}` (`conformal.alpha`), key `{conformal_key(f.cfg)}`"],
        ["smallest answerable set",
         f"{plural(min_answered(f.alpha), 'row')}, from `ceil(1/α − 1)`"],
        ["α floor at this n", f"`1/{f.fit['n'] + 1}` = {float(floor):.4f}"],
        *[[name, f"`{path}` sha256 `{digest[:12]}`"] for name, path, digest in f.digests],
    ]
    why = (
        f"RQ3 reported that no threshold is accepted at α = {f.alpha:.2f}, and stopped at *not "
        "measurable at this calibration size*. That sentence names the sample size as the cause "
        "without checking that it is the cause. Three things could each be it — the signal's "
        "resolution, the errors in the answers a threshold would release, and the finite-sample "
        "correction — and they have different fixes, so this file says how much of the refusal "
        "each one owns."
    )
    how = (
        "Nothing here is re-run and no model is called: every number is a function of the two "
        "files below, and the fit they describe is re-derived and compared before a line is "
        "written. The `test` half is not read."
    )
    return "\n".join([
        "# Day 8 — why the abstention gate selects no answers",
        "", why,
        "", how,
        "",
        "| | |",
        "| --- | --- |",
        *(f"| {a} | {b} |" for a, b in rows),
    ])


def render_saturation(f: Facts) -> str:
    """Section 1: what each signal's resolution lets a scan select."""
    rows = []
    for name in SIGNALS:
        s = f.sat[name]
        if len(s.unreachable) > 1:
            unreachable = f"{s.unreachable[0]}–{s.unreachable[-1]}"
        else:
            unreachable = str(s.unreachable[0]) if s.unreachable else "—"
        rows.append([f"`{name}`", str(s.n_rows), str(s.n_null), str(s.levels), str(s.at_top),
                     ", ".join(str(n) for n in s.sizes), unreachable])
    headers = ["signal", "rows", "null", "levels", "smallest cut", "sets it can answer",
               f"sets ≥ {min_answered(f.alpha)} it cannot"]
    lead = (
        "A scan can only cut where the signal changes value, so the answer sets it may choose "
        "from are fixed before a single label is read. Tied rows are answered together."
    )
    chosen = f.fit["chosen"]["signal"]
    s = f.sat[chosen]
    said = (
        f"`{chosen}`, the signal the fit chose, takes {s.levels} distinct values over its "
        f"{s.n_rows} rows and puts {s.at_top} of them on the single highest one. Its strictest "
        f"possible answer set is therefore {s.at_top} rows — no threshold anywhere answers fewer. "
        f"The correction would accept a clean set of {plural(min_answered(f.alpha), 'row')}; this "
        "signal cannot produce one, and the sets it cannot produce are the last column."
    )
    return "\n".join(["## 1. Saturation: which answer sets a signal can offer", "", lead, "",
                      md_table(headers, rows), "", said])


def render_cuts(f: Facts) -> str:
    """Section 2: every candidate threshold, and which of the three things stops it."""
    lead = (
        "Every distinct value as a candidate, strictest first. This is the whole curve rather "
        "than the prefix the rule tests — `eval.conformal.scan` stops at the first tested "
        "failure, and a diagnosis has to see past it. **Reading a threshold off this table would "
        "be the global selection the fixed-sequence rule refuses**: the `+1` does not cover a "
        "choice made over every candidate."
    )
    out = ["## 2. The cuts, and what stops each one", "", lead, ""]
    for name in SIGNALS:
        s = f.sat[name]
        rows = [[f"`{cut.t:g}`", str(cut.n), pct(cut.n / s.n_rows), str(cut.k), interval(cut),
                 num(cut.risk_hat),
                 "**yes**" if cut.acceptable else cut.blocker(f.alpha)]
                for cut in f.cuts[name]]
        headers = ["t", "answered", "coverage", "wrong",
                   f"raw error [{round(f.level * 100)}% CI]", "(k+1)/(n+1)", "accepted?"]
        out += [f"**`{name}`** — {plural(s.n_rows, 'row')}, {s.n_null} null.", "",
                md_table(headers, rows), ""]
    return "\n".join(out).rstrip("\n")


def render_decomposition(f: Facts) -> str:
    """Section 3: of the distance to α, how much is the errors and how much is the ``+1``."""
    rows = [[f"`{name}`", f"`{cut.t:g}`", str(cut.n), signed(cut.risk_hat - f.alpha),
             signed(cut.raw - f.alpha), signed(cut.risk_hat - cut.raw)]
            for name in SIGNALS for cut in f.cuts[name]]
    lead = (
        "`(k+1)/(n+1) − α` splits exactly into `k/n − α`, which is the answers being wrong, and "
        "`(k+1)/(n+1) − k/n`, which is what the correction adds at this `n`. A cut whose second "
        "column is negative and whose third is larger is one the correction alone refuses; a cut "
        "whose second column is positive is one the model refuses."
    )
    return "\n".join([
        "## 3. Of the distance to α, how much is the errors and how much is the correction",
        "", lead, "",
        md_table(["signal", "t", "answered", "total gap", "from the errors", "from the +1"], rows),
    ])


def render_sweep(f: Facts) -> str:
    """Section 4: the smallest α each signal would accept anything at."""
    rows = []
    for name in SIGNALS:
        values = f.table.signals[name]
        a = smallest_alpha(values, f.table.correct)
        if a is None:
            rows.append([f"`{name}`", "— (never)", "—", "—"])
            continue
        cut = chosen_cut(values, f.table.correct, a)
        assert cut is not None  # smallest_alpha returns an alpha where one exists
        rows.append([f"`{name}`", f"{float(a):.3f} (`{a}`)", f"`{cut.t:g}`",
                     f"{cut.n} of {f.sat[name].n_rows} ({cut.k} wrong)"])
    lead = (
        "Swept over the only values where acceptability can change — each cut's own "
        "`(k+1)/(n+1)` — and replayed through the real fixed-sequence rule rather than reasoned "
        "about. Acceptance is **not** monotone in α: raising it makes a smaller, stricter cut "
        "testable, and that cut can fail and stop the scan before a larger one that would have "
        "passed."
    )
    return "\n".join(["## 4. The smallest α that would accept anything", "", lead, "",
                      md_table(["signal", "smallest α that accepts", "t there",
                                "what it would answer"], rows)])


def render_sample_size(f: Facts) -> str:
    """Section 5: how many rows it would take, and whether the cut already has them."""
    chosen = f.fit["chosen"]["signal"]
    top = f.cuts[chosen][0]
    asked = [(f"{p:.2f}", p) for p in HYPOTHESISED]
    asked += [(f"{num(top.raw)} — observed", top.raw),
              (f"{num(top.lo)} — bottom of its interval", top.lo),
              (f"{num(top.hi)} — top of its interval", top.hi)]
    rows = []
    for label, p in asked:
        need = required_n(p, f.alpha)
        rows.append([label, "— (no n is enough)" if need is None else plural(need, "answered row"),
                     "—" if need is None else ("**yes**" if need <= top.n else "no")])
    lead = (
        f"If the true error rate at `{chosen}`'s strictest cut were `p`, the correction stops "
        "binding at `n ≥ (1 − α) / (α − p)`. It is not a power calculation: it treats `p` as "
        "exact and ignores that `k` is an integer with variance, so it is a floor on what would "
        "be needed rather than a size to plan with. Where `p ≥ α` no number of rows is enough, "
        "and that is the difference between *the sample was small* and *the answers were wrong*."
    )
    said = (
        f"**What refuses this cut is the wrong answers, not the row count.** Acceptance at "
        f"{top.n} rows needs at most {max_k(top.n, f.alpha)} wrong; there are {top.k}. The "
        f"correction's own requirement is already met at the bottom of the interval — "
        f"{plural(required_n(top.lo, f.alpha), 'row')}, and the cut answers {top.n} — though not "
        f"everywhere under α: {top.n} rows satisfy it for a true rate up to "
        f"{num(satisfied_below(top.n, f.alpha))}, and between that and α it would want more."
    )
    then = (
        f"So *would more calibration data have fixed this?* — not by being more data. The "
        f"observed {top.k} of {top.n} = {interval(top)} reaches below α and above it, so a larger "
        "half would re-estimate that rate rather than relax the requirement, and whether it then "
        "accepted would depend entirely on where the true rate actually is. The quantity to move "
        "is the error rate among the model's most confident answers; the number of rows only "
        "decides how precisely it is known."
    )
    headers = ["true error at the top cut", f"rows needed at α = {f.alpha:.2f}",
               f"the cut has {top.n} — enough?"]
    return "\n".join(["## 5. How many rows would it take?", "", lead, "",
                       md_table(headers, rows), "", said, "", then])


def render_answer(f: Facts) -> str:
    """Section 6: the three questions #140 asks, one sentence and one number each."""
    chosen = f.fit["chosen"]["signal"]
    s, top = f.sat[chosen], f.cuts[chosen][0]
    share = (top.risk_hat - top.raw) / (top.risk_hat - f.alpha)
    one = (
        f"**Does saturation block it?** Yes, and first: `{chosen}` offers no answer set smaller "
        f"than {s.at_top} rows, so the rule never gets to choose a small clean one. The "
        f"{plural(min_answered(f.alpha), 'row')} the correction would accept is a set this signal "
        "cannot make."
    )
    two = (
        f"**Do the observed errors block it?** Yes, and decisively: at that strictest cut {top.k} "
        f"of {top.n} answers are wrong, {interval(top)}, where acceptance needs at most "
        f"{max_k(top.n, f.alpha)}. The raw rate alone is {signed(top.raw - f.alpha)} past α "
        "before any correction is applied."
    )
    three = (
        f"**Does the finite-sample correction block it?** Not here, and not for want of rows. It "
        f"adds {signed(top.risk_hat - top.raw)} at n = {top.n}, which is {pct(share)} of the gap; "
        "removing it entirely would still not accept this cut, and at the bottom of that cut's "
        "interval its row count already satisfies the correction. What the correction forbids is "
        "the small sets — and this signal has none to offer, so on this fit it decides nothing."
    )
    return f"## 6. The answer\n\n{one}\n\n{two}\n\n{three}"


def render_limits(f: Facts) -> str:
    """What the diagnosis does not establish, and how to re-run it."""
    chosen = f.fit["chosen"]["signal"]
    top = f.cuts[chosen][0]
    half = (
        f"**One half, {f.fit['n']} rows.** Every rate here is over the calibration half, and the "
        "interval on the number that matters most is wide enough to contain both answers to the "
        "question section 5 asks."
    )
    fix = (
        "**A diagnosis is not a fix.** Nothing here changes the signal, the prompt, α or the "
        "gate; it says which of the three a fix would have to address. A finer-grained signal is "
        "what it points at, and that is not measured here."
    )
    menu = (
        "**The cut table is not a menu.** Section 2 prints every candidate because a diagnostic "
        "may look where a rule may not. Choosing a threshold from it would be selection over "
        "every candidate, which the `+1` does not cover."
    )
    labels = (
        "**The labels are the ones the fit used.** `correct` is one person's verdict with the "
        "judge's call displayed first, as Day 5's report records, and an error rate of "
        f"{num(top.raw)} at the top cut inherits that."
    )
    how = (
        "No key and no network: it reads the two committed files in the header, and writes the "
        "same bytes over unchanged inputs."
    )
    commands = (
        "```bash\n"
        "uv run python -m eval.scripts.saturation            # rewrite this file\n"
        "uv run python -m eval.scripts.saturation --check    # list the inputs; write nothing\n"
        "```"
    )
    return (f"## Limitations\n\n{half}\n\n{fix}\n\n{menu}\n\n{labels}\n\n"
            f"## Reproduction\n\n{commands}\n\n{how}")


def render_report(f: Facts) -> str:
    """The whole file, in the order #140 asks its three questions."""
    parts = [render_header(f), render_saturation(f), render_cuts(f), render_decomposition(f),
             render_sweep(f), render_sample_size(f), render_answer(f), render_limits(f)]
    return "\n\n".join(parts).rstrip("\n") + "\n"


def summary(f: Facts, path: Path) -> str:
    """One line for the terminal: where it went, and what it concluded."""
    chosen = f.fit["chosen"]["signal"]
    top, sat = f.cuts[chosen][0], f.sat[chosen]
    return (f"wrote {rel_to_repo(path)}: `{chosen}` saturates at {sat.at_top} of {sat.n_rows} "
            f"rows, and its strictest cut is wrong {top.k} times in {top.n}")


def render_check(cfg: Config, src: Sources, out: TextIO) -> int:
    """``--check``: the inputs, one per line, and what to run for each that is absent."""
    found = inputs(cfg, src)
    width = max(len(one.name) for one in found)
    out.write(f"report {rel_to_repo(report_path(cfg, src.report_dir))}  "
              f"conformal key {conformal_key(cfg)}\n\n")
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
    """Write the diagnosis, or with ``--check`` list its inputs and ``--dry-run`` print it."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.saturation",
        description="Write report/day8_saturation_{conformal_key}.md: why the fitted gate "
                    "accepts no threshold, in terms of saturation, observed errors and the "
                    "finite-sample correction.",
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
        return render_check(cfg, src, sys.stdout)
    if missing(cfg, src):
        render_check(cfg, src, sys.stderr)
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
