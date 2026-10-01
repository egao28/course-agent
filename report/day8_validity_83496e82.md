# Day 8 — what the abstention gate actually controls

#140 asked why the gate accepts nothing and answered it in arithmetic. This file asks the prior question: **by what theorem, over what randomness, does accepting a threshold control anything?** The denominator is the number of calibration rows the candidate selects, and the rule then walks across candidates; both are decided by the data. Exact arithmetic settled how the comparison is computed, not what it licenses.

The short answer is that the committed rule controls nothing with any stated confidence, and `eval/conformal.py` already says so — *"a finite-sample-corrected risk estimate, not a high-probability bound... nothing here computes one"*. What is added here is the rest of the sentence, and the correction of five reports that say otherwise.

| | |
| --- | --- |
| half | `calib`, 46 rows |
| α | `0.10` (`conformal.alpha`), the error budget |
| δ | `0.05`, a constant of this report (see below), not a config key |
| calib signals CSV | `eval/runs/signals_calib_37164723.csv` sha256 `2e6ef2df42ef` |
| conformal fit | `eval/runs/conformal_83496e82.json` sha256 `27987e369aaa` |

## 1. What would have to be controlled

The gate answers a row when its signal clears `t`. So the quantity that would have to be bounded is the **selective risk**

> **R(t) = P(wrong | signal ≥ t)**

for a future row exchangeable with the calibration rows — an error rate *among the answered*. Not a marginal error rate over all rows, and not a prediction set's coverage.

**The assumption.** Rows are drawn independently from one population, and the selection event is a function of the signal with `t` fixed before the labels are read. Then, conditional on how many rows clear `t`, the selected ones are independent draws from the law of a row given it clears `t`, and the error count is binomial in that number. Every statement below rests on that sentence.

**What could break it here.** The labels are one person's verdict with an LLM judge's call displayed first, so errors are not independent of each other in the way the sentence needs. The signals come from one sampled run. The questions were drafted in four designed buckets rather than sampled from the questions students ask, so "exchangeable with a future row" is a property of a future row drawn *from this eval set's population*, which is not the deployment population. None of that is fixed by choosing a better rule.

## 2. Why split conformal's theorem is not the one

Split conformal's correction is a **rank**. Its guarantee is that a prediction set built at the `⌈(n+1)(1−α)⌉`-th smallest calibration score covers a future label with probability at least `1 − α`, and the `n + 1` is there because the test point is one of `n + 1` exchangeable scores: the index moves by one to account for it.

The `+1` in `(k+1)/(n+1)` is added to a **count of errors** among a selected subset. It is not a quantile of scores, the subset is chosen by the threshold being tested, and the target is conditional on that selection rather than marginal. The formulas resemble each other; the objects do not. So the theorem's conclusion is not available here, however exactly the arithmetic is done — which is what #140's `Fraction` work did and did not settle.

This is true even for a single threshold fixed in advance. There, the selected rows are an honest sample and `k/n` is an honest estimate of `R(t)` — but `(k+1)/(n+1) ≤ α` says *the estimate is under budget*, not *the risk is under budget with confidence `1 − δ`*. There is no `δ` in the committed rule, so there is no probability attached to anything it accepts.

## 3. The two rules, side by side

| rule | accepts when | what it compares | what it controls |
| --- | --- | --- | --- |
| the committed rule (#90) | `(k+1)/(n+1) ≤ α` | a point estimate against the budget | nothing with a stated probability |
| a valid rule (Learn-then-Test) | `P(Bin(n, α) ≤ k) ≤ δ` | a valid p-value against δ | `P(R(t) ≤ α) ≥ 1 − δ`, over a family fixed in advance |

**What a theorem needs**, in the Learn-then-Test shape (Angelopoulos et al.) that `eval/conformal.py` names but does not implement: a p-value per candidate valid for `H_t: R(t) > α`, a family of candidates fixed before the labels are seen, and a procedure that controls the family-wise error rate at `δ`. Fixed-sequence testing is such a procedure, and the committed scan already has its ordering and its stopping rule — so what is missing is not the procedure but the test it is made of.

**One gap this report does not close.** Fixed-sequence testing controls the family-wise error rate when the family is fixed in advance. The candidates below are the observed calibration values, so the family is data-derived. `verbalized` and `self_consistency` have declared scales that would supply a fixed grid; `retrieval_score` is an unbounded BM25 score and has none. On these rows the distinction changes nothing, because no candidate is rejected at any threshold — but it would have to be settled before any number here was called a guarantee.

## 4. The same candidates, under both rules

The ordering, the skip and the stop are the committed scan's, so the two rules are compared candidate for candidate rather than across different families. `testable` is whether a set of that size could be rejected with no errors at all: it reads a count and no label, which is the argument the committed rule makes for its own skip.

**`retrieval_score`** — the valid rule rejects nothing, and never reaches a candidate it could test — all 10 candidates are too small.

| t | answered | wrong | (k+1)/(n+1) | committed accepts? | p-value | 95% upper bound | testable? | valid rule rejects? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `21.9308` | 1 | 0 | 0.500 | no | 0.9000 | 0.950 | no — too small to reject | no |
| `20.9108` | 2 | 0 | 0.333 | no | 0.8100 | 0.776 | no — too small to reject | no |
| `20.3316` | 3 | 1 | 0.500 | no | 0.9720 | 0.865 | no — too small to reject | no |
| `15.0333` | 4 | 1 | 0.400 | no | 0.9477 | 0.751 | no — too small to reject | no |
| `12.7687` | 5 | 1 | 0.333 | no | 0.9185 | 0.657 | no — too small to reject | no |
| `12.7469` | 6 | 2 | 0.429 | no | 0.9841 | 0.729 | no — too small to reject | no |
| `12.0408` | 7 | 3 | 0.500 | no | 0.9973 | 0.775 | no — too small to reject | no |
| `11.9755` | 8 | 4 | 0.556 | no | 0.9996 | 0.807 | no — too small to reject | no |
| `11.0561` | 9 | 4 | 0.500 | no | 0.9991 | 0.749 | no — too small to reject | no |
| `10.8379` | 10 | 5 | 0.545 | no | 0.9999 | 0.778 | no — too small to reject | no |

**`verbalized`** — the valid rule rejects nothing: the first candidate large enough to test answers 33 rows and fails on its errors, which ends the scan.

| t | answered | wrong | (k+1)/(n+1) | committed accepts? | p-value | 95% upper bound | testable? | valid rule rejects? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `100` | 28 | 5 | 0.207 | no | 0.9450 | 0.339 | no — too small to reject | no |
| `90` | 33 | 10 | 0.324 | no | 0.9998 | 0.460 | yes | no |

**`self_consistency`** — the valid rule rejects nothing: the first candidate large enough to test answers 43 rows and fails on its errors, which ends the scan.

| t | answered | wrong | (k+1)/(n+1) | committed accepts? | p-value | 95% upper bound | testable? | valid rule rejects? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | 43 | 17 | 0.409 | no | 1.0000 | 0.532 | yes | no |

## 5. What each rule needs before it can accept anything

| rule | criterion | smallest all-correct set |
| --- | --- | --- |
| the committed rule | `(k+1)/(n+1) ≤ α` | 9 rows |
| a valid rule | upper bound ≤ α at δ = 0.05 | 29 rows |

A valid bound needs 29 all-correct rows before it can clear α at all, against 9 rows for the point estimate. `verbalized`'s only selective cut answers 28 — **one row short**. So under a rule that carries a theorem, the one threshold this signal can offer is not rejected, not even tested: it is skipped as too small, and the scan moves on to a looser cut that fails on its errors.

That makes #140's conclusion stronger rather than weaker. The saturation it found is not an artefact of a lenient criterion — under the stricter and better-founded one, the gap between what the signal can select and what any rule could accept is wider.

## 6. The answer

**What does the committed rule control?** Nothing with a stated probability. It accepts a threshold when a point estimate of the selective risk is under the budget. That is a description of the calibration half, not a guarantee about a future row, and no `δ` appears anywhere in it.

**Does split conformal's theorem apply?** No. Its `n + 1` is a rank correction for a test point among exchangeable scores; this `+1` is added to a count of errors in a selected subset, and the target is conditional on the selection. Both the construction and the functional are different.

**What would?** Learn-then-Test, with a valid p-value per candidate and a `δ`. Run that way on these rows it accepts nothing for any signal, and `verbalized`'s only selective cut is not even large enough to test. The honest description of what is committed is an **abstention gate** fitted by a **threshold rule**, and that is what the reports now say.

## Limitations

**Nothing here changes the gate.** #92's dormant gate, `conformal.alpha` and every file named after `conformal_key` are untouched. The valid rule is reported, never fitted: putting a second rule in the pipeline would rename every committed artifact, and that is a decision this does not make.

**The identifiers keep their names.** `eval/conformal.py`, `conformal_key` and `conformal.alpha` still say *conformal* because renaming them would move every file the study cites. The name is historical; the prose around it is what #142 corrects.

**`δ = 0.05` is a choice, not a result.** It is the level every interval in the study already uses. A different δ moves the smallest testable set, and the report would have to be re-read at that δ — which is the point of its being named rather than buried.

**The family is data-derived.** Section 3 says why that matters and why it changes nothing on these rows. It would have to be settled before any number here was called a guarantee.

**The assumption in section 1 is not verified.** Independence and exchangeability with a future row are assumed, and section 1 lists three reasons they are doubtful here. A valid rule over invalid assumptions is not valid; it is only honest about its arithmetic.

## Reproduction

```bash
uv run python -m eval.scripts.validity            # rewrite this file
uv run python -m eval.scripts.validity --check    # list the inputs; write nothing
```

Two committed files, no model call, no resample, and the `test` half is not read.
