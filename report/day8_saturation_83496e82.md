# Day 8 — why the abstention gate selects no answers

RQ3 reported that no threshold is accepted at α = 0.10, and stopped at *not measurable at this calibration size*. That sentence names the sample size as the cause without checking that it is the cause. Three things could each be it — the signal's resolution, the errors in the answers a threshold would release, and the finite-sample correction — and they have different fixes, so this file says how much of the refusal each one owns.

Nothing here is re-run and no model is called: every number is a function of the two files below, and the fit they describe is re-derived and compared before a line is written. The `test` half is not read.

| | |
| --- | --- |
| half | `calib`, 46 rows |
| α | `0.10` (`conformal.alpha`), key `83496e82` |
| smallest answerable set | 9 rows, from `ceil(1/α − 1)` |
| α floor at this n | `1/47` = 0.0213 |
| calib signals CSV | `eval/runs/signals_calib_37164723.csv` sha256 `2e6ef2df42ef` |
| conformal fit | `eval/runs/conformal_83496e82.json` sha256 `27987e369aaa` |

## 1. Saturation: which answer sets a signal can offer

A scan can only cut where the signal changes value, so the answer sets it may choose from are fixed before a single label is read. Tied rows are answered together.

| signal | rows | null | levels | smallest cut | sets it can answer | sets ≥ 9 it cannot |
| --- | --- | --- | --- | --- | --- | --- |
| `retrieval_score` | 10 | 36 | 10 | 1 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | — |
| `verbalized` | 46 | 0 | 8 | 28 | 28, 33, 35, 38, 42, 43, 44, 46 | 9–45 |
| `self_consistency` | 46 | 0 | 4 | 43 | 43, 44, 45, 46 | 9–42 |

`verbalized`, the signal the fit chose, takes 8 distinct values over its 46 rows and puts 28 of them on the single highest one. Its strictest possible answer set is therefore 28 rows — no threshold anywhere answers fewer. The correction would accept a clean set of 9 rows; this signal cannot produce one, and the sets it cannot produce are the last column.

## 2. The cuts, and what stops each one

Every distinct value as a candidate, strictest first. This is the whole curve rather than the prefix the rule tests — `eval.conformal.scan` stops at the first tested failure, and a diagnosis has to see past it. **Reading a threshold off this table would be the global selection the fixed-sequence rule refuses**: the `+1` does not cover a choice made over every candidate.

**`retrieval_score`** — 10 rows, 36 null.

| t | answered | coverage | wrong | raw error [95% CI] | (k+1)/(n+1) | accepted? |
| --- | --- | --- | --- | --- | --- | --- |
| `21.9308` | 1 | 10.0% | 0 | 0.000 [0.000, 0.975] | 0.500 | correction (too few rows to pass with k = 0) |
| `20.9108` | 2 | 20.0% | 0 | 0.000 [0.000, 0.842] | 0.333 | correction (too few rows to pass with k = 0) |
| `20.3316` | 3 | 30.0% | 1 | 0.333 [0.008, 0.906] | 0.500 | correction (too few rows to pass with k = 0) |
| `15.0333` | 4 | 40.0% | 1 | 0.250 [0.006, 0.806] | 0.400 | correction (too few rows to pass with k = 0) |
| `12.7687` | 5 | 50.0% | 1 | 0.200 [0.005, 0.716] | 0.333 | correction (too few rows to pass with k = 0) |
| `12.7469` | 6 | 60.0% | 2 | 0.333 [0.043, 0.777] | 0.429 | correction (too few rows to pass with k = 0) |
| `12.0408` | 7 | 70.0% | 3 | 0.429 [0.099, 0.816] | 0.500 | correction (too few rows to pass with k = 0) |
| `11.9755` | 8 | 80.0% | 4 | 0.500 [0.157, 0.843] | 0.556 | correction (too few rows to pass with k = 0) |
| `11.0561` | 9 | 90.0% | 4 | 0.444 [0.137, 0.788] | 0.500 | observed errors (raw rate alone is over α) |
| `10.8379` | 10 | 100.0% | 5 | 0.500 [0.187, 0.813] | 0.545 | observed errors (raw rate alone is over α) |

**`verbalized`** — 46 rows, 0 null.

| t | answered | coverage | wrong | raw error [95% CI] | (k+1)/(n+1) | accepted? |
| --- | --- | --- | --- | --- | --- | --- |
| `100` | 28 | 60.9% | 5 | 0.179 [0.061, 0.369] | 0.207 | observed errors (raw rate alone is over α) |
| `90` | 33 | 71.7% | 10 | 0.303 [0.156, 0.487] | 0.324 | observed errors (raw rate alone is over α) |
| `85` | 35 | 76.1% | 12 | 0.343 [0.191, 0.522] | 0.361 | observed errors (raw rate alone is over α) |
| `80` | 38 | 82.6% | 14 | 0.368 [0.218, 0.540] | 0.385 | observed errors (raw rate alone is over α) |
| `70` | 42 | 91.3% | 18 | 0.429 [0.277, 0.590] | 0.442 | observed errors (raw rate alone is over α) |
| `40` | 43 | 93.5% | 18 | 0.419 [0.270, 0.579] | 0.432 | observed errors (raw rate alone is over α) |
| `20` | 44 | 95.7% | 19 | 0.432 [0.283, 0.590] | 0.444 | observed errors (raw rate alone is over α) |
| `0` | 46 | 100.0% | 20 | 0.435 [0.289, 0.589] | 0.447 | observed errors (raw rate alone is over α) |

**`self_consistency`** — 46 rows, 0 null.

| t | answered | coverage | wrong | raw error [95% CI] | (k+1)/(n+1) | accepted? |
| --- | --- | --- | --- | --- | --- | --- |
| `1` | 43 | 93.5% | 17 | 0.395 [0.250, 0.556] | 0.409 | observed errors (raw rate alone is over α) |
| `0.666667` | 44 | 95.7% | 18 | 0.409 [0.263, 0.568] | 0.422 | observed errors (raw rate alone is over α) |
| `0.333333` | 45 | 97.8% | 19 | 0.422 [0.277, 0.578] | 0.435 | observed errors (raw rate alone is over α) |
| `0` | 46 | 100.0% | 20 | 0.435 [0.289, 0.589] | 0.447 | observed errors (raw rate alone is over α) |

## 3. Of the distance to α, how much is the errors and how much is the correction

`(k+1)/(n+1) − α` splits exactly into `k/n − α`, which is the answers being wrong, and `(k+1)/(n+1) − k/n`, which is what the correction adds at this `n`. A cut whose second column is negative and whose third is larger is one the correction alone refuses; a cut whose second column is positive is one the model refuses.

| signal | t | answered | total gap | from the errors | from the +1 |
| --- | --- | --- | --- | --- | --- |
| `retrieval_score` | `21.9308` | 1 | +0.400 | −0.100 | +0.500 |
| `retrieval_score` | `20.9108` | 2 | +0.233 | −0.100 | +0.333 |
| `retrieval_score` | `20.3316` | 3 | +0.400 | +0.233 | +0.167 |
| `retrieval_score` | `15.0333` | 4 | +0.300 | +0.150 | +0.150 |
| `retrieval_score` | `12.7687` | 5 | +0.233 | +0.100 | +0.133 |
| `retrieval_score` | `12.7469` | 6 | +0.329 | +0.233 | +0.095 |
| `retrieval_score` | `12.0408` | 7 | +0.400 | +0.329 | +0.071 |
| `retrieval_score` | `11.9755` | 8 | +0.456 | +0.400 | +0.056 |
| `retrieval_score` | `11.0561` | 9 | +0.400 | +0.344 | +0.056 |
| `retrieval_score` | `10.8379` | 10 | +0.445 | +0.400 | +0.045 |
| `verbalized` | `100` | 28 | +0.107 | +0.079 | +0.028 |
| `verbalized` | `90` | 33 | +0.224 | +0.203 | +0.020 |
| `verbalized` | `85` | 35 | +0.261 | +0.243 | +0.018 |
| `verbalized` | `80` | 38 | +0.285 | +0.268 | +0.016 |
| `verbalized` | `70` | 42 | +0.342 | +0.329 | +0.013 |
| `verbalized` | `40` | 43 | +0.332 | +0.319 | +0.013 |
| `verbalized` | `20` | 44 | +0.344 | +0.332 | +0.013 |
| `verbalized` | `0` | 46 | +0.347 | +0.335 | +0.012 |
| `self_consistency` | `1` | 43 | +0.309 | +0.295 | +0.014 |
| `self_consistency` | `0.666667` | 44 | +0.322 | +0.309 | +0.013 |
| `self_consistency` | `0.333333` | 45 | +0.335 | +0.322 | +0.013 |
| `self_consistency` | `0` | 46 | +0.347 | +0.335 | +0.012 |

## 4. The smallest α that would accept anything

Swept over the only values where acceptability can change — each cut's own `(k+1)/(n+1)` — and replayed through the real fixed-sequence rule rather than reasoned about. Acceptance is **not** monotone in α: raising it makes a smaller, stricter cut testable, and that cut can fail and stop the scan before a larger one that would have passed.

| signal | smallest α that accepts | t there | what it would answer |
| --- | --- | --- | --- |
| `retrieval_score` | 0.333 (`1/3`) | `20.9108` | 2 of 10 (0 wrong) |
| `verbalized` | 0.207 (`6/29`) | `100` | 28 of 46 (5 wrong) |
| `self_consistency` | 0.409 (`9/22`) | `1` | 43 of 46 (17 wrong) |

## 5. How many rows would it take?

If the true error rate at `verbalized`'s strictest cut were `p`, the correction stops binding at `n ≥ (1 − α) / (α − p)`. It is not a power calculation: it treats `p` as exact and ignores that `k` is an integer with variance, so it is a floor on what would be needed rather than a size to plan with. Where `p ≥ α` no number of rows is enough, and that is the difference between *the sample was small* and *the answers were wrong*.

| true error at the top cut | rows needed at α = 0.10 | the cut has 28 — enough? |
| --- | --- | --- |
| 0.02 | 12 answered rows | **yes** |
| 0.05 | 18 answered rows | **yes** |
| 0.08 | 45 answered rows | no |
| 0.10 | — (no n is enough) | — |
| 0.179 — observed | — (no n is enough) | — |
| 0.061 — bottom of its interval | 23 answered rows | **yes** |
| 0.369 — top of its interval | — (no n is enough) | — |

**What refuses this cut is the wrong answers, not the row count.** Acceptance at 28 rows needs at most 1 wrong; there are 5. The correction's own requirement is already met at the bottom of the interval — 23 rows, and the cut answers 28 — though not everywhere under α: 28 rows satisfy it for a true rate up to 0.068, and between that and α it would want more.

So *would more calibration data have fixed this?* — not by being more data. The observed 5 of 28 = 0.179 [0.061, 0.369] reaches below α and above it, so a larger half would re-estimate that rate rather than relax the requirement, and whether it then accepted would depend entirely on where the true rate actually is. The quantity to move is the error rate among the model's most confident answers; the number of rows only decides how precisely it is known.

## 6. The answer

**Does saturation block it?** Yes, and first: `verbalized` offers no answer set smaller than 28 rows, so the rule never gets to choose a small clean one. The 9 rows the correction would accept is a set this signal cannot make.

**Do the observed errors block it?** Yes, and decisively: at that strictest cut 5 of 28 answers are wrong, 0.179 [0.061, 0.369], where acceptance needs at most 1. The raw rate alone is +0.079 past α before any correction is applied.

**Does the finite-sample correction block it?** Not here, and not for want of rows. It adds +0.028 at n = 28, which is 26.5% of the gap; removing it entirely would still not accept this cut, and at the bottom of that cut's interval its row count already satisfies the correction. What the correction forbids is the small sets — and this signal has none to offer, so on this fit it decides nothing.

## Limitations

**One half, 46 rows.** Every rate here is over the calibration half, and the interval on the number that matters most is wide enough to contain both answers to the question section 5 asks.

**A diagnosis is not a fix.** Nothing here changes the signal, the prompt, α or the gate; it says which of the three a fix would have to address. A finer-grained signal is what it points at, and that is not measured here.

**The cut table is not a menu.** Section 2 prints every candidate because a diagnostic may look where a rule may not. Choosing a threshold from it would be selection over every candidate, which the `+1` does not cover.

**The labels are the ones the fit used.** `correct` is one person's verdict with the judge's call displayed first, as Day 5's report records, and an error rate of 0.179 at the top cut inherits that.

## Reproduction

```bash
uv run python -m eval.scripts.saturation            # rewrite this file
uv run python -m eval.scripts.saturation --check    # list the inputs; write nothing
```

No key and no network: it reads the two committed files in the header, and writes the same bytes over unchanged inputs.
