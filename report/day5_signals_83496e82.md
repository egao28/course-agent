# Day 5: confidence signals and conformal gating

Config hash `c73b197d` · signals `37164723` · correctness `c54b1377` · conformal `83496e82` · full hash `65bb9643` · dataset sha256 `2a93182221bd`

On the test half (42 answered rows), `verbalized`, the signal chosen on calib, ranks correct above incorrect answers with AUROC 0.847 [0.722, 0.953]. The highest test AUROC is `retrieval_score`'s (1.000, on 10 non-null rows).
The calib fit accepted no threshold for any signal at alpha = 0.10, so the conformal gate answers nothing on test; a fixed 0.7 answers 42 of 42 at 0.405, against 0.405 for answering everything.

## Inputs

Every file below matched its sidecar, the file downstream of it, this config's keys and the dataset before a word of this report was written.

| input | path | sha256 |
| --- | --- | --- |
| dataset | `eval/datasets/v1.jsonl` | `2a93182221bd` |
| calib signals run | `eval/runs/signals_calib_37164723.jsonl` | `cd29fdf7e551` |
| calib judge run | `eval/runs/judge_calib_c54b1377.jsonl` | `469b39b4fb36` |
| calib signals CSV | `eval/runs/signals_calib_37164723.csv` | `2e6ef2df42ef` |
| test signals run | `eval/runs/signals_test_37164723.jsonl` | `abd246c1716f` |
| test judge run | `eval/runs/judge_test_c54b1377.jsonl` | `b6a949d0ecc7` |
| test signals CSV | `eval/runs/signals_test_37164723.csv` | `8db35639f2ba` |
| review log | `eval/runs/correctness_log.jsonl` | `66e404dd987d` |
| risk-coverage numbers | `eval/runs/risk_coverage_test_37164723.json` | `875e469b6160` |
| risk-coverage figure | `report/figures/risk_coverage_test_37164723.png` | `b1ac83a34f39` |
| conformal fit | `eval/runs/conformal_83496e82.json` | `27987e369aaa` |
| conformal test reading | `eval/runs/conformal_eval_83496e82.json` | `10fe779c58c5` |

## 1. Setup

| bucket | calib rows | calib answered | test rows | test answered |
| --- | --- | --- | --- | --- |
| fact | 29 | 25 | 29 | 21 |
| multihop | 12 | 12 | 12 | 12 |
| unanswerable | 13 | 4 | 12 | 3 |
| ambiguous | 10 | 5 | 10 | 6 |
| **total** | 64 | 46 | 63 | 42 |

*Rows* is the half of the frozen dataset; *answered* is what the signals run collected and #88 labelled. Skipped before an answer existed: calib: 15 no_evidence, 3 out_of_scope; test: 15 no_evidence, 6 out_of_scope. `no_evidence` rows have nothing for the answer node to cite, and `out_of_scope` rows never reach dispatch.

**The evidence gate was bypassed** (#87); the run's sidecar says: "bypassed: every row with evidence was answered; gate_ok is what the Day 2 top_score gate would have said". Rows the Day 2 `top_score` gate would have refused, and that were answered anyway: calib: 4 ambiguous, 2 unanswerable; test: 5 ambiguous.

**Nulls per signal.** calib: `retrieval_score` 36, `verbalized` 0, `self_consistency` 0 (of 46; 36 tool rows); test: `retrieval_score` 32, `verbalized` 0, `self_consistency` 0 (of 42; 32 tool rows). `retrieval_score` is null on every tool row (an exact catalog lookup has no retrieval score), never 1.0. Unparsed replies: calib 0 verbalized / 0 judge, test 0 verbalized / 0 judge.

**Model and sampling.** `gpt-4o-mini` for every call, retrieval `bm25`; self-consistency draws 3 samples at temperature 0.7 and judges all 3 pairs. Prompt sha256: answer_v1 `6de696d3d087`, judge `8fc5f97ebc57`, router_v2 `3314cfc1f5e9`, verbalized `f2545d66bcf4`.

**Cost.** Each answered row took 9 chat calls (route, answer, 1 verbalized, 3 samples, 3 pair judgements) and one embedding call, and #88's judge one more: 88 answered rows across both halves are 880 chat calls. That counts the committed runs only, not the router calls on skipped rows or any run these replaced. The runs did not record token counts, so there is no dollar figure here.

## 2. Labels

`correct` is a person's verdict on every row; the LLM judge's call was shown first (#88). Judge × human, with the human verdict second:

| half | n | judge ✓ human ✓ | judge ✓ human ✗ | judge ✗ human ✓ | judge ✗ human ✗ | judge unparsed | agreement | overturned | carried over | reviewer (rows) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calib | 46 | 26 | 0 | 0 | 20 | 0 | 46 / 46 (100.0%) | 0 | 37 | eve (46) |
| test | 42 | 25 | 0 | 0 | 17 | 0 | 42 / 42 (100.0%) | 0 | 0 | eve (42) |

*Carried over* counts verdicts matched to an earlier signals run on (id, answer_sha256): the answer text is byte-identical, so the person's earlier verdict is about the same answer.

Correct / answered per bucket:

| bucket | calib | test |
| --- | --- | --- |
| fact | 23 / 25 | 21 / 21 |
| multihop | 0 / 12 | 2 / 12 |
| unanswerable | 3 / 4 | 2 / 3 |
| ambiguous | 0 / 5 | 0 / 6 |
| **total** | 26 / 46 | 25 / 42 |

## 3. RQ2: which signal ranks answers best

![Risk-coverage curves, test half](figures/risk_coverage_test_37164723.png)

Written by `eval/scripts/risk_coverage.py` (#89) from the test CSV; its table, as it prints it:

test n = 42 {'ambiguous': 6, 'fact': 21, 'multihop': 12, 'unanswerable': 3}, B = 10000, seed 76, 95% paired percentile; answer-everything risk 0.405

| signal | n | null | AUROC [95% CI] | skipped resamples | AURC |
|---|---|---|---|---|---|
| retrieval_score | 10 | 32 | 1.000 [1.000, 1.000] | 454 | 0.371 |
| verbalized | 42 | 0 | 0.847 [0.722, 0.953] | 0 | 0.219 |
| self_consistency | 42 | 0 | 0.652 [0.536, 0.778] | 0 | 0.327 |

| AUROC difference | point [CI] | skipped | excludes 0 |
|---|---|---|---|
| retrieval_score - self_consistency | +0.348 [+0.222, +0.464] | 454 | yes |
| retrieval_score - verbalized | +0.153 [+0.048, +0.278] | 454 | yes |
| verbalized - self_consistency | +0.195 [+0.035, +0.346] | 0 | yes |

- `retrieval_score` is ranked on its 10 non-null rows of 42 (3 correct, 7 incorrect); 454 of 10000 resamples drew one class or none of those rows and were skipped.
- `retrieval_score`'s interval has zero width because every resample that could be scored gave 1.000: the rows are perfectly separated, which is what a sample of 10 can do. It is not a measured certainty.
- `retrieval_score - self_consistency` shares the resampled questions but not the rows: each AUROC ranks its own non-null rows (10 vs 42), so *excludes 0* here does not say one signal orders the same answers better than the other.
- `retrieval_score - verbalized` shares the resampled questions but not the rows: each AUROC ranks its own non-null rows (10 vs 42), so *excludes 0* here does not say one signal orders the same answers better than the other.
- Calib AUROC, the one #90 chose on: `retrieval_score` 0.760, `verbalized` 0.792, `self_consistency` 0.575. The test ordering differs: `retrieval_score`, `verbalized`, `self_consistency` on test against `verbalized`, `retrieval_score`, `self_consistency` on calib. The choice (highest calib AUROC; ties to more non-null rows, then to SIGNALS order) was made before the test labels were read.

## 4. RQ3: conformal against a fixed 0.7

Written by `eval/scripts/conformal_eval.py` (#91), which applied the calib fit to the test half once; its tables, as it prints them:

test n = 42 {'ambiguous': 6, 'fact': 21, 'multihop': 12, 'unanswerable': 3}, alpha = 0.10, 95% Clopper-Pearson; signal chosen on calib: verbalized

| method | t | target error | realized error [95% CI] | coverage (answered / n) |
|---|---|---|---|---|
| conformal (calib-fit t) | +inf | 0.10 | - (nothing answered) | 0 / 42 (0.0%) |
| fixed t = 0.7 | 70 | - (no guarantee) | 0.405 [0.256, 0.567] (17 wrong) | 42 / 42 (100.0%) |
| answer everything (reference) | - | - | 0.405 [0.256, 0.567] (17 wrong) | 42 / 42 (100.0%) |

Appendix: every signal x both methods

| signal | method | t | realized error [95% CI] | coverage | null | test range | what the threshold means |
|---|---|---|---|---|---|---|---|
| retrieval_score | conformal | +inf | - (nothing answered) | 0 / 10 (0.0%) | 32 | 10.6679-30.2896 | calib fit at alpha = 0.10 |
| retrieval_score | fixed | 0.7 | 0.700 [0.348, 0.933] (7 wrong) | 10 / 10 (100.0%) | 32 | 10.6679-30.2896 | raw BM25 top-1 >= 0.7 (unbounded score, not normalized) |
| verbalized | conformal | +inf | - (nothing answered) | 0 / 42 (0.0%) | 0 | 70-100 | calib fit at alpha = 0.10 |
| verbalized | fixed | 70 | 0.405 [0.256, 0.567] (17 wrong) | 42 / 42 (100.0%) | 0 | 70-100 | verbalized / 100 >= 0.7, i.e. verbalized >= 70 |
| self_consistency | conformal | +inf | - (nothing answered) | 0 / 42 (0.0%) | 0 | 0.333333-1 | calib fit at alpha = 0.10 |
| self_consistency | fixed | 0.7 | 0.314 [0.169, 0.493] (11 wrong) | 35 / 42 (83.3%) | 0 | 0.333333-1 | between 2/3 and 1: all 3 pairs agree |

dense_top1 (diagnostic, not a signal): max 0.740 on 42 rows; a raw 0.7 would answer 4

**Why the conformal threshold is where it is.** A threshold is accepted on calib when (k + 1) / (n + 1) <= alpha = 0.10 for the rows it answers, so nothing is accepted on fewer than 9 answered rows. The lowest value of (k + 1) / (n + 1) any threshold reached on calib: `retrieval_score` 0.333, `verbalized` 0.207, `self_consistency` 0.409.

**What 0.7 means on each scale.**

- `retrieval_score`: raw BM25 top-1 >= 0.7 (unbounded score, not normalized). Its test values run 10.6679 to 30.2896, so the threshold answers 10 of 10 -- every non-null row, the same rows as answering everything.
- `verbalized`: verbalized / 100 >= 0.7, i.e. verbalized >= 70. Its test values run 70 to 100, so the threshold answers 42 of 42 -- every non-null row, the same rows as answering everything.
- `self_consistency`: between 2/3 and 1: all 3 pairs agree. Its test values run 0.333333 to 1, so the threshold answers 35 of 42.

## 5. Wiring

**The agent runs `evidence.method: top_score`.** Its 6 nodes: `route_intent`, `dispatch`, `check_evidence`, `answer`, `out_of_scope`, `refuse_handoff`. The conformal gate (#92) is built and configured but dormant: `evidence.conformal` names `eval/runs/conformal_83496e82.json` by sha256, signal `verbalized`, threshold +inf (answer nothing), alpha 0.10, fitted on calib (46 rows).

Switched to `conformal`, the config hash moves from `c73b197d` to `2d44be6a`, the graph gains `confidence_gate`, the comparison happens after `answer`, and it adds 1 chat call per answered query. The switched gate was built while writing this report, so its pinned sha256, alpha, split, threshold and key chain all still match the fit.

It is not switched on because the fit's threshold would hand every question that reaches evidence to the refusal exit.

## 6. Limitations

- **Test n.** 42 answered rows of the half's 63. Answering everything is wrong on 17 of them, 0.405 with a 95% Clopper-Pearson interval of [0.256, 0.567]: a width of 0.311, which no difference between two gates on this half can be read inside of.
- **Live samples do not reproduce.** The self-consistency samples are drawn at temperature 0.7 and the dense side of a live run jitters, so a rerun of #87 is a new draw, not a check of this one. The committed run files are the sample, and nothing downstream re-queries.
- **This rule carries no guarantee.** It accepts a threshold when a plug-in estimate of the error among the rows it would answer is at most alpha = 0.10 -- an estimate against a budget, with no confidence level in it, so split conformal's theorem does not transfer to it (#142). With calib n = 46 the scan cannot accept a threshold that answers fewer than 9 rows, whatever the signal.
- **`self_consistency` is heavily tied.** It takes 3 values on test: 1.000 on 35 rows, 0.667 on 4 rows, 0.333 on 3 rows. Tied rows are answered or skipped together, so its curve has 3 points and no threshold reaches the coverages between them. `verbalized` has its own mass point: 26 of 42 test rows say 100.
- **One reviewer, and the judge was never overturned.** 88 labels, by eve, with the judge's call shown first, and 0 overturned; 37 calib verdicts were carried over from an earlier run's identical answers. There is no second annotator on Day 5's labels, so the labels are the judge's as accepted by one person, and agreement between two people is unmeasured.
- **Buckets with no correct answer:** calib `multihop`, `ambiguous`; test `ambiguous`. A signal sees those rows only as wrong answers to push below the right ones; there is no correct answer in them for it to find, and a fit made on such a half learns nothing about the bucket except how much of it to skip.
- **`retrieval_score` is null on 32 of 42 test rows** (the tool rows), so its curve and AUROC describe the retrieval rows only, and the gate on it would pass tool rows without a comparison.
- **The gate was bypassed to collect the signals**, so every label is about an answer the Day 2 gate might not have allowed. The numbers describe the answer node's output, not the deployed agent's replies.

## Reproduction

```bash
uv run python -m eval.scripts.day5_report
```

This file is a function of the files in the Inputs table and of `config.yaml`, which names it.
There is no timestamp in it, so a re-run over unchanged inputs rewrites it byte for byte. The
generator refuses to write if any sha256 in that chain has moved.
