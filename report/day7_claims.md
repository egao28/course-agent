# almabot — every claim, and the file behind it

One row per claim this study can support: the claim, the file and key path every number in it
comes from, the rows it rests on, and what it does not say. The file exists so that a claim made
out loud — on a resume line, in an interview, in a README somewhere else — can be answered with a
file rather than with a memory.

**The numbers here are not typed in.** `eval/scripts/final_report.py` formats every one of them
from the committed run files, and `--check` then reads each one back out of the file its own row
names, at the precision it is printed to. A key path is a path into a JSON file, a column of a
JSONL file, or a heading of a Markdown one. Editing a number in a table below fails CI; moving it
in the run file and regenerating is the only way to change one.

**The claims are quotations, and the bullets are not written here.** Each row states the
measurement plainly. A resume line is a different register and it is not this generator's to
write, so the bullets are left to the person whose claims they are — either in the section below,
where `--check` holds every number in them to this same table, or in private notes, in which case
this file is the evidence table it was commissioned as and the next section stays empty. The rule
is the same either way: a number on the line carries the `[src]` link of the row it quotes, and
where the last column is not empty, the caveat travels with the claim.

## The bullets

<!--
One bullet per claim, in your words, naming the row it quotes (`C1` … `C9`). Every number in a
bullet needs that row's own [src] link beside it: the prose here is read exactly as the prose in
report/README.md is read, so a number with no link, or one its linked file does not have, fails
CI rather than being found by whoever asks about it.
-->

_Not written yet._

## The claims

<!-- BEGIN gen:claims -->
| the claim | where every number in it comes from | what it rests on | what it does not say |
| --- | --- | --- | --- |
| **C1** — A hand-verified evaluation set, 127 questions in four buckets, split 64 calibration / 63 test and frozen before anything was measured, with a second annotator on a sample of it. | `eval/datasets/v1.jsonl` → `split` [src](../eval/datasets/v1.jsonl "split") | 127 rows; counts rather than estimates, so no interval. Digest `2a93182221bd`. | Not a double-annotated set. The second annotator read a fifteen-row sample, reported it verbally for transcription rather than filling the sheet in, and κ on `answer_correct` is zero there by construction — the percent agreement is the number to read ([Day 3's agreement report](day3_agreement_c73b197d.md)). The questions were drafted from the catalog by the people who built the system. |
| **C2** — A tuned hybrid retriever did not beat `bm25` on a code-heavy catalog: Recall@5 hybrid − bm25 = −0.004 [−0.085, 0.073], MRR@5 = −0.120 [−0.225, −0.023] — the first interval containing zero, the second excluding it *against* hybrid. | `eval/runs/bootstrap_baa560e9.json` → `metrics.recall_at_5.difference`, `metrics.mrr_at_5.difference`, `n.test`, `n_resamples` [src](../eval/runs/bootstrap_baa560e9.json "metrics.recall_at_5.difference, metrics.mrr_at_5.difference, n.test, n_resamples") | n = 41 test rows, paired bootstrap over 10,000 resamples, 95% percentile intervals. | Not an improvement in either direction that a reader should hear as one. This is why the agent ships `bm25`, and a cross-encoder rerank of hybrid's pool separates from neither metric ([Day 4's report](day4_ablation_c73b197d.md)). One corpus, where course codes make a lexical index hard to beat. |
| **C3** — The model's verbalized self-confidence ranks correct answers above wrong ones: test AUROC 0.847 [0.722, 0.953], for the signal chosen on the calibration half at 0.792. | `eval/runs/risk_coverage_test_37164723.json` → `signals.verbalized.auroc`, `n`, `n_resamples` [src](../eval/runs/risk_coverage_test_37164723.json "signals.verbalized.auroc, n, n_resamples") · `eval/runs/conformal_83496e82.json` → `signals.verbalized.auroc` [src](../eval/runs/conformal_83496e82.json "signals.verbalized.auroc") | n = 42 test rows, 10,000 bootstrap resamples, 95% percentile interval; the test half was read once. | Ranking, not calibration: AUROC says nothing about whether a stated confidence matches an observed rate, and the gate fitted on this signal accepts no threshold at all (C4). These rows were collected with the evidence gate bypassed, so they describe answers the deployed agent might never have released. |
| **C4** — Conformal calibration at α = 0.10 accepts no threshold on a calibration half of 46 rows, so the fitted gate answers nothing; the fixed 0.7 threshold it was built to replace answers 42 of 42 at a realized error of 0.405 [0.256, 0.567]. | `eval/runs/conformal_83496e82.json` → `alpha`, `n`, `min_answered_to_accept` [src](../eval/runs/conformal_83496e82.json "alpha, n, min_answered_to_accept") · `eval/runs/conformal_eval_83496e82.json` → `fixed`, `n`, `signals.verbalized.fixed` [src](../eval/runs/conformal_eval_83496e82.json "fixed, n, signals.verbalized.fixed") | Calibration n = 46 for the fit, by a fixed-sequence scan; n = 42 test rows for the reading, Clopper-Pearson 95%. | Not a reduction in anything. The conformal gate is built and dormant (#92) and the live config still runs the Day 2 `top_score` rule. The guarantee is marginal over calibration draws rather than conditional on this half, and at this size the scan cannot accept a threshold that answers fewer than 9 rows — so the finding is "not measurable here", not "conformal does not work". |
| **C5** — The intent router is right on 59 of 63 test rows, 0.937 [0.845, 0.982]. | `eval/runs/e2e_metrics_c73b197d.json` → `metrics.intent_accuracy` [src](../eval/runs/e2e_metrics_c73b197d.json "metrics.intent_accuracy") | 59 / 63, Clopper-Pearson 95%, on one end-to-end run. | The most forgiving number in the study: the gold label is a set per row, and the labels were written after the run existed, so nothing in the process proves the labeller had not seen the router's output. One of the five intents, `degree_requirement`, has no test row at all. |
| **C6** — 43 of 63 test rows end in the outcome their bucket calls correct, 0.683 [0.553, 0.794] — or 34 of 63, 0.540 [0.409, 0.666], if the 9 ambiguous hand-offs nobody read do not count. | `eval/runs/e2e_metrics_c73b197d.json` → `metrics.answer_correctness`, `caveats.ambiguous_handoff` [src](../eval/runs/e2e_metrics_c73b197d.json "metrics.answer_correctness, caveats.ambiguous_handoff") | 43 / 63 and 34 / 63, Clopper-Pearson 95%, on one run read once. A correct outcome is an answer a person judged right, a decline on an unanswerable row, or a clarifying question on an ambiguous one. | Three readings, not one: exit-only is 41 / 63 (0.651 [0.520, 0.767]), and 43 / 63 is the upper bound of the range rather than the number — the report asks for it never to be quoted alone. Correctness is one person's verdict and the judge's call was displayed before they made it, so the agreement between them is not independent confirmation. |
| **C7** — False release 2 of 22, 0.091 [0.011, 0.292]; false block 8 of 41, 0.195 [0.088, 0.349]. | `eval/runs/e2e_metrics_c73b197d.json` → `metrics.false_release`, `metrics.false_block` [src](../eval/runs/e2e_metrics_c73b197d.json "metrics.false_release, metrics.false_block") | 2 / 22 rows that should not have been answered and 8 / 41 that should have been, Clopper-Pearson 95%. | Read two ways, because the graph has no clarify exit: a reply that declines without asking the question counts as a release on the exit-only reading, which is 4 / 22 (0.182 [0.052, 0.403]). The false-block rate is what the gate costs — 8 answerable questions were handed to an advisor instead. |
| **C8** — Every one of the 20 failures carries a cause, assigned by a fixed rule order and confirmed by a reviewer: 10 retrieval, 8 evidence misread, 2 overconfident, 0 intent, 0 other. | `eval/runs/attribution_test_c73b197d.meta.json` → `by_class`, `actions`, `n_rows`, `n_failures` [src](../eval/runs/attribution_test_c73b197d.meta.json "by_class, actions, n_rows, n_failures") | 20 failure rows out of 63; a classification, so no interval. All 20 confirmed, none overridden. | Agreement with a rule, not a second opinion: the reviewer saw the rule's proposal on every row. The order charges each row to a single cause, so a row with two causes shows only the first, and `retrieval` leading is partly a property of that order. |
| **C9** — One university catalog, one term: 4,502 courses and 2,086 prerequisite edges, re-exported from the catalog CSV and pinned by digest. | `README.md` → `Data` [src](../README.md "Data") | Counts over the committed export rather than estimates; the input CSV is pinned by `data.source_sha256` and checked on every run. | One institution and one term (`2026-sp`), so "the catalog cannot answer this" is a property of this export rather than of the world. The edges are a reading of catalog prose, with the known failures written down ([`data/README.md`](../data/README.md)). Nothing was deployed, and no question here came from a real student. |
<!-- END gen:claims -->

## What may not go on the line

Each of these is a line this work could plausibly be described with, and each one is refused by a
number rather than by taste.

<!-- BEGIN gen:claims_refused -->
| the line | what the committed numbers say instead | where that is |
| --- | --- | --- |
| *Improved retrieval by X%* | RQ1 found no gap on Recall@5 (−0.004 [−0.085, 0.073]) and a gap *against* hybrid on MRR@5 (−0.120 [−0.225, −0.023]). The finding is that a tuned hybrid did not beat `bm25` on this corpus, which is the stronger line because it is the true one. | `eval/runs/bootstrap_baa560e9.json` → `metrics.recall_at_5.difference`, `metrics.mrr_at_5.difference` [src](../eval/runs/bootstrap_baa560e9.json "metrics.recall_at_5.difference, metrics.mrr_at_5.difference") |
| *Deployed / in production / served N users / p99 latency* | Nothing was deployed. No user has asked this agent anything; there is no load, no cost per session and no log of what anyone asks. The only timing in the repo is a cross-encoder's median per question on one laptop cpu ([Day 4's report](day4_ablation_c73b197d.md)), measured once, and a clean machine measured it differently (#118). | — |
| *Calibrated confidence gating reduced hallucination* | The calibrated gate answers nothing at α = 0.10 on 46 calibration rows, and the live config still runs `top_score`. What the calibration produced is a measurement of the calibration half, not a shipped improvement. | `eval/runs/conformal_83496e82.json` → `alpha`, `n` [src](../eval/runs/conformal_83496e82.json "alpha, n") |
| *A rate without its n, or a point estimate with the interval dropped* | Every rate above is printed as k / n with a 95% interval. At these sample sizes the interval *is* the finding: several of the comparisons have intervals wide enough to contain the opposite conclusion, and dropping one turns a measurement into an assertion. | — |
| *43 of 63 correct, with the lower reading left out* | Both readings or neither: 34 of 63 (0.540 [0.409, 0.666]) is the same run with the 9 unread ambiguous hand-offs counted wrong. Which one is right is a judgement about what a hand-off that declines without asking is worth, and that judgement belongs beside the number rather than behind it. | `eval/runs/e2e_metrics_c73b197d.json` → `metrics.answer_correctness`, `caveats.ambiguous_handoff` [src](../eval/runs/e2e_metrics_c73b197d.json "metrics.answer_correctness, caveats.ambiguous_handoff") |
<!-- END gen:claims_refused -->

## Checking this file

```bash
uv run python -m eval.scripts.final_report            # rewrite the tables above
uv run python -m eval.scripts.final_report --check    # read every number back out of its file
uv run python -m eval.scripts.final_report --inputs   # list every file behind them
```

The second is what CI runs on every push, over this page and the two it shares a generator with.
What it cannot do is read: it holds a row to the files the row cites, and whether the sentence
around a number is a fair reading of them is a person's judgement. That judgement is the last
column, and its longer form is [the report's limitations](README.md#7-limitations).
