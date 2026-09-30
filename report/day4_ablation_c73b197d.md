# Day 4 — RQ1: retrieval ablation

Recall@5 and MRR@5 for `dense`, `bm25` and `hybrid` through `retrieve(q, mode, k=5)`, on the frozen `test` half, and for `hybrid_rerank`: hybrid's top 20 reranked by a cross-encoder, cut to 5.

- **n** = 41 test rows (fact 29, multihop 12) — the rows that carry gold doc ids
- **dataset** `eval/datasets/v1.jsonl` sha256 `2a93182221bd8546d5bed8e121d0b4bde9a8b624a2197c67f5e87120d36cfce6`
- **config** `c73b197d` (dense index `094143d8`, bm25 index `877a6610`)
- **runs** `eval/runs/ablation_test_c73b197d.jsonl` (reported), `eval/runs/ablation_calib_c73b197d.jsonl` (chooses the strongest single mode only)
- **rerank** `eval/runs/rerank_test_3c68593d.jsonl` — `BAAI/bge-reranker-base` at revision `2cfc18c9415c`, rerank key `3c68593d`
- **bootstrap** `eval/runs/bootstrap_725d7385.json` — paired, B = 10,000, seed 76, 95% percentile intervals (full hash `725d7385`)

## Result

Test half, n = 41. Point estimate on all n rows, 95% paired-bootstrap interval in brackets.

| mode | Recall@5 | MRR@5 |
| --- | --- | --- |
| `dense` | 0.705 [0.571, 0.833] | 0.637 [0.502, 0.764] |
| `bm25` | 0.870 [0.778, 0.951] | 0.923 [0.850, 0.984] |
| `hybrid` | 0.866 [0.783, 0.935] | 0.802 [0.707, 0.891] |
| `hybrid_rerank` | 0.846 [0.748, 0.931] | 0.868 [0.774, 0.945] |
| *hybrid − bm25* | *−0.004 [−0.085, 0.073], not significant (CI contains 0)* | *−0.120 [−0.225, −0.023], significant (CI excludes 0)* |
| *hybrid_rerank − hybrid* | *−0.020 [−0.104, 0.061], not significant (CI contains 0)* | *+0.065 [−0.059, 0.186], not significant (CI contains 0)* |

- **Recall@5**: hybrid − bm25 = −0.004 [−0.085, 0.073]; the interval contains 0: these 41 questions do not separate them.
- **MRR@5**: hybrid − bm25 = −0.120 [−0.225, −0.023]; the interval excludes 0: on this metric `bm25` is higher.

## Strongest single mode

Chosen per metric on `calib`, then tested on `test`. Choosing it on the test rows would compare hybrid with the larger of two noisy means, which biases the gap against hybrid.

- **Recall@5**: `bm25` (calib n = 41: `bm25` 0.870, `dense` 0.750)
- **MRR@5**: `bm25` (calib n = 41: `bm25` 0.939, `dense` 0.621)

## By bucket (point estimates only)

| mode | fact Recall@5 (n = 29) | fact MRR@5 (n = 29) | multihop Recall@5 (n = 12) | multihop MRR@5 (n = 12) |
| --- | --- | --- | --- | --- |
| `dense` | 0.793 | 0.682 | 0.493 | 0.528 |
| `bm25` | 0.966 | 0.908 | 0.639 | 0.958 |
| `hybrid` | 0.966 | 0.829 | 0.625 | 0.739 |
| `hybrid_rerank` | 0.966 | 0.922 | 0.556 | 0.736 |

No intervals here, on purpose. At n = 12 one `multihop` question moves a mean by up to 1/12 = 0.083, so an interval on this split is too wide to be read as a finding. The split describes where the rows in the main table come from. It is not a second test.

## Reranking hybrid's top 20 (stretch, #79)

`hybrid_rerank` is `retrieve(q, "hybrid", k=20)`, rescored by `BAAI/bge-reranker-base` on (question, chunk text) pairs, and cut to 5. It is a step after `retrieve()`, not a fourth mode, so `hybrid` means what it meant above. Its interval uses the same resample indices as the three modes.

- **Recall@5**: hybrid_rerank − hybrid = −0.020 [−0.104, 0.061]; the interval contains 0: these 41 questions do not separate them.
- **MRR@5**: hybrid_rerank − hybrid = +0.065 [−0.059, 0.186]; the interval contains 0: these 41 questions do not separate them.
- **Ceiling**: hybrid Recall@20 = 0.949 over the same 41 rows. The reranker only reorders the pool, so its Recall@5 cannot exceed that.
- **Cost**: the cross-encoder takes a median of 724 ms per question (p90 987, max 1248) on cpu, on top of the hybrid@20 retrieval's own median of 176 ms. Model loading is excluded (sentence-transformers 6.0.1, torch 2.14.0, transformers 5.17.0).
- **Pairing**: the pool is a second live draw, and its top 5 matches the committed hybrid top 5 on 41 of 41 rows.

A reranker that does not separate from hybrid at n = 41 is a finding about this corpus and this sample size, not a failed experiment.

## Where hybrid misses (for Day 6)

Hybrid leaves at least one gold id out of its top 5 on 11 of 41 test rows: 1 of 29 `fact`, 10 of 12 `multihop`. Each is logged in `eval/runs/hybrid_failures_c73b197d.jsonl` with the ids it missed, the 5 it returned, and the rank each missing id has in `dense@20`, `bm25@20` and `hybrid@20`.

The failing rows are read from `eval/runs/ablation_test_c73b197d.jsonl`. The ranks at depth come from a second, live pass over those rows only, and its top 5 differs from the run's on 0 of 11. The log does not sort the misses by cause. That is Day 6.

## Definitions

- **MRR is MRR@5.** 1 / rank of the first gold id in the top 5, and 0 if none is there (`eval/metrics.py`, #74).
- **Recall@5 is fractional on multi-gold rows.** It is |gold ∩ top-5| / |gold|, not Hit@5. Rows with more than one gold id: 0 of 29 `fact`, 12 of 12 `multihop`.
- The ranked list is cut to 5 before anything is counted, and a repeated id in the top k counts once.

## Reproduction

```bash
uv run --env-file .env python -m eval.scripts.ablation --split calib
uv run --env-file .env python -m eval.scripts.ablation --split test
uv run --env-file .env --extra local python -m eval.scripts.rerank --split test
uv run python -m eval.scripts.bootstrap
uv run --env-file .env python -m eval.scripts.hybrid_failures
uv run python -m eval.scripts.day4_report
```

This file is a function of the two committed run files, the rerank run, the bootstrap result
computed from them, the failure log's sidecar, the dataset, and `config.yaml`. It has no timestamp,
so re-running the last command over unchanged inputs rewrites it byte for byte. Re-running the
first three does not reproduce the run files: the embeddings API returns slightly different query
vectors from call to call (two live calib runs differed in 1 of 41 dense lines), and the rerank run
records its own timings. The committed run files are the sample this table describes.
