# Day 7 — reproducing Day 4 from a fresh clone

[`report/day4_ablation_c73b197d.md`](day4_ablation_c73b197d.md) re-derived from a clone that held
nothing but the tracked tree, following [`README.md`](../README.md) and the report's own
Reproduction block and nothing else (#118).

**Verdict: it reproduces.** Every key matches, the `bm25` row matches to the digit, every point
estimate and every interval in the main table is identical, and the one line of the regenerated
report that moved is the cross-encoder's own latency — the line the report already says is a
property of the machine it ran on.

## What was run, and where

| | |
| --- | --- |
| machine | Apple M5, 10 cores, 24 GB |
| OS | macOS 26.6.2 (build 25G83), arm64 |
| Python | 3.12.13, installed by `uv` from `.python-version` |
| `uv` | 0.11.19 (Homebrew) |
| system Python (unused) | 3.14.1 |
| commit | `7833bc7`, `main`, the merge of #131 |
| clone | `git clone` into an empty directory; no file in it was ever edited by hand before a run |
| date | 2026-09-30, 21:45–22:05 CDT |

A fresh clone rather than a cleaned working copy, so that an untracked leftover could not quietly
be an input. `.env` is gitignored and so is absent from a clone; it was created from the committed
`.env.example` exactly as [Setup](../README.md#setup) says, and `OPENAI_API_KEY` is the only
variable in it.

The one deviation from #118's procedure: the Hugging Face cache was redirected with `HF_HOME` to an
empty scratch directory instead of deleting `~/.cache/huggingface`, which is shared with everything
else on this machine. The effect is the same — the cross-encoder downloaded cold, 1.1 GB, and the
timing below includes it — without destroying a cache this repo does not own.

The `uv` package cache was **not** cleared, so `uv sync` finished in under a second and the sync
time below is not the time a genuinely cold machine would see. Nothing else in the chain depends on
that cache, and the environment `--locked` produced is the same one either way.

## The five keys

Fixed in #118 before anything was run. All five hold.

| must be identical | expected | observed | |
| --- | --- | --- | --- |
| `cfg.hash` | `c73b197d` | `c73b197d` | ✓ |
| dense collection key | `094143d8` | `094143d8` | ✓ |
| BM25 index key | `877a6610` | `877a6610` | ✓ |
| dataset sha256 | `2a93182221bd…` | `2a93182221bd…` | ✓ |
| rerank key | `3c68593d` | `3c68593d` | ✓ |

The cross-encoder resolved to revision `2cfc18c9415c912f9d8155881c133215df768a70`, whose prefix is
the `2cfc18c9415c` the Day 4 report pins. `bootstrap_key` reproduced as `baa560e9` as well, which
#118 does not list but the report names its result after.

## The table, side by side

Test half, n = 41 (29 `fact`, 12 `multihop`). Point estimate on all n rows, 95% paired-bootstrap
interval in brackets, B = 10,000, seed 76.

| mode | Recall@5 committed | Recall@5 re-derived | MRR@5 committed | MRR@5 re-derived |
| --- | --- | --- | --- | --- |
| `dense` | 0.705 [0.571, 0.833] | 0.705 [0.571, 0.833] | 0.637 [0.502, 0.764] | 0.637 [0.502, 0.764] |
| `bm25` | 0.870 [0.778, 0.951] | 0.870 [0.778, 0.951] | 0.923 [0.850, 0.984] | 0.923 [0.850, 0.984] |
| `hybrid` | 0.866 [0.783, 0.935] | 0.866 [0.783, 0.935] | 0.802 [0.707, 0.891] | 0.802 [0.707, 0.891] |
| `hybrid_rerank` | 0.846 [0.748, 0.931] | 0.846 [0.748, 0.931] | 0.868 [0.774, 0.945] | 0.868 [0.774, 0.945] |
| *hybrid − bm25* | *−0.004 [−0.085, 0.073]* | *−0.004 [−0.085, 0.073]* | *−0.120 [−0.225, −0.023]* | *−0.120 [−0.225, −0.023]* |
| *hybrid_rerank − hybrid* | *−0.020 [−0.104, 0.061]* | *−0.020 [−0.104, 0.061]* | *+0.065 [−0.059, 0.186]* | *+0.065 [−0.059, 0.186]* |

Not "within tolerance" — equal, in every cell, including both significance verdicts. The strongest
single mode was chosen on `calib` and came out `bm25` on both metrics again, from the same calib
means the report quotes: Recall@5 `bm25` 0.870 / `dense` 0.750, MRR@5 `bm25` 0.939 / `dense` 0.621.
The per-bucket table and the Recall@20 ceiling of 0.949 are identical too; see the regenerated
report below, where they did not move.

The tolerance #118 set in advance was *every point estimate inside the committed 95% interval, and
every per-row difference listed by id*. The first half was not exercised — nothing moved far enough
to need it. The second half is the next section.

## Why identical, when three of the modes are live draws

The two live modes did draw differently. The run files are not byte-identical:

| run file | lines | lines that differ | what differs | metrics that moved |
| --- | --- | --- | --- | --- |
| `ablation_calib_c73b197d.jsonl` | 123 | 6 | `ranked` only | 0 |
| `ablation_test_c73b197d.jsonl` | 123 | 3 | `ranked` only | 0 |
| `rerank_test_3c68593d.jsonl` | 41 | 41 | `pool_ms`, `rerank_ms` (41), `pool` (8) | 0 |

Every differing line, by id, with the first rank at which the two lists part:

**calib** — `v1-0006` `dense` at rank 4, `v1-0006` `hybrid` at rank 5, `v1-0033` `dense` at rank 5,
`v1-0036` `dense` at rank 3, `v1-0036` `hybrid` at rank 4, `v1-1007` `hybrid` at rank 4.

**test** — `v1-0017` `hybrid` at rank 3, `v1-0032` `hybrid` at rank 4, `v1-0044` `dense` at rank 5.

Every one of them parts *below* the deepest gold id in that row, so `recall_at_5` and `mrr_at_5` are
unchanged on all 123 lines of both files. The 41 metric vectors the bootstrap resamples are
therefore the same 41 numbers per mode, and a seeded resample of identical inputs returns identical
intervals. That is the mechanism behind the table above: not luck, and not stability of the
embeddings API, but the fact that this corpus separates the gold chunk from the rest by a wide
enough margin that the jitter lands in ranks 3–5.

`bm25` differs on **0 of 41** lines on either split, as #118 required of it — it calls nothing.

The rerank file differs on all 41 rows because every row records its own two timings. Its `pool`,
the live hybrid@20 that feeds the cross-encoder, differs on 8 of 41 rows, and its top 5 still
matched the committed hybrid top 5 on 41 of 41 — the same pairing check the report reports.

`bootstrap_baa560e9.json` differs from the committed one in exactly three fields: the sha256 of the
two ablation runs and of the rerank run. Every statistic in it is byte-identical.

## The generator

Over the **committed** runs, `uv run python -m eval.scripts.day4_report` rewrote
`report/day4_ablation_c73b197d.md` byte for byte — zero diff, clean tree. Byte-stable, as it claims.

Over the **re-derived** runs it produced a report that differs from the committed one in one line:

```diff
-- **Cost**: the cross-encoder takes a median of 724 ms per question (p90 987, max 1248) on cpu, on top of the hybrid@20 retrieval's own median of 176 ms.
+- **Cost**: the cross-encoder takes a median of 741 ms per question (p90 1150, max 1243) on cpu, on top of the hybrid@20 retrieval's own median of 236 ms.
```

741 ms against 724 ms on a different machine, and `sentence-transformers` 6.0.1 / `torch` 2.14.0 /
`transformers` 5.17.0 are the versions the committed line names. Nothing else in the file moved.

## Cost and time

| step | wall time | cost |
| --- | --- | --- |
| `uv sync --locked --extra analysis` | 0.5 s (warm `uv` cache) | — |
| `ingest.chunk` + `ingest.build_bm25` | 1.2 s | — |
| `ingest.build_dense` | 22.8 s | **$0.0105**, 522,658 tokens billed, `text-embedding-3-small` |
| `ablation --split calib` | 20.0 s | 41 query embeddings ×2 modes |
| `ablation --split test` | 18.8 s | 41 query embeddings ×2 modes |
| `rerank --split test` | 3 min 32 s | 1.1 GB model download (cold), then ~0.74 s/question on CPU |
| `bootstrap` | 0.5 s | — |
| `hybrid_failures` | 6.2 s | 11 rows, live |
| `day4_report` | ~1 s | — |
| `pytest -q` | 32.7 s cold, 14.4 s warm | — |

**About one cent and about five minutes**, of which three and a half are the cross-encoder
downloading. Re-running it on a warm model cache is well under a minute. The corpus embedding is
the only metered call that is not trivial, and the README's `--dry-run` quotes it to the cent
before it is spent.

## The suite on a fresh clone

| state | result |
| --- | --- |
| after `ingest.chunk` + `ingest.build_bm25` | **2337 passed, 0 skipped, 0 failed** |
| before either had run | **2327 passed, 5 failed, 5 skipped** |

The 5 that **fail** before the corpus is built are all in `tests/test_draft_items.py`:
`test_the_short_tail_is_dropped`, `test_every_subject_is_reachable_early`,
`test_the_example_ids_are_not_real_ids`, `test_the_cli_drafts_into_the_directory_it_is_given`,
`test_the_cli_exits_non_zero_when_a_batch_fails`. They raise `FileNotFoundError` out of
`load_chunks`, whose message names the command to run.

The 5 that **skip** declare why: `tests/test_build_unanswerable.py:119` and
`tests/test_eval_validate.py:102` on "chunks.jsonl has not been built here";
`tests/test_build_unanswerable.py:605`, `:616` and `tests/test_rerank.py:259` on "the BM25 index has
not been built here". Both counts are what the README's [Without uv](../README.md#without-uv)
section already predicts — "without them 5 tests fail, 5 skip".

Nothing skipped at all once the two builds had run, including on a plain `uv sync --locked` with no
extra named (see D3).

## README defects

Four, found by typing what the file says and writing down each place it was wrong rather than
fixing it in passing.

**D1 — the Day 4 Reproduction block does not run from a clean clone. Filed, not fixed here.**
Its first command needs a dense index that the clone does not have and that none of its six
commands build:

```
FileNotFoundError: no collection 'courses_094143d8' in indexes/chroma;
run `uv run --env-file .env python -m ingest.build_dense` first
```

`README.md`'s own [Retrieval ablation](../README.md#retrieval-ablation) section had the same gap and
is **fixed in this PR**. The block inside `day4_ablation_c73b197d.md` is not: it is emitted from the
`REPRODUCTION` constant in `eval/scripts/day4_report.py`, so correcting it means editing the
generator and regenerating a committed report, and #118 rules number and report edits out of this
PR. Filed as **#132**. Severity is low in practice — the error names the exact command and costs a
reader one failed run, not a dead end — but a reproduction block that cannot be pasted is the one
kind of documentation this repo cannot leave wrong.

**D2 — `## Setup`'s three commands end in a `pytest` that fails. Fixed in this PR.** Read top to
bottom the file is fine: [Quickstart](../README.md#quickstart) builds the corpus and the BM25 index
several sections earlier. But `Setup` is where the Reference half sends a reader who skipped the
front page, and its block is `uv sync` → `cp .env.example .env` → `uv run pytest -q` with the
comment "smoke: imports resolve + config hash properties hold". On a clone where Quickstart has not
been run, that line reports 5 failures. A pointer to the two build commands now sits in the block.

**D3 — `--extra analysis` is claimed to prevent skips that it does not prevent. Fixed in this PR.**
Quickstart says the steps are CI's "including the `analysis` extra, without which part of the suite
skips". Measured here, with the default `dev` group present, `uv sync --locked` and
`uv sync --locked --extra analysis` both give **2337 passed, 0 skipped** — the extra is a no-op,
because `pyproject.toml` puts `almabot[analysis]` in `dev` precisely so that it would be. The 47
skips the claim is reaching for belong to the `--no-dev` / bare-`requirements.txt` install, which
the [Without uv](../README.md#without-uv) section already describes correctly. The same sentence
appears as a comment on the sync step in `.github/workflows/tests.yml`; both are corrected.

**D4 — nothing was wrong about the private CSV.** #118 asked whether an outside reader hits a wall
at step one over `data.source` (`~/Downloads/course-catalog.csv`, sha `0b7c50d4…`), which is not in
the repo. They do not. [Data](../README.md#data) says the exporter's outputs are committed "so a
clone can build indexes without the private CSV", Quickstart says the three commands need no key,
and nothing in the Day 4 chain reads the CSV or calls `verify_source`. Recorded as checked, not as
a defect.

## Reproduction

```bash
git clone https://github.com/egao28/course-agent.git && cd course-agent   # the repo rename is #113/#119
uv sync --locked --extra analysis
cp -n .env.example .env                                   # then fill in OPENAI_API_KEY
uv run python -m ingest.chunk && uv run python -m ingest.build_bm25
uv run --env-file .env python -m ingest.build_dense       # the step D1 is about
uv run --env-file .env python -m eval.scripts.ablation --split calib
uv run --env-file .env python -m eval.scripts.ablation --split test
uv run --env-file .env --extra local python -m eval.scripts.rerank --split test
uv run python -m eval.scripts.bootstrap
uv run --env-file .env python -m eval.scripts.hybrid_failures
uv run python -m eval.scripts.day4_report
```

This file is hand-written, not generated: it is one person's account of one run on one machine, and
a second run of these commands will not rewrite it. The run files that run produced were **not**
committed — `git checkout -- eval/runs report/` put the committed sample back before this branch
was cut, so every number above describes a comparison that no longer exists in the tree. The
committed runs remain the sample Day 4 describes.
