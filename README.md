# course-agent

Retrieval + agent layer for a UIUC course-planning assistant. Data comes from [AlmaBot](../almabot-web)
(React frontend; its `scripts/build_data.py` exports courses and the prerequisite DAG from `course-catalog.csv`).

## Layout

```
course-agent/
├── config.yaml    # embedding model, chunk strategy, BM25/RRF params, seed — every experiment output carries the config hash
├── config.py      # load_config() / config_hash()
├── data/          # courses.json, prereqs.json, requirements.json, chunks.jsonl
├── ingest/        # export → chunk → build indexes (dense / bm25)
├── retrieval/     # dense.py, bm25.py, fusion.py, retrieve(query, mode)
├── agent/         # graph.py, tools/          (Day 2+)
├── eval/          # datasets/, scripts/       (Day 1: smoke set only)
├── report/        # one markdown report per experiment, filename tagged with the config hash
└── indexes/       # Chroma persistence + BM25 pickle (gitignored, rebuildable)
```

## Day 1 goal

End-to-end `query → dense + BM25 recall → RRF → top-k`:

```python
from retrieval import retrieve
retrieve("machine learning courses", mode="hybrid")   # mode ∈ {dense, bm25, hybrid}
```

- Each retriever returns top-20; RRF `score(d) = Σ 1/(60 + rank_i(d))`; keep top-5 after fusion
- One interface for all three modes so the Day 4 ablation reuses it directly
- Acceptance: hand-check top-5 for 5 queries

Progress is tracked in GitHub Issues (milestone: Day 1).
