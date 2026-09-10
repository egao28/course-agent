# course-agent

UIUC 选课 agent 的检索 + agent 层。数据来自 [AlmaBot](../almabot-web)(React 前端,`scripts/build_data.py` 从 `course-catalog.csv` 导出课程与先修 DAG)。

## 布局

```
course-agent/
├── config.yaml    # embedding 模型、chunk 策略、BM25/RRF 参数、seed —— 每个实验输出带 config hash
├── config.py      # load_config() / config_hash()
├── data/          # courses.json, prereqs.json, requirements.json, chunks.jsonl
├── ingest/        # 导出 → 切块 → 建索引(dense / bm25)
├── retrieval/     # dense.py, bm25.py, fusion.py, retrieve(query, mode)
├── agent/         # graph.py, tools/          (Day 2+)
├── eval/          # datasets/, scripts/       (Day 1 只有 smoke 集)
├── report/        # 每次实验的 markdown 报告,文件名带 config hash
└── indexes/       # Chroma 持久化 + BM25 pickle(gitignored,可重建)
```

## Day 1 目标

跑通 `query → dense + bm25 两路召回 → RRF → top-k`:

```python
from retrieval import retrieve
retrieve("machine learning courses", mode="hybrid")   # mode ∈ {dense, bm25, hybrid}
```

- 两路各取 top-20,RRF `score(d) = Σ 1/(60 + rank_i(d))`,融合后取 top-5
- 三种 mode 一个接口,Day 4 消融直接复用
- 验收:5 个 query 手测 top-5 合理

进度见 GitHub Issues(milestone: Day 1)。
