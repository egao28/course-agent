"""Catalog CSV in, two indexes out -- everything that runs before a query is ever asked.

The pipeline is four steps in order: :mod:`ingest.export_almabot` re-exports
``course-catalog.csv`` into ``data/`` as the corpus, :mod:`ingest.chunk` renders one course into
one chunk of ``data/chunks.jsonl`` through the versioned template, and
:mod:`ingest.build_dense` / :mod:`ingest.build_bm25` each read that same file into ``indexes/``.
:mod:`ingest.majors` holds the degree requirements transcribed by hand, which no CSV column
carries, and :mod:`ingest.tokenize` is the one analyzer both sides of the BM25 index go through.

Every output is named by the config hash of the settings that produced it, so a changed template
or a changed field list orphans the old index rather than silently answering over it.
"""
