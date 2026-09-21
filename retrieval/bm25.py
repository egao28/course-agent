"""Lexical retrieval: tokenize the query, score it against the BM25 index.

Reads the pickle ``ingest.build_bm25`` wrote -- and only the one its own config names, since that
name carries the BM25 key. Nothing here opens ``data/chunks.jsonl``: the text and metadata a hit
carries were stored in the index, so a query works in a checkout that never ran the export.

    from config import load_config
    from retrieval.bm25 import BM25Retriever

    BM25Retriever(load_config()).search("CS 374", 5)

or from the shell::

    uv run python -m retrieval.bm25 "CS 374" -k 5

``score`` is the raw Okapi BM25 sum over the query's terms: higher is better, unbounded above, and
meaningless outside its own result list -- a 30 here says nothing about a cosine 0.9 from the dense
side, which is why fusion works on ``rank``. Documents that matched *nothing* score exactly 0 and
are dropped rather than padded out to ``k``: fusion gives a rank to every list it is handed, and a
document that no query term touched has not been retrieved, it has merely been counted. (A term in
more than half the corpus -- every template label is one -- gets rank_bm25's floored idf instead of
a negative one, which stays positive as long as the corpus' average idf is. On 4,502 courses it is,
so "dropped" means "matched nothing".)

This is the half of the hybrid that can find ``CS 374`` by name. Embeddings put it near "algorithms"
and "theory of computation"; only the lexical side knows that the string is a course code, and that
one of the 4,502 courses *is* it -- see ``ingest.tokenize``.
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import Config, bm25_key, bm25_path, load_config, rel_to_repo
from ingest.build_bm25 import load_index
from ingest.chunk import chunk_key
from ingest.tokenize import check_tokenizer, get_tokenizer, query_tokens
from retrieval.types import Hit, format_hit, top_k_each


def no_index(path: Path) -> str:
    """The message for a query that has no index to run against."""
    return (
        f"no BM25 index at {rel_to_repo(path)}; run "
        "`uv run python -m ingest.build_bm25` first (it takes about a second and costs nothing)"
    )


def open_index(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The index this config names, or a pointer at the command that builds it."""
    path = bm25_path(cfg, chunk_key(cfg))
    if not path.exists():
        raise FileNotFoundError(no_index(path))
    return load_index(path)


def check_index(index: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    """Refuse an index this config cannot read correctly, before any query runs.

    The file name already carries the BM25 key, so a mismatch means the file was moved or renamed
    rather than rebuilt -- unlikely, and silent if unchecked: a query tokenized one way against a
    corpus tokenized another returns a well-formed, wrong ranking. The parallel lists are measured
    against the index for the same reason. They are what turns a score's position into a course,
    so a short one would not fail, it would answer with the wrong course.
    """
    expected = bm25_key(cfg, chunk_key(cfg))
    found = index.get("bm25_key")
    if found != expected:
        raise ValueError(
            f"the BM25 index was built with bm25_key {found!r}, but config.yaml describes "
            f"{expected!r}; re-run `uv run python -m ingest.build_bm25`"
        )
    tokenizer = index.get("tokenizer")
    if tokenizer != check_tokenizer(cfg):
        raise ValueError(
            f"the BM25 index was tokenized by {tokenizer!r}, but config.yaml names "
            f"{check_tokenizer(cfg)!r}; a query cannot match terms it spells differently. Re-run "
            "`uv run python -m ingest.build_bm25`"
        )
    sizes = {key: len(index[key]) for key in ("doc_ids", "texts", "metadatas")}
    sizes["bm25"] = index["bm25"].corpus_size
    if len(set(sizes.values())) != 1:
        listed = ", ".join(f"{key} {n}" for key, n in sizes.items())
        raise ValueError(
            f"the BM25 index holds mismatched columns ({listed}), so a hit's rank would name the "
            "wrong course; re-run `uv run python -m ingest.build_bm25`"
        )


class BM25Retriever:
    """Okapi BM25 over the index named by ``cfg``.

    The index is opened in the constructor, like ``DenseRetriever``'s collection: a missing or
    stale index is a setup problem and should surface when the retriever is built, not on the
    first query of a batch of five. ``index`` can be injected, which is how the tests run against
    a corpus they built in-process.
    """

    def __init__(self, cfg: Config, index: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg
        #: Default recall depth: what fusion (#7) asks each retriever for.
        self.default_k = top_k_each(cfg)
        self.index = open_index(cfg) if index is None else index
        check_index(self.index, cfg)
        self.tokenizer = get_tokenizer(cfg)
        self.bm25 = self.index["bm25"]

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Top-``k`` chunks for ``query``, best first. ``k`` defaults to ``fusion.top_k_each``.

        Ties are broken by corpus order, so two documents that BM25 scores identically come back
        in the same order on every run and on every machine -- a ranking that shuffles under a
        stable index would make the Day 4 ablation measure the shuffle.
        """
        if not query.strip():
            raise ValueError("query is empty; nothing to tokenize")
        k = self.default_k if k is None else int(k)
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        terms = query_tokens(query, self.tokenizer)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        return [
            Hit(
                doc_id=self.index["doc_ids"][i],
                score=float(scores[i]),
                rank=rank,
                text=self.index["texts"][i],
                metadata=dict(self.index["metadatas"][i]),
            )
            for rank, i in enumerate((i for i in top if scores[i] > 0), start=1)
        ]


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Query the BM25 index and print the top hits.")
    parser.add_argument("query", nargs="+", help="query text")
    parser.add_argument(
        "-k", type=int, default=None, help="how many hits (default: fusion.top_k_each)"
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument("--terms", action="store_true", help="also print the query's terms")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    query = " ".join(args.query)
    retriever = BM25Retriever(cfg)
    hits = retriever.search(query, args.k)
    print(f"{query!r}  ->  {len(hits)} hits from bm25_{bm25_key(cfg, chunk_key(cfg))}")
    if args.terms:
        print(f"terms {query_tokens(query, retriever.tokenizer)}")
    for hit in hits:
        print(format_hit(hit))


if __name__ == "__main__":
    sys.exit(main())
