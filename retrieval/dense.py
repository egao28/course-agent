"""Dense retrieval: embed the query, ask Chroma for its nearest chunks.

Reads the collection ``ingest.build_dense`` wrote -- and only the one its own config names, since
that name carries the config hash. Nothing here touches ``data/chunks.jsonl``: the text and
metadata a hit carries come back out of the index, so a query works in a checkout that never ran
the export.

    from config import load_config
    from retrieval.dense import DenseRetriever

    DenseRetriever(load_config()).search("machine learning", 5)

or from the shell::

    uv run --env-file .env python -m retrieval.dense "machine learning" -k 5

``score`` is ``1 - cosine_distance``: higher is better, which is the direction every other
retriever reports and the direction fusion assumes. Chroma hands back a *distance*, so the
conversion happens here, once -- and the collection's space is checked rather than trusted,
because the same arithmetic on a squared-L2 distance would produce a plausible-looking number
that ranks nothing correctly.

Scores are comparable within one result list, never across retrievers -- see ``retrieval.types``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from config import Config, chroma_path, collection_name, load_config, rel_to_repo
from embedding import Embedder, make_embedder
from ingest.chunk import chunk_key, decode_chroma_metadata
from retrieval.types import Hit, format_hit, top_k_each

#: The only distance function this module can turn into a similarity; see :func:`similarity`.
COSINE = "cosine"
#: What a query needs back from Chroma to build a :class:`Hit`.
INCLUDE = ["documents", "metadatas", "distances"]


def no_index(name: str) -> str:
    """The message for a query that has no index to run against, whatever is missing."""
    return (
        f"no collection {name!r} in {rel_to_repo(chroma_path())}; run "
        "`uv run --env-file .env python -m ingest.build_dense` first"
    )


def open_collection(cfg: Mapping[str, Any], client: chromadb.ClientAPI | None = None) -> Any:
    """The collection this config names, or a pointer at the command that builds it.

    A store that is not on disk is answered from the filesystem rather than by opening a client:
    constructing a persistent Chroma client *creates* ``indexes/chroma``, so a query against an
    index that was never built would leave an empty store behind as a side effect of failing.
    """
    name = collection_name(cfg, chunk_key(cfg))
    if client is None:
        if not chroma_path().exists():
            raise FileNotFoundError(no_index(name))
        client = chromadb.PersistentClient(path=str(chroma_path()))
    try:
        return client.get_collection(name, embedding_function=None)
    except NotFoundError as exc:
        raise FileNotFoundError(no_index(name)) from exc


def collection_space(collection: Any) -> str | None:
    """The collection's distance function, or ``None`` when Chroma does not report one.

    Unknown is not the same as wrong: :func:`check_collection` only refuses a space it can read
    *and* does not support, so a Chroma version that reports its configuration differently costs
    a check rather than the ability to query.
    """
    configuration = getattr(collection, "configuration_json", None) or {}
    return (configuration.get("hnsw") or configuration.get("spann") or {}).get("space")


def check_collection(collection: Any, cfg: Mapping[str, Any]) -> None:
    """Refuse a collection this module cannot read correctly, before any query runs.

    Two ways that happens: it was built in another distance space (``1 - distance`` would no
    longer be a similarity), or from a corpus rendered by another chunk template. Neither raises
    at query time on its own -- both just return well-formed, wrong answers.
    """
    space = collection_space(collection)
    if space is not None and space != COSINE:
        raise ValueError(
            f"collection {collection.name!r} was built in {space!r} space, but scores here are "
            f"{COSINE} similarities; rebuild it with "
            "`uv run --env-file .env python -m ingest.build_dense --rebuild`"
        )
    built = (collection.metadata or {}).get("chunk_key")
    expected = chunk_key(cfg)
    if built is not None and built != expected:
        raise ValueError(
            f"collection {collection.name!r} was built from chunk_key {built!r}, but config.yaml "
            f"describes {expected!r}; rebuild it with "
            "`uv run --env-file .env python -m ingest.build_dense --rebuild`"
        )


def similarity(distance: float, space: str | None = COSINE) -> float:
    """Chroma's distance as a higher-is-better score.

    Cosine distance is ``1 - cos``, so the similarity is ``1 - distance``: 1.0 for an identical
    vector, 0.0 for an unrelated one, negative for an opposed one. ``None`` means the space could
    not be read, in which case the build's own guarantee (always cosine) stands.
    """
    if space is not None and space != COSINE:
        raise ValueError(f"cannot turn a {space!r} distance into a similarity")
    return 1.0 - distance


def to_hits(result: Mapping[str, Any], space: str | None = COSINE) -> list[Hit]:
    """Chroma's column-wise query result for one query, as a ranked list of :class:`Hit`.

    Chroma returns one list per query and we only ever send one, hence the ``[0]`` everywhere.
    Ranks are 1-based positions in the order Chroma returned, which is ascending distance.
    """

    def column(key: str) -> list[Any]:
        values = result.get(key)
        return list(values[0]) if values else []

    ids, documents = column("ids"), column("documents")
    metadatas, distances = column("metadatas"), column("distances")
    return [
        Hit(
            doc_id=doc_id,
            score=similarity(distance, space),
            rank=rank,
            text=document or "",
            metadata=decode_chroma_metadata(metadata or {}),
        )
        for rank, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        )
    ]


class DenseRetriever:
    """Vector search over the Chroma collection named by ``cfg``.

    The collection and the embedder are opened once, in the constructor: a missing index or an
    unset API key is a setup problem, and should surface when the retriever is built rather than
    on the first query of a batch of five. ``collection`` and ``embedder`` can be injected, which
    is how the tests run without an API key.
    """

    def __init__(
        self,
        cfg: Config,
        collection: Any | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Open the collection named by ``cfg`` and check it was built from these settings.

        ``collection`` and ``embedder`` are taken as given when passed, which is how a test runs
        without Chroma or an API key; the check runs either way, so neither can skip it.
        """
        self.cfg = cfg
        #: Default recall depth: what fusion (#7) asks each retriever for.
        self.default_k = top_k_each(cfg)
        self.collection = open_collection(cfg) if collection is None else collection
        check_collection(self.collection, cfg)
        self.space = collection_space(self.collection)
        self.embedder = make_embedder(cfg) if embedder is None else embedder

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Top-``k`` chunks for ``query``, best first. ``k`` defaults to ``fusion.top_k_each``."""
        if not query.strip():
            raise ValueError("query is empty; nothing to embed")
        k = self.default_k if k is None else int(k)
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        vector = self.embedder.embed([query]).vectors[0]
        result = self.collection.query(query_embeddings=[vector], n_results=k, include=INCLUDE)
        return to_hits(result, self.space)


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Query the dense index alone and print the hits, naming the collection they came from."""
    parser = argparse.ArgumentParser(description="Query the dense index and print the top hits.")
    parser.add_argument("query", nargs="+", help="query text")
    parser.add_argument(
        "-k", type=int, default=None, help="how many hits (default: fusion.top_k_each)"
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    query = " ".join(args.query)
    hits = DenseRetriever(cfg).search(query, args.k)
    print(f"{query!r}  ->  {len(hits)} hits from {collection_name(cfg, chunk_key(cfg))}")
    for hit in hits:
        print(format_hit(hit))


if __name__ == "__main__":
    sys.exit(main())
