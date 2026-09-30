"""Embed ``data/chunks.jsonl`` into Chroma -- the dense half of Day 1's hybrid retrieval.

Run from the repo root, after ``ingest.chunk``::

    uv run --env-file .env python -m ingest.build_dense            # build it if it is not there
    uv run --env-file .env python -m ingest.build_dense --rebuild  # drop it and embed again
    uv run python -m ingest.build_dense --dry-run                  # what it would cost, no API call

The collection is ``courses_{dense_key}`` inside ``indexes/chroma``, so a change to the chunk
template or the encoder builds a new collection beside the old one instead of overwriting it, and
``retrieval.dense`` can only read the one its own config names. It is keyed on ``dense_key`` rather
than the full config hash because tuning ``fusion`` or ``bm25`` -- which a Day 4 ablation does on
purpose -- cannot change a single vector, and re-embedding a byte-identical corpus to answer to a
new name would cost money for nothing; see :func:`config.dense_key`.

**Cosine, explicitly.** Chroma defaults to squared L2. OpenAI embeddings are meant to be compared
by cosine, and ``DenseRetriever`` reports ``1 - distance`` as its score, which is only a similarity
under cosine -- so the space is set at creation and re-checked at query time rather than assumed.

Running it twice costs nothing: a collection holding exactly as many rows as there are chunks is
left alone. Anything else -- a different count, an interrupted build -- stops and asks for
``--rebuild``, because embedding *into* a half-built collection would leave a corpus that looks
complete and silently is not.

The chunks are checked against the config's ``chunk_key`` before the first request. Embedding is
the one step here that costs money and minutes; discovering afterwards that the corpus came from
another template means paying for it twice.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import chromadb
from chromadb.errors import NotFoundError
from tqdm import tqdm

from config import Config, chroma_path, collection_name, config_hash, load_config, rel_to_repo
from embedding import (
    PRICE_PER_MTOK,
    Embedder,
    check_embedding_config,
    estimate_cost,
    make_embedder,
)
from ingest.chunk import (
    check_chunks,
    chroma_metadata,
    chunk_key,
    load_chunks,
    load_chunks_meta,
)

#: The distance function the collection is created with; see the module docstring.
SPACE = "cosine"


class Tally(NamedTuple):
    """What one build actually did -- the numbers the report prints."""

    vectors: int
    batches: int
    #: Tokens the provider billed, or ``None`` for a provider that reports none (local models).
    tokens: int | None
    #: Embedding dimension, read off the vectors rather than assumed from the model name.
    dim: int | None
    seconds: float


# --- the store ------------------------------------------------------------------------------------


def open_store(path: Path | None = None) -> chromadb.ClientAPI:
    """A persistent Chroma client on ``indexes/chroma`` (gitignored, rebuildable).

    This is the one place a store comes into existence: opening a persistent client writes a
    ``chroma.sqlite3``, so :func:`chroma_path` deliberately does not create the directory and
    read-only callers go through :func:`find_collection` instead.
    """
    path = chroma_path() if path is None else Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(client: chromadb.ClientAPI, name: str) -> Any | None:
    """The named collection, or ``None`` if it does not exist yet.

    ``embedding_function=None`` on every open: Chroma would otherwise attach its default ONNX
    model, which downloads ~80 MB and would silently embed a query with the wrong model if
    anything ever called ``query(query_texts=...)``. Every vector here comes from
    :mod:`embedding`.
    """
    try:
        return client.get_collection(name, embedding_function=None)
    except NotFoundError:
        return None


def find_collection(client: chromadb.ClientAPI | None, name: str) -> Any | None:
    """The named collection, without bringing a store into existence to find out.

    ``None`` covers both "no store yet" and "store, but no such collection": either way there is
    nothing to read and nothing has been created. A store that is not on disk cannot hold a
    collection, so the question is answered from the filesystem rather than by opening a client
    that would write one.
    """
    if client is None:
        if not chroma_path().exists():
            return None
        client = open_store()
    return get_collection(client, name)


def collection_metadata(
    cfg: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]], section: Mapping[str, Any]
) -> dict[str, Any]:
    """What the collection records about its own build.

    The name already carries the config hash, so this is for the human who opens an index six
    weeks later: which corpus, which model, when. ``chunk_key`` is the one field read back by
    code -- ``DenseRetriever`` refuses a collection built from another chunk template.
    """
    return {
        "chunk_key": chunk_key(cfg),
        "config_hash": config_hash(cfg),
        "provider": section["provider"],
        "model": section["model"],
        "space": SPACE,
        "chunks": len(chunks),
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- building -------------------------------------------------------------------------------------


def embed_into(
    collection: Any,
    chunks: Sequence[Mapping[str, Any]],
    embedder: Embedder,
    batch_size: int,
    progress: bool = True,
) -> Tally:
    """Embed every chunk and write it to ``collection``, one batch per request.

    Each batch is added as soon as it comes back rather than accumulating 4,502 vectors in memory
    and writing once: an interrupted run then leaves a short collection, which the count check
    catches, instead of nothing at all after paying for everything.
    """
    started = time.perf_counter()
    vectors = batches = 0
    tokens: int | None = 0
    dim: int | None = None
    with tqdm(total=len(chunks), unit="chunk", disable=not progress) as bar:
        for batch in itertools.batched(chunks, batch_size):
            texts = [c["text"] for c in batch]
            result = embedder.embed(texts)
            if len(result.vectors) != len(batch):
                raise RuntimeError(
                    f"embedder returned {len(result.vectors)} vectors for {len(batch)} chunks"
                )
            batch_dim = len(result.vectors[0])
            if dim is not None and batch_dim != dim:
                raise RuntimeError(
                    f"embedder returned {batch_dim}-dimensional vectors after {dim}-dimensional "
                    "ones; the collection would hold two incomparable spaces"
                )
            dim = batch_dim
            collection.add(
                ids=[c["doc_id"] for c in batch],
                embeddings=result.vectors,
                documents=texts,
                metadatas=[chroma_metadata(c["metadata"]) for c in batch],
            )
            vectors += len(batch)
            batches += 1
            tokens = None if tokens is None or result.tokens is None else tokens + result.tokens
            bar.update(len(batch))
    return Tally(vectors, batches, tokens, dim, time.perf_counter() - started)


def build(
    cfg: Config,
    rebuild: bool = False,
    client: chromadb.ClientAPI | None = None,
    embedder: Embedder | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Make ``courses_{hash}`` hold exactly the current chunks, and report what that took.

    Returns ``status`` ``"up-to-date"`` (nothing embedded, nothing charged) or ``"built"``.
    Everything that can fail without spending money -- the config, the corpus, the collection
    state, the API key -- is checked before the store is opened, so a run that cannot embed
    leaves nothing behind, not even an empty ``indexes/chroma``. What no guard here can foresee is
    the API refusing the first request (an exhausted credit balance, say), which does leave an
    empty collection behind -- so an empty one is read as absent, not as an interrupted build.

    What landed is counted afterwards as well. ``count()`` is the signal every later run reads to
    decide whether there is anything to do, so a build that reports success while the collection
    is short would make every future run demand a ``--rebuild`` that re-pays for the corpus and
    ends in exactly the same state.
    """
    section = check_embedding_config(cfg)
    chunks = load_chunks()
    check_chunks(chunks, load_chunks_meta(), cfg)

    name = collection_name(cfg, chunk_key(cfg))
    existing = find_collection(client, name)
    if existing is not None and not rebuild:
        count = existing.count()
        if count == len(chunks):
            return {"status": "up-to-date", "name": name, "chunks": count, "section": section}
        if count:
            raise ValueError(
                f"collection {name!r} holds {count} of the {len(chunks)} chunks -- an interrupted "
                "build, or a corpus that grew since. Re-run with --rebuild to drop it and "
                f"re-embed (about {len(chunks)} chunks)"
            )
        # Zero rows is not an interrupted build: the collection was created and the first request
        # was refused, so there is nothing in it to preserve and nothing --rebuild would do
        # differently. Demanding the flag here would ask the reader to rebuild nothing.

    embedder = make_embedder(cfg) if embedder is None else embedder
    client = open_store() if client is None else client
    if existing is not None:
        client.delete_collection(name)
    payload = collection_metadata(cfg, chunks, section)
    collection = client.create_collection(
        name,
        configuration={"hnsw": {"space": SPACE}},
        metadata=payload,
        embedding_function=None,
    )
    tally = embed_into(collection, chunks, embedder, int(section["batch_size"]), progress)
    collection.modify(metadata={**payload, "dim": tally.dim})
    stored = collection.count()
    if stored != len(chunks):
        raise RuntimeError(
            f"embedded {tally.vectors} chunks but collection {name!r} holds {stored} of "
            f"{len(chunks)}; the store dropped rows instead of the run being interrupted, so "
            "--rebuild would buy the same result again. The collection is left in place to "
            "inspect"
        )
    return {
        "status": "built",
        "name": name,
        "chunks": len(chunks),
        "section": section,
        "tally": tally,
    }


# --- reporting ------------------------------------------------------------------------------------


def cost_lines(model: str, tokens: int | None) -> list[str]:
    """The token and cost lines, saying plainly when either number is unavailable.

    A price that is not in the table prints as no price at all: a build report that quietly
    guesses is worse than one that says it does not know.
    """
    if tokens is None:
        return ["tokens billed          n/a  (provider reports no usage)"]
    cost = estimate_cost(model, tokens)
    lines = [f"tokens billed          {tokens}  ({model})"]
    if cost is None:
        lines.append(f"cost                   unknown  (no list price on file for {model})")
    else:
        lines.append(
            f"cost                   ${cost:.4f}  (at ${PRICE_PER_MTOK[model]:g} / 1M tokens)"
        )
    return lines


def summarize(result: Mapping[str, Any]) -> list[str]:
    """Human-readable build report."""
    name, section = result["name"], result["section"]
    if result["status"] == "up-to-date":
        return [
            f"collection             {name}  (already holds {result['chunks']} chunks)",
            f"store                  {rel_to_repo(chroma_path())}",
            "nothing to do          pass --rebuild to drop it and re-embed",
        ]
    tally: Tally = result["tally"]
    return [
        f"collection             {name}  ({SPACE}, dim {tally.dim})",
        (
            f"chunks embedded        {tally.vectors}  in {tally.batches} batches of "
            f"{section['batch_size']}"
        ),
        *cost_lines(section["model"], tally.tokens),
        f"elapsed                {tally.seconds:.1f}s",
        f"store                  {rel_to_repo(chroma_path())}",
    ]


def plan(cfg: Config, rebuild: bool = False, client: chromadb.ClientAPI | None = None) -> list[str]:
    """What a build would do, without embedding anything: the ``--dry-run`` report.

    The token count is the one ``ingest.chunk`` already measured on exactly this corpus, so the
    cost quoted here is the cost of the run -- not a re-estimate that could disagree with it.

    It is the cost of *this* run, too: a complete collection would be left alone by :func:`build`,
    so quoting the full corpus there would be a bill for work that is not going to happen. This is
    the command someone runs to decide whether to pay, so it takes ``--rebuild`` -- which does
    change the answer -- and, like ``build``, it creates nothing on the way to the report.
    """
    section = check_embedding_config(cfg)
    chunks = load_chunks()
    meta = load_chunks_meta()
    check_chunks(chunks, meta, cfg)
    name = collection_name(cfg, chunk_key(cfg))
    existing = find_collection(client, name)
    count = None if existing is None else existing.count()
    state = "does not exist yet" if count is None else f"holds {count} chunks"
    store = f"store                  {rel_to_repo(chroma_path())}"
    if count == len(chunks) and not rebuild:
        return [
            f"collection             {name}  ({state})",
            f"would embed            nothing  (already holds all {len(chunks)} chunks)",
            "cost                   $0.0000  (--rebuild drops it and embeds again)",
            store,
        ]
    verb = "drop and re-embed" if count is not None and rebuild else "would embed"
    tokens = (meta.get("tokens") or {}).get("total")
    return [
        f"collection             {name}  ({state})",
        (
            f"{verb:<22} {len(chunks)} chunks via {section['provider']} "
            f"in batches of {section['batch_size']}"
        ),
        *cost_lines(section["model"], tokens),
        store,
    ]


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Embed ``data/chunks.jsonl`` into the collection this config names.

    ``--dry-run`` reports the same plan without calling the API or writing anything, which is how
    the cost of a ``--rebuild`` is checked before it is paid.
    """
    parser = argparse.ArgumentParser(
        description="Embed data/chunks.jsonl into the Chroma collection named by the config hash."
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the collection and embed it again"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report what would be embedded and what it would cost; calls no API, writes nothing. "
            "Combines with --rebuild"
        ),
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    print(f"config hash {cfg.hash}  (chunk key {chunk_key(cfg)})")
    if args.dry_run:
        for line in plan(cfg, rebuild=args.rebuild):
            print(line)
        return
    for line in summarize(build(cfg, rebuild=args.rebuild)):
        print(line)


if __name__ == "__main__":
    sys.exit(main())
