"""Embed ``data/chunks.jsonl`` into Chroma -- the dense half of Day 1's hybrid retrieval.

Run from the repo root, after ``ingest.chunk``::

    uv run --env-file .env python -m ingest.build_dense            # build it if it is not there
    uv run --env-file .env python -m ingest.build_dense --rebuild  # drop it and embed again
    uv run python -m ingest.build_dense --dry-run                  # what it would cost, no API call

The collection is ``courses_{config_hash}`` inside ``indexes/chroma``, so a config change builds a
new collection beside the old one instead of overwriting it, and ``retrieval.dense`` can only read
the one its own config names.

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
import json
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
from ingest.chunk import CHUNKS_META_PATH, CHUNKS_PATH, chroma_metadata, chunk_key

#: The distance function the collection is created with; see the module docstring.
SPACE = "cosine"
#: Keys every line of chunks.jsonl must carry.
CHUNK_KEYS = frozenset({"doc_id", "text", "metadata"})


class Tally(NamedTuple):
    """What one build actually did -- the numbers the report prints."""

    vectors: int
    batches: int
    #: Tokens the provider billed, or ``None`` for a provider that reports none (local models).
    tokens: int | None
    #: Embedding dimension, read off the vectors rather than assumed from the model name.
    dim: int | None
    seconds: float


# --- inputs ---------------------------------------------------------------------------------------


def load_chunks(path: Path | None = None) -> list[dict[str, Any]]:
    """The chunk corpus, or a pointer at the command that writes it."""
    path = CHUNKS_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{rel_to_repo(path)} is missing; run `uv run python -m ingest.chunk` first"
        )
    chunks = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            row = json.loads(line)
            if CHUNK_KEYS - row.keys():
                missing = ", ".join(sorted(CHUNK_KEYS - row.keys()))
                raise ValueError(
                    f"{rel_to_repo(path)}:{lineno} is missing {missing}; "
                    "re-run `uv run python -m ingest.chunk`"
                )
            chunks.append(row)
    if not chunks:
        raise ValueError(f"{rel_to_repo(path)} is empty; re-run `uv run python -m ingest.chunk`")
    return chunks


def load_chunks_meta(path: Path | None = None) -> dict[str, Any]:
    """The sidecar written beside the corpus, which says what it was chunked under."""
    path = CHUNKS_META_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{rel_to_repo(path)} is missing; run `uv run python -m ingest.chunk` first"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_chunks(
    chunks: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], cfg: Mapping[str, Any]
) -> None:
    """Refuse a corpus that is not the one this config describes, before anything is embedded.

    ``chunk_key`` covers only what decides a chunk's *content* (term, strategy, template, fields),
    so tuning ``bm25`` or ``fusion`` between two runs does not read as a stale corpus -- see
    ``ingest.chunk.chunk_key``. The sidecar's own count is compared too: a truncated
    ``chunks.jsonl`` would otherwise build a collection that is short a few courses and says
    nothing about it.
    """
    expected = chunk_key(cfg)
    found = meta.get("chunk_key")
    if found != expected:
        raise ValueError(
            f"{rel_to_repo(CHUNKS_PATH)} was written with chunk_key {found!r}, but config.yaml "
            f"describes {expected!r}; re-run `uv run python -m ingest.chunk` before indexing"
        )
    if meta.get("chunks") != len(chunks):
        raise ValueError(
            f"{rel_to_repo(CHUNKS_META_PATH)} records {meta.get('chunks')} chunks but "
            f"{rel_to_repo(CHUNKS_PATH)} holds {len(chunks)}; re-run "
            "`uv run python -m ingest.chunk`"
        )


# --- the store ------------------------------------------------------------------------------------


def open_store(path: Path | None = None) -> chromadb.ClientAPI:
    """A persistent Chroma client on ``indexes/chroma`` (gitignored, rebuildable)."""
    return chromadb.PersistentClient(path=str(chroma_path() if path is None else path))


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
    state -- is checked before the embedder is even constructed.
    """
    section = check_embedding_config(cfg)
    chunks = load_chunks()
    check_chunks(chunks, load_chunks_meta(), cfg)

    client = open_store() if client is None else client
    name = collection_name(cfg)
    existing = get_collection(client, name)
    if existing is not None and not rebuild:
        count = existing.count()
        if count == len(chunks):
            return {"status": "up-to-date", "name": name, "chunks": count, "section": section}
        raise ValueError(
            f"collection {name!r} holds {count} of the {len(chunks)} chunks -- an interrupted "
            "build, or a corpus that grew since. Re-run with --rebuild to drop it and re-embed "
            f"(about {len(chunks)} chunks)"
        )

    embedder = make_embedder(cfg) if embedder is None else embedder
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


def plan(cfg: Config, client: chromadb.ClientAPI | None = None) -> list[str]:
    """What a build would do, without embedding anything: the ``--dry-run`` report.

    The token count is the one ``ingest.chunk`` already measured on exactly this corpus, so the
    cost quoted here is the cost of the run -- not a re-estimate that could disagree with it.
    """
    section = check_embedding_config(cfg)
    chunks = load_chunks()
    meta = load_chunks_meta()
    check_chunks(chunks, meta, cfg)
    name = collection_name(cfg)
    existing = get_collection(open_store() if client is None else client, name)
    state = f"holds {existing.count()} chunks" if existing is not None else "does not exist yet"
    tokens = (meta.get("tokens") or {}).get("total")
    return [
        f"collection             {name}  ({state})",
        (
            f"would embed            {len(chunks)} chunks via {section['provider']} "
            f"in batches of {section['batch_size']}"
        ),
        *cost_lines(section["model"], tokens),
        f"store                  {rel_to_repo(chroma_path())}",
    ]


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
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
        help="report what would be embedded and what it would cost; calls no API",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    print(f"config hash {cfg.hash}  (chunk key {chunk_key(cfg)})")
    if args.dry_run:
        for line in plan(cfg):
            print(line)
        return
    for line in summarize(build(cfg, rebuild=args.rebuild)):
        print(line)


if __name__ == "__main__":
    sys.exit(main())
