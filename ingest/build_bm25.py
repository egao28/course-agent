"""Tokenize ``data/chunks.jsonl`` into a BM25 index -- the lexical half of Day 1's retrieval.

Run from the repo root, after ``ingest.chunk``::

    uv run python -m ingest.build_bm25

No API key, no network, no cost: the whole 2026-sp corpus tokenizes and indexes in about a second.
That is the one design difference from ``ingest.build_dense``, and it runs all the way through --
there is no ``--dry-run`` to price a run that is free, and no idempotency check to skip one that is
cheaper than deciding whether to skip it. Every run rebuilds and overwrites the index its config
names. What the dense side does keep here is the freshness check on the corpus itself, through the
same :func:`ingest.chunk.check_chunks`: an index built from a stale ``chunks.jsonl`` is wrong in
the same way whichever retriever reads it.

The index is a pickle at ``indexes/bm25_{bm25_key}.pkl`` (gitignored, rebuildable). It is keyed on
``bm25_key`` -- the chunk key plus everything under ``bm25`` -- rather than the full config hash,
so tuning ``fusion.rrf_k`` on Day 4 re-fuses the two rankings it already has instead of rebuilding
an index that cannot have changed; see :func:`config.bm25_key`.

It holds the ``BM25Okapi`` itself and, in corpus order, the ``doc_id``, text and metadata of every
chunk -- so a query returns a whole hit without opening ``chunks.jsonl``, exactly as the dense side
returns one without it. The metadata lists stay lists here; only Chroma needed them flattened.

**It is a pickle, so it is local-only.** Unpickling executes whatever the file says; this one is
written by this command, ignored by git and never fetched from anywhere. Do not load one you did
not build. It also holds an object of a third-party class, so the rank_bm25 version that wrote it
is recorded in the header and a file that will not load is answered with the command that rewrites
it, rather than a traceback out of the pickle module.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

from rank_bm25 import BM25Okapi

from config import Config, bm25_key, bm25_path, config_hash, load_config, rel_to_repo
from ingest.chunk import check_chunks, chunk_key, load_chunks, load_chunks_meta
from ingest.tokenize import check_tokenizer, code_boost, corpus_tokens, get_tokenizer

#: Layout version of the pickle. Bumped when the payload's keys change, so an index written by an
#: older checkout is refused by name instead of read into the wrong shape.
FORMAT = 1
#: Payload keys a reader needs. Checked on load: a truncated or hand-made file fails saying which
#: key is missing, not with a KeyError three frames into a search.
REQUIRED_KEYS = ("format", "bm25_key", "tokenizer", "doc_ids", "texts", "metadatas", "bm25")


class Tally(NamedTuple):
    """What one build did -- the numbers the report prints."""

    docs: int
    #: Distinct terms in the index, i.e. the vocabulary BM25 computed an idf for.
    terms: int
    #: Total terms emitted, including the repeats ``bm25.code_boost`` adds.
    tokens: int
    seconds: float


def rank_bm25_version() -> str:
    """The installed rank_bm25 version, recorded in the header; ``"unknown"`` if unreadable."""
    try:
        return version("rank-bm25")
    except PackageNotFoundError:  # pragma: no cover -- an editable install without metadata
        return "unknown"


# --- config ---------------------------------------------------------------------------------------


def check_bm25_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the ``bm25`` section against what this module implements; return it.

    Every value in it is hashed into the index name (:func:`config.bm25_key`), so a config naming
    a tokenizer nothing implements, or a ``b`` outside its range, would tag an index with a
    description of how it was *not* built. Reported as a sentence naming the offending key -- the
    same rule as ``check_embedding_config`` and ``check_chunk_config``.
    """
    check_tokenizer(cfg)
    code_boost(cfg)
    section = cfg["bm25"]
    missing = [key for key in ("k1", "b") if key not in section]
    if missing:
        raise ValueError(
            f"config bm25 section is missing {', '.join(missing)}; BM25 needs both (k1 is term "
            "frequency saturation, b is length normalisation -- see config.yaml)"
        )
    k1, b = section["k1"], section["b"]
    if isinstance(k1, bool) or not isinstance(k1, (int, float)) or k1 < 0:
        raise ValueError(f"bm25.k1 must be a non-negative number, got {k1!r}")
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b <= 1:
        raise ValueError(f"bm25.b must be a number between 0 and 1, got {b!r}")
    return section


# --- building -------------------------------------------------------------------------------------


def index_payload(
    cfg: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    corpus: Sequence[Sequence[str]],
    bm25: BM25Okapi,
) -> dict[str, Any]:
    """Everything the pickle holds: how it was built, what it holds, and the index itself.

    The header is what a reader checks before trusting the body, and what the person who opens an
    index six weeks later reads to find out which corpus and which settings produced it. ``epsilon``
    is rank_bm25's own floor for negative idf; it is not in ``config.yaml`` and so not tuned on
    Day 1, but it changes scores, so the value that was in force is written down.
    """
    section = cfg["bm25"]
    return {
        "format": FORMAT,
        "bm25_key": bm25_key(cfg, chunk_key(cfg)),
        "chunk_key": chunk_key(cfg),
        "config_hash": config_hash(cfg),
        "tokenizer": section["tokenizer"],
        "code_boost": code_boost(cfg),
        "k1": bm25.k1,
        "b": bm25.b,
        "epsilon": bm25.epsilon,
        "rank_bm25": rank_bm25_version(),
        "chunks": len(chunks),
        "terms": len(bm25.idf),
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "doc_ids": [c["doc_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [c["metadata"] for c in chunks],
        "bm25": bm25,
    }


def write_index(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the payload, atomically: a build interrupted mid-write leaves the old index intact.

    Without the rename an interrupted write would leave a truncated file under the name its
    settings give it -- which is exactly the name the next query opens, and the one name nothing
    would think to doubt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as f:
            pickle.dump(dict(payload), f, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
    finally:
        # A no-op once the rename went through; on the way out of a failed write it is what keeps
        # a half-written file from sitting next to a good index.
        temp.unlink(missing_ok=True)
    return path


def load_index(path: Path) -> dict[str, Any]:
    """Read an index back, or say which command rewrites it.

    Two failures are answered rather than raised through: a file this checkout cannot unpickle
    (the rank_bm25 class it holds moved under it), and one missing a key a reader needs. Both mean
    the same thing -- rebuild -- and neither is worth a traceback from inside :mod:`pickle`.
    """
    try:
        with path.open("rb") as f:
            index = pickle.load(f)
    except Exception as exc:
        raise ValueError(
            f"{rel_to_repo(path)} could not be read ({type(exc).__name__}: {exc}); it is a "
            "rebuildable cache, so run `uv run python -m ingest.build_bm25` to rewrite it"
        ) from exc
    if not isinstance(index, dict):
        # The file being wrong is not the caller passing a wrong type: every other bad index
        # here is a ValueError, and they all mean the same thing -- rebuild it.
        raise ValueError(  # noqa: TRY004
            f"{rel_to_repo(path)} does not hold a BM25 index; run "
            "`uv run python -m ingest.build_bm25` to rewrite it"
        )
    missing = [key for key in REQUIRED_KEYS if key not in index]
    if missing:
        raise ValueError(
            f"{rel_to_repo(path)} is missing {', '.join(missing)}; re-run "
            "`uv run python -m ingest.build_bm25`"
        )
    if index["format"] != FORMAT:
        raise ValueError(
            f"{rel_to_repo(path)} is format {index['format']}, but this checkout writes "
            f"{FORMAT}; re-run `uv run python -m ingest.build_bm25`"
        )
    return index


def build(cfg: Config) -> dict[str, Any]:
    """Tokenize the corpus, index it, and write the pickle this config names.

    The corpus is checked against the config's ``chunk_key`` before anything is tokenized, for the
    same reason the dense build checks it before embedding: an index built from another template's
    chunks answers queries, plausibly, out of a corpus nobody meant to search.
    """
    check_bm25_config(cfg)
    chunks = load_chunks()
    check_chunks(chunks, load_chunks_meta(), cfg)

    started = time.perf_counter()
    corpus = corpus_tokens(chunks, get_tokenizer(cfg), code_boost(cfg))
    bm25 = BM25Okapi(corpus, k1=float(cfg["bm25"]["k1"]), b=float(cfg["bm25"]["b"]))
    path = write_index(bm25_path(cfg, chunk_key(cfg)), index_payload(cfg, chunks, corpus, bm25))
    tally = Tally(
        docs=len(chunks),
        terms=len(bm25.idf),
        tokens=sum(len(doc) for doc in corpus),
        seconds=time.perf_counter() - started,
    )
    return {"path": path, "tally": tally, "bm25": bm25}


# --- reporting ------------------------------------------------------------------------------------


def summarize(result: Mapping[str, Any]) -> list[str]:
    """Human-readable build report."""
    tally: Tally = result["tally"]
    path: Path = result["path"]
    bm25: BM25Okapi = result["bm25"]
    return [
        f"documents indexed      {tally.docs}",
        f"vocabulary             {tally.terms} terms",
        f"terms emitted          {tally.tokens}  (mean {tally.tokens / tally.docs:.1f} per chunk)",
        f"k1 / b                 {bm25.k1:g} / {bm25.b:g}",
        f"elapsed                {tally.seconds:.1f}s",
        f"wrote                  {rel_to_repo(path)}  ({path.stat().st_size // 1024} KB)",
    ]


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize data/chunks.jsonl into the BM25 index named by the bm25 key."
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    print(f"config hash {cfg.hash}  (bm25 key {bm25_key(cfg, chunk_key(cfg))})")
    for line in summarize(build(cfg)):
        print(line)


if __name__ == "__main__":
    sys.exit(main())
