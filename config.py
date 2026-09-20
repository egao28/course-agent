"""Experiment configuration and the config hash that tags every artifact.

Usage::

    from config import load_config

    cfg = load_config()        # read-only Config; cfg["fusion"]["rrf_k"] works like a dict
    cfg.hash                   # e.g. "3f9a1c2e", computed once at load time

The hash is sha256 over a canonical JSON dump of the config (sorted keys, no whitespace),
truncated to 8 hex chars. Changing any value, including ``seed`` and the ``data.source_sha256``
pin, changes it; reordering keys does not. List order is significant (``chunk.fields`` order
defines the chunk template). ``Config`` is immutable, so the hash cannot drift between two helper
calls inside one build; to derive a variant (an ablation), copy and rebuild::

    variant = Config({**cfg.to_dict(), "seed": 43})

Artifacts are named with the hash so results from different settings never mix::

    Chroma collection   courses_{dense_key}               collection_name(cfg, chunk_key)
    BM25 index          <repo>/indexes/bm25_{hash}.pkl    bm25_path(cfg)
    reports             <repo>/report/{stem}_{hash}.md    report_path(stem, cfg)

The Chroma collection is the one artifact *not* named after the full hash. It holds embeddings,
which move only when the chunk text or the encoder moves, so it is named after ``dense_key()``: an
ablation over ``fusion.rrf_k``, ``bm25.k1`` or ``seed`` then queries the vectors it already paid
for instead of re-embedding a byte-identical corpus into a second collection. The store *directory*
(``chroma_path()``) carries no key at all -- it holds one collection per config, so the key belongs
on the collection inside it -- and it is the one path helper that does not create its directory.

Those paths are absolute (anchored at this module, not the cwd) and their directory is created on
demand. Write them into reports and logs via ``rel_to_repo(path)`` so no local home directory
leaks into tracked files.

CLI: ``python config.py [path] [--canonical] [--source-digest]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
INDEXES_DIR = REPO_ROOT / "indexes"
REPORT_DIR = REPO_ROOT / "report"
HASH_LEN = 8


# --- canonical form + hash ----------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    """Deep copy of nested mappings / sequences into plain dicts / lists (JSON-serializable)."""
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _freeze(obj: Any) -> Any:
    """Deep conversion of nested dicts / lists into read-only mapping proxies / tuples."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def canonical_dump(cfg: Mapping[str, Any]) -> str:
    """Deterministic serialization: JSON with sorted keys and no whitespace.

    Nested mappings and tuples are converted to plain dicts / lists first -- ``json.dumps`` would
    otherwise fall back to ``str()`` for a ``mappingproxy`` and hash ``"{'a': 1}"``. Non-JSON
    scalars that YAML may produce (dates) are rendered via ``str``.
    """
    return json.dumps(_to_plain(cfg), sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Mapping[str, Any]) -> str:
    """First ``HASH_LEN`` hex chars of ``sha256(canonical_dump(cfg))``.

    For a :class:`Config` this returns the hash frozen at load time.
    """
    if isinstance(cfg, Config):
        return cfg.hash
    return hashlib.sha256(canonical_dump(cfg).encode("utf-8")).hexdigest()[:HASH_LEN]


class Config(Mapping[str, Any]):
    """Read-only view of a config, with its hash computed once at construction.

    Nested mappings are ``MappingProxyType`` and lists are tuples, so an in-place edit raises
    ``TypeError`` instead of silently changing what a later ``config_hash()`` call would see.
    """

    __slots__ = ("_data", "_hash", "path")

    def __init__(self, data: Mapping[str, Any], path: Path | None = None) -> None:
        if not isinstance(data, Mapping):
            raise TypeError(f"config must be a mapping, got {type(data).__name__}")
        plain = _to_plain(data)
        self._data = _freeze(plain)
        self._hash = config_hash(plain)
        self.path = path

    @property
    def hash(self) -> str:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        """Mutable deep copy, e.g. to build an ablation variant."""
        return _to_plain(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config(hash={self._hash!r}, path={self.path})"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a YAML file into a read-only :class:`Config` with its hash frozen.

    Values are kept exactly as written -- ``~`` in ``data.source`` is *not* expanded, so the same
    file hashes identically on every machine. Use :func:`source_path` to get a usable path.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return Config(raw, path=path)


# --- input data ---------------------------------------------------------------------------------


def source_path(cfg: Mapping[str, Any]) -> Path:
    """``data.source`` as a usable absolute path.

    ``~`` is expanded and a relative path is anchored at the repo root. The config itself is left
    untouched, so the hash is unaffected.
    """
    p = Path(cfg["data"]["source"]).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_source(cfg: Mapping[str, Any]) -> Path:
    """Return :func:`source_path` after checking the file matches ``data.source_sha256``.

    The pin is part of the config hash, so replacing the input data forces a new hash -- and new
    index / report names -- instead of silently reusing artifacts built from the old data.
    """
    path = source_path(cfg)
    pinned = cfg["data"].get("source_sha256")
    if not pinned:
        raise ValueError(
            "data.source_sha256 is not set; run `python config.py --source-digest` and pin it"
        )
    actual = file_sha256(path)
    if actual != pinned:
        raise ValueError(
            f"{rel_to_repo(path)} has sha256 {actual} but config pins {pinned}; "
            "update data.source_sha256 (python config.py --source-digest) to build against it"
        )
    return path


# --- artifact naming: single source of truth for the {hash} tag ----------------------------------


def dense_key(cfg: Mapping[str, Any], chunk_key: str) -> str:
    """Short hash of everything that decides a vector: the chunk text, and the encoder.

    The full config hash names most artifacts, but the dense index is the expensive one and it
    moves for far fewer reasons than the config does. Only ``chunk_key`` (what the text is) and
    ``embedding.provider`` / ``embedding.model`` (what turns it into a vector) can change a single
    number in it. A Day 4 ablation over ``fusion.rrf_k``, ``bm25.k1`` or ``seed`` moves
    ``cfg.hash`` without moving any of those, so naming the collection after the full hash would
    orphan the index, re-embed 4,502 byte-identical chunks and pay for them a second time -- the
    same reasoning that gave ``ingest.chunk`` its ``chunk_key``.

    ``chunk_key`` is passed in rather than imported: ``ingest.chunk`` owns the chunk template and
    imports this module, so the dependency runs one way only.
    """
    embedding = cfg.get("embedding") or {}
    missing = [key for key in ("provider", "model") if key not in embedding]
    if missing:
        raise ValueError(
            f"config embedding section is missing {', '.join(missing)}; both are part of the name "
            "of the collection the dense index lives in (see config.yaml)"
        )
    content = {
        "chunk_key": chunk_key,
        "provider": embedding["provider"],
        "model": embedding["model"],
    }
    return hashlib.sha256(canonical_dump(content).encode("utf-8")).hexdigest()[:HASH_LEN]


def collection_name(cfg: Mapping[str, Any], chunk_key: str) -> str:
    """Chroma collection name, ``courses_{dense_key}`` -- see :func:`dense_key`."""
    return f"courses_{dense_key(cfg, chunk_key)}"


def chroma_path() -> Path:
    """Chroma persistence directory, ``<repo>/indexes/chroma``. **Not** created if missing.

    Takes no config and carries no key on purpose: one store holds a collection per config
    (:func:`collection_name`), so an ablation adds a collection beside the baseline instead of a
    second copy of the store.

    Every other path helper here creates its directory; this one must not. Opening a persistent
    Chroma client on a directory *writes* a ``chroma.sqlite3`` into it, so a caller that only
    wants to know where the store would go -- ``build_dense --dry-run``, a query against an index
    that was never built, an error message quoting the path -- would leave an empty store behind
    just by asking. Whoever actually writes the store creates it: see
    ``ingest.build_dense.open_store``.
    """
    return INDEXES_DIR / "chroma"


def bm25_path(cfg: Mapping[str, Any]) -> Path:
    """BM25 pickle path, ``<repo>/indexes/bm25_{hash}.pkl``; directory created if missing."""
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    return INDEXES_DIR / f"bm25_{config_hash(cfg)}.pkl"


def report_path(stem: str, cfg: Mapping[str, Any]) -> Path:
    """Report path, ``<repo>/report/{stem}_{hash}.md``; directory created if missing."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR / f"{stem}_{config_hash(cfg)}.md"


def rel_to_repo(path: str | Path) -> str:
    """Path as written into reports and logs.

    Repo-relative POSIX when the path is inside the repo, otherwise with the home directory
    collapsed to ``~`` -- never a bare absolute path carrying a local user name.
    """
    path = Path(path)
    for base, prefix in ((REPO_ROOT, ""), (Path.home(), "~/")):
        if path.is_relative_to(base):
            return prefix + path.relative_to(base).as_posix()
    return path.as_posix()


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the config hash of a config.yaml.")
    parser.add_argument("path", nargs="?", default=DEFAULT_CONFIG_PATH, help="config file path")
    parser.add_argument("--canonical", action="store_true", help="also print the canonical dump")
    parser.add_argument(
        "--source-digest",
        action="store_true",
        help="print sha256 of data.source instead (the value to pin as data.source_sha256)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.path)
    if args.source_digest:
        print(file_sha256(source_path(cfg)))
        return
    print(cfg.hash)
    if args.canonical:
        print(canonical_dump(cfg))


if __name__ == "__main__":
    main()
