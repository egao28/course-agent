"""Experiment configuration and the config hash that tags every artifact.

Usage::

    from config import load_config, config_hash

    cfg = load_config()      # repo-root config.yaml by default
    h = config_hash(cfg)     # e.g. "3f9a1c2e"

The hash is sha256 over a canonical JSON dump of the loaded config (sorted keys, no whitespace),
truncated to 8 hex chars. Changing any value, including ``seed``, changes it; reordering keys does
not. List order is significant (``chunk.fields`` order defines the chunk template).

Artifacts are named with the hash so results from different settings never mix::

    Chroma collection   courses_{hash}            collection_name(cfg)
    BM25 index          indexes/bm25_{hash}.pkl   bm25_path(cfg)
    reports             report/{stem}_{hash}.md   report_path(stem, cfg)

Run ``python config.py`` to print the current hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
INDEXES_DIR = REPO_ROOT / "indexes"
REPORT_DIR = REPO_ROOT / "report"
HASH_LEN = 8

Config = dict[str, Any]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a YAML config file into a plain dict.

    Values are returned exactly as written (``~`` in ``data.source`` is *not* expanded), so the
    same file hashes identically on every machine. Expand paths at the point of use.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping, got {type(cfg).__name__}")
    return cfg


def canonical_dump(cfg: Config) -> str:
    """Deterministic serialization: JSON with sorted keys and no whitespace.

    Non-JSON scalars that YAML may produce (dates, paths) are rendered via ``str``.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Config) -> str:
    """First ``HASH_LEN`` hex chars of ``sha256(canonical_dump(cfg))``."""
    return hashlib.sha256(canonical_dump(cfg).encode("utf-8")).hexdigest()[:HASH_LEN]


# --- artifact naming: single source of truth for the {hash} tag ---------------------------------


def collection_name(cfg: Config) -> str:
    """Chroma collection name, ``courses_{hash}``."""
    return f"courses_{config_hash(cfg)}"


def bm25_path(cfg: Config) -> Path:
    """BM25 pickle path, ``indexes/bm25_{hash}.pkl``."""
    return INDEXES_DIR / f"bm25_{config_hash(cfg)}.pkl"


def report_path(stem: str, cfg: Config) -> Path:
    """Report path, ``report/{stem}_{hash}.md`` — e.g. ``report_path("day1_smoke", cfg)``."""
    return REPORT_DIR / f"{stem}_{config_hash(cfg)}.md"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the config hash of a config.yaml.")
    parser.add_argument("path", nargs="?", default=DEFAULT_CONFIG_PATH, help="config file path")
    parser.add_argument("--canonical", action="store_true", help="also print the canonical dump")
    args = parser.parse_args(argv)
    cfg = load_config(args.path)
    print(config_hash(cfg))
    if args.canonical:
        print(canonical_dump(cfg))


if __name__ == "__main__":
    main()
