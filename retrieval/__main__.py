"""The CLI for :func:`retrieval.retrieve`: one query, one mode, one table.

    uv run python -m retrieval "CS 374" --mode bm25
    uv run --env-file .env python -m retrieval "machine learning" --mode hybrid -k 10

``dense`` and ``hybrid`` embed the query, so they need ``OPENAI_API_KEY`` (hence ``--env-file``,
until #11 loads ``.env`` by itself) and a built dense index; ``bm25`` needs neither. Whichever is
missing is reported by the retriever that wanted it, with the command that builds it.

It is one query per invocation on purpose: the process exits afterwards, so the in-process cache
buys nothing here and every run pays the index load. Five queries in a row belong in #9's smoke
report, which calls ``retrieve`` in a loop and pays it once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from time import perf_counter

from config import load_config
from retrieval import MODE_SOURCES, MODES, format_result, retrieve


def duration(ms: float) -> str:
    """Milliseconds as the unit a human would have said: ``84 ms``, ``1.4 s``."""
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def main(argv: Sequence[str] | None = None) -> None:
    """Run one query in one mode and print its hits, then what opening the indexes cost.

    The opening time is printed separately and only when it rounds to a millisecond or more: it is
    paid once per process, so folding it into the query time would make every single-query run
    look far slower than the same query inside a report.
    """
    parser = argparse.ArgumentParser(
        prog="python -m retrieval",
        description="Query the course corpus in one of three modes and print the top hits.",
    )
    parser.add_argument("query", nargs="+", help="query text")
    parser.add_argument(
        "--mode", choices=MODES, default="hybrid", help="retrieval mode (default: hybrid)"
    )
    parser.add_argument(
        "-k", type=int, default=None, help="how many hits (default: fusion.top_k_final)"
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    query = " ".join(args.query)
    started = perf_counter()
    result = retrieve(query, args.mode, args.k, cfg)
    wall_ms = (perf_counter() - started) * 1000

    print(format_result(result))
    opening_ms = wall_ms - result.latency_ms
    if opening_ms >= 1:
        indexes = " and ".join(MODE_SOURCES[result.mode])
        print(
            f"\n(plus {duration(opening_ms)} opening the {indexes} index, once per process -- "
            "not counted in the query time above)"
        )


if __name__ == "__main__":
    sys.exit(main())
