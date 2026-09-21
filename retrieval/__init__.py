"""One entry point for all three retrieval modes: ``retrieve(query, mode)``.

    from retrieval import retrieve

    retrieve("machine learning", mode="hybrid")   # mode in {"dense", "bm25", "hybrid"}

or from the shell::

    uv run --env-file .env python -m retrieval "machine learning" --mode hybrid

``dense`` and ``bm25`` each return their own top-k; ``hybrid`` asks both for ``fusion.top_k_each``
and fuses the two rankings with RRF, keeping ``fusion.top_k_final``. All three return the same
:class:`RetrievalResult`, whose ``hits`` are the same :class:`~retrieval.types.Hit` whichever mode
produced them -- that sameness is the whole point of this module. The Day 4 ablation calls
``retrieve`` directly and varies ``mode``, so a mode must not be a different code path with a
different return shape; anything a caller needs to tell them apart is *data* here (``mode``,
``config_hash``, ``hit.sources``), not a different function to call.

Nothing is loaded until a mode asks for it. Opening the Chroma collection and unpickling the 6 MB
BM25 index cost about a second each, and the retrievers do it in their constructors, so they are
built at most once per process and cached -- five queries in a report, or a hundred in an
ablation, pay it once. It also means ``retrieve(q, "bm25")`` never opens Chroma, never imports
``chromadb`` or ``openai``, and never needs an API key: the lexical half of the pipeline stays
usable in a checkout that has no key and no dense index.

``latency_ms`` measures the **query**, not the process warm-up: the retrievers are built before the
clock starts. A number that included a one-off index load on the first query and not on the next
four would not be the latency of anything -- and the load is exactly what the cache exists to stop
paying. A cold ``hybrid`` call still takes a second or two in wall-clock; that is the index
opening, and the CLI prints it on its own line when it happens.

Adding a mode -- ``keyword``, the `searchCourses()` baseline from AlmaBot's ``planner.ts``, is the
one Day 4 wants to compare against -- means adding a builder to :data:`BUILDERS` and an entry to
:data:`MODE_SOURCES`, and nothing else: a single-retriever mode runs through the same code the
other two do. It is out of scope for Day 1; the room is deliberate, the implementation is not here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Literal, NamedTuple

from config import Config, config_hash, load_config
from retrieval.fusion import check_limit, fuse, top_k_final
from retrieval.types import Hit, Retriever, top_k_each

#: The modes ``retrieve`` accepts. ``Literal`` so a typo is a type error before it is a runtime one.
Mode = Literal["dense", "bm25", "hybrid"]

#: The same three, as data -- for CLI choices, error messages, and the ablation's loop.
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid")

#: Which retrievers each mode asks, by name. The names are what a fused hit reports in ``sources``.
MODE_SOURCES: dict[Mode, tuple[str, ...]] = {
    "dense": ("dense",),
    "bm25": ("bm25",),
    "hybrid": ("dense", "bm25"),
}


# --- building the retrievers, once per process ----------------------------------------------------


def _build_dense(cfg: Config) -> Retriever:
    """The dense retriever, imported at call time.

    ``retrieval.dense`` pulls in ``chromadb`` and (through ``embedding``) ``openai``; importing it
    here rather than at the top of this module is what keeps a BM25-only query -- and a plain
    ``import retrieval.bm25``, which runs this file first -- clear of both.
    """
    from retrieval.dense import DenseRetriever

    return DenseRetriever(cfg)


def _build_bm25(cfg: Config) -> Retriever:
    """The BM25 retriever, imported at call time for the same reason as :func:`_build_dense`."""
    from retrieval.bm25 import BM25Retriever

    return BM25Retriever(cfg)


#: How to build each retriever, by the name :data:`MODE_SOURCES` refers to it by. A test injects a
#: fake through here; a new mode adds an entry.
BUILDERS: dict[str, Callable[[Config], Retriever]] = {
    "dense": _build_dense,
    "bm25": _build_bm25,
}

#: Built retrievers, keyed by ``(config hash, name)``. Module-level, so it is per process.
_RETRIEVERS: dict[tuple[str, str], Retriever] = {}


def get_retriever(name: str, cfg: Config) -> Retriever:
    """The named retriever for ``cfg``, built on first use and cached for the rest of the process.

    Keyed by the **full** config hash rather than by the retriever's own index key
    (``dense_key`` / ``bm25_key``). That is stricter than it has to be: an ablation over
    ``fusion.rrf_k`` reopens two indexes whose contents did not change. It costs a second per
    variant, and it means a cached retriever can never outlive a setting the caller has since
    changed -- a stale retriever would not raise, it would answer.

    A build that fails is not cached: a query run before ``build_bm25`` was run should work once
    the index is there, without restarting the REPL.
    """
    key = (config_hash(cfg), name)
    retriever = _RETRIEVERS.get(key)
    if retriever is None:
        try:
            build = BUILDERS[name]
        except KeyError:
            known = ", ".join(sorted(BUILDERS))
            raise KeyError(f"no retriever named {name!r}; known retrievers: {known}") from None
        retriever = build(cfg)
        _RETRIEVERS[key] = retriever
    return retriever


def clear_cache() -> None:
    """Drop every cached retriever, so the next query reopens its index.

    For a process that rebuilt an index after querying it -- a test, or a notebook that ran
    ``ingest.build_bm25`` -- where the cached retriever still holds the old one.
    """
    _RETRIEVERS.clear()


# --- the interface --------------------------------------------------------------------------------


class RetrievalResult(NamedTuple):
    """What every mode returns: the hits, and enough about the run to file them under.

    ``config_hash`` is the full hash, not either index key: a result is the product of the whole
    configuration -- including the fusion parameters, which name no index -- and #9's report and
    the Day 4 ablation are both filed under it.
    """

    #: The query as asked, before any tokenizing or embedding.
    query: str
    #: Which mode produced ``hits``.
    mode: Mode
    #: ``cfg.hash`` -- the settings this ranking came out of, for the report that quotes it.
    config_hash: str
    #: Best first, ``rank`` 1..n. In ``hybrid`` each hit's ``sources`` says which retrievers found
    #: it and where; in the other two it is ``None``, the mode already being the answer.
    hits: list[Hit]
    #: Wall-clock milliseconds for the query alone -- index loading is excluded, see the module
    #: docstring. For ``hybrid`` this is both retrievers, run one after the other.
    latency_ms: float


def check_mode(mode: str) -> Mode:
    """Refuse a mode this module does not implement, naming the ones it does."""
    if mode not in MODE_SOURCES:
        known = ", ".join(MODES)
        raise ValueError(f"unknown mode {mode!r}; modes are {known}")
    return mode  # type: ignore[return-value]


def result_k(cfg: Mapping[str, Any], k: int | None) -> int:
    """How many hits the caller gets back: ``k``, or ``fusion.top_k_final`` when it is ``None``.

    One default for all three modes, so a mode switch cannot quietly change the length of the
    list a report is comparing.
    """
    return top_k_final(cfg) if k is None else check_limit(k)


def retrieve(
    query: str,
    mode: Mode = "hybrid",
    k: int | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> RetrievalResult:
    """Run ``query`` through ``mode`` and return the top ``k`` hits.

    ``k`` defaults to ``fusion.top_k_final`` (5) in every mode. ``cfg`` defaults to the live
    ``config.yaml``; a plain mapping is accepted and frozen into a :class:`~config.Config`, so a
    caller's later edit to their own dict cannot change what a cached retriever was built from.

    In ``hybrid``, each retriever is asked for ``fusion.top_k_each`` (20) and the fused list is
    cut to ``k``. Asking for more than the recall depth deepens it -- ``k=50`` asks each retriever
    for 50, since the alternative is silently returning at most the 40 documents two 20-deep lists
    can hold between them. At the default ``k`` nothing about the tuned pipeline moves.

    Raises before touching an index: an empty query, an unknown mode, a ``k`` that is not a
    positive integer. A missing index or an unset API key surfaces as the retriever is built --
    with the command that fixes it, from ``retrieval.dense`` / ``retrieval.bm25``.
    """
    if not query.strip():
        raise ValueError("query is empty; nothing to retrieve")
    mode = check_mode(mode)
    config = cfg if isinstance(cfg, Config) else load_config() if cfg is None else Config(cfg)
    k = result_k(config, k)

    names = MODE_SOURCES[mode]
    retrievers = {name: get_retriever(name, config) for name in names}

    start = perf_counter()
    if mode == "hybrid":
        depth = max(top_k_each(config), k)
        rankings = {name: retriever.search(query, depth) for name, retriever in retrievers.items()}
        hits = fuse(rankings, config, k)
    else:
        hits = retrievers[names[0]].search(query, k)
    latency_ms = (perf_counter() - start) * 1000

    return RetrievalResult(
        query=query, mode=mode, config_hash=config.hash, hits=hits, latency_ms=latency_ms
    )


# --- printing it ----------------------------------------------------------------------------------

#: The table's columns, in order. ``#`` is the hit's rank, ``sources`` its provenance.
HEADERS = ("#", "doc_id", "score", "name", "sources")
#: Columns whose cells are numbers, and so are right-aligned.
NUMERIC = {0, 2}
#: Course titles run long (``Introduction to Algorithms & Models of Computation``); past this the
#: cell is elided, so the ``sources`` column stays where the eye left it on the row above.
NAME_WIDTH = 46
#: What a retriever that did not return the document prints in ``sources``.
MISSING = "—"


def hit_name(hit: Hit) -> str:
    """The course title, from the chunk's first line (``CS 374 — Introduction to Algorithms``).

    The title is not in ``metadata`` -- it is in the text, which is the one place the corpus keeps
    it -- so it is read back off the rendered chunk and the code prefix, which the ``doc_id``
    column already carries, is dropped. Falls back to the ``doc_id`` for a hit whose text a future
    template renders differently.
    """
    first = hit.text.splitlines()[0].strip() if hit.text else ""
    code = str(hit.metadata.get("code", ""))
    if code and first.startswith(code):
        first = first[len(code) :].lstrip(" —-").strip()
    return first or hit.doc_id


def hit_sources(hit: Hit, mode: Mode) -> str:
    """Where the hit came from: ``dense 3, bm25 —`` after fusion, ``bm25 1`` in a single mode.

    A single-retriever mode leaves ``sources`` unset -- the mode is the answer -- so the column is
    filled in from the mode rather than left blank, and the three modes read the same way down the
    page.
    """
    sources = (
        hit.sources if hit.sources is not None else dict.fromkeys(MODE_SOURCES[mode], hit.rank)
    )
    return ", ".join(
        f"{name} {MISSING if rank is None else rank}" for name, rank in sources.items()
    )


def format_table(result: RetrievalResult) -> str:
    """The hits as an aligned table: rank, doc_id, score, name, sources.

    ``score`` is printed to four places because RRF scores live in the third and fourth
    (``1/61 = 0.0164`` against ``1/62 = 0.0161``); at three, a fused list reads as a column of
    ties. It remains comparable only within one table -- the scales are the retriever's own.
    """
    if not result.hits:
        return f"no hits for {result.query!r} in {result.mode} mode"
    rows = [
        (
            str(hit.rank),
            hit.doc_id,
            f"{hit.score:.4f}",
            elide(hit_name(hit), NAME_WIDTH),
            hit_sources(hit, result.mode),
        )
        for hit in result.hits
    ]
    widths = [max(len(cell) for cell in column) for column in zip(HEADERS, *rows, strict=True)]
    return "\n".join(row_line(row, widths) for row in (HEADERS, *rows))


def row_line(row: tuple[str, ...], widths: list[int]) -> str:
    """One table row, numbers right-aligned and text left-aligned, with no trailing spaces."""
    cells = [
        cell.rjust(width) if i in NUMERIC else cell.ljust(width)
        for i, (cell, width) in enumerate(zip(row, widths, strict=True))
    ]
    return "  ".join(cells).rstrip()


def elide(text: str, width: int) -> str:
    """``text`` cut to ``width`` characters, ending in ``…`` when anything was dropped."""
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def format_result(result: RetrievalResult) -> str:
    """A summary line and the table -- what the CLI prints, and what #9's report quotes."""
    summary = (
        f"{result.query!r}  ->  {len(result.hits)} hits, mode {result.mode}, "
        f"config {result.config_hash}, {result.latency_ms:.1f} ms"
    )
    return f"{summary}\n\n{format_table(result)}"
