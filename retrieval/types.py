"""What every retriever returns, so RRF can fuse two lists without knowing where they came from.

Dense (#5) and BM25 (#6) agree on this shape; fusion (#7) merges them; ``retrieve(query, mode)``
(#8) hands it to the caller. Keeping it in one module -- rather than each retriever returning its
own dict -- is what makes ``mode`` a parameter instead of three code paths.

The one thing to know about ``score``: it is **not** comparable across retrievers. A cosine
similarity of 0.42 and a BM25 score of 11.3 say nothing about each other, and even within one
retriever the scale shifts with the query. That is exactly why RRF fuses on ``rank``, and why
``rank`` is carried here rather than recomputed from a sorted list downstream.

``sources`` is the one field a retriever never fills: fusion (#7) sets it to the rank each
retriever gave the document, so a fused list still says where every result came from. Hits compare
as tuples, so a retriever's hit and the fused copy of it are unequal -- which is right, they are
different claims about the same chunk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol


class Hit(NamedTuple):
    """One retrieved chunk, as one retriever ranked it -- or, after fusion, as several did."""

    #: ``{term}:{SUBJ}-{NUM}``, minted by ``ingest.chunk.doc_id`` and nowhere else.
    doc_id: str
    #: Higher is better, whatever the retriever's own scale is. Only meaningful within one list.
    score: float
    #: 1-based position in that list.
    rank: int
    #: The chunk text that was indexed, i.e. what the score is about.
    text: str
    #: ``ingest.chunk.chunk_metadata`` shape: lists are lists, on both retrievers.
    metadata: dict[str, Any]
    #: Fusion only: the rank each retriever gave this document, ``None`` where it did not return
    #: it at all -- ``{"dense": 3, "bm25": None}``. ``None`` on a hit that came straight from one
    #: retriever, which already knows which one it is.
    sources: dict[str, int | None] | None = None


class Retriever(Protocol):
    """A thing that ranks chunks for a query.

    ``k`` defaults to ``fusion.top_k_each`` from the config, so a caller that does not care about
    recall depth gets the depth the fusion step was tuned for.
    """

    def search(self, query: str, k: int | None = None) -> list[Hit]: ...


def format_hit(hit: Hit) -> str:
    """One line per hit, for a CLI: rank, score, and the chunk's first line (``CODE - Title``).

    Shared by both retrievers' ``-m`` entry points so two rankings of the same corpus can be read
    side by side. ``score`` is printed to three places whatever its scale -- a cosine similarity
    just under 1, a BM25 score around 30 -- because the number is only ever meaningful against the
    others in its own list.
    """
    title = hit.text.splitlines()[0] if hit.text else hit.doc_id
    return f"{hit.rank:>3}. {hit.score:.3f}  {title}"


def fusion_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``fusion`` section, or a sentence saying what it is supposed to hold."""
    if "fusion" not in cfg:
        raise ValueError(
            "config has no `fusion` section; it must set method, rrf_k, top_k_each and "
            "top_k_final -- how the two rankings are combined (see config.yaml)"
        )
    return cfg["fusion"]


def fusion_int(cfg: Mapping[str, Any], key: str, gloss: str, minimum: int = 1) -> int:
    """One whole-number ``fusion`` setting, or a sentence naming the section, the key and the file.

    ``top_k_each`` below and ``fusion.rrf_k`` / ``fusion.top_k_final`` in ``retrieval.fusion`` all
    read through here: three modules take four values out of one section, and a config that never
    set one of them should read the same way whichever module touches it first. A bare
    ``KeyError`` from whichever line got there first names neither the key nor ``config.yaml``,
    and ``config.yaml`` is the file the reader has to go and fix.

    ``minimum`` is 1 for anything counting hits -- a retriever asked for zero hits is a typo, not
    a setting -- and 0 for ``rrf_k``, where zero damping is degenerate but meaningful.
    """
    section = fusion_section(cfg)
    if key not in section:
        raise ValueError(f"config fusion section is missing {key} -- {gloss} (see config.yaml)")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        wanted = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
        raise ValueError(f"fusion.{key} must be {wanted}, got {value!r}")
    return value


def top_k_each(cfg: Mapping[str, Any]) -> int:
    """``fusion.top_k_each`` -- the recall depth every retriever defaults to.

    Read through one guard rather than indexed in each constructor, so dense (#5), BM25 (#6) and
    fusion (#7) get the same number and, when a config never set it, the same sentence.
    """
    return fusion_int(cfg, "top_k_each", "how many hits each retriever returns before fusion")
