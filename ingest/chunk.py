"""Turn the exported corpus into ``data/chunks.jsonl`` -- the unit both indexes embed.

One course is one chunk. UIUC catalog descriptions are short -- the 2026-sp corpus renders at p50
111 tokens, p95 202, max 523, against an 8191-token embedding limit -- so splitting would only
scatter a single course's facts across several vectors and make a "what is CS 374 about" query
compete with itself. The token stats printed on every run are there to show that assumption still
holds when the data changes.

Run from the repo root, after ``ingest.export_almabot``::

    uv run python -m ingest.chunk                 # -> data/chunks.jsonl + data/chunks.meta.json
    uv run python -m ingest.chunk --show CS 374   # print one rendered chunk and exit

Each line is one JSON object::

    {"doc_id": "2026-sp:CS-374", "text": "CS 374 - ...", "metadata": {...}}

``doc_id`` is ``{term}:{SUBJ}-{NUM}``: readable in a ranked list, and term-scoped so a second
term can be added to the same collection without colliding.

The wording of ``text`` is a retrieval parameter, not a detail -- it is what gets embedded and
BM25-tokenized -- so it lives in :func:`render_chunk` alone, keyed by ``chunk.template_version``
in ``config.yaml``. Two templates ship: ``v1`` renders the catalog fields as they stand, ``v2``
drops the requirements block the catalog duplicates into ``desc`` and follows a
cross-reference-only description to the course that carries the real text. Switching between them
is one line of ``config.yaml`` and gets fresh index names for free -- the version is part of the
config hash -- so which one retrieves better is a Day 4 measurement, not a decision made here.
Nothing else in the pipeline may build chunk text.

Three properties the rest of Day 1 relies on, all asserted here rather than assumed downstream:
every ``doc_id`` is unique, every course carries the term ``config.yaml`` claims, and no chunk
exceeds the embedding input limit (#5 pays for a batch before the API rejects the long one).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import tiktoken

from config import HASH_LEN, Config, canonical_dump, load_config, rel_to_repo
from ingest.export_almabot import (
    COURSES_PATH,
    CROSS_REF_RE,
    DATA_DIR,
    REQUIREMENTS_PATH,
    matches_attrs,
)

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
#: Sidecar describing the run that wrote CHUNKS_PATH, so a consumer can refuse stale chunks.
CHUNKS_META_PATH = DATA_DIR / "chunks.meta.json"

#: The only ``chunk.strategy`` this module implements.
STRATEGY = "one_course_one_chunk"
#: Tokenizer of the text-embedding-3-* models; used for the size report, not for indexing.
TOKEN_ENCODING = "cl100k_base"
#: Hard per-input limit of text-embedding-3-small: the API rejects a longer chunk outright.
EMBEDDING_TOKEN_LIMIT = 8191
#: Chroma metadata values must be scalars, so list fields are joined on this.
LIST_SEPARATOR = "|"
#: The metadata fields that hold a list. Named rather than sniffed by type: after the join, a
#: decoder cannot tell "CS 173|CS 225" from a string that happens to contain a pipe.
LIST_FIELDS = ("prereq_codes", "gened_ids")

#: The label the catalog writes in front of the requirements block, which runs from there to the
#: end of ``desc``. Anchored at the end because ``v2`` strips it along with the block it opens.
REQUIREMENTS_LABEL_RE = re.compile(r"\s*Prerequisites?:\s*$")

#: The ``See X.`` sentence of a cross-reference-only description, naming the one course that holds
#: the real text. The subject is matched case-insensitively: the catalog writes "See Rel 514" in
#: one of the 497, which the export's upper-case ``CODE_RE`` misses.
SEE_RE = re.compile(r"\bSee\s+([A-Za-z]{2,4})\s?(\d{3})\b")

#: Course fields each template reads, in the order they appear in the rendered text. Checked
#: against ``chunk.fields`` in config.yaml, which is hashed: the hash must not claim a field set
#: the renderer does not actually use.
TEMPLATE_FIELDS: dict[str, tuple[str, ...]] = {
    "v1": ("code", "name", "credits", "term", "desc", "prereqRaw", "attrs"),
    # v2 reads the same seven fields *of this course*; what differs is how two of them are
    # rendered (see :func:`_render_v2`). The extra text it can pull in is another course's
    # ``desc``, which is already in the list. Identical lists are what make the switch one line:
    # ``chunk.fields`` is checked against this, so only ``template_version`` has to move.
    "v2": ("code", "name", "credits", "term", "desc", "prereqRaw", "attrs"),
}

#: Templates that follow a cross-reference to the course carrying the real description. Read by
#: :func:`summarize` so a build says which of the two things it did.
FOLLOWS_REFERENCES = frozenset({"v2"})

#: A ``{code: course}`` view of the corpus, for templates that read across courses.
Corpus = Mapping[str, Mapping[str, Any]]


# --- rendering ------------------------------------------------------------------------------------


def format_credits(credits: float) -> str:
    """Credit hours as they read in prose: ``3``, ``1.5``.

    ``0`` is a real value here -- ``parse_credits`` returns it for "hours to be arranged" -- and
    is rendered as written rather than hidden, so the chunk never implies a credit count the
    catalog does not give.
    """
    return f"{credits:g}"


def strip_requirements(desc: str, prereq_raw: str) -> str:
    """``desc`` without the trailing requirements block, its ``Prerequisite:`` label included.

    The catalog runs that block from the label to the end of the description, so ``prereqRaw`` is
    a *suffix* of ``desc`` -- true of all 2,732 courses in the 2026-sp corpus that have one.
    ``v1`` renders both fields as #4 specified, which embeds the text twice and doubles the BM25
    term frequency of every code and phrase inside it, biasing lexical retrieval toward courses
    with long prerequisite text. Removing it here leaves the topical text on the ``Description:``
    line and the requirements on the ``Prerequisites:`` one, each exactly once.

    A ``prereqRaw`` that is not a suffix is left alone. None are today, but the block is parsed
    out of prose: cutting on anything less than an exact match could truncate a real description.
    """
    if not prereq_raw or not desc.endswith(prereq_raw):
        return desc
    head = desc[: len(desc) - len(prereq_raw)]
    return REQUIREMENTS_LABEL_RE.sub("", head).rstrip()


def cross_reference(course: Mapping[str, Any]) -> str | None:
    """The course code a cross-reference-only description defers to, or ``None`` for a real one.

    497 descriptions (11% of the corpus) are nothing but ``"Same as HIST 283. See HIST 283."``:
    no topical text for either retriever to match on, only the code and the title. The ``Same as``
    clause may name up to seven sibling codes, but the ``See`` clause names exactly one -- the
    course the catalog files the real text under -- so that is the one read here.
    """
    desc = course["desc"]
    if not CROSS_REF_RE.match(desc):
        return None
    _, _, see = desc.partition(".")
    match = SEE_RE.search(see)
    if match is None:
        return None
    subject, number = match.groups()
    return f"{subject.upper()} {number}"


def inherited_desc(course: Mapping[str, Any], corpus: Corpus) -> tuple[str, str] | None:
    """``(source code, description)`` a cross-reference borrows, or ``None`` if it cannot.

    Three ways a reference does not resolve, all of which leave the course rendered as the
    catalog wrote it: it is not a cross-reference at all; the target is not offered this term
    (``BADM 205`` points at ``GGIS 205``, absent from the 2026-sp corpus); or the target is
    itself a cross-reference, which would only pass the pointer along. That last check is also
    what makes chains and mutual references safe without recursion -- this corpus has neither,
    but the guard costs one comparison and a later term may.

    The borrowed text goes through :func:`strip_requirements` too, so an inherited description
    never carries a requirements block: 274 of these sources have one, and it describes the
    *source*. This course's own ``prereqRaw`` is empty and so is its ``metadata.prereq_codes``;
    rendering the source's block here would have the chunk contradict its own metadata.
    """
    code = cross_reference(course)
    if code is None:
        return None
    source = corpus.get(code)
    if source is None or not source["desc"] or cross_reference(source) is not None:
        return None
    return code, strip_requirements(source["desc"], source["prereqRaw"])


def _render_v1(course: Mapping[str, Any], corpus: Corpus) -> str:
    """Template ``v1``: a labelled line per field, in ``TEMPLATE_FIELDS["v1"]`` order.

    Labels are spelled out ("Prerequisites", "Degree Attributes") because they are embedded along
    with the values and carry query vocabulary of their own. An empty field keeps its label --
    bare for ``desc`` (one course in the catalog has no description), with an explicit "None
    listed" / "None" for the two fields where absence is itself an answer ("does this need a
    prereq?").

    ``attrs`` is labelled as the catalog labels it, *not* "Gen Ed": the column also carries
    non-Gen-Ed attributes (ACE 436 and CMN 361 are "Global Classroom course." and satisfy no Gen
    Ed at all; ABE 232 mixes the two), so a "Gen Ed:" line would have the chunk assert a category
    the course does not hold. The authoritative answer is ``metadata.gened_ids``, computed from
    the rules in ``requirements.json``. ``v2`` keeps this line exactly as it is.

    ``corpus`` is unused: this template reads one course and nothing else. It is in the signature
    because every renderer shares one.
    """
    lines = [
        f"{course['code']} — {course['name']}",
        f"Credits: {format_credits(course['credits'])}",
        f"Term: {course['term']}",
        f"Description: {course['desc']}",
        f"Prerequisites: {course['prereqRaw'] or 'None listed'}",
        f"Degree Attributes: {course['attrs'] or 'None'}",
    ]
    return "\n".join(line.rstrip() for line in lines)


def _render_v2(course: Mapping[str, Any], corpus: Corpus) -> str:
    """Template ``v2``: ``v1``, minus the duplicated requirements block, plus resolved references.

    Two changes #4 left out of the baseline on purpose, so that each could be measured instead of
    assumed (#15). Both are described where they are implemented -- :func:`strip_requirements`
    and :func:`inherited_desc` -- and neither touches the other four lines, which render exactly
    as ``v1`` renders them.

    The ``Same as X. See X.`` sentence stays on the ``Description:`` line even when the reference
    resolves. It is the only place the sibling codes appear (``AAS 357`` lists four), so dropping
    it in favour of the borrowed text would trade one retrieval signal for another instead of
    adding one. The inherited text goes on its own line, labelled with the course it came from:
    a chunk never claims another course's description as its own, and a hit can be read back to
    its source.
    """
    lines = [
        f"{course['code']} — {course['name']}",
        f"Credits: {format_credits(course['credits'])}",
        f"Term: {course['term']}",
        f"Description: {strip_requirements(course['desc'], course['prereqRaw'])}",
    ]
    borrowed = inherited_desc(course, corpus)
    if borrowed is not None:
        source, text = borrowed
        lines.append(f"Description of {source}: {text}")
    lines += [
        f"Prerequisites: {course['prereqRaw'] or 'None listed'}",
        f"Degree Attributes: {course['attrs'] or 'None'}",
    ]
    return "\n".join(line.rstrip() for line in lines)


#: Every renderer, keyed by ``chunk.template_version``.
RENDERERS: dict[str, Callable[[Mapping[str, Any], Corpus], str]] = {
    "v1": _render_v1,
    "v2": _render_v2,
}


def render_chunk(
    course: Mapping[str, Any], template_version: str, corpus: Corpus | None = None
) -> str:
    """The text that gets embedded and BM25-tokenized, for one course.

    The single place chunk text is built, and pure -- courses in, string out -- so a template
    ablation can re-render the corpus without re-running the export.

    ``corpus`` is the ``{code: course}`` view a template needs to follow a reference to another
    course; :func:`build_chunks` passes the whole corpus, so every built chunk has it. It
    defaults to empty, where an unresolvable reference and an absent corpus render alike: the
    course as the catalog wrote it.
    """
    try:
        renderer = RENDERERS[template_version]
    except KeyError:
        known = ", ".join(sorted(RENDERERS))
        raise ValueError(
            f"unknown chunk.template_version {template_version!r}; known templates: {known}"
        ) from None
    return renderer(course, {} if corpus is None else corpus)


# --- ids and metadata -----------------------------------------------------------------------------


def doc_id(course: Mapping[str, Any]) -> str:
    """Stable, readable, term-scoped id: ``{term}:{SUBJ}-{NUM}``, e.g. ``2026-sp:CS-374``.

    Derived from fields the export writes, so re-running the export reproduces the same ids;
    nothing downstream may mint its own.
    """
    return f"{course['term']}:{course['subject']}-{course['number']}"


def load_gened_categories(path: Path | None = None) -> list[dict[str, Any]]:
    """The Gen Ed rules from ``data/requirements.json``, ready for :func:`matches_attrs`.

    Read from the file rather than from ``export_almabot.GENED`` so a chunk's ``gened_ids``
    describe the same export the corpus came from.
    """
    path = REQUIREMENTS_PATH if path is None else path
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)["genEd"]


def gened_ids(course: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ids of the Gen Ed categories this course satisfies, in ``requirements.json`` order."""
    return [c["id"] for c in categories if matches_attrs(c, course["attrs"])]


def chunk_metadata(
    course: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Filterable facts carried alongside the text (Chroma ``where``, and BM25 hits return it).

    Deliberately small: what a query filters or displays by, not a copy of the course record.
    ``prereq_codes`` is the unfiltered list from the export -- a reference to a course not offered
    this term stays visible here, unlike in ``prereqs.json``.
    """
    return {
        "code": course["code"],
        "subject": course["subject"],
        "number": course["number"],
        "credits": course["credits"],
        "term": course["term"],
        "prereq_codes": list(course["prereqCodes"]),
        "gened_ids": gened_ids(course, categories),
    }


def chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """``metadata`` with its list fields flattened, for a store that only holds scalars.

    Chroma rejects a list metadata value, so ``prereq_codes`` / ``gened_ids`` go in joined on
    ``LIST_SEPARATOR`` (``"CS 173|CS 225"``, ``""`` when empty). Flattening lives here so the
    dense and BM25 sides cannot disagree about the encoding; ``chunks.jsonl`` keeps the lists.

    A list field missing from :data:`LIST_FIELDS` is refused rather than flattened: it would
    write fine and come back from a query as a string nothing splits again.
    """
    unnamed = [k for k, v in metadata.items() if isinstance(v, list) and k not in LIST_FIELDS]
    if unnamed:
        raise ValueError(
            f"metadata field(s) {', '.join(unnamed)} hold a list but are not in LIST_FIELDS; "
            "add them there, or decode_chroma_metadata cannot restore them"
        )
    return {k: LIST_SEPARATOR.join(v) if isinstance(v, list) else v for k, v in metadata.items()}


def decode_chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`chroma_metadata`, for what a dense query reads back out of Chroma.

    BM25 (#6) takes its metadata straight from ``chunks.jsonl``, where the lists are still lists.
    The dense side only has what Chroma stored, so it splits them again here -- beside the join,
    so the two encodings cannot drift -- and a ``Hit`` means the same thing whichever retriever
    produced it. An empty string decodes to ``[]``, not ``[""]``.
    """
    return {
        k: [part for part in v.split(LIST_SEPARATOR) if part]
        if k in LIST_FIELDS and isinstance(v, str)
        else v
        for k, v in metadata.items()
    }


# --- building -------------------------------------------------------------------------------------


def check_chunk_config(cfg: Mapping[str, Any]) -> str:
    """Validate the ``chunk`` section against what this module implements; return the version.

    Every value here is part of the config hash, which names the indexes built from these chunks.
    A config claiming a strategy or field set the renderer does not honour would tag an index
    with a description of a corpus that was never built.

    An incomplete section is reported the same way as a wrong one: this function exists to turn a
    bad config into a sentence, so it must not be the place a bare ``KeyError`` escapes from.
    """
    if "chunk" not in cfg:
        raise ValueError(
            "config has no `chunk` section; it must set strategy, template_version and fields "
            "(see config.yaml) -- all three are hashed into the index names"
        )
    chunk = cfg["chunk"]
    missing = [key for key in ("strategy", "template_version", "fields") if key not in chunk]
    if missing:
        raise ValueError(
            f"config chunk section is missing {', '.join(missing)}; it must set strategy, "
            "template_version and fields (see config.yaml) -- all three are hashed into the "
            "index names"
        )
    if chunk["strategy"] != STRATEGY:
        raise ValueError(
            f"chunk.strategy is {chunk['strategy']!r}, but this module only implements {STRATEGY!r}"
        )
    version = chunk["template_version"]
    if version not in RENDERERS:
        known = ", ".join(sorted(RENDERERS))
        raise ValueError(f"unknown chunk.template_version {version!r}; known templates: {known}")
    fields = tuple(chunk["fields"])
    expected = TEMPLATE_FIELDS[version]
    if fields != expected:
        raise ValueError(
            f"chunk.fields is {list(fields)} but template {version} renders {list(expected)}; "
            "the hashed config must describe the chunks it names"
        )
    return version


def load_courses(path: Path | None = None) -> list[dict[str, Any]]:
    """The exported corpus, or a pointer at the command that writes it."""
    path = COURSES_PATH if path is None else path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{rel_to_repo(path)} is missing; run `uv run python -m ingest.export_almabot` first"
        )
    with path.open(encoding="utf-8") as f:
        courses = json.load(f)
    if not courses:
        raise ValueError(f"{rel_to_repo(path)} holds no courses; re-run the export")
    return courses


def check_terms(courses: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> None:
    """Fail unless every course carries the term ``config.yaml`` claims.

    ``data.term`` is hashed and goes into every ``doc_id``; a corpus exported under a different
    term would produce ids and index names that disagree about which term they describe. The
    export enforces this per section row -- re-checking here catches a stale ``courses.json``.
    """
    found = sorted({c["term"] for c in courses})
    expected = cfg["data"]["term"]
    if found != [expected]:
        raise ValueError(
            f"config data.term is {expected!r} but data/courses.json holds "
            f"{', '.join(repr(t) for t in found)}; re-run the export against the configured term"
        )


def build_chunks(
    courses: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    template_version: str,
) -> list[dict[str, Any]]:
    """One chunk per course, in ``courses`` order, with ``doc_id`` uniqueness enforced.

    The ``{code: course}`` view handed to the renderer is built once here, from the same corpus
    being chunked: a template that follows a reference resolves it against the courses going into
    this index, never against a different export.
    """
    corpus = {c["code"]: c for c in courses}
    chunks: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for course in courses:
        cid = doc_id(course)
        if cid in seen:
            raise ValueError(
                f"duplicate doc_id {cid!r} from {seen[cid]} and {course['code']}; "
                "the export should have de-duplicated these"
            )
        seen[cid] = course["code"]
        chunks.append(
            {
                "doc_id": cid,
                "text": render_chunk(course, template_version, corpus),
                "metadata": chunk_metadata(course, categories),
            }
        )
    return chunks


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one compact JSON object per line (JSONL: streamable, appendable, diff-friendly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def chunk_key(cfg: Mapping[str, Any]) -> str:
    """Short hash of just the config values that decide what a chunk *contains*.

    ``cfg.hash`` covers the whole file, so tuning ``fusion.rrf_k``, ``bm25.k1`` or ``seed`` --
    which a Day 4 ablation does on purpose -- would mark byte-identical chunks stale and train
    #5 / #6 to ignore the mismatch, disarming the guard for the case it was written for. This
    moves only when the term, strategy, template version or field list moves, which is exactly
    when ``chunks.jsonl`` has to be rebuilt. Consumers compare this; ``config_hash`` in the
    sidecar records which run wrote the file.
    """
    version = cfg["chunk"]["template_version"]
    content = {
        "term": cfg["data"]["term"],
        "strategy": STRATEGY,
        "template_version": version,
        "fields": list(TEMPLATE_FIELDS[version]),
    }
    return hashlib.sha256(canonical_dump(content).encode("utf-8")).hexdigest()[:HASH_LEN]


def chunks_meta(cfg: Config, chunks: Sequence[Mapping[str, Any]], stats: TokenStats | None) -> dict:
    """What the sidecar records: the settings this corpus was chunked under, plus its size.

    ``chunk_key`` is the point of it. The indexes built in #5 / #6 are named after the config
    hash while ``chunks.jsonl`` has a fixed name, so a rebuild under a changed ``chunk`` section
    would otherwise leave a stale file that a later index build reads without noticing.

    ``tokens`` is ``null`` when the tokenizer vocabulary was unavailable (see
    :func:`default_encode`) -- which also means this corpus was never size-checked against
    ``EMBEDDING_TOKEN_LIMIT``.
    """
    version = cfg["chunk"]["template_version"]
    return {
        "chunk_key": chunk_key(cfg),
        "config_hash": cfg.hash,
        "term": cfg["data"]["term"],
        "strategy": STRATEGY,
        "template_version": version,
        "fields": list(TEMPLATE_FIELDS[version]),
        "chunks": len(chunks),
        "tokens": None
        if stats is None
        else {
            "encoding": TOKEN_ENCODING,
            "total": stats.total,
            "p50": stats.p50,
            "p95": stats.p95,
            "max": stats.largest,
            "limit": EMBEDDING_TOKEN_LIMIT,
        },
    }


def write_meta(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the sidecar indented -- it is small and meant to be read by a human debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def default_encode() -> Callable[[str], Sequence[int]] | None:
    """The ``cl100k_base`` tokenizer, or ``None`` if its vocabulary cannot be had.

    tiktoken fetches the vocabulary over the network on first use and caches it. The chunks are
    the product of this module and the token report is a diagnostic, so a fresh clone with no
    network writes the corpus with an empty ``tokens`` block instead of failing the build after
    rendering all 4,502 chunks.
    """
    try:
        return tiktoken.get_encoding(TOKEN_ENCODING).encode
    except Exception as exc:  # noqa: BLE001 -- any failure to fetch a diagnostic is degradable
        print(
            f"warning: no {TOKEN_ENCODING} vocabulary ({type(exc).__name__}: {exc}); "
            "writing chunks without the size report, so they are NOT checked against the "
            f"{EMBEDDING_TOKEN_LIMIT}-token limit",
            file=sys.stderr,
        )
        return None


def check_sizes(stats: TokenStats) -> None:
    """Fail on a chunk the embeddings API would reject, before anything is written.

    Asserted like the other two invariants rather than merely printed: a corpus with one
    9,000-token description otherwise writes cleanly and exits 0, and #5 discovers it mid-batch,
    after paying for the chunks ahead of it.
    """
    if stats.over_limit:
        listed = ", ".join(stats.over_limit[:5])
        more = "" if len(stats.over_limit) <= 5 else f" (+{len(stats.over_limit) - 5} more)"
        raise ValueError(
            f"{len(stats.over_limit)} chunk(s) exceed the {EMBEDDING_TOKEN_LIMIT}-token "
            f"embedding limit and would be rejected: {listed}{more}; split them or render a "
            "shorter template (a new chunk.template_version) before building an index"
        )


def chunk(cfg: Config, encode: Callable[[str], Sequence[int]] | None = None) -> dict[str, Any]:
    """Read the exported corpus, write ``data/chunks.jsonl`` + its sidecar, and report sizes.

    ``encode`` defaults to :func:`default_encode`; pass one in to count tokens with anything
    else. Everything that can fail is checked before the first write, so a failed run leaves the
    previous ``chunks.jsonl`` in place rather than a half-written one.
    """
    version = check_chunk_config(cfg)
    courses = load_courses()
    check_terms(courses, cfg)
    categories = load_gened_categories()
    chunks = build_chunks(courses, categories, version)
    if encode is None:
        encode = default_encode()
    stats = None if encode is None else token_stats(chunks, encode)
    if stats is not None:
        check_sizes(stats)
    write_jsonl(CHUNKS_PATH, chunks)
    write_meta(CHUNKS_META_PATH, chunks_meta(cfg, chunks, stats))
    return {
        "chunks": chunks,
        "courses": courses,
        "template_version": version,
        "stats": stats,
    }


# --- size report ----------------------------------------------------------------------------------


class TokenStats(NamedTuple):
    """Chunk sizes in embedding tokens -- the budget #5 pays for and the limit it must fit."""

    count: int
    total: int
    p50: int
    p95: int
    largest: int
    #: doc_ids over :data:`EMBEDDING_TOKEN_LIMIT`, which the embeddings API would reject.
    over_limit: list[str]


def percentile(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted sequence (no interpolation).

    A real chunk length, which is what a size budget wants, rather than an average of two.
    """
    if not sorted_values:
        return 0
    rank = math.ceil(q * len(sorted_values))
    return sorted_values[min(max(rank, 1), len(sorted_values)) - 1]


def token_stats(
    chunks: Sequence[Mapping[str, Any]],
    encode: Callable[[str], Sequence[int]],
) -> TokenStats:
    """Token-count distribution over rendered chunks.

    ``encode`` is injected so the report can be produced with any tokenizer -- and so tests do not
    need the ``cl100k_base`` vocabulary file, which tiktoken downloads on first use.
    """
    counted = [(len(encode(c["text"])), c["doc_id"]) for c in chunks]
    lengths = sorted(n for n, _ in counted)
    return TokenStats(
        count=len(lengths),
        total=sum(lengths),
        p50=percentile(lengths, 0.50),
        p95=percentile(lengths, 0.95),
        largest=lengths[-1] if lengths else 0,
        over_limit=[doc for n, doc in counted if n > EMBEDDING_TOKEN_LIMIT],
    )


def crossref_counts(courses: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(cross-reference-only descriptions, how many of them resolve)``.

    The second number is a property of the corpus, not of the template: it is what a template
    that follows references *could* pick up. :func:`summarize` reports it either way, so a ``v1``
    build says out loud what it is leaving on the table.
    """
    corpus = {c["code"]: c for c in courses}
    total = sum(1 for c in courses if CROSS_REF_RE.match(c["desc"]))
    resolvable = sum(1 for c in courses if inherited_desc(c, corpus) is not None)
    return total, resolvable


def summarize(result: Mapping[str, Any]) -> list[str]:
    """Human-readable build stats: counts, token budget, metadata coverage.

    Reads the stats out of ``result`` rather than taking them alongside it, so there is one path
    to them and no way to describe one corpus with another's numbers.
    """
    chunks = result["chunks"]
    version = result["template_version"]
    stats = result["stats"]
    with_prereqs = sum(1 for c in chunks if c["metadata"]["prereq_codes"])
    with_gened = sum(1 for c in chunks if c["metadata"]["gened_ids"])
    crossrefs, resolvable = crossref_counts(result["courses"])
    followed = (
        f"{resolvable} resolved, {crossrefs - resolvable} left as written"
        if version in FOLLOWS_REFERENCES
        else f"not followed by template {version}"
    )
    tokens = (
        [f"tokens ({TOKEN_ENCODING})        unavailable, chunks not size-checked"]
        if stats is None
        else [
            f"tokens ({TOKEN_ENCODING})",
            f"  total                {stats.total}  (what #5 pays to embed once)",
            f"  p50 / p95 / max      {stats.p50} / {stats.p95} / {stats.largest}",
            (
                f"  over {EMBEDDING_TOKEN_LIMIT} limit        {len(stats.over_limit)}"
                f"  (headroom {EMBEDDING_TOKEN_LIMIT - stats.largest})"
            ),
        ]
    )
    return [
        f"courses read           {len(result['courses'])}",
        f"chunks written         {len(chunks)}  (strategy {STRATEGY}, template {version})",
        f"unique doc_ids         {len({c['doc_id'] for c in chunks})}",
        *tokens,
        f"metadata prereq_codes  {with_prereqs} chunks",
        f"metadata gened_ids     {with_gened} chunks",
        f"cross-reference descs  {crossrefs}  ({followed})",
    ]


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chunk data/courses.json into data/chunks.jsonl, one chunk per course."
    )
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    parser.add_argument(
        "--show",
        nargs="+",
        metavar="CODE",
        help="print the rendered chunk for one course (e.g. --show CS 374) and exit",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    if args.show:
        wanted = " ".join(args.show).upper()
        version = check_chunk_config(cfg)
        courses = load_courses()
        categories = load_gened_categories()
        match = next((c for c in courses if c["code"].upper() == wanted), None)
        if match is None:
            raise SystemExit(f"no course {wanted!r} in {rel_to_repo(COURSES_PATH)}")
        print(f"doc_id   {doc_id(match)}")
        print(f"metadata {json.dumps(chunk_metadata(match, categories), ensure_ascii=False)}")
        print(f"chroma   {json.dumps(chroma_metadata(chunk_metadata(match, categories)))}")
        print("text")
        print(render_chunk(match, version, {c["code"]: c for c in courses}))
        return

    result = chunk(cfg)
    print(f"config hash {cfg.hash}  (chunk key {chunk_key(cfg)})")
    for line in summarize(result):
        print(line)
    for path in (CHUNKS_PATH, CHUNKS_META_PATH):
        print(f"wrote    {rel_to_repo(path)}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
