"""Tests for ingest/chunk.py: the rendered text, doc_ids, metadata, and the config guards.

The corpus under test is built from ``tests/fixtures/catalog_sample.csv`` through the real
exporter, so a change to the export that would alter chunk text (a new field, different prereq
parsing) shows up here as a failing golden string rather than as a quietly re-embedded corpus.

``redirect_paths`` is autouse and points every path the module touches -- inputs *and* outputs --
into tmp_path: no test can read the live 3 MB corpus by accident, and none can overwrite the
``data/chunks.jsonl`` a later index build reads.

``token_stats`` is always driven with a fake tokenizer; the real ``cl100k_base`` vocabulary is a
download, and none of the properties tested here depend on which tokenizer counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Config, load_config
from ingest import chunk as ch
from ingest import export_almabot as ex

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "catalog_sample.csv"
BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"
PATHS = ("COURSES_PATH", "REQUIREMENTS_PATH", "CHUNKS_PATH", "CHUNKS_META_PATH")

#: The chunk text template v1 is expected to produce, for a course that fills every field.
CS374_CHUNK = """CS 374 — Introduction to Algorithms & Models
Credits: 4
Term: 2026-sp
Description: Analysis of algorithms, major paradigms of algorithm design including recursive \
algorithms, divide-and-conquer algorithms, dynamic programming, greedy algorithms, and graph \
algorithms. Formal models of computation including finite automata and Turing machines. Same as \
ECE 374. Prerequisite: One of CS 173, MATH 213; CS 225.
Prerequisites: One of CS 173, MATH 213; CS 225.
Degree Attributes: None"""

#: ...and for one that fills none of the optional ones.
CWL593_CHUNK = """CWL 593 — Directed Research
Credits: 1
Term: 2026-sp
Description:
Prerequisites: None listed
Degree Attributes: None"""

#: How ``v1`` renders a field whose value is not the bare string. Keyed by course field, so the
#: coverage test can be derived from TEMPLATE_FIELDS instead of listing fields by hand.
FIELD_AS_RENDERED = {"credits": ch.format_credits}


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch) -> dict[str, Path]:
    """Point every module-level path at tmp_path, for every test in this module."""
    paths = {name: tmp_path / f"{name.removesuffix('_PATH').lower()}.json" for name in PATHS}
    paths["CHUNKS_PATH"] = tmp_path / "chunks.jsonl"
    paths["CHUNKS_META_PATH"] = tmp_path / "chunks.meta.json"
    for name, path in paths.items():
        monkeypatch.setattr(ch, name, path)
    return paths


@pytest.fixture
def cfg() -> Config:
    """Config for the fixture corpus, with a ``chunk`` section spelled out here.

    Not read from the live ``config.yaml``: tuning it must not break the suite. That the shipped
    config is still one this module can build from is its own test, below.
    """
    raw = load_config(BASE_CONFIG).to_dict()
    raw["data"]["term"] = "2026-sp"
    raw["chunk"] = {
        "strategy": ch.STRATEGY,
        "template_version": "v1",
        "fields": list(ch.TEMPLATE_FIELDS["v1"]),
    }
    return Config(raw)


@pytest.fixture
def courses() -> list[dict]:
    return ex.read_courses(FIXTURE_CSV).courses


@pytest.fixture
def by_code(courses) -> dict[str, dict]:
    return {c["code"]: c for c in courses}


@pytest.fixture
def categories(courses, cfg) -> list[dict]:
    """Gen Ed rules in the shape ``requirements.json`` carries them."""
    return ex.build_requirements(courses, cfg)["genEd"]


@pytest.fixture
def corpus(courses, categories, cfg, redirect_paths) -> dict[str, Path]:
    """Write the fixture corpus where :func:`ch.chunk` will look for it."""
    ex.write_json(redirect_paths["COURSES_PATH"], courses)
    ex.write_json(redirect_paths["REQUIREMENTS_PATH"], {"genEd": categories})
    return redirect_paths


def fake_encode(text: str) -> list[int]:
    """A whitespace tokenizer: deterministic, offline, and enough to exercise the stats."""
    return [len(w) for w in text.split()]


@pytest.fixture
def no_tokenizer(monkeypatch) -> None:
    """tiktoken with no vocabulary available -- a fresh clone with no network."""

    def offline(name: str) -> None:
        raise RuntimeError(f"could not fetch {name}")

    monkeypatch.setattr(ch.tiktoken, "get_encoding", offline)


# --- rendering ----------------------------------------------------------------------------------


def test_render_chunk_v1_is_stable(by_code):
    # Golden: this exact string is what gets embedded and BM25-tokenized. Changing it changes
    # retrieval, so it must go through a new template_version, not through an edit here.
    assert ch.render_chunk(by_code["CS 374"], "v1") == CS374_CHUNK


def test_render_chunk_v1_labels_absent_fields(by_code):
    # No description, no requirements block, no Degree Attributes.
    assert ch.render_chunk(by_code["CWL 593"], "v1") == CWL593_CHUNK
    assert "Description:\n" in CWL593_CHUNK  # bare label, not a trailing space


def test_render_chunk_covers_every_configured_field(courses):
    # Derived from TEMPLATE_FIELDS, not from a hand-written list: adding a field to the hashed
    # field set and forgetting it in _render_v1 must fail here. check_chunk_config only compares
    # the config against the constant, so this is the only check that the template renders them.
    for field in ch.TEMPLATE_FIELDS["v1"]:
        # A course that actually fills the field -- an empty value would assert nothing, and no
        # single fixture course fills all seven (none has both a prereq block and attributes).
        course = next((c for c in courses if c[field]), None)
        assert course is not None, f"no fixture course fills {field}"
        rendered = FIELD_AS_RENDERED.get(field, str)(course[field])
        assert rendered in ch.render_chunk(course, "v1"), field


def test_render_chunk_labels_attrs_as_the_catalog_does(by_code):
    # Not "Gen Ed": the column also carries non-Gen-Ed attributes, so that label would have the
    # embedded text claim a category the course does not hold. gened_ids is the real answer.
    text = ch.render_chunk(by_code["ECON 102"], "v1")
    assert "Degree Attributes: Social & Beh Sci - Soc Sci course." in text
    assert "Gen Ed" not in text


def test_render_chunk_rejects_an_unknown_template(by_code):
    with pytest.raises(ValueError, match="unknown chunk.template_version 'v9'"):
        ch.render_chunk(by_code["CS 374"], "v9")


@pytest.mark.parametrize(
    ("credits", "expected"),
    [(3.0, "3"), (4.0, "4"), (1.5, "1.5"), (0.5, "0.5"), (0.0, "0")],
)
def test_format_credits_drops_a_trailing_zero(credits, expected):
    assert ch.format_credits(credits) == expected


# --- doc_id -------------------------------------------------------------------------------------


def test_doc_id_is_term_scoped_and_readable(by_code):
    assert ch.doc_id(by_code["CS 374"]) == "2026-sp:CS-374"
    assert ch.doc_id(by_code["AAS 100"]) == "2026-sp:AAS-100"


def test_doc_ids_are_unique_across_the_catalog(courses, categories):
    chunks = ch.build_chunks(courses, categories, "v1")
    ids = [c["doc_id"] for c in chunks]
    assert len(ids) == len(courses) == len(set(ids))


def test_build_chunks_rejects_a_duplicate_doc_id(courses, categories):
    # Same subject+number in the same term, which the export should never emit.
    twin = {**courses[0], "code": "CS 999"}
    with pytest.raises(ValueError, match="duplicate doc_id"):
        ch.build_chunks([*courses, twin], categories, "v1")


# --- metadata -----------------------------------------------------------------------------------


def test_metadata_carries_exactly_the_filterable_fields(by_code, categories):
    meta = ch.chunk_metadata(by_code["CS 374"], categories)
    assert meta == {
        "code": "CS 374",
        "subject": "CS",
        "number": 374,
        "level": 300,
        "credits": 4.0,
        "term": "2026-sp",
        # Unfiltered: MATH 213 is not offered this term but stays visible in the metadata.
        "prereq_codes": ["CS 173", "CS 225", "MATH 213"],
        "gened_ids": [],
    }


def test_gened_ids_follow_the_exported_rules(by_code, categories):
    assert ch.gened_ids(by_code["STAT 100"], categories) == ["qr1"]
    assert ch.gened_ids(by_code["MATH 101"], categories) == ["qr2"]  # not qr1: the rule excludes it
    assert ch.gened_ids(by_code["CS 374"], categories) == []
    assert "cultus" in ch.gened_ids(by_code["AAS 100"], categories)


def test_gened_ids_read_the_file_the_corpus_came_with(corpus, by_code):
    # Rules come from requirements.json, not from the GENED constant, so the ids on a chunk
    # describe the same export the courses came from.
    ex.write_json(corpus["REQUIREMENTS_PATH"], {"genEd": []})
    assert ch.gened_ids(by_code["STAT 100"], ch.load_gened_categories()) == []


def test_catalog_number_is_an_int_and_level_rounds_it_down(by_code, categories):
    # Chroma compares numbers, not numeric strings: "400-level" has to be an int to be filterable.
    meta = ch.chunk_metadata(by_code["STAT 100"], categories)
    assert (meta["number"], meta["level"]) == (100, 100)


def test_catalog_number_refuses_a_number_that_is_not_digits(by_code, categories):
    # The export copies the Number column verbatim; one string among the ints would be skipped by
    # every range filter without saying so.
    odd = {**by_code["CS 374"], "number": "374X"}
    with pytest.raises(ValueError, match="not digits"):
        ch.chunk_metadata(odd, categories)


def test_chroma_metadata_keeps_the_lists_as_lists(by_code, categories):
    # Chroma 1.5.9 stores a non-empty list and filters it with $contains; joining the values into
    # one string left them matchable only in full.
    stored = ch.chroma_metadata(ch.chunk_metadata(by_code["CS 374"], categories))
    assert stored["prereq_codes"] == ["CS 173", "CS 225", "MATH 213"]
    assert stored["credits"] == 4.0  # scalars are left alone


def test_chroma_metadata_drops_an_empty_list_because_chroma_refuses_one(by_code, categories):
    # "Expected metadata list value ... to be non-empty in add" -- CS 374 satisfies no Gen Ed.
    stored = ch.chroma_metadata(ch.chunk_metadata(by_code["CS 374"], categories))
    assert "gened_ids" not in stored
    assert all(v != [] for v in stored.values())


def test_chroma_metadata_round_trips_through_the_decoder(by_code, categories):
    # A dense hit must carry the same metadata a BM25 hit does, or `Hit` means two things.
    metadata = ch.chunk_metadata(by_code["CS 374"], categories)
    assert ch.decode_chroma_metadata(ch.chroma_metadata(metadata)) == metadata


def test_decode_chroma_metadata_restores_a_dropped_field_as_an_empty_list():
    decoded = ch.decode_chroma_metadata({"code": "CS 374", "prereq_codes": ["CS 173"]})
    assert decoded == {"code": "CS 374", "prereq_codes": ["CS 173"], "gened_ids": []}


def test_decode_chroma_metadata_leaves_scalars_alone():
    # The two list fields are restored; every scalar comes back as it went in, the int included.
    decoded = ch.decode_chroma_metadata({"code": "CS 374", "credits": 4.0, "number": 374})
    assert decoded == {
        "code": "CS 374",
        "credits": 4.0,
        "number": 374,
        "prereq_codes": [],
        "gened_ids": [],
    }


def test_chroma_metadata_refuses_a_list_field_the_decoder_could_not_restore():
    # Chroma accepts it; the query side would read back a string nothing splits again.
    with pytest.raises(ValueError, match="LIST_FIELDS"):
        ch.chroma_metadata({"code": "CS 374", "instructors": ["Chekuri"]})


# --- config guards ------------------------------------------------------------------------------


def test_check_chunk_config_returns_the_template_version(cfg):
    assert ch.check_chunk_config(cfg) == "v1"


def test_the_shipped_config_is_one_this_module_can_build_from():
    # The live config.yaml, not a fixture: its chunk section names the strategy, template and
    # field list this module implements, and all three are hashed into the index names.
    assert ch.check_chunk_config(load_config()) in ch.RENDERERS


def test_check_chunk_config_rejects_another_strategy(cfg):
    other = Config({**cfg.to_dict(), "chunk": {**cfg["chunk"], "strategy": "sliding_window"}})
    with pytest.raises(ValueError, match="chunk.strategy"):
        ch.check_chunk_config(other)


def test_check_chunk_config_rejects_an_unknown_template(cfg):
    other = Config({**cfg.to_dict(), "chunk": {**cfg["chunk"], "template_version": "v9"}})
    with pytest.raises(ValueError, match="unknown chunk.template_version"):
        ch.check_chunk_config(other)


def test_check_chunk_config_rejects_fields_the_template_does_not_render(cfg):
    # The field list is hashed, so it must describe the chunks the hash names.
    fields = [*ch.TEMPLATE_FIELDS["v1"], "sectionCount"]
    other = Config({**cfg.to_dict(), "chunk": {**cfg["chunk"], "fields": fields}})
    with pytest.raises(ValueError, match="chunk.fields"):
        ch.check_chunk_config(other)


def test_check_chunk_config_reports_a_missing_section_as_a_sentence(cfg):
    # This function's job is turning a bad config into an actionable error, so a config it cannot
    # read must not escape as a bare KeyError from inside it.
    raw = cfg.to_dict()
    raw.pop("chunk")
    with pytest.raises(ValueError, match="no `chunk` section"):
        ch.check_chunk_config(Config(raw))


def test_check_chunk_config_names_the_keys_it_needs(cfg):
    partial = {k: v for k, v in cfg["chunk"].items() if k != "template_version"}
    with pytest.raises(ValueError, match="missing template_version"):
        ch.check_chunk_config(Config({**cfg.to_dict(), "chunk": partial}))


def test_check_terms_rejects_a_corpus_from_another_term(courses, cfg):
    stale = [{**c, "term": "2025-fa"} for c in courses]
    with pytest.raises(ValueError, match="2025-fa"):
        ch.check_terms(stale, cfg)


def test_check_terms_accepts_the_configured_term(courses, cfg):
    ch.check_terms(courses, cfg)


def test_load_courses_points_at_the_export_when_the_corpus_is_missing():
    with pytest.raises(FileNotFoundError, match="ingest.export_almabot"):
        ch.load_courses()


def test_load_courses_rejects_an_empty_corpus(corpus):
    ex.write_json(corpus["COURSES_PATH"], [])
    with pytest.raises(ValueError, match="no courses"):
        ch.load_courses()


# --- token stats --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0.0, 1), (0.5, 5), (0.95, 10), (1.0, 10)],
)
def test_percentile_is_nearest_rank(q, expected):
    assert ch.percentile(list(range(1, 11)), q) == expected


def test_percentile_of_nothing_is_zero():
    assert ch.percentile([], 0.5) == 0


def test_token_stats_describe_the_rendered_chunks(courses, categories):
    chunks = ch.build_chunks(courses, categories, "v1")
    stats = ch.token_stats(chunks, fake_encode)
    lengths = sorted(len(fake_encode(c["text"])) for c in chunks)
    assert stats.count == len(chunks)
    assert stats.total == sum(lengths)
    assert (stats.p50, stats.largest) == (ch.percentile(lengths, 0.5), lengths[-1])
    assert stats.p50 <= stats.p95 <= stats.largest
    assert stats.over_limit == []


def test_token_stats_name_the_chunks_over_the_embedding_limit():
    oversized = {"doc_id": "2026-sp:CS-999", "text": "x " * (ch.EMBEDDING_TOKEN_LIMIT + 1)}
    small = {"doc_id": "2026-sp:CS-100", "text": "x"}
    stats = ch.token_stats([small, oversized], fake_encode)
    assert stats.over_limit == ["2026-sp:CS-999"]


def test_check_sizes_names_the_offenders_and_the_fix():
    stats = ch.token_stats(
        [{"doc_id": "2026-sp:CS-999", "text": "x " * (ch.EMBEDDING_TOKEN_LIMIT + 1)}], fake_encode
    )
    with pytest.raises(ValueError, match="2026-sp:CS-999"):
        ch.check_sizes(stats)


def test_check_sizes_passes_a_corpus_within_the_limit(courses, categories):
    ch.check_sizes(ch.token_stats(ch.build_chunks(courses, categories, "v1"), fake_encode))


def test_default_encode_degrades_instead_of_failing_the_build(no_tokenizer, capsys):
    # A fresh clone with no network must still get its chunks: the size report is a diagnostic.
    assert ch.default_encode() is None
    assert "no cl100k_base vocabulary" in capsys.readouterr().err


# --- end to end ---------------------------------------------------------------------------------


def test_chunk_writes_one_line_per_course_with_unique_ids(cfg, corpus, courses):
    result = ch.chunk(cfg, encode=fake_encode)

    lines = corpus["CHUNKS_PATH"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(courses)  # the acceptance check: wc -l == course count
    rows = [json.loads(line) for line in lines]
    assert len({r["doc_id"] for r in rows}) == len(courses)
    assert all(set(r) == {"doc_id", "text", "metadata"} for r in rows)
    assert rows[0]["doc_id"] == ch.doc_id(courses[0])
    assert rows == result["chunks"]


def test_chunk_records_the_config_hash_alongside_the_corpus(cfg, corpus):
    ch.chunk(cfg, encode=fake_encode)

    meta = json.loads(corpus["CHUNKS_META_PATH"].read_text(encoding="utf-8"))
    assert meta["chunk_key"] == ch.chunk_key(cfg)  # what a consumer compares
    assert meta["config_hash"] == cfg.hash
    assert meta["template_version"] == "v1"
    assert meta["strategy"] == ch.STRATEGY
    assert meta["fields"] == list(ch.TEMPLATE_FIELDS["v1"])
    assert meta["chunks"] == len(corpus["CHUNKS_PATH"].read_text(encoding="utf-8").splitlines())
    assert meta["tokens"]["limit"] == ch.EMBEDDING_TOKEN_LIMIT


def test_chunk_refuses_a_corpus_that_does_not_match_the_config(cfg, corpus, courses):
    ex.write_json(corpus["COURSES_PATH"], [{**c, "term": "2025-fa"} for c in courses])
    with pytest.raises(ValueError, match="data.term"):
        ch.chunk(cfg, encode=fake_encode)
    assert not corpus["CHUNKS_PATH"].exists()  # nothing written on a failed run


def test_chunk_refuses_to_write_a_chunk_over_the_embedding_limit(cfg, corpus, courses):
    # Not merely reported: #5 would otherwise pay for the batch ahead of the rejected chunk.
    huge = {**courses[0], "desc": "x " * (ch.EMBEDDING_TOKEN_LIMIT + 1)}
    ex.write_json(corpus["COURSES_PATH"], [huge, *courses[1:]])
    with pytest.raises(ValueError, match=f"exceed the {ch.EMBEDDING_TOKEN_LIMIT}-token"):
        ch.chunk(cfg, encode=fake_encode)
    assert not corpus["CHUNKS_PATH"].exists()


def test_chunk_writes_the_corpus_when_the_tokenizer_is_unavailable(
    cfg, corpus, courses, no_tokenizer
):
    # The diagnostic degrades, the product does not: an offline run still gets its chunks, and
    # the sidecar says they were never size-checked.
    result = ch.chunk(cfg)

    assert result["stats"] is None
    lines = corpus["CHUNKS_PATH"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(courses)
    assert json.loads(corpus["CHUNKS_META_PATH"].read_text(encoding="utf-8"))["tokens"] is None
    assert "not size-checked" in "\n".join(ch.summarize(result))


# --- the chunk key ------------------------------------------------------------------------------


def test_chunk_key_ignores_settings_that_do_not_change_the_chunks(cfg):
    # A Day 4 ablation tunes these on purpose; the chunks on disk stay valid, so the key must not
    # move or #5 / #6 learn to ignore it.
    tuned = Config({**cfg.to_dict(), "seed": 43, "fusion": {**cfg["fusion"], "rrf_k": 10}})
    assert tuned.hash != cfg.hash
    assert ch.chunk_key(tuned) == ch.chunk_key(cfg)


def test_chunk_key_moves_with_the_term(cfg):
    other = Config({**cfg.to_dict(), "data": {**cfg["data"], "term": "2026-fa"}})
    assert ch.chunk_key(other) != ch.chunk_key(cfg)


def test_chunk_key_moves_with_the_template(cfg, monkeypatch):
    monkeypatch.setitem(ch.TEMPLATE_FIELDS, "v2", ("code", "desc"))
    v2 = Config({**cfg.to_dict(), "chunk": {**cfg["chunk"], "template_version": "v2"}})
    assert ch.chunk_key(v2) != ch.chunk_key(cfg)
