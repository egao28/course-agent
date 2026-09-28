"""Tests for config.py: load_config / Config / config_hash, the source pin, artifact naming.

Hash properties are exercised on ``tests/fixtures/config_fixture.yaml`` so that tuning the live
``config.yaml`` (a Day 4 ablation changing ``top_k_final``, say) can never break them. The live
file gets exactly one smoke test.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

import config
from config import (
    Config,
    bm25_key,
    bm25_path,
    chroma_path,
    collection_name,
    config_hash,
    dense_key,
    full_hash,
    load_config,
    rel_to_repo,
    report_path,
    signals_key,
    source_path,
    verify_source,
)

FIXTURE = Path(__file__).parent / "fixtures" / "config_fixture.yaml"
HEX8 = re.compile(r"^[0-9a-f]{8}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SECTIONS = {
    "seed",
    "data",
    "chunk",
    "embedding",
    "bm25",
    "fusion",
    "llm",
    "agent",
    "evidence",
    # #61: eval.split_seed, the provenance of the frozen calibration/test split.
    "eval",
}


@pytest.fixture
def cfg() -> Config:
    """Read-only Config loaded from the fixture file."""
    return load_config(FIXTURE)


@pytest.fixture
def raw(cfg: Config) -> dict:
    """Mutable plain-dict copy of the fixture, for building variants."""
    return cfg.to_dict()


# --- helpers --------------------------------------------------------------------------------------


def leaf_paths(node: Mapping, prefix: tuple = ()):
    """Yield the key path of every non-mapping leaf in a nested mapping."""
    for key, value in node.items():
        path = (*prefix, key)
        if isinstance(value, Mapping):
            yield from leaf_paths(value, path)
        else:
            yield path


def get_at(node: Mapping, path: tuple):
    for key in path:
        node = node[key]
    return node


def set_at(node: dict, path: tuple, value) -> dict:
    """Deep copy of ``node`` with the leaf at ``path`` replaced by ``value``."""
    new = copy.deepcopy(node)
    target = new
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return new


def mutate(value):
    """A value that differs from ``value``, same type where one exists."""
    if value is None:
        return "was_null"
    if isinstance(value, bool):  # before int: bool is an int subclass
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.1
    if isinstance(value, str):
        return value + "_changed"
    if isinstance(value, list):
        return [*value, "extra"]
    raise TypeError(f"no mutation rule for {type(value).__name__}")


def reversed_keys(node):
    """Deep plain copy with the insertion order of every mapping reversed."""
    if isinstance(node, Mapping):
        return {k: reversed_keys(node[k]) for k in reversed(list(node))}
    if isinstance(node, (list, tuple)):
        return [reversed_keys(v) for v in node]
    return node


LEAVES = sorted(leaf_paths(yaml.safe_load(FIXTURE.read_text())))


# --- live config.yaml: one smoke test only ----------------------------------------------------


def test_live_config_smoke():
    live = load_config()
    assert REQUIRED_SECTIONS <= set(live)
    assert HEX8.match(live.hash)
    assert HEX64.match(live["data"]["source_sha256"])


# --- load_config / Config ----------------------------------------------------------------------


def test_load_config_returns_read_only_config(cfg):
    assert isinstance(cfg, Config)
    assert isinstance(cfg, Mapping)
    assert cfg.path == FIXTURE
    assert isinstance(cfg["chunk"]["fields"], tuple)
    with pytest.raises(TypeError):
        cfg["seed"] = 1
    with pytest.raises(TypeError):
        cfg["data"]["source"] = "/elsewhere.csv"


def test_load_config_does_not_expand_home(cfg):
    # The hash must be machine-independent, so "~" stays literal until source_path() is called.
    assert cfg["data"]["source"].startswith("~/")


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(bad)


def test_to_dict_is_a_mutable_deep_copy(cfg):
    d = cfg.to_dict()
    d["fusion"]["top_k_final"] = 10
    assert cfg["fusion"]["top_k_final"] == 5
    assert isinstance(d["chunk"]["fields"], list)
    assert Config(d).hash != cfg.hash


def test_config_round_trips_through_to_dict(cfg):
    assert Config(cfg.to_dict()).hash == cfg.hash


def test_fixture_covers_null_and_bool_leaves(raw):
    # Keeps the parametrized "any field" test honest about the value types it exercises.
    values = [get_at(raw, p) for p in LEAVES]
    assert any(v is None for v in values)
    assert any(isinstance(v, bool) for v in values)


# --- config_hash ---------------------------------------------------------------------------------


def test_hash_is_8_hex_and_deterministic(cfg):
    assert HEX8.match(cfg.hash)
    assert load_config(FIXTURE).hash == cfg.hash
    assert config_hash(cfg) == cfg.hash
    assert config_hash(cfg.to_dict()) == cfg.hash


def test_canonical_dump_treats_frozen_containers_like_plain_ones(cfg, raw):
    # json.dumps(default=str) would otherwise render a mappingproxy as "{'a': 1}".
    assert config.canonical_dump(cfg) == config.canonical_dump(raw)
    assert "mappingproxy" not in config.canonical_dump(cfg)


def test_hash_is_frozen_at_load_and_shared_by_every_helper(cfg):
    h = cfg.hash
    expanded = source_path(cfg)  # "expand at the point of use" must not move the hash
    assert expanded.is_absolute()
    assert "~" not in str(expanded)
    assert cfg["data"]["source"].startswith("~/")
    assert cfg.hash == h
    assert report_path("x", cfg).name == f"x_{h}.md"


def test_collection_name_ignores_settings_that_cannot_change_a_vector(cfg):
    # A Day 4 ablation tunes fusion / bm25 / seed. Every other artifact gets a new name for it;
    # the dense index must not, or the ablation re-embeds a byte-identical corpus and pays again.
    ablation = Config({**cfg.to_dict(), "seed": 43, "fusion": {**cfg["fusion"], "rrf_k": 61}})
    assert ablation.hash != cfg.hash
    assert collection_name(ablation, "c0ffee00") == collection_name(cfg, "c0ffee00")


def test_collection_name_moves_with_the_chunks_and_with_the_encoder(cfg):
    name = collection_name(cfg, "c0ffee00")
    assert HEX8.match(name.removeprefix("courses_"))
    assert collection_name(cfg, "deadbeef") != name  # other chunks, other vectors
    other_model = {**cfg["embedding"], "model": "text-embedding-3-large"}
    assert collection_name(Config({**cfg.to_dict(), "embedding": other_model}), "c0ffee00") != name


def test_bm25_key_ignores_settings_that_cannot_change_a_posting_list(cfg):
    # The dense argument, for the cheaper index: tuning fusion or the seed moves the config hash
    # without moving a single term, and so must not move the name the BM25 index answers to.
    ablation = Config({**cfg.to_dict(), "seed": 43, "fusion": {**cfg["fusion"], "rrf_k": 61}})
    assert ablation.hash != cfg.hash
    assert bm25_key(ablation, "c0ffee00") == bm25_key(cfg, "c0ffee00")


def test_bm25_key_moves_with_the_chunks_and_with_every_bm25_setting(cfg):
    key = bm25_key(cfg, "c0ffee00")
    assert HEX8.match(key)
    assert bm25_key(cfg, "deadbeef") != key  # other chunks, other terms
    tuned = {**cfg["bm25"], "k1": 1.2}
    assert bm25_key(Config({**cfg.to_dict(), "bm25": tuned}), "c0ffee00") != key


def test_bm25_key_covers_a_setting_nobody_has_added_yet(cfg):
    # The section is hashed whole rather than field by field, so the next knob added to it cannot
    # be quietly left out of the name of the index it changes.
    extended = {**cfg["bm25"], "delta": 1.0}
    assert bm25_key(Config({**cfg.to_dict(), "bm25": extended}), "c0ffee00") != bm25_key(
        cfg, "c0ffee00"
    )


def test_the_two_index_keys_do_not_move_each_other(cfg):
    # Rebuilding the BM25 index must not orphan the one that cost money, and re-embedding must not
    # invalidate a posting list that no embedding setting can reach.
    tuned = Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "k1": 1.2}})
    assert collection_name(tuned, "c0ffee00") == collection_name(cfg, "c0ffee00")
    other_model = {**cfg["embedding"], "model": "text-embedding-3-large"}
    remodelled = Config({**cfg.to_dict(), "embedding": other_model})
    assert bm25_key(remodelled, "c0ffee00") == bm25_key(cfg, "c0ffee00")


def test_a_new_section_moves_the_config_hash_and_neither_index_key(cfg):
    # Day 2 adds `llm:` to config.yaml, which is the general case: a section that no index is a
    # function of must rename the reports and leave both indexes answering to the names they
    # already have -- otherwise arriving at Day 2 orphans a dense index that cost money to build.
    with_llm = Config(
        {**cfg.to_dict(), "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0}}
    )
    assert with_llm.hash != cfg.hash
    assert collection_name(with_llm, "c0ffee00") == collection_name(cfg, "c0ffee00")
    assert bm25_key(with_llm, "c0ffee00") == bm25_key(cfg, "c0ffee00")


def test_the_eval_section_moves_the_hash_and_leaves_all_three_index_keys_alone():
    # #61 adds `eval: {split_seed: ...}` to the live file -- the second instance of the general
    # case above, and the claim the PR makes about the artifacts already in the tree: the reports
    # are renamed, the two indexes answer to the names they have. Over config.yaml itself rather
    # than the fixture, because that is the file whose hash moved.
    live = load_config()
    assert isinstance(live["eval"]["split_seed"], int)
    without = Config({k: v for k, v in live.to_dict().items() if k != "eval"})
    assert live.hash != without.hash
    assert collection_name(live, "c0ffee00") == collection_name(without, "c0ffee00")
    assert bm25_key(live, "c0ffee00") == bm25_key(without, "c0ffee00")
    reseeded = Config({**live.to_dict(), "eval": {"split_seed": live["eval"]["split_seed"] + 1}})
    assert reseeded.hash != live.hash
    assert bm25_key(reseeded, "c0ffee00") == bm25_key(live, "c0ffee00")


def test_bm25_key_says_so_when_there_is_no_section_to_hash(cfg):
    without = {k: v for k, v in cfg.to_dict().items() if k != "bm25"}
    with pytest.raises(ValueError, match="no `bm25` section"):
        bm25_key(Config(without), "c0ffee00")


def test_dense_key_names_the_embedding_keys_it_needs(cfg):
    # config.yaml is the file the reader has to go and fix, so it gets named -- see the same rule
    # in check_embedding_config / check_chunk_config.
    bare = {k: v for k, v in cfg["embedding"].items() if k != "model"}
    with pytest.raises(ValueError, match="embedding section is missing model"):
        dense_key(Config({**cfg.to_dict(), "embedding": bare}), "c0ffee00")


def test_reordering_keys_does_not_change_hash(raw):
    shuffled = reversed_keys(raw)
    assert list(shuffled) != list(raw)  # sanity: the order really did change
    assert config_hash(shuffled) == config_hash(raw)


def test_reordering_keys_in_yaml_file_does_not_change_hash(cfg, tmp_path):
    shuffled = tmp_path / "config.yaml"
    shuffled.write_text(yaml.safe_dump(reversed_keys(cfg), sort_keys=False))
    assert shuffled.read_text() != FIXTURE.read_text()
    assert load_config(shuffled).hash == cfg.hash


@pytest.mark.parametrize("path", LEAVES, ids=[".".join(p) for p in LEAVES])
def test_changing_any_field_changes_hash(raw, path):
    changed = set_at(raw, path, mutate(get_at(raw, path)))
    assert config_hash(changed) != config_hash(raw)


def test_seed_participates_in_hash(raw):
    assert config_hash({**raw, "seed": raw["seed"] + 1}) != config_hash(raw)


def test_source_pin_participates_in_hash(raw):
    assert config_hash(set_at(raw, ("data", "source_sha256"), "a" * 64)) != config_hash(raw)


def test_list_order_is_significant(raw):
    # chunk.fields order defines the chunk template, so reversing it must change the hash.
    fields = raw["chunk"]["fields"]
    assert config_hash(set_at(raw, ("chunk", "fields"), fields[::-1])) != config_hash(raw)


def test_adding_or_removing_a_key_changes_hash(raw):
    assert config_hash({**raw, "extra": 1}) != config_hash(raw)
    without_seed = {k: v for k, v in raw.items() if k != "seed"}
    assert config_hash(without_seed) != config_hash(raw)


# --- input data ----------------------------------------------------------------------------------


def test_source_path_expands_home_and_anchors_relative_paths(raw):
    assert source_path(Config(raw)) == Path.home() / "somewhere" / "data.csv"
    raw["data"]["source"] = "data/raw.csv"
    assert source_path(Config(raw)) == config.REPO_ROOT / "data" / "raw.csv"


def test_verify_source_accepts_matching_file(raw, tmp_path):
    csv = tmp_path / "catalog.csv"
    csv.write_bytes(b"code,name\nCS 225,Data Structures\n")
    raw["data"]["source"] = str(csv)
    raw["data"]["source_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert verify_source(Config(raw)) == csv


def test_verify_source_rejects_changed_content(raw, tmp_path):
    # Issue #3 re-exports the CSV: the hash must move, not silently reuse the old index.
    csv = tmp_path / "catalog.csv"
    csv.write_bytes(b"old\n")
    raw["data"]["source"] = str(csv)
    raw["data"]["source_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    before = Config(raw)
    assert verify_source(before) == csv

    csv.write_bytes(b"new\n")
    with pytest.raises(ValueError, match="pins"):
        verify_source(before)

    raw["data"]["source_sha256"] = hashlib.sha256(b"new\n").hexdigest()
    after = Config(raw)
    assert verify_source(after) == csv
    assert after.hash != before.hash


def test_verify_source_requires_a_pin(raw, tmp_path):
    csv = tmp_path / "catalog.csv"
    csv.write_bytes(b"x")
    raw["data"]["source"] = str(csv)
    raw["data"]["source_sha256"] = None
    with pytest.raises(ValueError, match="not set"):
        verify_source(Config(raw))


def test_verify_source_missing_file(raw, tmp_path):
    raw["data"]["source"] = str(tmp_path / "missing.csv")
    raw["data"]["source_sha256"] = "a" * 64
    with pytest.raises(FileNotFoundError):
        verify_source(Config(raw))


# --- artifact naming ------------------------------------------------------------------------------


def test_a_report_path_is_absolute_and_creates_its_directory(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "report")
    r = report_path("day1_smoke", cfg)
    assert r == tmp_path / "report" / f"day1_smoke_{cfg.hash}.md"
    assert r.parent.is_dir()


def test_the_index_paths_only_name(cfg, tmp_path, monkeypatch):
    # Both are asked for by readers looking for an index -- a query, an error message -- and a read
    # that creates what it failed to find is how a failed one leaves an empty store behind. The
    # builds create the directory they write into.
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    p = bm25_path(cfg, "c0ffee00")
    assert p == tmp_path / "indexes" / f"bm25_{bm25_key(cfg, 'c0ffee00')}.pkl"
    assert not p.parent.exists()
    assert not chroma_path().exists()


def test_chroma_path_is_one_store_for_every_config(cfg, tmp_path, monkeypatch):
    # The key belongs on the collection inside the store, not on the store: an ablation should
    # add a collection beside the baseline, not a second copy of the whole directory.
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    path = chroma_path()
    assert path == tmp_path / "indexes" / "chroma"
    assert cfg.hash not in str(path)


def test_chroma_path_does_not_create_the_store(tmp_path, monkeypatch):
    # The one path helper that must not: opening a persistent Chroma client on this directory
    # writes a chroma.sqlite3 into it, so a caller that only wants to *name* the store -- a dry
    # run, an error message quoting the path -- would leave one behind just by asking.
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    assert not chroma_path().exists()


def test_artifact_directories_exist_in_a_fresh_checkout():
    # indexes/ contents are gitignored, but the directory itself must exist in a fresh clone
    # (via .gitkeep, like report/), so the first index build does not hit FileNotFoundError.
    assert (config.INDEXES_DIR / ".gitkeep").is_file()
    assert (config.REPORT_DIR / ".gitkeep").is_file()


def test_rel_to_repo_never_leaks_home_directory(cfg):
    assert rel_to_repo(bm25_path(cfg, "c0ffee00")) == (
        f"indexes/bm25_{bm25_key(cfg, 'c0ffee00')}.pkl"
    )
    assert rel_to_repo(report_path("day1_smoke", cfg)) == f"report/day1_smoke_{cfg.hash}.md"
    assert rel_to_repo(Path.home() / "Downloads" / "x.csv") == "~/Downloads/x.csv"
    assert rel_to_repo(Path("/opt/data.csv")) == "/opt/data.csv"


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_prints_hash_for_given_path(cfg, capsys):
    config.main([str(FIXTURE)])
    assert capsys.readouterr().out.strip() == cfg.hash


def test_cli_source_digest(raw, tmp_path, capsys):
    csv = tmp_path / "c.csv"
    csv.write_bytes(b"x")
    raw["data"]["source"] = str(csv)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(raw))
    config.main([str(cfg_file), "--source-digest"])
    assert capsys.readouterr().out.strip() == hashlib.sha256(b"x").hexdigest()


def test_the_bootstrap_settings_move_only_the_full_hash():
    # #76: eval.bootstrap is read after every run file exists, so it must not rename a single
    # artifact made before it -- config.yaml's hash stays where #61 left it, and only the full
    # hash, which names the bootstrap's own result, sees it.
    live = load_config()
    assert "bootstrap" in live["eval"]
    assert live.hash == "c73b197d"
    assert live.full_hash != live.hash
    data = live.to_dict()
    without = {
        **{k: v for k, v in data.items() if k not in {"signals", "correctness", "conformal"}},
        "eval": {k: v for k, v in data["eval"].items() if k != "bootstrap"},
    }
    assert Config(without).hash == Config(without).full_hash == live.hash
    reseeded = {**data, "eval": {**data["eval"], "bootstrap": {"n_resamples": 10, "seed": 1}}}
    assert Config(reseeded).hash == live.hash
    assert Config(reseeded).full_hash != live.full_hash
    # Anything a run does see still moves both.
    moved = {**data, "eval": {**data["eval"], "split_seed": 62}}
    assert Config(moved).hash != live.hash and Config(moved).full_hash != live.full_hash
    # A plain mapping hashes the same way as the Config built from it.
    assert config_hash(data) == live.hash and full_hash(data) == live.full_hash
    # An eval section holding nothing but bootstrap hashes as if it were absent.
    only = {**without, "eval": {"bootstrap": data["eval"]["bootstrap"]}}
    bare = {k: v for k, v in without.items() if k != "eval"}
    assert config_hash(only) == config_hash(bare)


def test_the_signals_section_moves_only_its_own_key_and_the_full_hash():
    # #87: Day 5's sampling settings and prompts reach nothing made before the signals run, so
    # cfg.hash stays where #61 left it -- the Day 4 run files are found by it -- and the run is
    # named after signals_key, which does see every word of both prompts.
    live = load_config()
    assert "signals" in live
    assert live.hash == "c73b197d"
    data = live.to_dict()
    assert Config({k: v for k, v in data.items() if k != "signals"}).hash == live.hash
    edited = {
        **data,
        "signals": {**data["signals"], "judge_prompt": data["signals"]["judge_prompt"] + "!"},
    }
    assert Config(edited).hash == live.hash
    assert Config(edited).full_hash != live.full_hash
    assert signals_key(Config(edited)) != signals_key(live)
    # And a setting the agent does see moves the signals key too, through cfg.hash.
    moved = {**data, "agent": {"retrieval_mode": "dense"}}
    assert signals_key(Config(moved)) != signals_key(live)
    with pytest.raises(ValueError, match="no `signals` section"):
        signals_key(Config({k: v for k, v in data.items() if k != "signals"}))
