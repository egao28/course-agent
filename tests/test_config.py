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
    bm25_path,
    chroma_path,
    collection_name,
    config_hash,
    load_config,
    rel_to_repo,
    report_path,
    source_path,
    verify_source,
)

FIXTURE = Path(__file__).parent / "fixtures" / "config_fixture.yaml"
HEX8 = re.compile(r"^[0-9a-f]{8}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SECTIONS = {"seed", "data", "chunk", "embedding", "bm25", "fusion"}


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
    assert collection_name(cfg) == f"courses_{h}"
    assert bm25_path(cfg).name == f"bm25_{h}.pkl"
    assert report_path("x", cfg).name == f"x_{h}.md"


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


def test_artifact_paths_are_absolute_and_create_their_directory(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "report")
    p = bm25_path(cfg)
    r = report_path("day1_smoke", cfg)
    assert p == tmp_path / "indexes" / f"bm25_{cfg.hash}.pkl"
    assert p.parent.is_dir()
    assert r == tmp_path / "report" / f"day1_smoke_{cfg.hash}.md"
    assert r.parent.is_dir()


def test_chroma_path_is_one_store_for_every_config(cfg, tmp_path, monkeypatch):
    # The hash belongs on the collection inside the store, not on the store: an ablation should
    # add a collection beside the baseline, not a second copy of the whole directory.
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    path = chroma_path()
    assert path == tmp_path / "indexes" / "chroma"
    assert path.is_dir()
    assert cfg.hash not in str(path)


def test_artifact_directories_exist_in_a_fresh_checkout():
    # indexes/ contents are gitignored, but the directory itself must exist in a fresh clone
    # (via .gitkeep, like report/), so the first index build does not hit FileNotFoundError.
    assert (config.INDEXES_DIR / ".gitkeep").is_file()
    assert (config.REPORT_DIR / ".gitkeep").is_file()


def test_rel_to_repo_never_leaks_home_directory(cfg):
    assert rel_to_repo(bm25_path(cfg)) == f"indexes/bm25_{cfg.hash}.pkl"
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
