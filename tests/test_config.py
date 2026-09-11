"""Tests for config.load_config / config_hash and the artifact naming helpers."""

from __future__ import annotations

import copy
import re

import pytest
import yaml

import config
from config import (
    DEFAULT_CONFIG_PATH,
    bm25_path,
    collection_name,
    config_hash,
    load_config,
    report_path,
)

HEX8 = re.compile(r"^[0-9a-f]{8}$")


@pytest.fixture
def cfg() -> dict:
    return load_config()


# --- helpers ------------------------------------------------------------------------------------


def leaf_paths(node: dict, prefix: tuple = ()):
    """Yield the key path of every non-mapping leaf in a nested dict."""
    for key, value in node.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from leaf_paths(value, path)
        else:
            yield path


def get_at(node: dict, path: tuple):
    for key in path:
        node = node[key]
    return node


def set_at(node: dict, path: tuple, value) -> dict:
    """Return a deep copy of ``node`` with the leaf at ``path`` replaced by ``value``."""
    new = copy.deepcopy(node)
    target = new
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return new


def mutate(value):
    """Return a same-typed value that differs from ``value``."""
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
    """Deep copy with the insertion order of every mapping reversed."""
    if isinstance(node, dict):
        return {k: reversed_keys(node[k]) for k in reversed(list(node))}
    return node


LEAVES = sorted(leaf_paths(load_config()))


# --- load_config --------------------------------------------------------------------------------


def test_default_config_has_expected_sections_and_day1_params(cfg):
    assert set(cfg) == {"seed", "data", "chunk", "embedding", "bm25", "fusion"}
    assert cfg["seed"] == 42
    assert cfg["fusion"] == {"method": "rrf", "rrf_k": 60, "top_k_each": 20, "top_k_final": 5}


def test_load_config_does_not_expand_home(cfg):
    # Hash must be machine-independent, so "~" stays literal until the point of use.
    assert cfg["data"]["source"].startswith("~/")


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(bad)


# --- config_hash --------------------------------------------------------------------------------


def test_hash_is_8_hex_chars_and_deterministic(cfg):
    h = config_hash(cfg)
    assert HEX8.match(h)
    assert config_hash(load_config()) == h
    assert config_hash(copy.deepcopy(cfg)) == h


def test_reordering_keys_does_not_change_hash(cfg):
    shuffled = reversed_keys(cfg)
    assert list(shuffled) != list(cfg)  # sanity: the order really did change
    assert config_hash(shuffled) == config_hash(cfg)


def test_reordering_keys_in_yaml_file_does_not_change_hash(cfg, tmp_path):
    shuffled = tmp_path / "config.yaml"
    shuffled.write_text(yaml.safe_dump(reversed_keys(cfg), sort_keys=False))
    assert shuffled.read_text() != DEFAULT_CONFIG_PATH.read_text()
    assert config_hash(load_config(shuffled)) == config_hash(cfg)


@pytest.mark.parametrize("path", LEAVES, ids=[".".join(p) for p in LEAVES])
def test_changing_any_field_changes_hash(cfg, path):
    changed = set_at(cfg, path, mutate(get_at(cfg, path)))
    assert config_hash(changed) != config_hash(cfg)


def test_seed_participates_in_hash(cfg):
    assert config_hash({**cfg, "seed": 43}) != config_hash(cfg)


def test_list_order_is_significant(cfg):
    # chunk.fields order defines the chunk template, so reversing it must change the hash.
    fields = cfg["chunk"]["fields"]
    assert config_hash(set_at(cfg, ("chunk", "fields"), fields[::-1])) != config_hash(cfg)


def test_adding_or_removing_a_key_changes_hash(cfg):
    assert config_hash({**cfg, "extra": 1}) != config_hash(cfg)
    without_seed = {k: v for k, v in cfg.items() if k != "seed"}
    assert config_hash(without_seed) != config_hash(cfg)


# --- artifact naming ----------------------------------------------------------------------------


def test_artifact_names_carry_hash(cfg):
    h = config_hash(cfg)
    assert collection_name(cfg) == f"courses_{h}"
    assert bm25_path(cfg) == config.INDEXES_DIR / f"bm25_{h}.pkl"
    assert report_path("day1_smoke", cfg) == config.REPORT_DIR / f"day1_smoke_{h}.md"


def test_cli_prints_hash(cfg, capsys):
    config.main([])
    assert capsys.readouterr().out.strip() == config_hash(cfg)
