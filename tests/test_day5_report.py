"""Tests for eval/scripts/day5_report.py: the Day 5 closing report, and its refusals.

The fixture is a copy of the committed Day 5 files in tmp_path, so each refusal is tested by
breaking exactly one link of the real chain and nothing else.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from io import StringIO
from pathlib import Path

import pytest

from config import REPO_ROOT, load_config
from eval.scripts import day5_report as d5
from eval.scripts.review import DATASET_PATH
from eval.scripts.risk_coverage import FIGURES_DIR
from eval.scripts.signals import RUNS_DIR

pytest.importorskip("scipy")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def src(tmp_path, cfg):
    """Every committed input, copied, so a test can break one of them."""
    runs = tmp_path / "runs"
    runs.mkdir()
    for one in d5.inputs(cfg, d5.Sources()):
        if one.path.parent == RUNS_DIR:
            shutil.copy2(one.path, runs / one.path.name)
    figures = tmp_path / "figures"
    figures.mkdir()
    for png in FIGURES_DIR.glob("risk_coverage_test_*.png"):
        shutil.copy2(png, figures / png.name)
    dataset = tmp_path / "v1.jsonl"
    shutil.copy2(DATASET_PATH, dataset)
    return d5.Sources(runs, dataset, figures)


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refused(cfg, src, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        d5.read_facts(cfg, src)


# --- the committed report -----------------------------------------------------------------------


def test_the_committed_report_is_what_the_generator_writes(cfg):
    path = d5.report_path(cfg)
    assert path.name == "day5_signals_83496e82.md"
    assert path.read_text(encoding="utf-8") == d5.render_report(d5.read_facts(cfg, d5.Sources()))


def test_the_report_is_named_after_the_conformal_key_not_the_config_hash(cfg):
    assert d5.report_path(cfg).name == f"day5_signals_{d5.conformal_key(cfg)}.md"
    assert cfg.hash not in d5.report_path(cfg).name


def test_a_copy_renders_the_same_report_twice_and_carries_no_clock(cfg, src):
    first = d5.render_report(d5.read_facts(cfg, src))
    assert first == d5.render_report(d5.read_facts(cfg, src))
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}", first)


def test_the_six_sections_are_there_in_the_issues_order(cfg):
    body = d5.render_report(d5.read_facts(cfg, d5.Sources()))
    heads = [line for line in body.splitlines() if line.startswith("## ")]
    assert heads == ["## Inputs", "## 1. Setup", "## 2. Labels",
                     "## 3. RQ2: which signal ranks answers best",
                     "## 4. RQ3: conformal against a fixed 0.7", "## 5. Wiring",
                     "## 6. Limitations", "## Reproduction"]


def test_the_rq2_and_rq3_tables_are_the_owning_scripts_own(cfg):
    facts = d5.read_facts(cfg, d5.Sources())
    body = d5.render_report(facts)
    assert d5.risk_coverage.summary(facts.rc) in body
    assert d5.conformal_eval.summary(facts.reading) in body


def test_numbers_come_from_the_files(cfg, src):
    """A number moved in an input moves in the report: nothing was typed in."""
    before = d5.render_report(d5.read_facts(cfg, src))
    path = d5.risk_coverage.result_path("test", cfg, src.runs_dir)
    edit_json(path, lambda d: d["signals"]["verbalized"]["auroc"].update(point=0.123456))
    after = d5.render_report(d5.read_facts(cfg, src))
    assert "0.123" in after and "0.123" not in before


def test_the_degenerate_interval_and_the_unpaired_rows_are_said(cfg):
    notes = d5.rq2_notes(d5.read_facts(cfg, d5.Sources()))
    assert any("zero width" in note and "`retrieval_score`" in note for note in notes)
    assert any("not the rows" in note for note in notes)


def test_the_generator_never_calls_a_model_or_resamples():
    tree = ast.parse(Path(d5.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    for forbidden in ("openai", "llm", "agent.llm", "resample_indices", "bootstrap_auroc",
                      "verbalized", "self_consistency", "fit"):
        assert forbidden not in imported
    source = Path(d5.__file__).read_text(encoding="utf-8")
    assert "risk_coverage.analyse" not in source and "conformal.fit(" not in source


# --- the refusals -------------------------------------------------------------------------------


def test_check_names_a_missing_input_and_how_to_make_it(cfg, src):
    (src.runs_dir / d5.conformal_eval.result_path(cfg).name).unlink()
    out = StringIO()
    assert d5.render_check(cfg, src, out) == 1
    assert "MISSING" in out.getvalue() and "eval.scripts.conformal_eval" in out.getvalue()


def test_check_passes_on_the_committed_files(cfg):
    assert d5.render_check(cfg, d5.Sources(), StringIO()) == 0


def test_a_changed_csv_is_refused(cfg, src):
    path = d5.csv_path("test", cfg, src.runs_dir)
    path.write_text(path.read_text(encoding="utf-8").replace(",1,1,eve", ",1,0,eve", 1),
                    encoding="utf-8")
    refused(cfg, src, "sha256")


def test_an_incomplete_review_is_refused(cfg, src):
    path = d5.csv_path("calib", cfg, src.runs_dir)
    meta = path.with_name(path.name + ".meta.json")
    edit_json(meta, lambda d: d["by_bucket"]["fact"].update(reviewed=d["by_bucket"]["fact"]
                                                             ["rows"] - 1))
    refused(cfg, src, "calib review is incomplete")


def test_a_changed_signals_run_is_refused(cfg, src):
    path = d5.run_path("calib", cfg, src.runs_dir)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    refused(cfg, src, "not the file its sidecar describes")


def test_another_dataset_is_refused(cfg, src):
    src.dataset.write_text(src.dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    refused(cfg, src, "another dataset")


def test_a_redrawn_figure_is_refused(cfg, src):
    png = d5.risk_coverage.figure_path("test", cfg, src.figures_dir)
    png.write_bytes(png.read_bytes() + b"\0")
    refused(cfg, src, "is not the figure")


def test_a_reading_under_another_key_is_refused(cfg, src):
    edit_json(d5.conformal_eval.result_path(cfg, src.runs_dir),
              lambda d: d.update(conformal_key="00000000"))
    refused(cfg, src, "conformal_key 00000000")


def test_a_reading_that_evaluate_does_not_give_is_refused(cfg, src):
    edit_json(d5.conformal_eval.result_path(cfg, src.runs_dir),
              lambda d: d["answer_everything"].update(wrong=0))
    refused(cfg, src, "not what #91's evaluate gives")


def test_main_writes_nothing_when_it_refuses(cfg, src, monkeypatch, tmp_path):
    shutil.rmtree(src.runs_dir)
    monkeypatch.setattr(d5, "Sources", lambda: src)
    monkeypatch.setattr(d5, "report_path", lambda cfg: tmp_path / "out.md")
    assert d5.main([]) == 1
    assert not (tmp_path / "out.md").exists()


def test_every_report_path_is_repo_relative(cfg):
    body = d5.render_report(d5.read_facts(cfg, d5.Sources()))
    assert str(REPO_ROOT) not in body and str(Path.home()) not in body
