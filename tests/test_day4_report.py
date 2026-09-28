"""Tests for eval/scripts/day4_report.py: the table is #76's numbers, and nothing else.

A whole Day 4 is built in ``tmp_path`` -- a dataset, two run files in #75's format, and the
bootstrap result #76 computes from them -- and the report is rendered from it. The numbers in the
report are parsed back out and compared with the bootstrap JSON at the printed precision. One test
regenerates the committed report from the committed artifacts, byte for byte.
"""

from __future__ import annotations

import json
import re
from io import StringIO

import pytest

from config import file_sha256, load_config, report_path
from eval.scripts import ablation as ab
from eval.scripts import bootstrap as bs
from eval.scripts import day4_report as report
from eval.scripts import hybrid_failures as hf
from eval.scripts.draft_items import write_json
from retrieval import MODES


@pytest.fixture
def cfg():
    return load_config()


#: (bucket, gold, {mode: (recall, mrr)}) per row. On calib bm25 is strongest on both metrics.
CALIB = [
    ("fact", ["a"], {"dense": (0.0, 0.0), "bm25": (1.0, 1.0), "hybrid": (1.0, 0.5)}),
    ("fact", ["b"], {"dense": (1.0, 0.5), "bm25": (1.0, 1.0), "hybrid": (1.0, 1.0)}),
    ("multihop", ["c", "d"], {"dense": (0.5, 1.0), "bm25": (0.5, 1.0), "hybrid": (0.5, 0.5)}),
]
TEST = [
    ("fact", ["a"], {"dense": (0.0, 0.0), "bm25": (1.0, 1.0), "hybrid": (1.0, 0.5)}),
    ("fact", ["b"], {"dense": (1.0, 0.2), "bm25": (1.0, 1.0), "hybrid": (1.0, 1.0)}),
    ("fact", ["e"], {"dense": (1.0, 1.0), "bm25": (0.0, 0.0), "hybrid": (1.0, 0.25)}),
    ("multihop", ["c", "d"], {"dense": (0.5, 1.0), "bm25": (1.0, 1.0), "hybrid": (0.5, 0.5)}),
    (
        "multihop",
        ["f", "g", "h"],
        {"dense": (0.0, 0.0), "bm25": (1 / 3, 0.5), "hybrid": (0.0, 0.0)},
    ),
]


def write_run(cfg, runs_dir, split, rows, dataset_sha, **meta_overrides):
    ids = [f"v1-{split}-{i:02d}" for i in range(len(rows))]
    lines = [
        {
            "id": row_id,
            "bucket": bucket,
            "mode": mode,
            "ranked": [],
            "gold": gold,
            "recall_at_5": scores[mode][0],
            "mrr_at_5": scores[mode][1],
        }
        for row_id, (bucket, gold, scores) in zip(ids, rows, strict=True)
        for mode in MODES
    ]
    path = ab.write_lines(ab.run_path(split, cfg, runs_dir), lines)
    buckets: dict[str, int] = {}
    for bucket, _, _ in rows:
        buckets[bucket] = buckets.get(bucket, 0) + 1
    write_json(
        path.with_suffix(".meta.json"),
        {
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "dense_key": "d" * 8,
            "bm25_key": "b" * 8,
            "dataset_sha256": dataset_sha,
            "split": split,
            "k": 5,
            "modes": list(MODES),
            "n": len(rows),
            "buckets": buckets,
            "ids": ids,
            **meta_overrides,
        },
    )
    return path


@pytest.fixture
def day4(cfg, tmp_path):
    """A dataset, both runs and the bootstrap over them, all in ``tmp_path``."""
    dataset = tmp_path / "v1.jsonl"
    dataset.write_text('{"id": "v1-0001"}\n')
    sha = file_sha256(dataset)
    write_run(cfg, tmp_path, "calib", CALIB, sha)
    write_run(cfg, tmp_path, "test", TEST, sha)
    bs.bootstrap(cfg, StringIO(), tmp_path)
    write_failures(cfg, tmp_path)
    return report.Sources(dataset, tmp_path)


def write_failures(cfg, runs_dir):
    """#78's log in the shape its sidecar has, read off the test run above: 2 fact, 2 multihop."""
    path = hf.failures_path(cfg, runs_dir)
    path.write_text('{"id": "v1-test-00"}\n', encoding="utf-8")
    write_json(
        path.with_suffix(".meta.json"),
        {
            "log": "eval/runs/" + path.name,
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "run": "eval/runs/" + ab.run_path("test", cfg, runs_dir).name,
            "run_sha256": file_sha256(ab.run_path("test", cfg, runs_dir)),
            "k": 5,
            "depth": 20,
            "n": len(TEST),
            "scored": {"fact": 3, "multihop": 2},
            "failing": {"fact": 0, "multihop": 2},
            "live_top5_differs": [],
        },
    )


def numbers(text: str) -> list[float]:
    return [float(x.replace("−", "-")) for x in re.findall(r"[+−]?\d+\.\d{3}", text)]


def test_every_cell_is_the_bootstrap_result_to_three_places(cfg, day4):
    facts = report.read_facts(cfg, day4)
    body = report.render_report(facts)
    table = body.split("## Result")[1].split("## Strongest")[0]
    rows = {line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("|")}
    result = json.loads(bs.result_path(cfg, day4.runs_dir).read_text())
    for mode in MODES:
        expected = []
        for metric in bs.METRICS:
            iv = result["metrics"][metric]["test"][mode]
            expected += [round(iv["point"], 3), round(iv["lo"], 3), round(iv["hi"], 3)]
        assert numbers(rows[f"`{mode}`"]) == pytest.approx(expected, abs=1e-9)
    gap_row = rows["*hybrid − bm25*"]
    expected = []
    for metric in bs.METRICS:
        gap = result["metrics"][metric]["difference"]
        assert gap["subtrahend"] == "bm25"
        expected += [round(gap["point"], 3), round(gap["lo"], 3), round(gap["hi"], 3)]
    assert numbers(gap_row) == pytest.approx(expected, abs=1e-9)
    assert "significant (CI" in gap_row


def test_the_bucket_split_is_points_only_and_the_multi_gold_rows_are_counted(cfg, day4):
    facts = report.read_facts(cfg, day4)
    assert facts.bucket_n == {"fact": 3, "multihop": 2}
    assert facts.multi_gold == {"fact": 0, "multihop": 2}
    assert facts.by_bucket["multihop"]["bm25"]["recall_at_5"] == pytest.approx((1 + 1 / 3) / 2)
    section = report.render_buckets(facts)
    assert "[" not in section.split("No intervals")[0]
    assert "1/2 = 0.500" in section


def test_the_report_names_n_split_digest_hash_b_and_seed(cfg, day4):
    body = report.render_report(report.read_facts(cfg, day4))
    for text in (
        "n** = 5 test rows (fact 3, multihop 2)",
        f"sha256 `{file_sha256(day4.dataset)}`",
        f"`{cfg.hash}`",
        "B = 10,000, seed 76",
        "chosen per metric on `calib`".capitalize(),
        "MRR is MRR@5",
        "fractional on multi-gold rows",
    ):
        assert text in body


def test_the_failure_count_is_stated_by_bucket_with_a_pointer_to_the_log(cfg, day4):
    body = report.render_report(report.read_facts(cfg, day4))
    assert "on 2 of 5 test rows: 0 of 3 `fact`, 2 of 2 `multihop`" in body
    assert f"`eval/runs/hybrid_failures_{cfg.hash}.jsonl`" in body
    assert "differs from the run's on 0 of 2." in body


def test_a_failure_log_from_another_test_run_is_refused(cfg, day4):
    rows = [TEST[0], *TEST[1:-1], TEST[0]]  # another draw of the test half
    write_run(cfg, day4.runs_dir, "test", rows, file_sha256(day4.dataset))
    bs.bootstrap(cfg, StringIO(), day4.runs_dir)
    with pytest.raises(ValueError, match="not read from the test run on disk"):
        report.read_facts(cfg, day4)


def test_a_rerun_is_byte_identical_and_carries_no_clock_or_local_path(cfg, day4):
    first = report.render_report(report.read_facts(cfg, day4))
    assert report.render_report(report.read_facts(cfg, day4)) == first
    assert not re.search(r"\d{4}-\d{2}-\d{2}T|\d{2}:\d{2}:\d{2}", first)
    assert "/Users/" not in first and "/home/" not in first


def test_different_strongest_singles_get_one_comparison_each(cfg, day4):
    facts = report.read_facts(cfg, day4)
    facts.result["metrics"]["mrr_at_5"]["difference"]["subtrahend"] = "dense"
    table = report.render_table(facts)
    assert "*hybrid − strongest single*" in table
    assert "*vs bm25:" in table and "*vs dense:" in table


def test_check_lists_what_is_missing_and_the_command_that_makes_it(cfg, day4):
    bs.result_path(cfg, day4.runs_dir).unlink()
    out = StringIO()
    assert report.render_check(cfg, day4, out) == 1
    assert "MISSING  bootstrap" in out.getvalue()
    assert "uv run python -m eval.scripts.bootstrap" in out.getvalue()
    assert "not written" in out.getvalue()


def test_main_refuses_to_write_while_an_input_is_missing(cfg, day4, capsys):
    ab.run_path("test", cfg, day4.runs_dir).unlink()
    before = report_path(report.STEM, cfg)
    existed = before.exists() and before.read_bytes()
    assert report.main(["--runs-dir", str(day4.runs_dir), "--dataset", str(day4.dataset)]) == 1
    assert "--split test" in capsys.readouterr().err
    assert (before.exists() and before.read_bytes()) == existed


def test_a_bootstrap_from_another_run_is_refused(cfg, day4):
    path = ab.run_path("test", cfg, day4.runs_dir)
    rows = [TEST[0], *TEST[1:-1], TEST[0]]
    write_run(cfg, day4.runs_dir, "test", rows, file_sha256(day4.dataset))
    assert path.exists()
    with pytest.raises(ValueError, match="not computed from the test run on disk"):
        report.read_facts(cfg, day4)


def test_a_run_over_another_dataset_is_refused(cfg, day4):
    day4.dataset.write_text('{"id": "v1-0002"}\n')
    with pytest.raises(ValueError, match="different .*v1.jsonl"):
        report.read_facts(cfg, day4)


def test_minus_signs_are_real_and_negative_zero_is_not_printed():
    assert report.num(-0.12) == "−0.120"
    assert report.num(0.0004, signed=True) == "+0.000"
    assert report.num(-0.0004, signed=True) == "+0.000"
    assert report.num(0.07317) == "0.073"


# --- the committed report -------------------------------------------------------------------------


def test_the_committed_report_regenerates_byte_identically(cfg):
    """From the committed runs, bootstrap and dataset; reads them, writes nothing."""
    body = report.render_report(report.read_facts(cfg, report.Sources()))
    assert body == report_path(report.STEM, cfg).read_text(encoding="utf-8")
