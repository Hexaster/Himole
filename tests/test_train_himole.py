"""Command-line regression tests for the HiMoLE training entry point."""

import pytest

from scripts.train_himole import parse_args
from scripts.compare_baseline_himole import parse_args as parse_compare_args


def test_skip_stage1_flag_is_available_for_smoke_runs():
    args = parse_args(["--tiny", "--steps", "5", "--samples", "8", "--skip-stage1"])

    assert args.tiny
    assert args.steps == 5
    assert args.samples == 8
    assert args.skip_stage1


def test_stage1_runs_by_default():
    assert not parse_args([]).skip_stage1


def test_micro_batch_size_is_configurable():
    assert parse_args(["--micro-batch-size", "4"]).micro_batch_size == 4


def test_stage2_can_use_a_smaller_micro_batch():
    args = parse_args(["--micro-batch-size", "4", "--stage2-micro-batch-size", "1"])

    assert args.micro_batch_size == 4
    assert args.stage2_micro_batch_size == 1


def test_stage1_parallelism_can_be_resumed_or_disabled():
    args = parse_args(["--resume-stage1", "--sequential-stage1"])

    assert args.resume_stage1
    assert args.sequential_stage1


def test_completed_stage1_checkpoints_have_a_strict_load_mode():
    args = parse_args(["--load-stage1-checkpoints", "--stage2-micro-batch-size", "1"])

    assert args.load_stage1_checkpoints


def test_loading_stage1_checkpoints_conflicts_with_other_stage_modes():
    with pytest.raises(SystemExit):
        parse_args(["--load-stage1-checkpoints", "--skip-stage1"])


def test_stage1_only_flag_is_available_for_initialization_runs():
    args = parse_args(["--tiny", "--stage1-only", "--samples", "24", "--steps", "3"])

    assert args.stage1_only
    assert args.samples == 24
    assert args.steps == 3


def test_stage_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--skip-stage1", "--stage1-only"])


def test_comparison_report_accepts_the_documented_arguments():
    args = parse_compare_args(["--samples", "200", "--out", "outputs/himole_compare.json"])

    assert args.samples == 200
    assert args.out == "outputs/himole_compare.json"
