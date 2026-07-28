import os
from pathlib import Path
from typing import NamedTuple

import pytest

from geo_vlms.datasets.vhr10 import CLASS_MAP, build_counting_dataset, parse_annotation

DATA_DIR = Path(__file__).parents[1] / "data" / "vhr10"


class Vhr10Dirs(NamedTuple):
    pos: Path
    gt: Path
    neg: Path


@pytest.fixture
def vhr10_dirs(tmp_path):
    """Fixture to create a dummy VHR10 mock dataset."""
    dirs = Vhr10Dirs(
        pos=tmp_path / "positive_image_set",
        gt=tmp_path / "ground_truth",
        neg=tmp_path / "negative_image_set",
    )
    for d in dirs:
        d.mkdir()

    (dirs.pos / "001.jpg").touch()  # images are never opened, empty files suffice
    (dirs.gt / "001.txt").write_text(
        "(563,478),(630,573),1\n(310,150),(420,240),1\n(100,100),(150,150),2\n"
    )
    (dirs.neg / "900.jpg").touch()
    return dirs


def test_parse_annotation(vhr10_dirs):
    gt_df = parse_annotation(vhr10_dirs.gt / "001.txt")

    # One row per annotated object
    assert len(gt_df) == 3

    # category_id is the only column counting consumes; it must parse as ints
    assert gt_df.category_id.tolist() == [1, 1, 2]

    # The separator regex must not split on the commas inside corner tuples
    for corner_col in (gt_df.corner1, gt_df.corner2):
        assert corner_col.str.fullmatch(r"\(\d+,\d+\)").all()


def test_build_counting_dataset(vhr10_dirs):
    dataset = build_counting_dataset(
        pos_dir=vhr10_dirs.pos, gt_dir=vhr10_dirs.gt, neg_dir=vhr10_dirs.neg
    )

    # Each image should have one Example per category id
    assert len(dataset) == (
        len(os.listdir(vhr10_dirs.pos)) + len(os.listdir(vhr10_dirs.neg))
    ) * len(CLASS_MAP)

    by_id = {e.id: e for e in dataset}

    # Test some cases for the positive image
    assert by_id["pos/001:airplane"].expected == 2
    assert by_id["pos/001:ship"].expected == 1
    assert all(
        e.expected == 0
        for e in dataset
        if e.id.startswith("pos/")
        and not e.id.endswith(":airplane")
        and not e.id.endswith(":ship")
    )

    # Test the negative image
    assert all(e.expected == 0 for e in dataset if e.id.startswith("neg/"))


def test_missing_gt_raises(vhr10_dirs):
    """Make sure a missing ground truth file raises a FileNotFoundError"""
    (vhr10_dirs.pos / "002.jpg").touch()
    with pytest.raises(FileNotFoundError):
        build_counting_dataset(vhr10_dirs.pos, vhr10_dirs.gt, vhr10_dirs.neg)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="VHR10 not downloaded")
def test_real_vhr10_builds():
    examples = build_counting_dataset(
        DATA_DIR / "positive_image_set",
        DATA_DIR / "ground_truth",
        DATA_DIR / "negative_image_set",
    )
    assert len(examples) == (650 + 150) * len(CLASS_MAP)
    assert all(e.expected >= 0 for e in examples)
