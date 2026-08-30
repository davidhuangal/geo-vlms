from pathlib import Path
from typing import NamedTuple

import pytest

from geo_vlms.datasets.vhr10 import (
    CLASS_MAP,
    build_counting_dataset,
    build_existence_dataset,
    parse_annotation,
)

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

    # Images aren't used, empty file suffices
    (dirs.pos / "001.jpg").touch()
    # Two planes, one ship
    (dirs.gt / "001.txt").write_text(
        "(563,478),(630,573),1\n(310,150),(420,240),1\n(100,100),(150,150),2\n"
    )

    # Images aren't used, empty file suffices
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

    # Every image, positive or negative, yields one Example per category.
    assert len(dataset) == 2 * len(CLASS_MAP)

    by_id = {e.id: e for e in dataset}

    # Test some cases for the positive image
    assert by_id["pos/001:airplane"].expected == 2
    assert by_id["pos/001:ship"].expected == 1

    # Unannotated categories on positive images assert a zero count
    assert by_id["pos/001:vehicle"].expected == 0

    # Test the negative image
    assert all(e.expected == 0 for e in dataset if e.id.startswith("neg/"))


def test_build_existence_dataset(vhr10_dirs):
    dataset = build_existence_dataset(
        pos_dir=vhr10_dirs.pos, gt_dir=vhr10_dirs.gt, neg_dir=vhr10_dirs.neg
    )

    # Every image, positive or negative, yields one Example per category.
    assert len(dataset) == 2 * len(CLASS_MAP)

    by_id = {e.id: e for e in dataset}

    assert by_id["pos/001:airplane"].expected is True
    assert by_id["pos/001:ship"].expected is True

    # Unannotated categories on positive images assert absence
    assert by_id["pos/001:vehicle"].expected is False

    assert all(e.expected is False for e in dataset if e.id.startswith("neg/"))


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
    pos = [e for e in examples if e.id.startswith("pos/")]
    neg = [e for e in examples if e.id.startswith("neg/")]

    # Every image contributes one Example per category.
    num_pos_images = len(list((DATA_DIR / "positive_image_set").glob("*.jpg")))
    num_neg_images = len(list((DATA_DIR / "negative_image_set").glob("*.jpg")))
    assert len(pos) == num_pos_images * len(CLASS_MAP)
    assert len(neg) == num_neg_images * len(CLASS_MAP)

    # Golden count of annotated (image, category) pairs in the standard
    # download. Therefore, a change means the annotations or the parser shifted.
    assert sum(1 for e in pos if e.expected >= 1) == 888

    assert all(e.expected == 0 for e in neg)


def make_vhr10_layout(root, num_pos, num_neg) -> Vhr10Dirs:
    # every pos image gets a trivial one-airplane GT; content doesn't matter,
    # sampling tests only care about *which* images are chosen
    dirs = Vhr10Dirs(
        pos=root / "positive_image_set",
        gt=root / "ground_truth",
        neg=root / "negative_image_set",
    )
    for d in dirs:
        d.mkdir()

    for i in range(1, num_pos + 1):
        (dirs.pos / f"{i}.jpg").touch()
        (dirs.gt / f"{i}.txt").write_text("(1,1),(2,2),1\n")
    for i in range(1, num_neg + 1):
        (dirs.neg / f"{i}_neg.jpg").touch()  # _neg to make neg's easier to identify

    return dirs


def test_subset_deterministic(tmp_path):
    num_pos = 20
    num_neg = 10

    dirs = make_vhr10_layout(root=tmp_path, num_pos=num_pos, num_neg=num_neg)

    a = build_counting_dataset(
        pos_dir=dirs.pos,
        gt_dir=dirs.gt,
        neg_dir=dirs.neg,
        num_pos_images=num_pos - 10,
        num_neg_images=num_neg - 5,
        seed=0,
    )
    b = build_counting_dataset(
        pos_dir=dirs.pos,
        gt_dir=dirs.gt,
        neg_dir=dirs.neg,
        num_pos_images=num_pos - 10,
        num_neg_images=num_neg - 5,
        seed=0,
    )

    assert [e.id for e in a] == [e.id for e in b]


def test_pos_and_neg_subsets_independent(tmp_path):
    num_pos = 20
    num_neg = 10

    dirs = make_vhr10_layout(root=tmp_path, num_pos=num_pos, num_neg=num_neg)
    a = build_counting_dataset(
        pos_dir=dirs.pos,
        gt_dir=dirs.gt,
        neg_dir=dirs.neg,
        num_pos_images=num_pos - 10,
        num_neg_images=num_neg - 5,
        seed=0,
    )
    b = build_counting_dataset(
        pos_dir=dirs.pos,
        gt_dir=dirs.gt,
        neg_dir=dirs.neg,
        num_pos_images=num_pos - 15,
        num_neg_images=num_neg - 5,
        seed=0,
    )

    assert [e.id for e in a if e.id.startswith("neg")] == [
        e.id for e in b if e.id.startswith("neg")
    ]


def test_counting_metadata(vhr10_dirs):
    dataset = build_counting_dataset(
        pos_dir=vhr10_dirs.pos, gt_dir=vhr10_dirs.gt, neg_dir=vhr10_dirs.neg
    )
    by_id = {e.id: e for e in dataset}

    assert by_id["pos/001:airplane"].metadata == {
        "dataset": "vhr10",
        "split": "positive",
        "category": "airplane",
    }
    assert by_id["pos/001:ship"].metadata == {
        "dataset": "vhr10",
        "split": "positive",
        "category": "ship",
    }
    assert by_id["neg/900:vehicle"].metadata == {
        "dataset": "vhr10",
        "split": "negative",
        "category": "vehicle",
    }
