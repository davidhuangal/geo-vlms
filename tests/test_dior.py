import subprocess
import sys
from pathlib import Path

import pytest

from geo_vlms.datasets.dior import (
    CATEGORIES,
    build_counting_dataset,
    build_existence_dataset,
    load_prepared,
)

REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "dior"
PREPARE_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dior.py"


def _prepare(
    data_dir: Path, out_path: Path | None = None
) -> subprocess.CompletedProcess:
    command = [sys.executable, str(PREPARE_SCRIPT), "--data-dir", str(data_dir)]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _annotation_xml(image_id: str, categories: list[str]) -> str:
    objects = "".join(
        f"<object><name>{category}</name></object>" for category in categories
    )
    return f"<annotation><filename>{image_id}.jpg</filename>{objects}</annotation>"


@pytest.fixture
def dior_dir(tmp_path: Path) -> Path:
    annotation_dir = tmp_path / "Annotations" / "Horizontal Bounding Boxes"
    annotation_dir.mkdir(parents=True)
    (tmp_path / "JPEGImages-trainval").mkdir()
    (tmp_path / "JPEGImages-test").mkdir()
    (tmp_path / "Main").mkdir()

    images = {
        "00001": ("train", ["golffield", "ship", "ship"]),
        "00002": ("val", ["vehicle"]),
        "00003": ("test", ["ship"]),
    }
    for image_id, (split, categories) in images.items():
        image_dir = (
            tmp_path / "JPEGImages-test"
            if split == "test"
            else tmp_path / "JPEGImages-trainval"
        )
        (image_dir / f"{image_id}.jpg").touch()
        (annotation_dir / f"{image_id}.xml").write_text(
            _annotation_xml(image_id, categories)
        )
        (tmp_path / "Main" / f"{split}.txt").write_text(f"{image_id}\n")

    _prepare(tmp_path)
    return tmp_path


def test_prepare_dataset_writes_all_image_category_pairs(dior_dir):
    counts = load_prepared(dior_dir)

    assert len(counts) == 3 * len(CATEGORIES)
    row = counts[(counts.image_id == "00001") & (counts.category == "ship")].iloc[0]
    assert row.raw_category == "ship"
    assert row["count"] == 2
    assert (
        counts[(counts.image_id == "00001") & (counts.category == "vehicle")][
            "count"
        ].item()
        == 0
    )


def test_prepare_dataset_is_deterministic(dior_dir, tmp_path):
    second = tmp_path / "second.csv"
    _prepare(dior_dir, second)

    assert second.read_bytes() == (dior_dir / "counts.csv").read_bytes()


def test_prepare_dataset_rejects_missing_annotation(dior_dir):
    (dior_dir / "Annotations/Horizontal Bounding Boxes/00002.xml").unlink()

    with pytest.raises(subprocess.CalledProcessError) as error:
        _prepare(dior_dir)

    assert "Annotation mismatch" in error.value.stderr


def test_build_counting_dataset(dior_dir):
    examples = build_counting_dataset(
        dior_dir,
        "train",
        categories=["golf field", "ship", "vehicle"],
    )
    by_category = {example.metadata["category"]: example for example in examples}

    assert by_category["golf field"].expected == 1
    assert by_category["ship"].expected == 2
    assert by_category["vehicle"].expected == 0
    assert by_category["golf field"].metadata["raw_category"] == "golffield"


def test_build_existence_dataset(dior_dir):
    examples = build_existence_dataset(
        dior_dir,
        "test",
        categories=["ship", "vehicle"],
    )
    by_category = {example.metadata["category"]: example for example in examples}

    assert by_category["ship"].expected is True
    assert by_category["vehicle"].expected is False


def test_load_prepared_requires_preparation(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"prepare_dior\.py"):
        load_prepared(tmp_path)


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "counts.csv").exists(), reason="DIOR not prepared"
)
def test_real_dior_prepared_counts():
    counts = load_prepared(REAL_DATA_DIR)

    assert len(counts) == 23_463 * len(CATEGORIES)
    assert counts["count"].sum() == 192_518
    assert set(counts.split) == {"train", "val", "test"}
