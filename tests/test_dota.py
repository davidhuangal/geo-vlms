import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from geo_vlms.analysis import with_min_cover
from geo_vlms.coverage import expected_at
from geo_vlms.datasets.dota import CATEGORIES, CLASS_MAP, build_dataset, load_prepared
from geo_vlms.tasks import Counting, Existence

REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "dota"
PREPARE_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dota.py"


def _prepare(data_dir: Path, tile_size: int = 100) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--tile-size",
        str(tile_size),
        "--workers",
        "2",
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _box(x0: int, y0: int, x1: int, y1: int, category: str, difficult: int) -> str:
    return f"{x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1} {category} {difficult}"


def _write_image(
    data_dir: Path,
    image_id: str,
    size: tuple[int, int],
    labels: list[str],
    gsd: str = "0.1",
):
    Image.new("RGB", size, "white").save(
        data_dir / "val" / "images" / f"{image_id}.png"
    )
    (data_dir / "val" / "labelTxt" / f"{image_id}.txt").write_text(
        f"imagesource:GoogleEarth\ngsd:{gsd}\n" + "\n".join(labels) + "\n"
    )


def _make_dirs(data_dir: Path):
    (data_dir / "val" / "images").mkdir(parents=True)
    (data_dir / "val" / "labelTxt").mkdir(parents=True)


@pytest.fixture
def dota_dir(tmp_path: Path) -> Path:
    _make_dirs(tmp_path)
    # 250x150 at tile 100 gives x origins 0, 100, 150 and y origins 0, 50.
    _write_image(
        tmp_path,
        "P0000",
        (250, 150),
        [
            _box(10, 10, 30, 30, "ship", 0),  # inside tile 0_0
            _box(90, 10, 130, 30, "ship", 0),  # 1/4 in 0_0, 3/4 in 100_0
            _box(10, 110, 30, 140, "plane", 1),  # difficult, inside 0_50 only
        ],
    )
    _write_image(
        tmp_path, "P0001", (60, 60), [_box(10, 10, 20, 20, "plane", 0)], gsd="null"
    )
    _prepare(tmp_path)
    return tmp_path


def _objects(data_dir: Path, tile_id: str) -> pd.DataFrame:
    _, objects = load_prepared(data_dir, "val")
    return objects[objects.tile_id == tile_id].set_index("obj_idx")


def test_prepare_tiles_without_overlap(dota_dir):
    tiles, _ = load_prepared(dota_dir, "val")

    assert list(tiles.tile_id) == [
        "P0000_0_0",
        "P0000_100_0",
        "P0000_150_0",
        "P0000_0_50",
        "P0000_100_50",
        "P0000_150_50",
        "P0001_0_0",
    ]


def test_prepare_records_tile_attributes(dota_dir):
    tiles, _ = load_prepared(dota_dir, "val")
    by_id = tiles.set_index("tile_id")

    assert by_id.loc["P0000_0_0", "image_source"] == "GoogleEarth"
    assert by_id.loc["P0000_0_0", "gsd"] == 0.1
    assert by_id.loc["P0000_0_0", "pad_frac"] == 0
    assert pd.isna(by_id.loc["P0001_0_0", "gsd"])
    assert by_id.loc["P0001_0_0", "pad_frac"] == pytest.approx(1 - 0.36)


def test_prepare_writes_padded_tiles(dota_dir):
    with Image.open(dota_dir / "tiles" / "val" / "P0001_0_0.png") as tile:
        assert tile.size == (100, 100)
        assert tile.getpixel((99, 99)) == (0, 0, 0)
        assert tile.getpixel((0, 0)) == (255, 255, 255)


def test_prepare_records_coverage_per_tile(dota_dir):
    left = _objects(dota_dir, "P0000_0_0")
    right = _objects(dota_dir, "P0000_100_0")

    assert list(left.index) == [0, 1]
    assert left.loc[0, "coverage"] == 1.0
    assert left.loc[1, "coverage"] == 0.25
    assert right.loc[1, "coverage"] == 0.75
    assert _objects(dota_dir, "P0000_150_0").empty


def test_prepare_records_object_shape(dota_dir):
    plane = _objects(dota_dir, "P0000_0_50").loc[2]

    assert plane.raw_category == "plane"
    assert plane.difficult == 1
    assert (plane.area_px, plane.long_px, plane.short_px) == (600, 30, 20)
    assert plane.angle == 90
    assert (plane.cx, plane.cy) == (20, 75)


def test_prepare_measures_rotated_objects(tmp_path):
    _make_dirs(tmp_path)
    # A diamond centered on the x=100 tile edge sits half in each tile.
    diamond = "100 40 120 60 100 80 80 60 ship 0"
    _write_image(tmp_path, "P0000", (200, 100), [diamond])
    _prepare(tmp_path)

    for tile_id in ("P0000_0_0", "P0000_100_0"):
        ship = _objects(tmp_path, tile_id).loc[0]
        assert ship.coverage == 0.5
        assert ship.angle == 45


def test_prepare_is_deterministic(dota_dir):
    split_dir = dota_dir / "tiles" / "val"
    names = ("tiles.csv", "objects.csv")
    first = [(split_dir / name).read_bytes() for name in names]
    _prepare(dota_dir)

    assert [(split_dir / name).read_bytes() for name in names] == first


def test_prepare_removes_stale_tiles(dota_dir):
    _prepare(dota_dir, tile_size=200)
    tiles, _ = load_prepared(dota_dir, "val")

    assert not (dota_dir / "tiles" / "val" / "P0000_150_0.png").exists()
    assert set(tiles.tile_id) == {"P0000_0_0", "P0000_50_0", "P0001_0_0"}


def test_prepare_rejects_missing_label(dota_dir):
    (dota_dir / "val" / "labelTxt" / "P0001.txt").unlink()

    with pytest.raises(subprocess.CalledProcessError) as error:
        _prepare(dota_dir)

    assert "Image/label mismatch" in error.value.stderr


def test_prepare_rejects_missing_label_dir(dota_dir):
    (dota_dir / "val" / "labelTxt").rename(dota_dir / "val" / "labels")

    with pytest.raises(subprocess.CalledProcessError) as error:
        _prepare(dota_dir)

    assert "labelTxt does not exist" in error.value.stderr


def test_prepare_rejects_unknown_category(dota_dir):
    (dota_dir / "val" / "labelTxt" / "P0001.txt").write_text(
        _box(10, 10, 20, 20, "vehicle", 0)
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        _prepare(dota_dir)

    assert "unknown category 'vehicle'" in error.value.stderr


def test_class_map_categories():
    names = [category.name for category in CLASS_MAP.values()]

    assert tuple(names) == CATEGORIES
    assert len(set(names)) == len(names) == 16
    assert all(c.plural and c.plural != c.name for c in CLASS_MAP.values())


def test_expected_at_thresholds_coverage():
    coverage, difficult = [1.0, 0.25, 0.75], [False, False, True]

    assert expected_at(Counting(), coverage, difficult, 0.5) == [1, 2]
    assert expected_at(Counting(), coverage, difficult, 0.1) == [2, 3]
    assert expected_at(Counting(), coverage, difficult, 0.9) == 1
    assert expected_at(Counting(), [], [], 0.5) == 0
    assert expected_at(Existence(), [0.25], [False], 0.5) is False
    assert expected_at(Existence(), [0.75], [True], 0.5) is True


def test_counting_uses_min_cover(dota_dir):
    examples = build_dataset(dota_dir, Counting(), categories=["ship", "plane"])
    by_id = {example.id: example for example in examples}

    assert len(examples) == 7 * 2
    assert by_id["val/P0000_0_0:ship"].expected == 1
    assert by_id["val/P0000_100_0:ship"].expected == 1
    assert by_id["val/P0000_0_50:plane"].expected == [0, 1]
    assert by_id["val/P0000_150_0:ship"].expected == 0
    assert by_id["val/P0000_150_0:ship"].prompt.startswith("How many ships ")

    loose = build_dataset(dota_dir, Counting(), categories=["ship"], min_cover=0.1)
    assert {e.id: e.expected for e in loose}["val/P0000_0_0:ship"] == 2


def test_existence_uses_min_cover(dota_dir):
    examples = build_dataset(dota_dir, Existence(), categories=["ship", "plane"])
    by_id = {example.id: example for example in examples}

    assert by_id["val/P0000_0_0:ship"].expected is True
    assert by_id["val/P0000_0_50:plane"].expected is True
    assert by_id["val/P0000_150_0:ship"].expected is False


def test_examples_carry_coverage_metadata(dota_dir):
    examples = build_dataset(dota_dir, Counting(), categories=["ship"])
    metadata = {e.id: e.metadata for e in examples}["val/P0000_0_0:ship"]

    assert metadata["dataset"] == "dota"
    assert metadata["raw_category"] == "ship"
    assert metadata["image_id"] == "P0000"
    assert metadata["image_source"] == "GoogleEarth"
    assert metadata["coverage"] == [1.0, 0.25]
    assert metadata["difficult"] == [False, False]
    assert metadata["min_cover"] == 0.5


def test_examples_point_at_tiles(dota_dir):
    example = build_dataset(dota_dir, Counting(), categories=["plane"])[-1]

    assert Path(example.image_path).is_file()
    assert example.metadata["gsd"] is None


def test_with_min_cover_rescores_records(dota_dir):
    examples = build_dataset(dota_dir, Counting(), categories=["ship"])
    records = pd.DataFrame([{"expected": e.expected, **e.metadata} for e in examples])

    rescored = with_min_cover(records, Counting(), 0.1)

    assert rescored.expected.tolist() == [2, 1, 0, 0, 0, 0, 0]
    assert (rescored.min_cover == 0.1).all()
    assert records.expected.tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_with_min_cover_needs_coverage():
    with pytest.raises(ValueError, match="coverage metadata"):
        with_min_cover(pd.DataFrame({"expected": [1]}), Counting(), 0.5)


def test_num_tiles_samples_whole_tiles(dota_dir):
    examples = build_dataset(dota_dir, Existence(), num_tiles=2, seed=3)
    tile_ids = {example.id.split(":")[0] for example in examples}

    assert len(tile_ids) == 2
    assert examples == build_dataset(dota_dir, Existence(), num_tiles=2, seed=3)


def test_build_dataset_rejects_unknown_task(dota_dir):
    with pytest.raises(ValueError, match="Unsupported task"):
        build_dataset(dota_dir, object())


def test_build_dataset_rejects_unknown_category(dota_dir):
    with pytest.raises(ValueError, match="Unknown DOTA categories"):
        build_dataset(dota_dir, Counting(), categories=["vehicle"])


def test_load_prepared_requires_preparation(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"prepare_dota\.py"):
        load_prepared(tmp_path, "val")


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "tiles" / "val" / "objects.csv").exists(),
    reason="DOTA val not prepared",
)
def test_real_dota_val_images():
    tiles, _ = load_prepared(REAL_DATA_DIR, "val")

    assert tiles.image_id.nunique() == 458
