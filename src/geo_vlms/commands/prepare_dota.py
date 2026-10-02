import argparse
import csv
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from PIL import Image

from geo_vlms.datasets.dota import CLASS_MAP, SPLITS

# DOTA images reach ~20000 px per side, past PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

Point = tuple[float, float]
Box = tuple[int, int, int, int]

TILE_FIELDS = (
    "tile_id",
    "image_id",
    "tile_path",
    "x0",
    "y0",
    "image_source",
    "gsd",
    "pad_frac",
)
OBJECT_FIELDS = (
    "tile_id",
    "obj_idx",
    "raw_category",
    "category",
    "coverage",
    "difficult",
    "area_px",
    "long_px",
    "short_px",
    "angle",
    "cx",
    "cy",
)


@dataclass
class Label:
    polygon: list[Point]
    category: str
    difficult: bool


def _read_labels(label_path: Path) -> tuple[dict[str, str], list[Label]]:
    header = {"imagesource": "", "gsd": ""}
    labels = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue

        key, _, value = line.partition(":")
        if key in header:
            header[key] = "" if value.strip() == "null" else value.strip()
            continue

        if len(fields) != 10:
            raise ValueError(f"{label_path} has a malformed line: {line!r}")

        coords = [float(value) for value in fields[:8]]
        category, difficult = fields[8], fields[9]
        if category not in CLASS_MAP:
            raise ValueError(f"{label_path} has unknown category {category!r}")
        if difficult not in ("0", "1"):
            raise ValueError(f"{label_path} has bad difficult flag {difficult!r}")

        polygon = list(zip(coords[0::2], coords[1::2], strict=True))
        labels.append(Label(polygon, category, difficult == "1"))
    return header, labels


def _area(polygon: list[Point]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def _keeps(point: Point, axis: int, bound: float, below: bool) -> bool:
    return point[axis] <= bound if below else point[axis] >= bound


def _clip(polygon: list[Point], box: Box) -> list[Point]:
    """Sutherland-Hodgman clip of `polygon` to an axis-aligned box."""
    x0, y0, x1, y1 = box
    for axis, bound, below in (
        (0, x0, False),
        (0, x1, True),
        (1, y0, False),
        (1, y1, True),
    ):
        clipped = []
        for i, current in enumerate(polygon):
            previous = polygon[i - 1]
            current_in = _keeps(current, axis, bound, below)
            if current_in != _keeps(previous, axis, bound, below):
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(
                    (
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    )
                )
            if current_in:
                clipped.append(current)
        polygon = clipped
        if not polygon:
            break
    return polygon


def _center(polygon: list[Point]) -> Point:
    return (
        sum(x for x, _ in polygon) / len(polygon),
        sum(y for _, y in polygon) / len(polygon),
    )


def _coverage(polygon: list[Point], box: Box) -> float:
    """Fraction of the polygon's area inside `box`."""
    area = _area(polygon)
    if area == 0:
        # Degenerate label: fall back to whether its center is in the box.
        cx, cy = _center(polygon)
        x0, y0, x1, y1 = box
        return float(x0 <= cx < x1 and y0 <= cy < y1)
    return _area(_clip(polygon, box)) / area


def _shape(polygon: list[Point]) -> tuple[float, float, float]:
    """Long side, short side, and long-side angle in [0, 180) of a quadrilateral."""
    p0, p1, p2, p3 = polygon
    side_a = (math.dist(p0, p1) + math.dist(p2, p3)) / 2
    side_b = (math.dist(p1, p2) + math.dist(p3, p0)) / 2
    start, end = (p0, p1) if side_a >= side_b else (p1, p2)
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180
    return max(side_a, side_b), min(side_a, side_b), angle


def _origins(length: int, tile_size: int) -> list[int]:
    # No overlap, except the last tile shifts back to end at the border.
    if length <= tile_size:
        return [0]
    return [*range(0, length - tile_size, tile_size), length - tile_size]


def _object_rows(tile_id: str, labels: list[Label], box: Box) -> list[dict]:
    rows = []
    for obj_idx, label in enumerate(labels):
        coverage = _coverage(label.polygon, box)
        if coverage <= 0:
            continue

        long_px, short_px, angle = _shape(label.polygon)
        cx, cy = _center(label.polygon)
        rows.append(
            {
                "tile_id": tile_id,
                "obj_idx": obj_idx,
                "raw_category": label.category,
                "category": CLASS_MAP[label.category].name,
                "coverage": round(coverage, 4),
                "difficult": int(label.difficult),
                "area_px": round(_area(label.polygon), 1),
                "long_px": round(long_px, 1),
                "short_px": round(short_px, 1),
                "angle": round(angle, 1),
                "cx": round(cx - box[0], 1),
                "cy": round(cy - box[1], 1),
            }
        )
    return rows


def _tile_image(
    data_dir: Path, split: str, image_id: str, tile_size: int
) -> tuple[list[dict], list[dict]]:
    """Write one image's tiles and return their tile and object rows."""
    header, labels = _read_labels(data_dir / split / "labelTxt" / f"{image_id}.txt")
    tile_rows, object_rows = [], []
    with Image.open(data_dir / split / "images" / f"{image_id}.png") as image:
        image = image.convert("RGB")
        width, height = image.size

        for y0 in _origins(height, tile_size):
            for x0 in _origins(width, tile_size):
                box = (x0, y0, x0 + tile_size, y0 + tile_size)
                tile_id = f"{image_id}_{x0}_{y0}"
                tile_path = Path("tiles") / split / f"{tile_id}.png"
                # Crop pads past the image border with black.
                image.crop(box).save(data_dir / tile_path)

                inside = (min(width, box[2]) - x0) * (min(height, box[3]) - y0)
                tile_rows.append(
                    {
                        "tile_id": tile_id,
                        "image_id": image_id,
                        "tile_path": tile_path.as_posix(),
                        "x0": x0,
                        "y0": y0,
                        "image_source": header["imagesource"],
                        "gsd": header["gsd"],
                        "pad_frac": round(1 - inside / tile_size**2, 4),
                    }
                )
                object_rows.extend(_object_rows(tile_id, labels, box))
    return tile_rows, object_rows


def _check_pairs(image_dir: Path, label_dir: Path) -> list[str]:
    for folder in (image_dir, label_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} does not exist")

    image_ids = {path.stem for path in image_dir.glob("*.png")}
    label_ids = {path.stem for path in label_dir.glob("*.txt")}
    if not image_ids or image_ids != label_ids:
        missing = sorted(image_ids - label_ids)
        extra = sorted(label_ids - image_ids)
        raise ValueError(
            f"Image/label mismatch in {image_dir.parent}: "
            f"no label for {missing}, no image for {extra}"
        )
    return sorted(image_ids)


def prepare_split(
    data_dir: str | Path, split: str, tile_size: int, workers: int | None = None
) -> tuple[int, int]:
    """Tile one split and write deterministic tile and object tables."""
    data_dir = Path(data_dir)
    image_ids = _check_pairs(data_dir / split / "images", data_dir / split / "labelTxt")

    # The split's tile dir is fully generated, so stale tiles from an earlier
    # tile size must not survive.
    out_dir = data_dir / "tiles" / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tile_count = 0
    object_count = 0
    tile = partial(_tile_image, data_dir, split, tile_size=tile_size)
    with (
        (out_dir / "tiles.csv.tmp").open("w", newline="") as tiles_out,
        (out_dir / "objects.csv.tmp").open("w", newline="") as objects_out,
        ProcessPoolExecutor(workers) as pool,
    ):
        tile_writer = csv.DictWriter(tiles_out, fieldnames=TILE_FIELDS)
        object_writer = csv.DictWriter(objects_out, fieldnames=OBJECT_FIELDS)
        tile_writer.writeheader()
        object_writer.writeheader()
        # map yields in input order, so the tables are the same for any worker count.
        for tile_rows, object_rows in pool.map(tile, image_ids):
            tile_writer.writerows(tile_rows)
            object_writer.writerows(object_rows)
            tile_count += len(tile_rows)
            object_count += len(object_rows)

    (out_dir / "tiles.csv.tmp").replace(out_dir / "tiles.csv")
    (out_dir / "objects.csv.tmp").replace(out_dir / "objects.csv")
    return tile_count, object_count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geo-vlms prepare-dota",
        description="Tile DOTA-v1.5 for existence and counting evaluation.",
    )
    parser.add_argument("--data-dir", default="data/dota", help="Raw DOTA root.")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=["val"], help="Splits to tile."
    )
    parser.add_argument("--tile-size", type=int, default=896, help="Tile side in px.")
    parser.add_argument(
        "--workers", type=int, default=None, help="Processes. Default: CPU count."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    for split in args.splits:
        tiles, objects = prepare_split(
            args.data_dir, split, args.tile_size, args.workers
        )
        print(f"Prepared {tiles} {split} tiles with {objects} object rows")
