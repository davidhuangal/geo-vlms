import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from geo_vlms.datasets.dior import CLASS_MAP, SPLITS


def _read_split_ids(data_dir: Path) -> dict[str, list[str]]:
    splits = {}
    seen = set()
    for split in SPLITS:
        split_path = data_dir / "Main" / f"{split}.txt"
        image_ids = [line.strip() for line in split_path.read_text().splitlines()]
        image_ids = [image_id for image_id in image_ids if image_id]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{split_path} contains duplicate image ids")
        overlap = seen & set(image_ids)
        if overlap:
            raise ValueError(f"{split_path} overlaps another split: {sorted(overlap)}")
        seen.update(image_ids)
        splits[split] = image_ids
    return splits


def _counts_for(annotation_path: Path, image_name: str) -> Counter[str]:
    root = ET.parse(annotation_path).getroot()
    if root.findtext("filename") != image_name:
        raise ValueError(f"{annotation_path} does not describe {image_name}")

    categories = []
    for obj in root.findall("object"):
        category = obj.findtext("name")
        if category is None:
            raise ValueError(f"{annotation_path} has an object without a category")
        categories.append(category)
    counts = Counter(categories)
    unknown = set(counts) - set(CLASS_MAP)
    if unknown:
        raise ValueError(f"{annotation_path} has unknown categories: {sorted(unknown)}")
    return counts


def prepare_dataset(data_dir: str | Path, out_path: str | Path) -> tuple[int, int]:
    """Convert DIOR XML annotations into a deterministic count table."""
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    splits = _read_split_ids(data_dir)
    annotation_dir = data_dir / "Annotations" / "Horizontal Bounding Boxes"

    expected_ids = {image_id for image_ids in splits.values() for image_id in image_ids}
    annotation_ids = {path.stem for path in annotation_dir.glob("*.xml")}
    if annotation_ids != expected_ids:
        missing = sorted(expected_ids - annotation_ids)
        extra = sorted(annotation_ids - expected_ids)
        raise ValueError(f"Annotation mismatch: missing={missing}, extra={extra}")
    for image_dir in ("JPEGImages-trainval", "JPEGImages-test"):
        image_ids = {path.stem for path in (data_dir / image_dir).glob("*.jpg")}
        stray = sorted(image_ids - expected_ids)
        if stray:
            raise ValueError(f"{image_dir} has images in no split: {stray}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
    row_count = 0
    with temporary_path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "image_id",
                "split",
                "image_path",
                "raw_category",
                "category",
                "count",
            ),
        )
        writer.writeheader()
        for split in SPLITS:
            image_dir = "JPEGImages-test" if split == "test" else "JPEGImages-trainval"
            for image_id in splits[split]:
                image_name = f"{image_id}.jpg"
                relative_image_path = Path(image_dir) / image_name
                if not (data_dir / relative_image_path).is_file():
                    raise FileNotFoundError(data_dir / relative_image_path)
                counts = _counts_for(annotation_dir / f"{image_id}.xml", image_name)
                for raw_category, category in CLASS_MAP.items():
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "split": split,
                            "image_path": relative_image_path.as_posix(),
                            "raw_category": raw_category,
                            "category": category,
                            "count": counts[raw_category],
                        }
                    )
                    row_count += 1

    temporary_path.replace(out_path)
    return len(expected_ids), row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DIOR counts for existence and counting evaluation."
    )
    parser.add_argument("--data-dir", default="data/dior", help="Raw DIOR root.")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output CSV. Default: <data-dir>/counts.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(args.data_dir) / "counts.csv"
    images, rows = prepare_dataset(args.data_dir, out_path)
    print(f"Prepared {rows} rows for {images} images in {out_path}")


if __name__ == "__main__":
    main()
