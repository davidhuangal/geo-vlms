import random
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from geo_vlms.example import Example
from geo_vlms.tasks import Counting, Existence

CLASS_MAP = {
    "airplane": "airplane",
    "airport": "airport",
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "bridge": "bridge",
    "chimney": "chimney",
    "dam": "dam",
    "Expressway-Service-area": "expressway service area",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "harbor": "harbor",
    "overpass": "overpass",
    "ship": "ship",
    "stadium": "stadium",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
    "vehicle": "vehicle",
    "windmill": "windmill",
}
CATEGORIES = tuple(CLASS_MAP.values())
SPLITS = ("train", "val", "test")


def load_prepared(data_dir: str | Path) -> pd.DataFrame:
    """Load the counts table produced by scripts/prepare_dior.py."""
    counts_path = Path(data_dir) / "counts.csv"
    if not counts_path.is_file():
        raise FileNotFoundError(
            f"{counts_path} does not exist; run scripts/prepare_dior.py first"
        )
    return pd.read_csv(counts_path, dtype={"image_id": str})


def _select_rows(
    counts: pd.DataFrame,
    split: str,
    num_images: int | None,
    seed: int,
    categories: Collection[str] | None,
) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown DIOR split {split!r}; choose from {SPLITS}")

    rows = counts[counts.split == split]
    image_ids = sorted(rows.image_id.unique())
    if num_images is not None:
        if not 0 <= num_images <= len(image_ids):
            raise ValueError(
                f"num_images must be between 0 and {len(image_ids)}, got {num_images}"
            )
        rng = random.Random(f"{seed}-{split}")
        image_ids = sorted(rng.sample(image_ids, num_images))
        rows = rows[rows.image_id.isin(image_ids)]

    if categories is not None:
        unknown = set(categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown DIOR categories: {sorted(unknown)}")
        rows = rows[rows.category.isin(categories)]

    return rows


def _build_dataset(
    data_dir: str | Path,
    split: str,
    task: Counting | Existence,
    num_images: int | None,
    seed: int,
    categories: Collection[str] | None,
) -> list[Example]:
    data_dir = Path(data_dir)
    rows = _select_rows(load_prepared(data_dir), split, num_images, seed, categories)
    examples = []
    for row in rows.itertuples(index=False):
        count = int(row.count)
        expected = count if isinstance(task, Counting) else count > 0
        examples.append(
            Example(
                id=f"{split}/{row.image_id}:{row.category}",
                image_path=str(data_dir / row.image_path),
                prompt=task.format_prompt(category_name=row.category),
                expected=expected,
                metadata={
                    "dataset": "dior",
                    "split": split,
                    "category": row.category,
                    "raw_category": row.raw_category,
                },
            )
        )
    return examples


def build_counting_dataset(
    data_dir: str | Path,
    split: str,
    num_images: int | None = None,
    seed: int = 0,
    categories: Collection[str] | None = None,
) -> list[Example]:
    """Build DIOR counting examples from one official split."""
    return _build_dataset(data_dir, split, Counting(), num_images, seed, categories)


def build_existence_dataset(
    data_dir: str | Path,
    split: str,
    num_images: int | None = None,
    seed: int = 0,
    categories: Collection[str] | None = None,
) -> list[Example]:
    """Build DIOR existence examples from one official split."""
    return _build_dataset(data_dir, split, Existence(), num_images, seed, categories)
