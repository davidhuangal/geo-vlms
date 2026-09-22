import random
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from geo_vlms.example import Example
from geo_vlms.tasks import Category, Counting, Existence, Task

CLASS_MAP = {
    "airplane": Category("airplane", "airplanes"),
    "airport": Category("airport", "airports"),
    "baseballfield": Category("baseball field", "baseball fields"),
    "basketballcourt": Category("basketball court", "basketball courts"),
    "bridge": Category("bridge", "bridges"),
    "chimney": Category("chimney", "chimneys"),
    "dam": Category("dam", "dams"),
    "Expressway-Service-area": Category(
        "expressway service area", "expressway service areas"
    ),
    "Expressway-toll-station": Category(
        "expressway toll station", "expressway toll stations"
    ),
    "golffield": Category("golf field", "golf fields"),
    "groundtrackfield": Category("ground track field", "ground track fields"),
    "harbor": Category("harbor", "harbors"),
    "overpass": Category("overpass", "overpasses"),
    "ship": Category("ship", "ships"),
    "stadium": Category("stadium", "stadiums"),
    "storagetank": Category("storage tank", "storage tanks"),
    "tenniscourt": Category("tennis court", "tennis courts"),
    "trainstation": Category("train station", "train stations"),
    "vehicle": Category("vehicle", "vehicles"),
    "windmill": Category("windmill", "windmills"),
}
CATEGORIES = tuple(category.name for category in CLASS_MAP.values())
_BY_NAME = {category.name: category for category in CLASS_MAP.values()}
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
                prompt=task.format_prompt(_BY_NAME[row.category]),
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


def build_dataset(
    data_dir: str | Path,
    task: Task,
    seed: int = 0,
    split: str = "test",
    num_images: int | None = None,
    categories: Collection[str] | None = None,
) -> list[Example]:
    """Build DIOR examples for `task` from one official split."""
    if not isinstance(task, Counting | Existence):
        raise ValueError(f"Unsupported task for dior: {type(task).__name__}")
    return _build_dataset(data_dir, split, task, num_images, seed, categories)
