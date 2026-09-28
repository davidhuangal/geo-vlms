import random
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from geo_vlms.coverage import expected_at
from geo_vlms.example import Example
from geo_vlms.tasks import Category, Counting, Existence, Task

CLASS_MAP = {
    "baseball-diamond": Category("baseball diamond", "baseball diamonds"),
    "basketball-court": Category("basketball court", "basketball courts"),
    "bridge": Category("bridge", "bridges"),
    "container-crane": Category("container crane", "container cranes"),
    "ground-track-field": Category("ground track field", "ground track fields"),
    "harbor": Category("harbor", "harbors"),
    "helicopter": Category("helicopter", "helicopters"),
    "large-vehicle": Category("large vehicle", "large vehicles"),
    "plane": Category("plane", "planes"),
    "roundabout": Category("roundabout", "roundabouts"),
    "ship": Category("ship", "ships"),
    "small-vehicle": Category("small vehicle", "small vehicles"),
    "soccer-ball-field": Category("soccer ball field", "soccer ball fields"),
    "storage-tank": Category("storage tank", "storage tanks"),
    "swimming-pool": Category("swimming pool", "swimming pools"),
    "tennis-court": Category("tennis court", "tennis courts"),
}
CATEGORIES = tuple(category.name for category in CLASS_MAP.values())
SPLITS = ("train", "val")


def load_prepared(
    data_dir: str | Path, split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the tile and object tables produced by scripts/prepare_dota.py."""
    if split not in SPLITS:
        raise ValueError(f"Unknown DOTA split {split!r}; choose from {SPLITS}")

    split_dir = Path(data_dir) / "tiles" / split
    paths = (split_dir / "tiles.csv", split_dir / "objects.csv")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist; run scripts/prepare_dota.py first"
            )

    dtype = {"tile_id": str, "image_id": str}
    return pd.read_csv(paths[0], dtype=dtype), pd.read_csv(paths[1], dtype=dtype)


def _sample_tiles(
    tiles: pd.DataFrame, num_tiles: int | None, seed: int
) -> pd.DataFrame:
    if num_tiles is None:
        return tiles

    tile_ids = sorted(tiles.tile_id)
    if not 0 <= num_tiles <= len(tile_ids):
        raise ValueError(
            f"num_tiles must be between 0 and {len(tile_ids)}, got {num_tiles}"
        )
    rng = random.Random(f"{seed}-dota")
    return tiles[tiles.tile_id.isin(rng.sample(tile_ids, num_tiles))]


def _objects_by_row(objects: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {key: group for key, group in objects.groupby(["tile_id", "category"])}


def build_dataset(
    data_dir: str | Path,
    task: Task,
    seed: int = 0,
    split: str = "val",
    num_tiles: int | None = None,
    categories: Collection[str] | None = None,
    min_cover: float = 0.5,
) -> list[Example]:
    """Build DOTA examples for `task` from the tiles of one split."""
    if not isinstance(task, Counting | Existence):
        raise ValueError(f"Unsupported task for dota: {type(task).__name__}")
    if categories is not None:
        unknown = set(categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown DOTA categories: {sorted(unknown)}")

    data_dir = Path(data_dir)
    tiles, objects = load_prepared(data_dir, split)
    tiles = _sample_tiles(tiles, num_tiles, seed)
    by_row = _objects_by_row(objects)
    no_objects = objects.iloc[:0]

    examples = []
    for tile in tiles.itertuples(index=False):
        for raw_category, category in CLASS_MAP.items():
            if categories is not None and category.name not in categories:
                continue

            found = by_row.get((tile.tile_id, category.name), no_objects)
            coverage = [float(c) for c in found.coverage]
            difficult = [bool(d) for d in found.difficult]
            examples.append(
                Example(
                    id=f"{split}/{tile.tile_id}:{category.name}",
                    image_path=str(data_dir / tile.tile_path),
                    prompt=task.format_prompt(category),
                    expected=expected_at(task, coverage, difficult, min_cover),
                    metadata={
                        "dataset": "dota",
                        "split": split,
                        "category": category.name,
                        "raw_category": raw_category,
                        "image_id": tile.image_id,
                        "x0": int(tile.x0),
                        "y0": int(tile.y0),
                        "image_source": tile.image_source,
                        "gsd": None if pd.isna(tile.gsd) else float(tile.gsd),
                        "pad_frac": float(tile.pad_frac),
                        "min_cover": min_cover,
                        "coverage": coverage,
                        "difficult": difficult,
                    },
                )
            )
    return examples
