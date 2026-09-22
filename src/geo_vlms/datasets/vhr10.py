import os
import random
from pathlib import Path

import pandas as pd

from geo_vlms.example import Example
from geo_vlms.tasks import Category, Counting, Existence, Task

CLASS_MAP = {
    1: Category("airplane", "airplanes"),
    2: Category("ship", "ships"),
    3: Category("storage tank", "storage tanks"),
    4: Category("baseball diamond", "baseball diamonds"),
    5: Category("tennis court", "tennis courts"),
    6: Category("basketball court", "basketball courts"),
    7: Category("ground track field", "ground track fields"),
    8: Category("harbor", "harbors"),
    9: Category("bridge", "bridges"),
    10: Category("vehicle", "vehicles"),
}


def parse_annotation(annotation_path: os.PathLike) -> pd.DataFrame:
    """Parse VHR10 ground truth annotation as a Pandas DataFrame"""
    return pd.read_csv(
        annotation_path,
        names=["corner1", "corner2", "category_id"],
        sep=r",(?![^()]*\))",
        engine="python",
    )


def _sorted_jpgs(
    image_dir: os.PathLike, num_images: int | None, rng: random.Random
) -> list[Path]:
    # Sorted so the dataset order (and therefore example ids and results
    # files) is stable across runs and filesystems. Filtered to .jpg so
    # filesystem debris (e.g. .DS_Store) never becomes an Example.
    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    if num_images is not None:
        image_paths = sorted(rng.sample(image_paths, num_images))
    return image_paths


def _category_counts(image_path: Path, gt_dir: os.PathLike) -> dict[Category, int]:
    gt_path = Path(gt_dir) / image_path.with_suffix(".txt").name
    if not gt_path.exists():
        raise FileNotFoundError(f"No such file {gt_path}")

    gt_df = parse_annotation(gt_path)

    counts_per_category = {category_id: 0 for category_id in CLASS_MAP}
    for _, row in gt_df.iterrows():
        counts_per_category[row.category_id] += 1

    return {
        CLASS_MAP[category_id]: count
        for category_id, count in counts_per_category.items()
    }


def build_counting_dataset(
    pos_dir: os.PathLike,
    gt_dir: os.PathLike,
    neg_dir: os.PathLike | None = None,
    num_pos_images: int | None = None,
    num_neg_images: int | None = None,
    seed: int = 0,
) -> list[Example]:
    task = Counting()
    examples = []

    pos_rng = random.Random(f"{seed}-pos")
    for image_path in _sorted_jpgs(pos_dir, num_pos_images, pos_rng):
        for category, count in _category_counts(image_path, gt_dir).items():
            examples.append(
                Example(
                    id=f"pos/{image_path.stem}:{category.name}",
                    image_path=str(image_path),
                    prompt=task.format_prompt(category),
                    expected=count,
                    metadata={
                        "dataset": "vhr10",
                        "split": "positive",
                        "category": category.name,
                    },
                )
            )

    if neg_dir is not None:
        neg_rng = random.Random(f"{seed}-neg")
        for image_path in _sorted_jpgs(neg_dir, num_neg_images, neg_rng):
            for category in CLASS_MAP.values():
                examples.append(
                    Example(
                        id=f"neg/{image_path.stem}:{category.name}",
                        image_path=str(image_path),
                        prompt=task.format_prompt(category),
                        expected=0,
                        metadata={
                            "dataset": "vhr10",
                            "split": "negative",
                            "category": category.name,
                        },
                    )
                )

    return examples


def build_existence_dataset(
    pos_dir: os.PathLike,
    gt_dir: os.PathLike,
    neg_dir: os.PathLike | None = None,
    num_pos_images: int | None = None,
    num_neg_images: int | None = None,
    seed: int = 0,
) -> list[Example]:
    task = Existence()
    examples = []

    pos_rng = random.Random(f"{seed}-pos")
    for image_path in _sorted_jpgs(pos_dir, num_pos_images, pos_rng):
        for category, count in _category_counts(image_path, gt_dir).items():
            examples.append(
                Example(
                    id=f"pos/{image_path.stem}:{category.name}",
                    image_path=str(image_path),
                    prompt=task.format_prompt(category),
                    expected=count >= 1,
                    metadata={
                        "dataset": "vhr10",
                        "split": "positive",
                        "category": category.name,
                    },
                )
            )

    if neg_dir is not None:
        neg_rng = random.Random(f"{seed}-neg")
        for image_path in _sorted_jpgs(neg_dir, num_neg_images, neg_rng):
            for category in CLASS_MAP.values():
                examples.append(
                    Example(
                        id=f"neg/{image_path.stem}:{category.name}",
                        image_path=str(image_path),
                        prompt=task.format_prompt(category),
                        expected=False,
                        metadata={
                            "dataset": "vhr10",
                            "split": "negative",
                            "category": category.name,
                        },
                    )
                )

    return examples


def build_dataset(
    data_dir: os.PathLike,
    task: Task,
    seed: int = 0,
    num_pos: int | None = None,
    num_neg: int | None = None,
    no_neg: bool = False,
) -> list[Example]:
    """Build VHR-10 examples for `task` from the unpacked dataset layout."""
    data_dir = Path(data_dir)
    if isinstance(task, Counting):
        build = build_counting_dataset
    elif isinstance(task, Existence):
        build = build_existence_dataset
    else:
        raise ValueError(f"Unsupported task for vhr10: {type(task).__name__}")
    return build(
        pos_dir=data_dir / "positive_image_set",
        gt_dir=data_dir / "ground_truth",
        neg_dir=None if no_neg else data_dir / "negative_image_set",
        num_pos_images=num_pos,
        num_neg_images=num_neg,
        seed=seed,
    )
