import os
import random
from pathlib import Path

import pandas as pd

from geo_vlms.example import Example
from geo_vlms.tasks import Counting, Existence, Task

CLASS_MAP = {
    1: "airplane",
    2: "ship",
    3: "storage tank",
    4: "baseball diamond",
    5: "tennis court",
    6: "basketball court",
    7: "ground track field",
    8: "harbor",
    9: "bridge",
    10: "vehicle",
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


def _category_counts(image_path: Path, gt_dir: os.PathLike) -> dict[str, int]:
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
        for category_name, count in _category_counts(image_path, gt_dir).items():
            examples.append(
                Example(
                    id=f"pos/{image_path.stem}:{category_name}",
                    image_path=str(image_path),
                    prompt=task.format_prompt(category_name=category_name),
                    expected=count,
                    metadata={
                        "dataset": "vhr10",
                        "split": "positive",
                        "category": category_name,
                    },
                )
            )

    if neg_dir is not None:
        neg_rng = random.Random(f"{seed}-neg")
        for image_path in _sorted_jpgs(neg_dir, num_neg_images, neg_rng):
            for category_name in CLASS_MAP.values():
                examples.append(
                    Example(
                        id=f"neg/{image_path.stem}:{category_name}",
                        image_path=str(image_path),
                        prompt=task.format_prompt(category_name=category_name),
                        expected=0,
                        metadata={
                            "dataset": "vhr10",
                            "split": "negative",
                            "category": category_name,
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
        for category_name, count in _category_counts(image_path, gt_dir).items():
            examples.append(
                Example(
                    id=f"pos/{image_path.stem}:{category_name}",
                    image_path=str(image_path),
                    prompt=task.format_prompt(category_name=category_name),
                    expected=count >= 1,
                    metadata={
                        "dataset": "vhr10",
                        "split": "positive",
                        "category": category_name,
                    },
                )
            )

    if neg_dir is not None:
        neg_rng = random.Random(f"{seed}-neg")
        for image_path in _sorted_jpgs(neg_dir, num_neg_images, neg_rng):
            for category_name in CLASS_MAP.values():
                examples.append(
                    Example(
                        id=f"neg/{image_path.stem}:{category_name}",
                        image_path=str(image_path),
                        prompt=task.format_prompt(category_name=category_name),
                        expected=False,
                        metadata={
                            "dataset": "vhr10",
                            "split": "negative",
                            "category": category_name,
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
