import os
import random
from pathlib import Path

import pandas as pd

from geo_vlms.counting import format_prompt
from geo_vlms.example import Example

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


def _counting_question(category_name: str) -> str:
    return f"How many {category_name} objects are in this image?"


def build_counting_dataset(
    pos_dir: os.PathLike,
    gt_dir: os.PathLike,
    neg_dir: os.PathLike | None = None,
    num_pos_images: int | None = None,
    num_neg_images: int | None = None,
    seed: int = 0,
) -> list[Example]:
    examples = []
    pos_rng = random.Random(f"{seed}-pos")
    neg_rng = random.Random(f"{seed}-neg")

    # ----- Handle positive samples -----
    # Sorted so the dataset order (and therefore example ids and results
    # files) is stable across runs and filesystems.
    pos_image_paths = [Path(pos_dir) / img for img in sorted(os.listdir(pos_dir))]

    if num_pos_images is not None:
        pos_image_paths = sorted(pos_rng.sample(pos_image_paths, num_pos_images))

    for pos_image_path in pos_image_paths:
        # Find the corresponding GT file
        gt_path = Path(gt_dir) / pos_image_path.with_suffix(".txt").name
        if not gt_path.exists():
            raise FileNotFoundError(f"No such file {gt_path}")

        # Parse GT file to DataFrame
        gt_df = parse_annotation(gt_path)

        # Measure number of objects of each category in the image
        counts_per_category = {category_id: 0 for category_id in CLASS_MAP}
        for _, row in gt_df.iterrows():
            counts_per_category[row.category_id] += 1

        # Create Examples based on the object counts. Categories absent from
        # a GT file are skipped rather than emitted as expected=0: positive
        # images are only annotated for their target classes, so absence
        # means "not labeled", not "not present" (e.g. parking lots full of
        # unannotated vehicles). Trustworthy zeros come from the negative
        # set, which is curated to contain none of the ten classes.
        for category_id, object_count in counts_per_category.items():
            if object_count < 1:
                continue
            category_name = CLASS_MAP[category_id]
            examples.append(
                Example(
                    id=f"pos/{pos_image_path.stem}:{category_name}",
                    image_path=str(pos_image_path),
                    prompt=format_prompt(_counting_question(category_name)),
                    expected=object_count,
                )
            )

    # ----- Handle negative samples -----
    if neg_dir is not None:
        neg_image_paths = [Path(neg_dir) / img for img in sorted(os.listdir(neg_dir))]
        if num_neg_images is not None:
            neg_image_paths = sorted(neg_rng.sample(neg_image_paths, num_neg_images))
        for neg_image_path in neg_image_paths:
            # Create an expected=0 example per category
            for category_name in CLASS_MAP.values():
                examples.append(
                    Example(
                        id=f"neg/{neg_image_path.stem}:{category_name}",
                        image_path=str(neg_image_path),
                        prompt=format_prompt(_counting_question(category_name)),
                        expected=0,
                    )
                )

    return examples
