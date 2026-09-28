import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from geo_vlms.coverage import expected_at
from geo_vlms.tasks import Task


def load_records(results_path: os.PathLike) -> pd.DataFrame:
    """Read a raw `.jsonl` of records back into a DataFrame."""
    lines = Path(results_path).read_text().splitlines()
    records = [json.loads(x) for x in lines]

    records_df = pd.DataFrame(records)

    return records_df


def with_min_cover(
    records_df: pd.DataFrame, task: Task, min_cover: float
) -> pd.DataFrame:
    """Recompute `expected` counting only objects at least `min_cover` in view."""
    if "coverage" not in records_df.columns:
        raise ValueError("--min-cover needs records with coverage metadata, like dota")

    records_df = records_df.copy()
    records_df["expected"] = [
        expected_at(task, coverage, difficult, min_cover)
        for coverage, difficult in zip(
            records_df["coverage"], records_df["difficult"], strict=True
        )
    ]
    records_df["min_cover"] = min_cover
    return records_df


def score_records(
    records_df: pd.DataFrame, parse: Callable, score: Callable
) -> pd.DataFrame:
    """Perform scoring of records using given parsing and scoring functions."""
    scores = []

    for _, row in records_df.iterrows():
        model_response = row["output"]
        expected_response = row["expected"]
        parsed_response = parse(model_response)
        scores.append(score(parsed_response, expected_response))

    scores_df = pd.DataFrame(scores)
    metrics_df = pd.concat(
        [records_df.reset_index(drop=True), scores_df],
        axis="columns",
        verify_integrity=True,
    )
    metrics_df.index = records_df.index
    return metrics_df


def summarize(
    metrics_df: pd.DataFrame, group_by: list[str] | None, metric_cols: list[str]
) -> pd.DataFrame:
    """Summarize results from a metrics DataFrame.

    When `group_by` is None, summarizes over the whole DataFrame as a
    one-row result labeled "mean".
    """
    if group_by is None:
        return metrics_df[metric_cols].mean().to_frame("mean").T
    return metrics_df.groupby(group_by)[metric_cols].mean()
