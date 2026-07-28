import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def load_records(results_path: os.PathLike) -> pd.DataFrame:
    """Read a raw `.jsonl` of records back into a DataFrame."""
    lines = Path(results_path).read_text().splitlines()
    records = [json.loads(x) for x in lines]

    records_df = pd.DataFrame(records)

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

    metrics_df = pd.DataFrame(scores, index=records_df.index)
    metrics_df = records_df.join(metrics_df)
    return metrics_df


def summarize(
    metrics_df: pd.DataFrame, group_by: list[str], metric_cols: list
) -> pd.DataFrame:
    """Summarize results from a metrics DataFrame."""
    return metrics_df.groupby(group_by)[metric_cols].mean()
