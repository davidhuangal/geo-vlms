import argparse
from pathlib import Path

import pandas as pd

from geo_vlms import counting
from geo_vlms.analysis import load_records, score_records, summarize


def parse_args() -> argparse.Namespace:
    """Parse CLI for counting analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze records from a VLM counting inference job.",
    )
    parser.add_argument(
        "-r",
        "--records",
        type=str,
        required=True,
        help="Path to counting records JSONL file.",
    )
    parser.add_argument(
        "-g",
        "--groupby",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Columns to group by in a summary.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Desired metrics to view.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        raise FileNotFoundError(f"Records file {records_path} does not exist.")

    records_df = load_records(records_path)

    records_df = records_df.join(pd.json_normalize(records_df["metadata"])).drop(
        columns=["metadata"]
    )

    metrics_df = score_records(
        records_df=records_df, parse=counting.parse_response, score=counting.score
    )

    metric_cols = (
        [c for c in metrics_df.columns if c not in records_df.columns]
        if args.metrics is None
        else args.metrics
    )

    summary_df = summarize(
        metrics_df=metrics_df, group_by=args.groupby, metric_cols=metric_cols
    )

    print(summary_df)


if __name__ == "__main__":
    main()
