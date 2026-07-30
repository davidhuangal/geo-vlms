import argparse
from pathlib import Path

from geo_vlms import counting
from geo_vlms.analysis import load_records, score_records


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
    return parser.parse_args()


def main():
    args = parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        raise FileNotFoundError(f"Records file {records_path} does not exist.")

    records_df = load_records(records_path)

    metrics_df = score_records(
        records_df=records_df, parse=counting.parse_response, score=counting.score
    )

    print(metrics_df)


if __name__ == "__main__":
    main()
