import argparse
import difflib
import json
import sys
from pathlib import Path

import pandas as pd

from geo_vlms.analysis import load_records, score_records, summarize, with_min_cover
from geo_vlms.tasks import TASKS


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI for analysis."""
    parser = argparse.ArgumentParser(
        prog="geo-vlms analyze",
        description="Analyze records from a VLM inference job.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=False,
        default=None,
        choices=TASKS,
        help="Target task. Default: the task in <records stem>.meta.json.",
    )
    parser.add_argument(
        "-r",
        "--records",
        type=str,
        required=True,
        help="Path to records JSONL file.",
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
    parser.add_argument(
        "--min-cover",
        type=float,
        required=False,
        default=None,
        help="Rescore counting only objects with this fraction in view.",
    )
    return parser.parse_args(argv)


def usage_error(message: str):
    print(message, file=sys.stderr)
    sys.exit(2)


def unknown_names(kind: str, names: list[str], valid: list[str]) -> list[str]:
    """One message per name not in `valid`, then the valid names."""
    messages = []
    for name in names:
        if name in valid:
            continue
        message = f"Unknown {kind} '{name}'."
        close = difflib.get_close_matches(name, valid, n=1)
        if close:
            message += f" Did you mean '{close[0]}'?"
        messages.append(message)

    if messages:
        messages.append(f"{kind.capitalize()}s: {', '.join(valid)}")
    return messages


def resolve_task(records_path: Path, task: str | None) -> str:
    """The run's task from its `.meta.json`, checked against `--task` if given."""
    meta_path = records_path.with_suffix(".meta.json")
    if not meta_path.exists():
        if task is None:
            usage_error(f"{meta_path} not found; pass --task")
        print(
            f"Warning: {meta_path} not found; scoring as {task}. "
            "A wrong --task gives wrong results.",
            file=sys.stderr,
        )
        return task

    meta_task = json.loads(meta_path.read_text()).get("args", {}).get("task")
    if meta_task is None:
        usage_error(f"{meta_path} has no args.task")
    if task is not None and task != meta_task:
        usage_error(f"--task {task} does not match {meta_task} in {meta_path}")
    return meta_task


def main(argv: list[str]):
    args = parse_args(argv)

    records_path = Path(args.records)
    if not records_path.exists():
        raise FileNotFoundError(f"Records file {records_path} does not exist.")
    task = TASKS[resolve_task(records_path, args.task)]()

    records_df = load_records(records_path)

    records_df = records_df.join(pd.json_normalize(records_df["metadata"])).drop(
        columns=["metadata"]
    )
    if args.min_cover is not None:
        records_df = with_min_cover(records_df, task, args.min_cover)

    metrics_df = score_records(
        records_df=records_df, parse=task.parse_response, score=task.score
    )

    numeric = list(metrics_df.select_dtypes(include=["number", "bool"]).columns)
    problems = unknown_names("metric", args.metrics or [], numeric)
    problems += unknown_names("column", args.groupby or [], list(metrics_df.columns))
    if problems:
        usage_error("\n".join(problems))

    metric_cols = (
        [c for c in metrics_df.columns if c not in records_df.columns]
        if args.metrics is None
        else args.metrics
    )

    summary_df = summarize(
        metrics_df=metrics_df, group_by=args.groupby, metric_cols=metric_cols
    )

    print(summary_df.to_string())
