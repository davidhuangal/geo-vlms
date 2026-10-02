import argparse
import shlex
import sys

from geo_vlms.shards import merge_shards


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI for merging."""
    parser = argparse.ArgumentParser(
        prog="geo-vlms merge-shards",
        description="Merge the shards of a sharded run into one records file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Merged records path; shards are read from <out stem>.shards/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing merged file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]):
    args = parse_args(argv)
    count = merge_shards(args.out, shlex.join(sys.argv), overwrite=args.overwrite)
    print(f"Wrote {count} records to {args.out}")
