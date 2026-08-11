import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from geo_vlms.datasets.vhr10 import build_counting_dataset, build_existence_dataset
from geo_vlms.inference import run_inference
from geo_vlms.provenance import collect_provenance
from geo_vlms.runs import (
    drop_truncated_tail,
    finished_ids,
    note_resume,
    validate_resume,
)
from geo_vlms.vlm import build_model_and_processor

DATASET_BUILDERS = {
    "counting": build_counting_dataset,
    "existence": build_existence_dataset,
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse CLI for a VHR10 inference run."""
    parser = argparse.ArgumentParser(
        description="Run a VLM over a VHR10 task dataset and write records.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        choices=DATASET_BUILDERS,
        help="Target task.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name, e.g. Qwen/Qwen2.5-VL-3B-Instruct.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Path to the output records JSONL file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite an existing output file.",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from an existing output file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/vhr10",
        help="VHR10 root containing positive_image_set/, ground_truth/, "
        "and negative_image_set/.",
    )
    parser.add_argument(
        "--num-pos",
        type=int,
        default=None,
        help="Number of positive images to sample. Default: all.",
    )
    parser.add_argument(
        "--num-neg",
        type=int,
        default=None,
        help="Number of negative images to sample. Default: all.",
    )
    parser.add_argument(
        "--no-neg",
        action="store_true",
        help="Skip the negative image set entirely.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Torch device for the model. Default: auto-detect.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens the model may generate per example.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset = DATASET_BUILDERS[args.task]

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    provenance_out = out_path.with_suffix(".meta.json")
    if (not args.overwrite and not args.resume) and out_path.exists():
        raise FileExistsError(
            f"{args.out} already exists; choose a new --out path, "
            "use --resume, or pass --overwrite"
        )
    if args.resume and not out_path.exists():
        raise FileNotFoundError(
            f"{args.out} does not exist; --resume may only be used with an "
            "existing file"
        )
    if args.resume and not provenance_out.exists():
        raise FileNotFoundError(
            f"{provenance_out} does not exist; --resume needs the original "
            "run's provenance file to validate the config"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_dataset(
        pos_dir=data_dir / "positive_image_set",
        gt_dir=data_dir / "ground_truth",
        neg_dir=None if args.no_neg else data_dir / "negative_image_set",
        num_pos_images=args.num_pos,
        num_neg_images=args.num_neg,
        seed=args.seed,
    )
    print(
        f"Built {len(examples)} {args.task} examples. "
        f"Loading {args.model} on {args.device}."
    )

    # ----- Resume checks -----
    prev_meta = None
    if args.resume:
        with open(provenance_out) as f:
            prev_meta = json.load(f)
        validate_resume(prev_meta, examples, vars(args))
        if drop_truncated_tail(out_path):
            print("Dropping truncated final record; its example will rerun.")
        done = finished_ids(out_path)
        examples = [e for e in examples if e.id not in done]

    # ----- Model -----
    model, processor = build_model_and_processor(args.model, args.device)

    # ----- Handling provenance -----
    if prev_meta is not None:
        provenance = note_resume(
            prev_meta,
            command=shlex.join(sys.argv),
            started_at=datetime.now(UTC).isoformat(),
        )
    else:
        provenance = collect_provenance(
            command=shlex.join(sys.argv),
            args=vars(args),
            started_at=datetime.now(UTC).isoformat(),
            model=model,
            examples=examples,
        )
    with open(provenance_out, "w") as f:
        json.dump(provenance, f, indent=4)
    print(f"Wrote run provenance to {provenance_out}")

    # ----- Inference -----
    run_inference(
        examples=examples,
        model=model,
        processor=processor,
        out_path=out_path,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        append=args.resume,
    )
    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
