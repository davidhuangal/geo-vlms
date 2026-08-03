import argparse
from pathlib import Path

import torch

from geo_vlms.datasets.vhr10 import build_counting_dataset
from geo_vlms.inference import run_inference
from geo_vlms.vlm import build_model_and_processor


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse CLI for a counting inference run."""
    parser = argparse.ArgumentParser(
        description="Run a VLM over the VHR10 counting dataset and write records.",
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

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_counting_dataset(
        pos_dir=data_dir / "positive_image_set",
        gt_dir=data_dir / "ground_truth",
        neg_dir=None if args.no_neg else data_dir / "negative_image_set",
        num_pos_images=args.num_pos,
        num_neg_images=args.num_neg,
        seed=args.seed,
    )
    print(f"Built {len(examples)} examples. Loading {args.model} on {args.device}.")

    model, processor = build_model_and_processor(args.model, args.device)

    run_inference(
        examples=examples,
        model=model,
        processor=processor,
        out_path=out_path,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
