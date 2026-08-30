import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

from geo_vlms.backends import Backend
from geo_vlms.datasets import dior, vhr10
from geo_vlms.example import Example
from geo_vlms.inference import run_inference
from geo_vlms.provenance import collect_provenance
from geo_vlms.runs import (
    drop_truncated_tail,
    finished_ids,
    note_resume,
    validate_resume,
)

DATASET_BUILDERS = {
    ("vhr10", "counting"): vhr10.build_counting_dataset,
    ("vhr10", "existence"): vhr10.build_existence_dataset,
    ("dior", "counting"): dior.build_counting_dataset,
    ("dior", "existence"): dior.build_existence_dataset,
}
TASKS = ("counting", "existence")
DATASETS = ("vhr10", "dior")


def build_backend(
    backend_type: str, model_name: str, device: str | None, base_url: str | None
) -> Backend:
    if backend_type == "huggingface":
        import torch

        from geo_vlms.backends.huggingface import HuggingFaceBackend

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return HuggingFaceBackend(model_name=model_name, device=device)
    if backend_type == "llama-server":
        from geo_vlms.backends.llama_server import LlamaServerBackend

        backend = LlamaServerBackend(base_url=base_url)
        alias = backend.describe().get("model_alias")
        if alias is not None and alias != model_name:
            print(
                f"Warning: --model {model_name} does not match the server's "
                f"model {alias}; records will be labeled {model_name}."
            )
        return backend
    raise ValueError(f"Unknown backend: {backend_type}")


def build_examples(args: argparse.Namespace) -> list[Example]:
    build_dataset = DATASET_BUILDERS[(args.dataset, args.task)]
    data_dir = Path(args.data_dir)
    if args.dataset == "vhr10":
        return build_dataset(
            pos_dir=data_dir / "positive_image_set",
            gt_dir=data_dir / "ground_truth",
            neg_dir=None if args.no_neg else data_dir / "negative_image_set",
            num_pos_images=args.num_pos,
            num_neg_images=args.num_neg,
            seed=args.seed,
        )
    return build_dataset(
        data_dir=data_dir,
        split=args.split,
        num_images=args.num_images,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI for an inference run."""
    parser = argparse.ArgumentParser(
        description="Run a VLM over a task dataset and write records.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        choices=TASKS,
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
        "-b",
        "--backend",
        type=str,
        required=True,
        choices=["huggingface", "llama-server"],
        help="The desired backend. Allowed: %(choices)s",
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
        default=None,
        help="Dataset root. Default: data/<dataset>",
    )
    parser.add_argument(
        "--num-pos",
        type=int,
        default=None,
        help="vhr10: number of positive images to sample. Default: all.",
    )
    parser.add_argument(
        "--num-neg",
        type=int,
        default=None,
        help="vhr10: number of negative images to sample. Default: all.",
    )
    parser.add_argument(
        "--no-neg",
        action="store_true",
        help="vhr10: skip the negative image set entirely.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=dior.SPLITS,
        help="dior: official split to evaluate. Default: test",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="dior: number of images to sample from the split. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for the huggingface backend. Default: auto-detect.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens the model may generate per example.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the llama-server, e.g. http://localhost:8080/v1",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"
    vhr10_only = args.num_pos is not None or args.num_neg is not None or args.no_neg
    dior_only = args.split is not None or args.num_images is not None
    if args.dataset != "vhr10" and vhr10_only:
        parser.error("--num-pos, --num-neg, and --no-neg only apply to --dataset vhr10")
    if args.dataset != "dior" and dior_only:
        parser.error("--split and --num-images only apply to --dataset dior")
    if args.dataset == "dior" and args.split is None:
        args.split = "test"
    if args.backend == "llama-server" and args.base_url is None:
        parser.error("--backend llama-server requires --base-url")
    if args.backend != "llama-server" and args.base_url is not None:
        parser.error("--base-url only applies to --backend llama-server")
    if args.backend != "huggingface" and args.device is not None:
        parser.error("--device only applies to --backend huggingface")

    return args


def main():
    args = parse_args()

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

    examples = build_examples(args)
    print(
        f"Built {len(examples)} {args.dataset} {args.task} examples. "
        f"Using {args.model} via {args.backend}."
    )

    # ----- Backend -----
    backend = build_backend(
        backend_type=args.backend,
        model_name=args.model,
        device=args.device,
        base_url=args.base_url,
    )

    # ----- Resume checks -----
    prev_meta = None
    if args.resume:
        with open(provenance_out) as f:
            prev_meta = json.load(f)
        validate_resume(
            prev_meta=prev_meta, examples=examples, args=vars(args), backend=backend
        )
        if drop_truncated_tail(out_path):
            print("Dropping truncated final record; its example will rerun.")
        done = finished_ids(out_path)
        examples = [e for e in examples if e.id not in done]

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
            backend=backend,
            examples=examples,
        )
    with open(provenance_out, "w") as f:
        json.dump(provenance, f, indent=4)
    print(f"Wrote run provenance to {provenance_out}")

    # ----- Inference -----
    run_inference(
        examples=examples,
        backend=backend,
        out_path=out_path,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        append=args.resume,
    )
    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
