import json
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from geo_vlms.backends import Backend
from geo_vlms.config import register_configs
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

register_configs()

DATASET_BUILDERS = {
    ("vhr10", "counting"): vhr10.build_counting_dataset,
    ("vhr10", "existence"): vhr10.build_existence_dataset,
    ("dior", "counting"): dior.build_counting_dataset,
    ("dior", "existence"): dior.build_existence_dataset,
}


def build_backend(
    cfg: DictConfig,
) -> Backend:
    if cfg.backend.name == "huggingface":
        import torch

        from geo_vlms.backends.huggingface import HuggingFaceBackend

        device = cfg.backend.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return HuggingFaceBackend(model_name=cfg.model_name, device=device)
    if cfg.backend.name == "llama_server":
        from geo_vlms.backends.llama_server import LlamaServerBackend

        backend = LlamaServerBackend(
            base_url=cfg.backend.base_url,
            api_key=os.environ.get("GEO_VLMS_LLAMA_API_KEY", "unused"),
            temperature=cfg.backend.temperature,
            top_k=cfg.backend.top_k,
        )
        alias = backend.describe().get("model_alias")
        if alias is not None and alias != cfg.model_name:
            print(
                f"Warning: --model {cfg.model_name} does not match the server's "
                f"model {alias}; records will be labeled {cfg.model_name}."
            )
        return backend
    raise ValueError(f"Unknown backend: {cfg.backend.name}")


def build_examples(cfg: DictConfig) -> list[Example]:
    build_dataset = DATASET_BUILDERS[(cfg.dataset.name, cfg.task)]
    data_dir = Path(cfg.dataset.data_dir)

    if cfg.dataset.name == "vhr10":
        return build_dataset(
            pos_dir=data_dir / "positive_image_set",
            gt_dir=data_dir / "ground_truth",
            neg_dir=None if cfg.dataset.no_neg else data_dir / "negative_image_set",
            num_pos_images=cfg.dataset.num_pos,
            num_neg_images=cfg.dataset.num_neg,
            seed=cfg.seed,
        )

    return build_dataset(
        data_dir=data_dir,
        split=cfg.dataset.split,
        num_images=cfg.dataset.num_images,
        categories=cfg.dataset.categories,
        seed=cfg.seed,
    )


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if (cfg.dataset.name, cfg.task) not in DATASET_BUILDERS:
        raise ValueError(
            f"No builder for dataset={cfg.dataset.name}, task={cfg.task}; "
            f"tasks: counting, existence"
        )

    out_path = Path(cfg.out)
    provenance_out = out_path.with_suffix(".meta.json")

    if cfg.overwrite and cfg.resume:
        raise ValueError("Only one of overwrite or resume can be true.")

    if (not cfg.overwrite and not cfg.resume) and out_path.exists():
        raise FileExistsError(
            f"{cfg.out} already exists; choose a new --out path, "
            "use --resume, or pass --overwrite"
        )
    if cfg.resume and not out_path.exists():
        raise FileNotFoundError(
            f"{cfg.out} does not exist; --resume may only be used with an existing file"
        )
    if cfg.resume and not provenance_out.exists():
        raise FileNotFoundError(
            f"{provenance_out} does not exist; --resume needs the original "
            "run's provenance file to validate the config"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Dataset Creation -----
    examples = build_examples(cfg=cfg)
    print(
        f"Built {len(examples)} {cfg.dataset.name} {cfg.task} examples. "
        f"Using {cfg.model_name} via {cfg.backend.name}."
    )

    # ----- Backend -----
    backend = build_backend(cfg=cfg)

    # ----- Resume checks -----
    prev_meta = None
    if cfg.resume:
        with open(provenance_out) as f:
            prev_meta = json.load(f)
        validate_resume(
            prev_meta=prev_meta,
            examples=examples,
            args=OmegaConf.to_container(cfg, resolve=True),
            backend=backend,
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
            args=OmegaConf.to_container(cfg, resolve=True),
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
        model_name=cfg.model_name,
        max_new_tokens=cfg.max_new_tokens,
        append=cfg.resume,
    )
    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
