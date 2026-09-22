import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from geo_vlms.backends import Backend
from geo_vlms.config import register_configs
from geo_vlms.example import Example
from geo_vlms.inference import run_inference
from geo_vlms.provenance import collect_provenance
from geo_vlms.runs import (
    check_backend,
    check_keys,
    drop_truncated_tail,
    finished_ids,
    note_resume,
)
from geo_vlms.tasks import TASKS

register_configs()


def build_backend(cfg: DictConfig) -> Backend:
    """Construct the backend named by `cfg.backend._target_`."""
    return instantiate(cfg.backend)


def build_examples(cfg: DictConfig) -> list[Example]:
    """Call the dataset builder named by `cfg.dataset._target_`."""
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task {cfg.task}; tasks: {', '.join(TASKS)}")
    return instantiate(cfg.dataset, task=TASKS[cfg.task](), seed=cfg.seed)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
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
    choices = HydraConfig.get().runtime.choices
    print(
        f"Built {len(examples)} {choices['dataset']} {cfg.task} examples. "
        f"Using {cfg.model_name} via {choices['backend']}."
    )

    # ----- Backend -----
    backend = build_backend(cfg=cfg)

    # ----- Provenance and resume checks -----
    command = shlex.join(sys.argv)
    started_at = datetime.now(UTC).isoformat()
    provenance = collect_provenance(
        command=command,
        args=OmegaConf.to_container(cfg, resolve=True),
        started_at=started_at,
        backend=backend,
        examples=examples,
    )
    if cfg.resume:
        with open(provenance_out) as f:
            prev_meta = json.load(f)
        check_keys(
            prev_meta,
            provenance,
            ["args.model_name", "args.max_new_tokens", "dataset.sha256"],
        )
        check_backend(prev_meta, provenance)
        if drop_truncated_tail(out_path):
            print("Dropping truncated final record; its example will rerun.")
        done = finished_ids(out_path)
        examples = [e for e in examples if e.id not in done]
        provenance = note_resume(prev_meta, command=command, started_at=started_at)

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
        top_logprobs=cfg.top_logprobs,
    )
    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
