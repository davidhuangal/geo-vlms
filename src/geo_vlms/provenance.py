import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import transformers

from geo_vlms.example import Example


def git_info() -> dict[str, Any] | None:
    """
    Collect the git state of the geo_vlms source tree.

    Returns:
        The commit SHA and a dirty flag, or None when the source is not a
        git checkout (e.g. installed as a package).
    """
    # Run git from the package's own directory, not the caller's cwd, so an
    # unrelated repo the script happens to run inside is never reported.
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=cwd
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=cwd
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return {"sha": sha, "dirty": bool(status.strip())}


def env_info(device: str) -> dict[str, Any]:
    """
    Collect the software and hardware environment of a run.

    Args:
        device: The torch device the run targets. E.g. 'cuda' or 'mps'.

    Returns:
        Library versions, platform, and device name.
    """
    device_name = torch.cuda.get_device_name() if "cuda" in device else device

    return {
        "python": platform.python_version(),
        "geo_vlms": version("geo_vlms"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device_name": device_name,
    }


def dataset_sha256(examples: list[Example]) -> str:
    """
    Fingerprint an example list, independent of ordering.

    Args:
        examples: The examples to hash.

    Returns:
        A sha256 hex digest of the sorted, serialized examples.
    """
    examples_sorted = sorted(examples, key=lambda x: x.id)
    examples_json = json.dumps([asdict(e) for e in examples_sorted], sort_keys=True)
    return hashlib.sha256(examples_json.encode()).hexdigest()


def collect_provenance(
    command: str,
    args: dict,
    started_at: str,
    model,
    examples: list[Example],
) -> dict[str, Any]:
    """
    Assemble the provenance metadata for an inference run.

    Args:
        command: The command line that launched the run.
        args: The resolved CLI arguments.
        started_at: ISO-8601 timestamp of when the run started.
        model: The loaded VLM model.
        examples: The examples the run will execute.

    Returns:
        A JSON-serializable dict describing the run.
    """
    meta = {}

    # Simple copy into meta
    meta["command"] = command
    meta["args"] = args
    meta["started_at"] = started_at

    # ----- Model Meta -----
    model_meta = {}
    model_meta["name"] = args["model"]
    model_meta["commit_hash"] = model.config._commit_hash
    model_meta["attn_implementation"] = model.config._attn_implementation
    model_meta["dtype"] = str(model.dtype).removeprefix("torch.")
    meta["model"] = model_meta

    # ----- Dataset Meta -----
    dataset_meta = {}
    dataset_meta["num_examples"] = len(examples)
    dataset_meta["sha256"] = dataset_sha256(examples)
    meta["dataset"] = dataset_meta

    # ----- Environment Meta -----
    meta["env"] = env_info(args["device"])
    meta["git"] = git_info()

    return meta
