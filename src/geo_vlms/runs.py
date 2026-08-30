import json
import os
from pathlib import Path
from typing import Any

from geo_vlms.analysis import load_records
from geo_vlms.backends import Backend
from geo_vlms.example import Example
from geo_vlms.provenance import dataset_sha256


def validate_resume(
    prev_meta: dict[str, Any],
    examples: list[Example],
    args: dict[str, Any],
    backend: Backend,
) -> None:
    """
    Check that a resuming run's config matches the original run's sidecar.

    Args:
        prev_meta: The original run's parsed provenance sidecar.
        examples: The full example list built from the current args.
        args: The resolved config of the resuming run.
        backend: The backend being used for this resumed run.

    Raises:
        ValueError: When the dataset or a generation-affecting arg differs.
    """
    if dataset_sha256(examples) != prev_meta["dataset"]["sha256"]:
        raise ValueError(
            "Dataset does not match the original run; resume with the "
            "same task, data_dir, seed, and sampling options"
        )
    for key in ("model_name", "max_new_tokens"):
        if args[key] != prev_meta["args"][key]:
            raise ValueError(
                f"{key}={args[key]} does not "
                f"match the original run's {prev_meta['args'][key]}"
            )

    # base_url is ignored since the same server may be reached through a
    # different tunnel
    prev_backend = {
        k: v for k, v in prev_meta.get("backend", {}).items() if k != "base_url"
    }
    curr_backend = {k: v for k, v in backend.describe().items() if k != "base_url"}
    if prev_backend != curr_backend:
        raise ValueError(
            "Backend does not match the original run's.\n"
            f"Old: {prev_backend}\nNew: {curr_backend}"
        )


def drop_truncated_tail(out_path: str | os.PathLike) -> bool:
    """
    Repair a records file tail left by a hard kill mid-write.

    Args:
        out_path: Path to the records `.jsonl`.

    Returns:
        Whether a truncated final record was found and removed.
    """
    out_path = Path(out_path)
    text = out_path.read_text()
    lines = text.splitlines()
    if not lines:
        return False

    try:
        json.loads(lines[-1])
    except json.JSONDecodeError:
        kept = lines[:-1]
        out_path.write_text("\n".join(kept) + "\n" if kept else "")
        return True

    # A kill can also land between a record's JSON and its newline; restore
    # the newline so appended records start on a fresh line.
    if not text.endswith("\n"):
        with open(out_path, "a") as f:
            f.write("\n")

    return False


def finished_ids(out_path: str | os.PathLike) -> set[str]:
    """
    Collect the example ids already recorded in an output file.

    Args:
        out_path: Path to the records `.jsonl`.

    Returns:
        The recorded example ids; empty when the file has no records.
    """
    records = load_records(out_path)
    if len(records) == 0:
        return set()
    return set(records.id.unique())


def note_resume(
    prev_meta: dict[str, Any], command: str, started_at: str
) -> dict[str, Any]:
    """
    Record a resume invocation in an existing provenance sidecar.

    Args:
        prev_meta: The original run's parsed provenance sidecar.
        command: The command line that launched the resume.
        started_at: ISO-8601 timestamp of when the resume started.

    Returns:
        The sidecar dict with the invocation appended to its `resumes` list.
    """
    prev_meta.setdefault("resumes", []).append(
        {"command": command, "started_at": started_at}
    )
    return prev_meta
