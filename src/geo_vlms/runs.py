import json
import os
from pathlib import Path
from typing import Any

_MISSING = object()


def _lookup(meta: dict[str, Any], key: str) -> Any:
    """Fetch a dotted key such as `args.model_name`; `_MISSING` when absent."""
    value: Any = meta
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def check_keys(
    prev_meta: dict[str, Any], curr_meta: dict[str, Any], keys: list[str]
) -> None:
    """
    Check that a resuming run's sidecar matches the original run's.

    Args:
        prev_meta: The original run's parsed provenance sidecar.
        curr_meta: The resuming run's provenance.
        keys: Dotted keys that must agree, e.g. `args.model_name`.

    Raises:
        ValueError: On the first key whose values differ or is missing.
    """
    for key in keys:
        old = _lookup(prev_meta, key)
        new = _lookup(curr_meta, key)
        if old is _MISSING or new is _MISSING or old != new:
            old_s = "<missing>" if old is _MISSING else repr(old)
            new_s = "<missing>" if new is _MISSING else repr(new)
            raise ValueError(f"{key}={new_s} does not match the original run's {old_s}")


def check_backend(
    prev_meta: dict[str, Any],
    curr_meta: dict[str, Any],
    ignore: tuple[str, ...] = ("base_url",),
) -> None:
    """
    Check that a resuming run's backend description matches the original's.

    Args:
        prev_meta: The original run's parsed provenance sidecar.
        curr_meta: The resuming run's provenance.
        ignore: Backend keys allowed to differ. `base_url` is ignored by
            default since the same server may be reached through a
            different tunnel.

    Raises:
        ValueError: When any other backend key differs.
    """
    prev = {k: v for k, v in prev_meta.get("backend", {}).items() if k not in ignore}
    curr = {k: v for k, v in curr_meta.get("backend", {}).items() if k not in ignore}
    if prev != curr:
        raise ValueError(
            f"Backend does not match the original run's.\nOld: {prev}\nNew: {curr}"
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


def finished_ids(out_path: str | os.PathLike, key: str = "id") -> set[str]:
    """
    Collect the ids already recorded in an output file.

    Args:
        out_path: Path to the records `.jsonl`.
        key: The record field holding the id.

    Returns:
        The recorded ids; empty when the file has no records.
    """
    lines = Path(out_path).read_text().splitlines()
    return {json.loads(line)[key] for line in lines if line.strip()}


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


class RecordWriter:
    """Write JSON records one per line, flushing each so a crash loses none."""

    def __init__(self, out_path: str | os.PathLike, append: bool = False):
        self._path = Path(out_path)
        self._mode = "a" if append else "w"
        self._file = None

    def __enter__(self) -> "RecordWriter":
        self._file = open(self._path, self._mode)
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()
        self._file = None

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
