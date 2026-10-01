import json
import os
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

from geo_vlms.example import Example
from geo_vlms.provenance import dataset_sha256
from geo_vlms.runs import check_backend

_SHARD_NAME = re.compile(r"shard(\d+)of(\d+)\.jsonl")


def check_shard(shard: int, num_shards: int) -> None:
    """Raise ValueError unless `shard` is one of `num_shards` (at least 1)."""
    if num_shards < 1:
        raise ValueError(f"num_shards={num_shards} must be at least 1")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard={shard} must be in [0, {num_shards})")


def shard_path(out: str | os.PathLike, shard: int, num_shards: int) -> Path:
    """Where a shard writes: `<out stem>.shards/shard<shard>of<num_shards>.jsonl`."""
    shard_dir = Path(out).with_suffix(".shards")
    return shard_dir / f"shard{shard}of{num_shards}.jsonl"


def image_groups(paths: list[str | None]) -> list[list[int]]:
    """
    Group positions by image path, in order of first appearance.

    Args:
        paths: One image path per example; None (text_only) is its own group.

    Returns:
        The positions of each group.
    """
    groups: list[list[int]] = []
    by_path: dict[str, list[int]] = {}
    for i, path in enumerate(paths):
        if path is None:
            groups.append([i])
        elif path in by_path:
            by_path[path].append(i)
        else:
            by_path[path] = [i]
            groups.append(by_path[path])

    return groups


def take_shard(examples: list[Example], shard: int, num_shards: int) -> list[Example]:
    """
    Select one shard's examples: image k goes to shard k % num_shards.

    Keeping an image's questions together lets llama-server reuse the image
    across them, and lets `merge_shards` restore the unsharded order.

    Raises:
        ValueError: When one image's questions are not consecutive.
    """
    groups = image_groups([e.image_path for e in examples])
    for group in groups:
        if group[-1] - group[0] + 1 != len(group):
            raise ValueError(
                f"Questions about {examples[group[0]].image_path} are not "
                "consecutive; sharding needs each image's questions together"
            )

    return [examples[i] for group in groups[shard::num_shards] for i in group]


def find_shards(out: str | os.PathLike) -> list[Path]:
    """
    List the shard files of a sharded run, in shard order.

    Raises:
        FileNotFoundError: When the shard folder or any shard is missing.
        ValueError: When the files disagree on the number of shards.
    """
    shard_dir = Path(out).with_suffix(".shards")
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"{shard_dir} does not exist")

    found = []
    for path in shard_dir.glob("shard*of*.jsonl"):
        match = _SHARD_NAME.fullmatch(path.name)
        if match:
            found.append((int(match[1]), int(match[2])))
    totals = {total for _, total in found}
    if not totals:
        raise FileNotFoundError(f"{shard_dir} has no shard files")
    if len(totals) > 1:
        raise ValueError(f"{shard_dir} mixes shard counts {sorted(totals)}")

    (num_shards,) = totals
    shards = {shard for shard, _ in found}
    missing = sorted(set(range(num_shards)) - shards)
    if missing:
        raise FileNotFoundError(f"{shard_dir} is missing shards {missing}")
    extra = sorted(shards - set(range(num_shards)))
    if extra:
        raise ValueError(f"{shard_dir} has shards {extra} out of range")

    return [shard_path(out, i, num_shards) for i in range(num_shards)]


def read_shard(path: Path) -> tuple[list[dict], dict[str, Any]]:
    """
    Read a shard's records and provenance sidecar.

    Raises:
        ValueError: When a line is not valid JSON, e.g. cut off by a kill.
    """
    records = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(
                f"{path} line {n} is truncated; finish the shard with resume=true"
            ) from None

    with open(path.with_suffix(".meta.json")) as f:
        meta = json.load(f)

    return records, meta


def to_example(record: dict) -> Example:
    """The Example a record was made from."""
    return Example(**{f.name: record[f.name] for f in fields(Example)})


def _run_args(meta: dict[str, Any]) -> dict[str, Any]:
    """Config args that must agree across shards."""
    args = {k: v for k, v in meta["args"].items() if k not in ("shard", "out")}
    # Each array task starts its own server, so the port differs.
    args["backend"] = {k: v for k, v in args["backend"].items() if k != "base_url"}
    return args


def check_shards(paths: list[Path], shards: list[tuple[list, dict]]) -> None:
    """
    Check that shards are complete, distinct parts of one run.

    Raises:
        ValueError: On the first problem found.
    """
    num_shards = len(shards)
    first_meta = shards[0][1]
    for i, (path, (records, meta)) in enumerate(zip(paths, shards, strict=True)):
        if (meta["args"]["shard"], meta["args"]["num_shards"]) != (i, num_shards):
            raise ValueError(
                f"{path} provenance says shard {meta['args']['shard']} "
                f"of {meta['args']['num_shards']}"
            )
        if _run_args(meta) != _run_args(first_meta):
            raise ValueError(f"{path} config does not match {paths[0]}")
        check_backend(first_meta, meta)

        expected = meta["dataset"]["num_examples"]
        if len(records) != expected:
            raise ValueError(
                f"{path} has {len(records)} of {expected} records; "
                "finish the shard with resume=true"
            )

    counts = Counter(r["id"] for records, _ in shards for r in records)
    repeated = sorted(id_ for id_, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"Example ids appear more than once: {repeated[:5]}")

    for path, (records, meta) in zip(paths, shards, strict=True):
        examples = [to_example(r) for r in records]
        if dataset_sha256(examples) != meta["dataset"]["sha256"]:
            raise ValueError(f"{path} records do not match its dataset.sha256")


def merge_order(shards: list[list[dict]]) -> list[dict]:
    """
    Interleave shard records back into the unsharded order.

    Inverts `take_shard`: image k is the (k // n)-th image of shard k % n.

    Raises:
        ValueError: When the shards' image counts can't come from one run.
    """
    num_shards = len(shards)
    groups = [
        [
            [records[i] for i in group]
            for group in image_groups([r["image_path"] for r in records])
        ]
        for records in shards
    ]
    total = sum(len(g) for g in groups)
    for i, shard_groups in enumerate(groups):
        expected = len(range(i, total, num_shards))
        if len(shard_groups) != expected:
            raise ValueError(
                f"Shard {i} has {len(shard_groups)} images, expected {expected}"
            )

    return [
        record
        for k in range(total)
        for record in groups[k % num_shards][k // num_shards]
    ]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def merge_shards(out: str | os.PathLike, command: str, overwrite: bool = False) -> int:
    """
    Merge a sharded run's records and sidecars into `out`.

    Args:
        out: The merged records path; shards are read from `<out stem>.shards/`.
        command: The merge command line, recorded in the merged sidecar.
        overwrite: Whether to replace an existing `out`.

    Returns:
        The number of merged records.
    """
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass --overwrite")

    paths = find_shards(out)
    shards = [read_shard(path) for path in paths]
    check_shards(paths, shards)
    records = merge_order([records for records, _ in shards])

    metas = [meta for _, meta in shards]
    merged = dict(metas[0])
    merged.pop("resumes", None)
    merged["command"] = command
    merged["started_at"] = min(meta["started_at"] for meta in metas)
    merged["args"] = metas[0]["args"] | {"out": str(out), "shard": 0, "num_shards": 1}
    merged["dataset"] = {
        "num_examples": len(records),
        "sha256": dataset_sha256([to_example(r) for r in records]),
    }
    merged["shards"] = [
        {
            "path": str(path),
            "command": meta["command"],
            "started_at": meta["started_at"],
            "resumes": meta.get("resumes", []),
            "num_records": meta["dataset"]["num_examples"],
            "git": meta["git"],
        }
        for path, meta in zip(paths, metas, strict=True)
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(out, "".join(json.dumps(r) + "\n" for r in records))
    _write_atomic(out.with_suffix(".meta.json"), json.dumps(merged, indent=4))

    return len(records)
