import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from geo_vlms.example import Example
from geo_vlms.shards import (
    check_shard,
    image_groups,
    merge_order,
    shard_path,
    take_shard,
)

TESTS = Path(__file__).parent
NUM_SHARDS = 3


def make_examples(paths: list[str | None]) -> list[Example]:
    return [Example(id=f"{i}", image_path=p, prompt="q") for i, p in enumerate(paths)]


def test_image_groups_follow_first_appearance():
    assert image_groups(["/a", "/a", "/b", None, None, "/c"]) == [
        [0, 1],
        [2],
        [3],
        [4],
        [5],
    ]


def test_take_shard_keeps_images_together_round_robin():
    paths = [f"/{i}.jpg" for i in range(7) for _ in range(3)]
    examples = make_examples(paths)

    shards = [take_shard(examples, i, 3) for i in range(3)]

    images = [sorted({e.image_path for e in shard}) for shard in shards]
    assert images == [
        ["/0.jpg", "/3.jpg", "/6.jpg"],
        ["/1.jpg", "/4.jpg"],
        ["/2.jpg", "/5.jpg"],
    ]
    assert [len(shard) for shard in shards] == [9, 6, 6]


def test_take_shard_splits_text_only_by_example():
    examples = make_examples([None] * 5)

    shards = [take_shard(examples, i, 2) for i in range(2)]

    assert [[e.id for e in shard] for shard in shards] == [["0", "2", "4"], ["1", "3"]]


def test_take_shard_rejects_scattered_image():
    examples = make_examples(["/a", "/b", "/a"])

    with pytest.raises(ValueError, match="not consecutive"):
        take_shard(examples, 0, 2)


@pytest.mark.parametrize("paths", [["/a", "/a", "/b", "/c", "/c", "/c"], [None] * 4])
@pytest.mark.parametrize("num_shards", [1, 2, 3, 5])
def test_merge_order_inverts_take_shard(paths, num_shards):
    examples = make_examples(paths)
    shards = [
        [asdict(e) for e in take_shard(examples, i, num_shards)]
        for i in range(num_shards)
    ]

    assert merge_order(shards) == [asdict(e) for e in examples]


@pytest.mark.parametrize("shard, num_shards", [(0, 0), (-1, 2), (2, 2), (1, 1)])
def test_check_shard_rejects_out_of_range(shard, num_shards):
    with pytest.raises(ValueError):
        check_shard(shard, num_shards)


def test_shard_path_format():
    assert shard_path("results/x/records_seed0.jsonl", 1, 4) == Path(
        "results/x/records_seed0.shards/shard1of4.jsonl"
    )


def run_cli(conf_dir: Path, *overrides: str) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "geo_vlms.cli",
        f"--config-dir={conf_dir}",
        "backend=echo",
        "dataset=grouped",
        "model_name=fake/echo",
        *overrides,
    ]
    env = os.environ | {"PYTHONPATH": str(TESTS)}
    return subprocess.run(command, env=env, capture_output=True, text=True)


def merge(out: Path, *flags: str) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "geo_vlms.cli", "merge-shards", "--out", str(out)]
    command += flags
    return subprocess.run(command, capture_output=True, text=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_meta(path: Path) -> dict:
    return json.loads(path.with_suffix(".meta.json").read_text())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One unsharded run and the same run as NUM_SHARDS shards."""
    root = tmp_path_factory.mktemp("runs")
    (root / "conf" / "backend").mkdir(parents=True)
    (root / "conf" / "dataset").mkdir()
    (root / "conf" / "backend" / "echo.yaml").write_text(
        "_target_: fake_plugins.EchoBackend\nmodel_name: ${model_name}\nreply: '7'\n"
    )
    (root / "conf" / "dataset" / "grouped.yaml").write_text(
        "_target_: fake_plugins.build_grouped\nimages: 7\n"
    )

    unsharded = root / "unsharded" / "records.jsonl"
    run_cli(root / "conf", f"out={unsharded}").check_returncode()

    sharded = root / "sharded" / "records.jsonl"
    for i in range(NUM_SHARDS):
        run_cli(
            root / "conf", f"out={sharded}", f"shard={i}", f"num_shards={NUM_SHARDS}"
        ).check_returncode()

    return root / "conf", unsharded, sharded


@pytest.fixture
def out(runs, tmp_path):
    """A fresh copy of the shards, merging into the returned path."""
    _, _, sharded = runs
    shutil.copytree(sharded.with_suffix(".shards"), tmp_path / "records.shards")
    return tmp_path / "records.jsonl"


def test_unsharded_run_keeps_out_path(runs):
    _, unsharded, _ = runs

    assert unsharded.exists()
    assert not unsharded.with_suffix(".shards").exists()


def test_shards_write_beside_out(runs):
    _, _, sharded = runs
    shard_dir = sharded.with_suffix(".shards")

    assert not sharded.exists()
    assert sorted(p.name for p in shard_dir.iterdir()) == [
        f"shard{i}of3.{ext}" for i in range(3) for ext in ("jsonl", "meta.json")
    ]
    meta = read_meta(shard_dir / "shard1of3.jsonl")
    assert (meta["args"]["shard"], meta["args"]["num_shards"]) == (1, 3)
    assert meta["dataset"]["num_examples"] == 4


def test_merge_matches_unsharded(runs, out):
    _, unsharded, _ = runs

    result = merge(out)

    assert result.returncode == 0, result.stderr
    assert read_jsonl(out) == read_jsonl(unsharded)
    merged, base = read_meta(out), read_meta(unsharded)
    assert merged["dataset"] == base["dataset"]
    assert merged["args"] == base["args"] | {"out": str(out)}
    assert [s["num_records"] for s in merged["shards"]] == [6, 4, 4]
    assert "resumes" not in merged


def test_merge_leaves_shards(out):
    merge(out).check_returncode()

    assert len(list(out.with_suffix(".shards").glob("*.jsonl"))) == NUM_SHARDS
    assert not list(out.parent.glob("*.tmp"))


def test_merge_refuses_missing_shard(out):
    (out.with_suffix(".shards") / "shard1of3.jsonl").unlink()

    result = merge(out)

    assert result.returncode != 0
    assert "missing shards [1]" in result.stderr
    assert not out.exists()


def test_merge_refuses_mixed_shard_counts(out):
    shard_dir = out.with_suffix(".shards")
    shutil.copy(shard_dir / "shard0of3.jsonl", shard_dir / "shard0of2.jsonl")

    result = merge(out)

    assert result.returncode != 0
    assert "mixes shard counts [2, 3]" in result.stderr


def test_merge_refuses_short_shard(out):
    path = out.with_suffix(".shards") / "shard2of3.jsonl"
    path.write_text("".join(f"{line}\n" for line in path.read_text().splitlines()[:-1]))

    result = merge(out)

    assert result.returncode != 0
    assert "has 3 of 4 records" in result.stderr


def test_merge_refuses_truncated_line(out):
    path = out.with_suffix(".shards") / "shard2of3.jsonl"
    path.write_text(path.read_text()[:-20])

    result = merge(out)

    assert result.returncode != 0
    assert "line 4 is truncated" in result.stderr


def test_merge_refuses_config_mismatch(out):
    meta_path = out.with_suffix(".shards") / "shard1of3.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["args"]["max_new_tokens"] = 8
    meta_path.write_text(json.dumps(meta))

    result = merge(out)

    assert result.returncode != 0
    assert "config does not match" in result.stderr


def test_merge_refuses_duplicate_id(out):
    shard_dir = out.with_suffix(".shards")
    first = (shard_dir / "shard0of3.jsonl").read_text().splitlines()[0]
    path = shard_dir / "shard1of3.jsonl"
    path.write_text(path.read_text() + first + "\n")
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["dataset"]["num_examples"] += 1
    meta_path.write_text(json.dumps(meta))

    result = merge(out)

    assert result.returncode != 0
    assert "more than once: ['0:ship']" in result.stderr


def test_merge_refuses_existing_out(runs, out):
    _, unsharded, _ = runs
    out.write_text("old\n")

    refused = merge(out)
    replaced = merge(out, "--overwrite")

    assert refused.returncode != 0
    assert "already exists" in refused.stderr
    assert replaced.returncode == 0, replaced.stderr
    assert read_jsonl(out) == read_jsonl(unsharded)


def test_resume_finishes_one_shard(runs, out):
    conf, unsharded, _ = runs
    path = out.with_suffix(".shards") / "shard1of3.jsonl"
    full = path.read_text()
    path.write_text(full.splitlines()[0] + "\n" + full.splitlines()[1][:10])

    result = run_cli(conf, f"out={out}", "shard=1", "num_shards=3", "resume=true")

    assert result.returncode == 0, result.stderr
    assert path.read_text() == full
    assert len(read_meta(path)["resumes"]) == 1
    merge(out).check_returncode()
    assert read_jsonl(out) == read_jsonl(unsharded)


def test_empty_shard_warns_and_merges(runs, tmp_path):
    conf, _, _ = runs
    out = tmp_path / "records.jsonl"

    results = [
        run_cli(conf, f"out={out}", "dataset.images=2", f"shard={i}", "num_shards=3")
        for i in range(3)
    ]

    assert [r.returncode for r in results] == [0, 0, 0]
    assert "Warning: shard 2 of 3 is empty; num_shards exceeds the 2 images." in (
        results[2].stderr
    )
    assert "Warning" not in results[0].stderr
    merge(out).check_returncode()
    assert len(read_jsonl(out)) == 4


def test_cli_rejects_bad_shard(runs, tmp_path):
    conf, _, _ = runs

    result = run_cli(conf, f"out={tmp_path}/records.jsonl", "shard=3", "num_shards=3")

    assert result.returncode != 0
    assert "shard=3 must be in [0, 3)" in result.stderr
    assert not list(tmp_path.iterdir())


def test_merge_refuses_edited_record(out):
    path = out.with_suffix(".shards") / "shard0of3.jsonl"
    records = read_jsonl(path)
    records[0]["expected"] = 99
    path.write_text("".join(json.dumps(r) + "\n" for r in records))

    result = merge(out)

    assert result.returncode != 0
    assert "do not match its dataset.sha256" in result.stderr
