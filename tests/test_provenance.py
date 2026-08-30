import copy
import json
import random
import re
import subprocess

import pytest

from geo_vlms.example import Example
from geo_vlms.provenance import collect_provenance, git_info

STUB_DESCRIPTION = {
    "kind": "stub",
    "name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "commit_hash": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "dtype": "bfloat16",
}


class StubBackend:
    """Canned backend to avoid an expensive model load."""

    def generate(self, prompt, image_paths, max_new_tokens):
        return "canned response"

    def describe(self):
        return dict(STUB_DESCRIPTION)


def generate_examples(n: int) -> list[Example]:
    """Generate n number of random samples."""
    rng = random.Random(0)

    low = 1
    high = 1000
    id_ints = rng.sample(range(low, high + 1), n)

    return [
        Example(
            id=f"{id_int}",
            image_path=f"/{id_int}.jpeg",
            prompt=f"prompt for example id {id_int}",
            metadata={"category": "ship", "split": "positive"},
        )
        for id_int in id_ints
    ]


@pytest.fixture
def sample_run():
    """Fixture to create a dummy inference run."""
    command = (
        "scripts/run.py -d vhr10 -t counting -m Qwen/Qwen2.5-VL-3B-Instruct "
        "-b huggingface -o results/dummy.jsonl --data-dir data/vhr10 "
        "--num-pos 1 --num-neg 1 --seed 0 --device mps"
    )

    args = {
        "task": "counting",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "backend": "huggingface",
        "base_url": None,
        "out": "results/dummy.jsonl",
        "data_dir": "data/vhr10",
        "num_pos": 1,
        "num_neg": 1,
        "no_neg": False,
        "seed": 0,
        "device": "mps",
        "max_new_tokens": 64,
    }

    started_at = "2026-08-04T18:43:46.279457+00:00"

    examples = generate_examples(n=50)

    return command, args, started_at, examples


@pytest.fixture
def provenance(sample_run):
    """Fixture to collect provenance for the sample run."""
    command, args, started_at, examples = sample_run
    return collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        backend=StubBackend(),
        examples=examples,
    )


def test_provenance_keys(provenance):
    """
    Ensure we have the expected keys. Set as exact == so that it breaks if
    new keys are added to the provenance code.
    """
    assert {"command", "args", "started_at", "backend", "dataset", "env", "git"} == set(
        provenance
    )

    data_meta = provenance["dataset"]
    assert {"num_examples", "sha256"} == set(data_meta)

    env_meta = provenance["env"]
    assert {"python", "geo_vlms", "platform"} == set(env_meta)

    git_meta = provenance["git"]
    if git_meta is not None:
        assert {"sha", "dirty"} == set(git_meta)


def test_provenance_backend(provenance):
    # The backend section is the backend's self-description, verbatim
    assert provenance["backend"] == STUB_DESCRIPTION


def test_provenance_valid_data(provenance, sample_run):
    command, args, started_at, examples = sample_run
    dataset_hash = provenance["dataset"]["sha256"]
    assert re.fullmatch(r"[a-f0-9]{64}", dataset_hash)

    # ----- Shuffling the examples should have no effect on the output -----
    rng = random.Random(0)
    shuffled_examples = rng.sample(examples, len(examples))
    provenance_shuffled = collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        backend=StubBackend(),
        examples=shuffled_examples,
    )
    assert provenance_shuffled["dataset"]["sha256"] == dataset_hash

    # ----- Different number of examples should produce a different hash -----
    diff_examples = generate_examples(n=49)
    provenance_diff_examples = collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        backend=StubBackend(),
        examples=diff_examples,
    )
    assert provenance_diff_examples["dataset"]["sha256"] != dataset_hash

    # ----- Even one differing prompt should produce a different hash -----
    one_off_examples = copy.deepcopy(examples)
    one_off_examples[0].prompt = "I'm different!"
    provenance_one_off_examples = collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        backend=StubBackend(),
        examples=one_off_examples,
    )
    assert provenance_one_off_examples["dataset"]["sha256"] != dataset_hash


def test_provenance_serializable(provenance):
    json_string = json.dumps(provenance)
    assert json.loads(json_string) == provenance


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), subprocess.CalledProcessError(128, "git")],
    ids=["git-missing", "not-a-repo"],
)
def test_git_info_unavailable(monkeypatch, error):
    """git_info falls back to None when git is absent or the tree isn't a repo."""

    def fake_check_output(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "geo_vlms.provenance.subprocess.check_output", fake_check_output
    )
    assert git_info() is None
