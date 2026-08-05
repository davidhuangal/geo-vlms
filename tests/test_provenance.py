import copy
import json
import random
import re
from types import SimpleNamespace

import pytest

from geo_vlms.example import Example
from geo_vlms.provenance import collect_provenance


@pytest.fixture
def dummy_model():
    """Fixture to create dummy model to avoid expensive download / load"""
    model = SimpleNamespace()
    model.config = SimpleNamespace()
    model.config._commit_hash = "66285546d2b821cf421d4f5eb2576359d3770cd3"
    model.config._attn_implementation = "eager"
    model.dtype = "torch.bfloat16"
    return model


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
        "scripts/run_vhr10.py -t counting -m Qwen/Qwen2.5-VL-3B-Instruct "
        "-o results/dummy.jsonl --data-dir data/vhr10 --num-pos 1 --num-neg 1 "
        "--seed 0 --device mps"
    )

    args = {
        "task": "counting",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
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
def provenance(sample_run, dummy_model):
    """Fixture to collect provenance for the sample run."""
    command, args, started_at, examples = sample_run
    return collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        model=dummy_model,
        examples=examples,
    )


def test_provenance_keys(provenance):
    """
    Ensure we have the expected keys. Set as exact == so that it breaks if
    new keys are added to the provenance code.
    """
    assert {"command", "args", "started_at", "model", "dataset", "env", "git"} == set(
        provenance
    )

    model_meta = provenance["model"]
    assert {"name", "commit_hash", "attn_implementation", "dtype"} == set(model_meta)

    data_meta = provenance["dataset"]
    assert {"num_examples", "sha256"} == set(data_meta)

    env_meta = provenance["env"]
    assert {
        "python",
        "geo_vlms",
        "torch",
        "transformers",
        "platform",
        "device_name",
    } == set(env_meta)

    git_meta = provenance["git"]
    if git_meta is not None:
        assert {"sha", "dirty"} == set(git_meta)


def test_provenance_model(provenance):
    model_meta = provenance["model"]

    assert model_meta["name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert model_meta["commit_hash"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert model_meta["attn_implementation"] == "eager"
    assert model_meta["dtype"] == "bfloat16"


def test_provenance_valid_data(provenance, sample_run, dummy_model):
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
        model=dummy_model,
        examples=shuffled_examples,
    )
    assert provenance_shuffled["dataset"]["sha256"] == dataset_hash

    # ----- Different number of examples should produce a different hash -----
    diff_examples = generate_examples(n=49)
    provenance_diff_examples = collect_provenance(
        command=command,
        args=args,
        started_at=started_at,
        model=dummy_model,
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
        model=dummy_model,
        examples=one_off_examples,
    )
    assert provenance_one_off_examples["dataset"]["sha256"] != dataset_hash


def test_provenance_serializable(provenance):
    json_string = json.dumps(provenance)
    assert json.loads(json_string) == provenance
