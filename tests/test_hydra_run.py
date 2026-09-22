from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException, HydraException
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

from geo_vlms.config import register_configs
from geo_vlms.inference import run_inference
from geo_vlms.tasks import TASKS

register_configs()


def make_cfg(*overrides: str):
    """Compose like `hydra.main` does, so `${hydra:runtime.choices...}` resolves."""
    with initialize(config_path="../conf", version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        del cfg["hydra"]
    return cfg


def test_out_path_follows_config_group_choices():
    cfg = make_cfg("dataset=dior", "backend=huggingface", "model_name=org/m")

    assert cfg.out == "results/dior/counting/org/m/huggingface/records_seed0.jsonl"


def test_backend_config_carries_model_name():
    cfg = make_cfg("backend=huggingface", "model_name=org/m")

    assert cfg.backend._target_.endswith("HuggingFaceBackend")
    assert cfg.backend.model_name == "org/m"


def test_outside_backend_and_dataset_selected_through_config(tmp_path):
    # A user's own config dir, layered on the package's via the search path
    # like `--config-dir` does. Nothing under src/ knows these classes exist.
    (tmp_path / "backend").mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "backend" / "echo.yaml").write_text(
        "_target_: fake_plugins.EchoBackend\nmodel_name: ${model_name}\nreply: '7'\n"
    )
    (tmp_path / "dataset" / "tiny.yaml").write_text(
        "_target_: fake_plugins.build_dataset\nn: 3\n"
    )
    cfg = make_cfg(
        f"hydra.searchpath=[file://{tmp_path}]",
        "backend=echo",
        "dataset=tiny",
        "model_name=fake/echo",
        f"out={tmp_path}/records.jsonl",
    )

    assert cfg.out == f"{tmp_path}/records.jsonl"
    examples = instantiate(cfg.dataset, task=TASKS[cfg.task](), seed=cfg.seed)
    backend = instantiate(cfg.backend)
    records = run_inference(examples, backend, Path(cfg.out), cfg.model_name)

    assert [r["output"] for r in records] == ["7", "7", "7"]
    assert backend.describe() == {"kind": "echo", "name": "fake/echo"}


def test_defaults_follow_dataset():
    vhr10 = make_cfg("dataset=vhr10")
    dior = make_cfg("dataset=dior")

    assert vhr10.dataset.data_dir == "data/vhr10"
    assert "split" not in vhr10.dataset
    assert (dior.dataset.data_dir, dior.dataset.split) == ("data/dior", "test")


@pytest.mark.parametrize(
    "dataset, override",
    [
        ("dior", "dataset.num_pos=1"),
        ("dior", "dataset.no_neg=true"),
        ("vhr10", "dataset.split=val"),
        ("vhr10", "dataset.num_images=1"),
    ],
)
def test_rejects_keys_of_other_dataset(dataset, override):
    with pytest.raises(ConfigCompositionException):
        make_cfg(f"dataset={dataset}", override)


@pytest.mark.parametrize(
    "override",
    ["dataset.num_ps=1", "overwrite=flase", "seed=abc"],
)
def test_rejects_bad_overrides(override):
    with pytest.raises(HydraException):
        make_cfg(override)


def test_top_logprobs_defaults_off_and_accepts_int():
    assert make_cfg().top_logprobs is None
    assert make_cfg("top_logprobs=5").top_logprobs == 5


def test_llama_server_requires_base_url():
    cfg = make_cfg("backend=llama_server")

    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
