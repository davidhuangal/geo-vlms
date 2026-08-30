import pytest
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException, HydraException
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from geo_vlms.config import register_configs

register_configs()


def make_cfg(*overrides: str):
    with initialize(config_path="../conf", version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


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


def test_llama_server_requires_base_url():
    cfg = make_cfg("backend=llama_server")

    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
