import sys
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException, HydraException
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

from geo_vlms.cli import COMMANDS, build_examples, find_missing, hide_schemas, main
from geo_vlms.config import register_configs
from geo_vlms.inference import run_inference
from geo_vlms.tasks import TASKS

register_configs()


def make_cfg(*overrides: str):
    """Compose like `hydra.main` does, so `${hydra:runtime.choices...}` resolves."""
    with initialize(config_path="../src/geo_vlms/conf", version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        del cfg["hydra"]
    return cfg


def test_text_only_drops_images(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "tiny.yaml").write_text(
        "_target_: fake_plugins.build_dataset\nn: 2\n"
    )
    cfg = make_cfg(f"hydra.searchpath=[file://{tmp_path}]", "dataset=tiny")

    sighted = build_examples(cfg)
    cfg.text_only = True
    text_only = build_examples(cfg)

    assert [e.image_path for e in sighted] == ["/0.jpg", "/1.jpg"]
    assert all(e.image_path is None for e in text_only)
    assert [(e.id, e.prompt, e.expected) for e in text_only] == [
        (e.id, e.prompt, e.expected) for e in sighted
    ]


def test_prompt_override_reaches_examples(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "tiny.yaml").write_text(
        "_target_: fake_plugins.build_dataset\nn: 1\n"
    )
    cfg = make_cfg(
        f"hydra.searchpath=[file://{tmp_path}]",
        "dataset=tiny",
        "prompt='How many {plural}? Answer 0 if there are none.'",
    )

    assert (
        build_examples(cfg)[0].prompt == "How many ships? Answer 0 if there are none."
    )


def test_prompt_defaults_to_task_prompt():
    assert make_cfg().prompt is None


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
    dota = make_cfg("dataset=dota")

    assert vhr10.dataset.data_dir == "data/vhr10"
    assert "split" not in vhr10.dataset
    assert (dior.dataset.data_dir, dior.dataset.split) == ("data/dior", "test")
    assert (dota.dataset.data_dir, dota.dataset.split) == ("data/dota", "val")


@pytest.mark.parametrize(
    "dataset, override",
    [
        ("dior", "dataset.num_pos=1"),
        ("dior", "dataset.no_neg=true"),
        ("vhr10", "dataset.split=val"),
        ("vhr10", "dataset.num_images=1"),
        ("dota", "dataset.num_images=1"),
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


@pytest.mark.parametrize(
    "overrides, missing",
    [
        (["backend=huggingface"], ["out", "model_name"]),
        (["backend=llama_server"], ["backend.base_url", "out", "model_name"]),
        (["backend=huggingface", "model_name=org/m", "out=r.jsonl"], []),
    ],
)
def test_find_missing(overrides, missing):
    assert find_missing(make_cfg(*overrides)) == missing


def test_bare_run_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geo-vlms"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "backend: huggingface, llama_server\n" in out
    assert "dataset: dior, dota, vhr10\n" in out
    assert "base_" not in out


def test_hide_schemas_keeps_unpaired_base():
    text = "backend: base_hf, base_mine, hf\nPick: one\n"

    assert hide_schemas(text) == "backend: base_mine, hf\nPick: one\n"


def test_missing_values_exit_with_message(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geo-vlms", "backend=llama_server"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert capsys.readouterr().err == (
        "Some required arguments are missing: backend.base_url, out, model_name\n"
    )


def test_empty_dataset_exits_before_backend(tmp_path, monkeypatch, capsys):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "empty.yaml").write_text(
        "_target_: fake_plugins.build_dataset\nn: 0\n"
    )
    out = tmp_path / "results" / "records.jsonl"
    argv = ["geo-vlms", f"--config-dir={tmp_path}", "dataset=empty"]
    argv += ["backend=llama_server", "backend.base_url=http://unreachable"]
    monkeypatch.setattr(sys, "argv", [*argv, "model_name=org/m", f"out={out}"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == (
        "Dataset creation resulted in 0 counting examples from empty. "
        "Check that the dataset is available and configured correctly.\n"
    )
    assert not out.parent.exists()


def test_help_lists_every_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geo-vlms"])

    with pytest.raises(SystemExit):
        main()

    out = capsys.readouterr().out
    assert all(f"\n  {name}  " in out for name in COMMANDS)


@pytest.mark.parametrize("name", ["analyze", "merge-shards", "prepare-dior"])
def test_bare_command_prints_its_help(name, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geo-vlms", name])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: geo-vlms {name} ")


def test_unknown_command_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geo-vlms", "analyse"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.startswith("Unknown command 'analyse'. Commands: ")
