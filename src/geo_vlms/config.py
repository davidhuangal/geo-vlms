from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Config:
    dataset: Any
    backend: Any
    out: str = (
        "results/${hydra:runtime.choices.dataset}/${task}/${model_name}"
        "/${hydra:runtime.choices.backend}"
        "/records_seed${seed}${suffix_if:${text_only},_text_only}.jsonl"
    )
    model_name: str = "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
    task: str = "counting"
    seed: int = 0
    overwrite: bool = False
    resume: bool = False
    max_new_tokens: int = 64
    top_logprobs: int | None = None
    text_only: bool = False


@dataclass
class VHR10Config:
    _target_: str = "geo_vlms.datasets.vhr10.build_dataset"
    data_dir: str = "data/vhr10"
    num_pos: int | None = None
    num_neg: int | None = None
    no_neg: bool = False


@dataclass
class DIORConfig:
    _target_: str = "geo_vlms.datasets.dior.build_dataset"
    data_dir: str = "data/dior"
    split: str = "test"
    num_images: int | None = None
    categories: list[str] | None = None


@dataclass
class HuggingFaceConfig:
    _target_: str = "geo_vlms.backends.huggingface.HuggingFaceBackend"
    model_name: str = "${model_name}"
    device: str | None = None


@dataclass
class LlamaServerConfig:
    base_url: str
    _target_: str = "geo_vlms.backends.llama_server.LlamaServerBackend"
    model_name: str = "${model_name}"
    temperature: float = 0.0
    top_k: int = 1


def register_configs() -> None:
    OmegaConf.register_new_resolver(
        "suffix_if", lambda flag, suffix: suffix if flag else "", replace=True
    )
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="dataset", name="base_vhr10", node=VHR10Config)
    cs.store(group="dataset", name="base_dior", node=DIORConfig)
    cs.store(group="backend", name="base_huggingface", node=HuggingFaceConfig)
    cs.store(group="backend", name="base_llama_server", node=LlamaServerConfig)
