from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    dataset: Any
    backend: Any
    out: str = (
        "results/${dataset.name}/${task}/${model_name}/${backend.name}"
        "/records_seed${seed}.jsonl"
    )
    model_name: str = "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
    task: str = "counting"
    seed: int = 0
    overwrite: bool = False
    resume: bool = False
    max_new_tokens: int = 64


@dataclass
class VHR10Config:
    name: str = "vhr10"
    data_dir: str = "data/vhr10"
    num_pos: int | None = None
    num_neg: int | None = None
    no_neg: bool = False


@dataclass
class DIORConfig:
    name: str = "dior"
    data_dir: str = "data/dior"
    split: str = "test"
    num_images: int | None = None
    categories: list[str] | None = None


@dataclass
class HuggingFaceConfig:
    name: str = "huggingface"
    device: str | None = None


@dataclass
class LlamaServerConfig:
    base_url: str
    name: str = "llama_server"
    temperature: float = 0.0
    top_k: int = 1


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="dataset", name="base_vhr10", node=VHR10Config)
    cs.store(group="dataset", name="base_dior", node=DIORConfig)
    cs.store(group="backend", name="base_huggingface", node=HuggingFaceConfig)
    cs.store(group="backend", name="base_llama_server", node=LlamaServerConfig)
