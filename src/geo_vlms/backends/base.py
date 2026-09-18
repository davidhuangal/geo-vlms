from dataclasses import dataclass
from typing import Protocol


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top: dict[str, float]


@dataclass
class Generation:
    text: str
    tokens: list[TokenLogprob] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None


class Backend(Protocol):
    def generate(
        self,
        prompt: str,
        images: list[str | bytes] | None,
        max_new_tokens: int,
        top_logprobs: int | None = None,
    ) -> Generation: ...
    def describe(self) -> dict: ...
