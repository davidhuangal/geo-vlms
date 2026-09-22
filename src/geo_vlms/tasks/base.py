from abc import ABC, abstractmethod
from typing import NamedTuple


class Category(NamedTuple):
    """An object class as a dataset names it and as a prompt phrases it."""

    name: str  # Singular; used in example ids and metadata
    plural: str  # Used in prompts


class Task[PredT](ABC):
    @abstractmethod
    def parse_response(self, response: str) -> PredT | None: ...

    @abstractmethod
    def score(self, prediction: PredT | None, expected: PredT) -> dict[str, float]: ...

    @abstractmethod
    def format_prompt(self, category: Category) -> str: ...
