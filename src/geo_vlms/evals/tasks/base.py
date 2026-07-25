from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    @abstractmethod
    def parse_response(self, response: str) -> str: ...

    @abstractmethod
    def score(self, prediction: Any, expected: Any) -> dict[str, float]: ...

    @abstractmethod
    def format_prompt(self, question: str) -> str: ...
