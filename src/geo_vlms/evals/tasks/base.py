from abc import ABC, abstractmethod
from typing import Any


class Task[PredT](ABC):
    """A single evaluation task, generic over the type it parses responses into.

    ``PredT`` pins ``parse_response``'s output to ``score``'s input, so the
    runner can always feed one into the other.
    """

    @abstractmethod
    def parse_response(self, response: str) -> PredT: ...

    @abstractmethod
    def score(self, prediction: PredT, expected: Any) -> dict[str, float]: ...

    @abstractmethod
    def format_prompt(self, question: str) -> str: ...
