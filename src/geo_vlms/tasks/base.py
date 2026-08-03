from abc import ABC, abstractmethod


class Task[PredT](ABC):
    @abstractmethod
    def parse_response(self, response: str) -> PredT | None: ...

    @abstractmethod
    def score(self, prediction: PredT | None, expected: PredT) -> dict[str, float]: ...
