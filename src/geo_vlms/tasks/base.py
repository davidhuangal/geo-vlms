from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple


class Category(NamedTuple):
    """An object class as a dataset names it and as a prompt phrases it."""

    name: str  # Singular; used in example ids and metadata
    plural: str  # Used in prompts


class Task[PredT](ABC):
    # A str.format template filled with `{name}` and `{plural}`
    default_prompt: ClassVar[str]

    def __init__(self, prompt: str | None = None):
        self.prompt = self.default_prompt if prompt is None else prompt
        if "{name}" not in self.prompt and "{plural}" not in self.prompt:
            raise ValueError(f"Prompt must contain {{name}} or {{plural}}: {prompt!r}")
        try:
            self.prompt.format(name="", plural="")
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"Bad prompt template {prompt!r}: {error}") from error

    def format_prompt(self, category: Category) -> str:
        return self.prompt.format(name=category.name, plural=category.plural)

    @abstractmethod
    def parse_response(self, response: str) -> PredT | None: ...

    @abstractmethod
    def score(self, prediction: PredT | None, expected: PredT) -> dict[str, float]: ...
