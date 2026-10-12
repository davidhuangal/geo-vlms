from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple

import pandas as pd


class Category(NamedTuple):
    """An object class as a dataset names it and as a prompt phrases it."""

    name: str  # Singular; used in example ids and metadata
    plural: str  # Used in prompts


class Task[PredT](ABC):
    # A str.format template filled with `{name}` and `{plural}`
    default_prompt: ClassVar[str]
    default_answer_tokens: ClassVar[dict[str, list[str]]] = {}

    def __init__(
        self,
        prompt: str | None = None,
        answer_tokens: dict[str, list[str]] | None = None,
    ):
        self.prompt = self.default_prompt if prompt is None else prompt
        if "{name}" not in self.prompt and "{plural}" not in self.prompt:
            raise ValueError(f"Prompt must contain {{name}} or {{plural}}: {prompt!r}")
        try:
            self.prompt.format(name="", plural="")
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"Bad prompt template {prompt!r}: {error}") from error

        overrides = answer_tokens or {}
        unknown = set(overrides) - set(self.default_answer_tokens)
        if unknown:
            raise ValueError(
                f"Unknown answer classes {sorted(unknown)}. "
                f"Classes: {sorted(self.default_answer_tokens)}"
            )

        merged = self.default_answer_tokens | overrides
        self.answer_tokens = {
            key: [token.strip().lower() for token in tokens]
            for key, tokens in merged.items()
        }

    def format_prompt(self, category: Category) -> str:
        return self.prompt.format(name=category.name, plural=category.plural)

    @abstractmethod
    def parse_response(self, response: str) -> PredT | None: ...

    @abstractmethod
    def score(self, prediction: PredT | None, expected: PredT) -> dict[str, float]: ...

    def score_tokens(
        self, tokens: list[dict[str, Any]], expected: PredT
    ) -> dict[str, float]:
        """Per-row scores from a record's `tokens`. None by default."""
        return {}

    def summarize(self, rows: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
        """Summary of one group of scored rows."""
        return rows[metric_cols].mean()
