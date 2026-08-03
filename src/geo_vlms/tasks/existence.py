"""Existence task."""

import re

from .base import Task


class Existence(Task[bool]):
    def parse_response(self, response: str) -> bool | None:
        match = re.search(r"\b(yes|no|[yn])\b", response, re.IGNORECASE)

        if match is None:
            return None
        return match.group().lower().startswith("y")

    def score(self, prediction: bool | None, expected: bool) -> dict[str, float]:
        if prediction is None:
            return {"valid": 0.0, "correct": float("nan")}

        correct = float(prediction == expected)
        return {"valid": 1.0, "correct": correct}

    def _existence_question(self, category_name: str) -> str:
        return f"Are there any {category_name} objects in this image?"

    def format_prompt(self, category_name: str) -> str:
        question = self._existence_question(category_name=category_name)
        return (
            f"{question}\nRespond with a single character: Y for yes or N for no, "
            "and no other prose or punctuation."
        )
