"""Existence task."""

import re

from .base import Task


class Existence(Task[bool]):
    default_prompt = "Are there any {plural} in this image? Answer yes or no."

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
