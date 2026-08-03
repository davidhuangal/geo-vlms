"""Counting Task"""

import re

from .base import Task


class Counting(Task[int]):
    def parse_response(self, response: str) -> int | None:
        # \d+ matches one or more consecutive digits
        match = re.search(r"\d+", response.replace(",", ""))

        # Either return the match or None
        if match:
            first_int = int(match.group())
            return first_int
        else:
            return None

    def format_prompt(self, category_name: str) -> str:
        question = self._counting_question(category_name=category_name)
        return (
            f"{question}\nRespond with one integer only and no other prose or "
            "punctuation. Assume this will be passed to a Python int(response)."
        )

    def score(self, prediction: int | None, expected: int) -> dict[str, float]:
        # Every branch returns the same keys, so an aggregate over one column is
        # never silently computed over a different set of rows than another.
        if prediction is None:
            return {
                "valid": 0.0,
                "exact_match": 0.0,
                "absolute_error": float("nan"),  # 'nan' to not affect calculated mean
                "signed_error": float("nan"),
                "within_1": 0.0,
            }

        abs_error = abs(prediction - expected)
        signed_error = prediction - expected
        return {
            "valid": 1.0,
            "exact_match": float(prediction == expected),
            "absolute_error": float(abs_error),
            "signed_error": float(signed_error),
            "within_1": float(abs_error <= 1),
        }

    def _counting_question(self, category_name: str) -> str:
        return f"How many {category_name} objects are in this image?"
