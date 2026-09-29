"""Counting Task"""

import re
from collections.abc import Sequence

from .base import Task

NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"none": 0}


class Counting(Task[int]):
    default_prompt = (
        "How many {plural} are there in this image? Answer with a number only."
    )

    def parse_response(self, response: str) -> int | None:
        # \d+ matches one or more consecutive digits
        match = re.search(r"\d+", response.replace(",", ""))

        # Either return the match or None
        if match:
            first_int = int(match.group())
            return first_int

        # Fall back to a spelled-out number, e.g. "zero" or "None."
        for word in re.findall(r"[a-z]+", response.lower()):
            if word in NUMBER_WORDS:
                return NUMBER_WORDS[word]
        return None

    def score(
        self, prediction: int | None, expected: int | Sequence[int]
    ) -> dict[str, float]:
        """Score against a count, or a `[lo, hi]` range of acceptable counts."""
        # Every branch returns the same keys, so an aggregate over one column is
        # never silently computed over a different set of rows than another.
        if prediction is None:
            return {
                "valid": 0.0,
                "exact_match": 0.0,
                "absolute_error": float("nan"),  # 'nan' to not affect calculated mean
                "signed_error": float("nan"),
                "relative_error": float("nan"),
                "within_1": 0.0,
            }

        lo, hi = expected if isinstance(expected, Sequence) else (expected, expected)
        # Errors are measured to the nearest count in the range
        target = min(max(prediction, lo), hi)

        abs_error = abs(prediction - target)
        signed_error = prediction - target
        # Undefined on empty images, so its mean is MAPE over non-empty ones
        relative_error = abs_error / target if target > 0 else float("nan")
        return {
            "valid": 1.0,
            "exact_match": float(abs_error == 0),
            "absolute_error": float(abs_error),
            "signed_error": float(signed_error),
            "relative_error": float(relative_error),
            "within_1": float(abs_error <= 1),
        }
