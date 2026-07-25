import re

from .base import Task


class CountingTask(Task):
    def format_prompt(self, question):
        return f"{question}. Respond with one integer only."

    def parse_response(self, response: str) -> int | None:
        # \d+ matches one or more consecutive digits
        match = re.search(r"\d+", response)

        # Either return the match or None
        if match:
            first_int = int(match.group())
            return first_int
        else:
            return None

    def score(self, prediction: int | None, expected: int) -> dict[str, float]:
        if prediction is None:
            return {"valid": 0.0, "exact_match": 0.0, "absolute_error": float("nan")}

        error = abs(prediction - expected)
        return {
            "valid": 1.0,
            "exact_match": float(prediction == expected),
            "absolute_error": float(error),
            "within_1": float(error <= 1),
        }
