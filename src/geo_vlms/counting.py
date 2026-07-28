"""The counting task: how to ask, how to parse the answer, how to score it."""

import re


def format_prompt(question: str) -> str:
    # The newline marks the instruction boundary without assuming the question
    # carries (or lacks) terminal punctuation.
    return (
        f"{question}\nRespond with one integer only and no other prose or "
        "punctuation. Assume this will be passed to a Python int(response)."
    )


def parse_response(response: str) -> int | None:
    # \d+ matches one or more consecutive digits
    match = re.search(r"\d+", response.replace(",", ""))

    # Either return the match or None
    if match:
        first_int = int(match.group())
        return first_int
    else:
        return None


def score(prediction: int | None, expected: int) -> dict[str, float]:
    # Every branch returns the same keys, so an aggregate over one column is
    # never silently computed over a different set of rows than another.
    if prediction is None:
        return {
            "valid": 0.0,
            "exact_match": 0.0,
            "absolute_error": float("nan"),  # 'nan' to not affect calculated mean
            "within_1": 0.0,
        }

    error = abs(prediction - expected)
    return {
        "valid": 1.0,
        "exact_match": float(prediction == expected),
        "absolute_error": float(error),
        "within_1": float(error <= 1),
    }
