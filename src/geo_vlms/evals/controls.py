"""Reliability controls.

Defines the evaluation conditions (REAL vs. shuffled-image vs. text-only) and
the logic that expands each example into one run per condition. This is the
research core: comparing model behavior across these conditions is how we
distinguish genuine visual grounding from language priors.
"""

import random
from dataclasses import dataclass
from enum import Enum


class Condition(Enum):
    REAL = 1
    SHUFFLED = 2
    TEXT_ONLY = 3


@dataclass
class Example:
    id: str  # Stable label
    image_path: str  # Path to the input image
    prompt: str  # The text prompt
    expected: str | None = None  # Ground truth for scoring


@dataclass
class Run:
    example_id: str
    condition: Condition
    prompt: str
    image_paths: list[str] | None
    expected: str | None


def build_runs(examples: list[Example], seed: int) -> list[Run]:
    rng = random.Random(seed)
    runs = []

    for ex in examples:
        # ----- REAL ------
        runs.append(
            Run(
                example_id=ex.id,
                condition=Condition.REAL,
                prompt=ex.prompt,
                image_paths=[ex.image_path],
                expected=ex.expected,
            )
        )
        # ----- SHUFFLED -----
        others = [o for o in examples if o.id != ex.id]
        if others:
            other_ex = rng.choice(others)
            runs.append(
                Run(
                    example_id=ex.id,
                    condition=Condition.SHUFFLED,
                    prompt=ex.prompt,
                    image_paths=[other_ex.image_path],
                    expected=ex.expected,
                )
            )
        # ----- TEXT ONLY -----
        runs.append(
            Run(
                example_id=ex.id,
                condition=Condition.TEXT_ONLY,
                prompt=ex.prompt,
                image_paths=None,
                expected=ex.expected,
            )
        )

    return runs
