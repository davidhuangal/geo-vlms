from dataclasses import dataclass
from typing import Any


@dataclass
class Example:
    id: str  # Stable label
    image_path: str  # Path to the input image
    prompt: str  # The text prompt
    expected: Any = None  # Ground truth for scoring
