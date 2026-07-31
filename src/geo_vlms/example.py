from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    id: str  # Stable label
    # image_path: str (not PathLike) so asdict() output stays JSON-serializable.
    # None runs the prompt with no image, for text-only robustness controls.
    image_path: str | None
    prompt: str  # The text prompt
    expected: Any = None  # Ground truth for scoring
    metadata: dict[str, Any] = field(default_factory=dict)  # Any extra metadata
