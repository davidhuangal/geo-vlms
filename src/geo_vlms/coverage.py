from collections.abc import Sequence

from geo_vlms.tasks import Counting, Existence, Task


def expected_at(
    task: Task,
    coverage: Sequence[float],
    difficult: Sequence[bool],
    min_cover: float,
) -> int | list[int] | bool:
    """Ground truth counting objects with at least `min_cover` of their area in view.

    Difficult objects are optional, so they widen a count into a `[lo, hi]` range.
    """
    kept = [d for c, d in zip(coverage, difficult, strict=True) if c >= min_cover]
    hi = len(kept)
    lo = hi - sum(kept)

    if isinstance(task, Existence):
        return hi > 0
    if isinstance(task, Counting):
        return lo if lo == hi else [lo, hi]
    raise ValueError(f"Unsupported task for coverage: {type(task).__name__}")
