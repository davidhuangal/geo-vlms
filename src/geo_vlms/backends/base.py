from typing import Protocol


class Backend(Protocol):
    def generate(
        self, prompt: str, image_paths: list[str] | None, max_new_tokens: int
    ) -> str: ...
    def describe(self) -> dict: ...
