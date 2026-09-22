"""Backend and dataset defined outside the package, selected through config."""

from geo_vlms.backends.base import Generation
from geo_vlms.example import Example
from geo_vlms.tasks import Category, Task


class EchoBackend:
    def __init__(self, model_name: str, reply: str):
        self.model_name = model_name
        self.reply = reply

    def generate(self, prompt, images, max_new_tokens, top_logprobs=None):
        return Generation(text=self.reply)

    def describe(self):
        return {"kind": "echo", "name": self.model_name}


def build_dataset(task: Task, seed: int, n: int) -> list[Example]:
    return [
        Example(
            id=f"{i}",
            image_path=f"/{i}.jpg",
            prompt=task.format_prompt(Category("ship", "ships")),
        )
        for i in range(n)
    ]
