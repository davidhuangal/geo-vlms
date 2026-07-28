import json
import os
from dataclasses import dataclass
from typing import Any

from geo_vlms.models.vlm import build_model_and_processor, prompt_model


@dataclass
class Example:
    id: str  # Stable label
    image_path: str  # Path to the input image
    prompt: str  # The text prompt
    expected: Any = None  # Ground truth for scoring


def build_record(example, output, model_name):
    record = {
        "id": example.id,
        "image_path": example.image_path,
        "prompt": example.prompt,
        "expected": example.expected,
        "output": output,
        "model_name": model_name,
    }
    return record


class InferenceRunner:
    def __init__(self, model_name: str, device: str):
        # Denote the user settings
        self.model_name = model_name
        self.device = device

        # Leave these uninitialized for now to save on memory
        self.model, self.processor = None, None

    def _maybe_init_model(self):
        if self.model is None or self.processor is None:
            self.model, self.processor = build_model_and_processor(
                model_name=self.model_name, device=self.device
            )

    def infer(self, examples: list[Example], out_path: str | os.PathLike) -> list[dict]:
        """
        Run inference with the VLM and produce outputs.

        Args:
            examples: List of Example objects each representing an inference job.
            out_path: Desired output path for each example's outcome.

        Returns:
            The list of records which correspond to the examples.
        """
        # Init model if needed
        self._maybe_init_model()

        # Run inference
        records = []
        with open(out_path, "w") as f:
            for example in examples:
                # Run through the model and record the output
                output = prompt_model(
                    example.prompt, [example.image_path], self.model, self.processor
                )
                record = build_record(
                    example=example, output=output, model_name=self.model_name
                )
                f.write(json.dumps(record) + "\n")
                f.flush()
                records.append(record)

        return records
