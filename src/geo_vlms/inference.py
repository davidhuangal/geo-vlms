import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from geo_vlms.vlm import prompt_model


@dataclass
class Example:
    id: str  # Stable label
    image_path: str  # Path to the input image
    prompt: str  # The text prompt
    expected: Any = None  # Ground truth for scoring


def run_inference(
    examples: list[Example],
    model,
    processor,
    out_path: str | os.PathLike,
    model_name: str,
) -> list[dict]:
    """
    Run the VLM over a list of examples and write the raw outputs to disk.

    Args:
        examples: The examples to run.
        model: The VLM model.
        processor: The processor associated with the model.
        out_path: Path to the output `.jsonl`, one record per example.
        model_name: The model's HuggingFace name, recorded for provenance.

    Returns:
        The records corresponding to the examples.
    """
    records = []
    with open(out_path, "w") as f:
        for example in examples:
            output = prompt_model(
                example.prompt, [example.image_path], model, processor
            )
            record = asdict(example) | {"output": output, "model_name": model_name}

            # Flush per record so a mid-run crash still leaves the completed
            # examples on disk.
            f.write(json.dumps(record) + "\n")
            f.flush()

            records.append(record)

    return records
