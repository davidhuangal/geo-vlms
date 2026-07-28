import json
import os
from dataclasses import asdict

from geo_vlms.example import Example
from geo_vlms.vlm import prompt_model


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
            # A None image_path means a text-only control: pass no images
            # rather than a list containing None.
            image_paths = None if example.image_path is None else [example.image_path]
            output = prompt_model(example.prompt, image_paths, model, processor)
            record = asdict(example) | {"output": output, "model_name": model_name}

            # Flush per record so a mid-run crash still leaves the completed
            # examples on disk.
            f.write(json.dumps(record) + "\n")
            f.flush()

            records.append(record)

    return records
