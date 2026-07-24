"""Multi-condition evaluation runner.

Orchestrates a VLM over the runs produced by :mod:`geo_vlms.controls`,
writing one raw record per run to a JSONL file before any metrics are computed.
"""

import json

from geo_vlms.controls import build_runs
from geo_vlms.vlm import build_model_and_processor, prompt_model


def build_record(run, output, model_name, seed):
    record = {
        "example_id": run.example_id,
        "condition": run.condition.name,
        "prompt": run.prompt,
        "image_paths": run.image_paths,
        "expected": run.expected,
        "output": output,
        "model_name": model_name,
        "seed": seed,
    }
    return record


class MultiConditionRunner:
    def __init__(self, model_name: str, examples: list, device: str):
        # Denote the user settings
        self.model_name = model_name
        self.examples = examples
        self.device = device

        # Leave these uninitialized for now to save on memory
        self.model, self.processor = None, None

    def _maybe_init_model(self):
        if self.model is None or self.processor is None:
            self.model, self.processor = build_model_and_processor(
                model_name=self.model_name, device=self.device
            )

    def evaluate(self, out_path: str, seed: int = 0):
        # Init model if needed
        self._maybe_init_model()

        # Generate the runs per example
        runs = build_runs(examples=self.examples, seed=seed)

        # Run evaluation
        records = []
        with open(out_path, "w") as f:
            for run in runs:
                # Run through the model and record the output
                output = prompt_model(
                    run.prompt, run.image_paths, self.model, self.processor
                )
                record = build_record(
                    run=run, output=output, model_name=self.model_name, seed=seed
                )
                f.write(json.dumps(record) + "\n")
                f.flush()
                records.append(record)

        return records
