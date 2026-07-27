import json
import os

from geo_vlms.evals.controls import Run
from geo_vlms.models.vlm import build_model_and_processor, prompt_model


def build_record(run, output, model_name):
    record = {
        "example_id": run.example_id,
        "condition": run.condition.name,
        "prompt": run.prompt,
        "image_paths": run.image_paths,
        "expected": run.expected,
        "output": output,
        "model_name": model_name,
    }
    return record


class InferenceRunner:
    def __init__(self, model_name: str, list, device: str):
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

    def infer(self, runs: list[Run], out_path: str | os.PathLike) -> list[dict]:
        """
        Run inference with the VLM and produce outputs.

        Args:
            runs: List of Run objects each representing an inference job.
            out_path: Desired output path for each run's outcome.

        Returns:
            The list of records which correspond to the runs.
        """
        # Init model if needed
        self._maybe_init_model()

        # Run evaluation
        records = []
        with open(out_path, "w") as f:
            for run in runs:
                # Run through the model and record the output
                output = prompt_model(
                    run.prompt, run.image_paths, self.model, self.processor
                )
                record = build_record(
                    run=run, output=output, model_name=self.model_name
                )
                f.write(json.dumps(record) + "\n")
                f.flush()
                records.append(record)

        return records
