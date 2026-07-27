import json

from geo_vlms.evals import inference
from geo_vlms.evals.controls import Condition, Example, Run, build_runs
from geo_vlms.evals.inference import InferenceRunner, build_record

MOCK_EXAMPLES = [
    Example(id="a", image_path="/a.jpg", prompt="q"),
    Example(id="b", image_path="/b.jpg", prompt="q"),
    Example(id="c", image_path="/c.jpg", prompt="q"),
]


def test_record_is_json_serializable():
    # Create a sample Run
    run = Run(
        example_id="a",
        condition=Condition.REAL,
        prompt="q",
        image_paths=["/a.jpg"],
        expected=None,
    )

    # Generate a record and make sure it is serializable
    record = build_record(run=run, output="some model response", model_name="my_model")
    dumped_s = json.dumps(record)

    # Make sure the Condition Enum renders correctly
    assert json.loads(dumped_s)["condition"] == "REAL"


def test_inference_writes_one_record_per_run(monkeypatch, tmp_path):
    # Monkey patching to avoid expensive model load
    monkeypatch.setattr(
        target=inference,
        name="build_model_and_processor",
        value=lambda model_name, device: (object(), object()),
    )
    monkeypatch.setattr(
        target=inference,
        name="prompt_model",
        value=lambda prompt, image_paths, model, processor: "canned response",
    )

    # Denote the output path
    out_path = tmp_path / "runs.jsonl"

    # Init the inference runner
    runner = InferenceRunner(model_name="fake-model", device="cpu")

    # Run inference and read the output
    runs = build_runs(examples=MOCK_EXAMPLES, seed=0)
    records = runner.infer(out_path=out_path, runs=runs)
    lines = out_path.read_text().splitlines()

    # Should write 3 lines per example
    assert len(lines) == len(records) == len(MOCK_EXAMPLES) * 3

    first = json.loads(lines[0])
    # Make sure the response was threaded through
    assert first["output"] == "canned response"

    # These keys should be in one output line
    assert {"example_id", "condition", "prompt", "image_paths", "expected"} <= set(
        first
    )
