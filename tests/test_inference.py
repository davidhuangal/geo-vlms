import json

from geo_vlms.evals import inference
from geo_vlms.evals.inference import Example, InferenceRunner

MOCK_EXAMPLES = [
    Example(id="a", image_path="/a.jpg", prompt="q"),
    Example(id="b", image_path="/b.jpg", prompt="q"),
    Example(id="c", image_path="/c.jpg", prompt="q"),
]


def test_inference_writes_one_record_per_example(monkeypatch, tmp_path):
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
    records = runner.infer(examples=MOCK_EXAMPLES, out_path=out_path)
    lines = out_path.read_text().splitlines()

    # Making sure one-line-per-example is true
    assert len(lines) == len(records) == len(MOCK_EXAMPLES)

    first = json.loads(lines[0])
    # Make sure the response was threaded through
    assert first["output"] == "canned response"

    # These keys should be in one output line
    assert {"id", "image_path", "prompt", "expected", "model_name"} <= set(first)
