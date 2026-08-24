import json

from geo_vlms.example import Example
from geo_vlms.inference import run_inference

MOCK_EXAMPLES = [
    Example(id="a", image_path="/a.jpg", prompt="q"),
    Example(id="b", image_path="/b.jpg", prompt="q"),
    Example(id="c", image_path="/c.jpg", prompt="q"),
]


class StubBackend:
    """Canned backend to avoid an expensive model call."""

    def generate(self, prompt, image_paths, max_new_tokens):
        return "canned response"

    def describe(self):
        return {"kind": "stub"}


def test_inference_writes_one_record_per_example(tmp_path):
    # Denote the output path
    out_path = tmp_path / "runs.jsonl"

    # Run inference and read the output
    records = run_inference(
        examples=MOCK_EXAMPLES,
        backend=StubBackend(),
        out_path=out_path,
        model_name="fake-model",
    )
    lines = out_path.read_text().splitlines()

    # Making sure one-line-per-example is true
    assert len(lines) == len(records) == len(MOCK_EXAMPLES)

    first = json.loads(lines[0])
    # Make sure the response was threaded through
    assert first["output"] == "canned response"

    # These keys should be in one output line
    assert {"id", "image_path", "prompt", "expected", "model_name"} <= set(first)
