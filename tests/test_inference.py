import json

from geo_vlms.backends.base import Generation, TokenLogprob
from geo_vlms.example import Example
from geo_vlms.inference import run_inference

MOCK_EXAMPLES = [
    Example(id="a", image_path="/a.jpg", prompt="q"),
    Example(id="b", image_path="/b.jpg", prompt="q"),
    Example(id="c", image_path="/c.jpg", prompt="q"),
]


class StubBackend:
    """Canned backend to avoid an expensive model call."""

    def generate(self, prompt, images, max_new_tokens, top_logprobs=None):
        return Generation(text="canned response")

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
    assert "tokens" not in first


def test_inference_records_logprobs(tmp_path):
    class LogprobBackend(StubBackend):
        def generate(self, prompt, images, max_new_tokens, top_logprobs=None):
            assert top_logprobs == 2
            return Generation(
                text="yes",
                tokens=[TokenLogprob("yes", -0.1, {"yes": -0.1, "no": -2.3})],
                prompt_tokens=10,
                completion_tokens=1,
                latency_s=0.5,
            )

    out_path = tmp_path / "runs.jsonl"
    run_inference(
        examples=MOCK_EXAMPLES[:1],
        backend=LogprobBackend(),
        out_path=out_path,
        model_name="fake-model",
        top_logprobs=2,
    )

    (record,) = [json.loads(x) for x in out_path.read_text().splitlines()]
    assert record["output"] == "yes"
    assert record["tokens"] == [
        {"token": "yes", "logprob": -0.1, "top": {"yes": -0.1, "no": -2.3}}
    ]
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 1
    assert record["latency_s"] == 0.5
