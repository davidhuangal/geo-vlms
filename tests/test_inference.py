import json

from geo_vlms.controls import Condition, Example, Run, build_runs
from geo_vlms.evals import inference
from geo_vlms.evals.inference import MultiConditionRunner, build_record

MOCK_EXAMPLES = [
    Example(id="a", image_path="/a.jpg", prompt="q"),
    Example(id="b", image_path="/b.jpg", prompt="q"),
    Example(id="c", image_path="/c.jpg", prompt="q"),
]


def test_build_runs_num_created_runs():
    # There should be three runs per example
    runs = build_runs(examples=MOCK_EXAMPLES, seed=0)
    assert len(runs) == len(MOCK_EXAMPLES) * 3


def test_build_runs_is_deterministic():
    # Using the same seed should produce identical results
    assert build_runs(MOCK_EXAMPLES, seed=0) == build_runs(MOCK_EXAMPLES, seed=0)


def test_build_runs_seed_changes_shuffle():
    # Varying seeds should produce different outcomes
    # NOTE: In theory, this could not hold true, especially over a large
    # number of trials
    assert build_runs(MOCK_EXAMPLES, seed=0) != build_runs(MOCK_EXAMPLES, seed=1)


def test_build_runs_shuffle_correctness():
    # Build runs
    runs = build_runs(examples=MOCK_EXAMPLES, seed=0)
    # Map from example id to its original image path
    own_image = {ex.id: ex.image_path for ex in MOCK_EXAMPLES}
    # Check each run
    for run in runs:
        if run.condition == Condition.SHUFFLED:
            # A shuffled run should not have the same image_path as the original
            assert run.image_paths != [own_image[run.example_id]]


def test_build_runs_text_only_correctness():
    # Build runs
    runs = build_runs(examples=MOCK_EXAMPLES, seed=0)
    # Find the TEXT_ONLY runs
    text_only = next(run for run in runs if run.condition == Condition.TEXT_ONLY)
    # Ensure image paths aren't present
    assert text_only.image_paths is None


def test_build_runs_single_example():
    # Create runs with only one example
    runs = build_runs(examples=MOCK_EXAMPLES[:1], seed=0)
    # Denote shuffled
    shuffled_runs = [run for run in runs if run.condition == Condition.SHUFFLED]
    # There should be no shuffled runs
    assert len(shuffled_runs) == 0


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
    record = build_record(
        run=run, output="some model response", model_name="my_model", seed=0
    )
    dumped_s = json.dumps(record)

    # Make sure the Condition Enum renders correctly
    assert json.loads(dumped_s)["condition"] == "REAL"


def test_evaluate_writes_one_record_per_run(monkeypatch, tmp_path):
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

    # Init the evaluation runner
    runner = MultiConditionRunner(
        model_name="fake-model", examples=MOCK_EXAMPLES, device="cpu"
    )

    # Run evaluation and read the output
    records = runner.evaluate(out_path=out_path, seed=0)
    lines = out_path.read_text().splitlines()

    # Should write 3 lines per example
    assert len(lines) == len(records) == len(MOCK_EXAMPLES) * 3

    first = json.loads(lines[0])
    # Make sure the response was threaded through
    assert first["output"] == "canned response"

    # These keys should be in one output line
    assert {"example_id", "condition", "seed"} <= set(first)
