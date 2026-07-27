from geo_vlms.evals.controls import Condition, Example, build_runs

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
