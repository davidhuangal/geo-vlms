import math

import pandas as pd
import pytest

from geo_vlms.tasks import Category, Existence


@pytest.fixture
def task() -> Existence:
    return Existence()


def test_existence_prompt_uses_plural(task):
    assert task.format_prompt(Category("ship", "ships")) == (
        "Are there any ships in this image? Answer yes or no."
    )


def test_existence_custom_prompt():
    task = Existence("Is at least one {name} visible? Answer yes or no.")

    assert task.format_prompt(Category("ship", "ships")) == (
        "Is at least one ship visible? Answer yes or no."
    )


def test_existence_parse_words(task):
    assert task.parse_response("Yes") is True
    assert task.parse_response("No.") is False
    assert task.parse_response("yes, there are two.") is True


def test_existence_parse_single_char(task):
    assert task.parse_response("Y") is True
    assert task.parse_response("y") is True
    assert task.parse_response("N") is False
    assert task.parse_response("n") is False


def test_existence_char_with_period(task):
    assert task.parse_response("Y.") is True
    assert task.parse_response("N.") is False


def test_existence_char_with_spaces(task):
    assert task.parse_response(" Y") is True
    assert task.parse_response("Y ") is True
    assert task.parse_response(" N") is False
    assert task.parse_response("N ") is False


def test_existence_parse_in_text(task):
    assert task.parse_response("Yes there are ships in this image.") is True
    assert task.parse_response("No planes are in this image.") is False


@pytest.mark.xfail(
    reason="parsing difficult responses not yet implemented", strict=True
)
def test_existence_parse_difficult_response(task):
    assert task.parse_response("Many planes are present.") is True
    assert task.parse_response("Your image did not contain any harbors.") is False


def test_existence_scoring(task):
    # ----- Scenario 1: Pred: True, Expected: True
    metrics = task.score(prediction=True, expected=True)
    assert metrics["valid"] == 1.0
    assert metrics["correct"] == 1.0

    # ----- Scenario 2: Pred: False, Expected: False
    metrics = task.score(prediction=False, expected=False)
    assert metrics["valid"] == 1.0
    assert metrics["correct"] == 1.0

    # ----- Scenario 3: Pred: True, Expected: False
    metrics = task.score(prediction=True, expected=False)
    assert metrics["valid"] == 1.0
    assert metrics["correct"] == 0.0

    # ----- Scenario 4: Pred: False, Expected: True
    metrics = task.score(prediction=False, expected=True)
    assert metrics["valid"] == 1.0
    assert metrics["correct"] == 0.0

    # ----- Scenario 5: Pred: None, Expected: True
    metrics = task.score(prediction=None, expected=True)
    assert metrics["valid"] == 0.0
    assert math.isnan(metrics["correct"])

    # ----- Scenario 6: Pred: None, Expected: False
    metrics = task.score(prediction=None, expected=False)
    assert metrics["valid"] == 0.0
    assert math.isnan(metrics["correct"])


def test_parse_invalid_response(task):
    assert task.parse_response("maybe") is None
    assert task.parse_response("") is None


def first_token(top: dict[str, float]) -> list[dict]:
    alternatives = [
        {"id": i, "token": token, "logprob": logprob}
        for i, (token, logprob) in enumerate(top.items())
    ]
    return [{"id": 0, "token": "x", "logprob": 0.0, "top": alternatives}]


def test_score_tokens_renormalizes_yes_no(task):
    scores = task.score_tokens(
        first_token({"Yes": math.log(0.6), "No": math.log(0.2), "I": math.log(0.2)}),
        expected=True,
    )

    assert scores["covered"] == 1.0
    assert scores["p_yes"] == pytest.approx(0.75)


def test_score_tokens_sums_variants(task):
    scores = task.score_tokens(
        first_token({"Yes": math.log(0.3), " yes": math.log(0.3), "no": math.log(0.2)}),
        expected=True,
    )

    assert scores["p_yes"] == pytest.approx(0.75)


def test_score_tokens_one_sided(task):
    scores = task.score_tokens(first_token({"no": -0.1, "There": -2.0}), expected=True)

    assert scores == {"covered": 1.0, "p_yes": 0.0}


def test_score_tokens_uncovered(task):
    for tokens in (first_token({"There": -0.1}), []):
        scores = task.score_tokens(tokens, expected=True)

        assert scores["covered"] == 0.0
        assert math.isnan(scores["p_yes"])


def test_answer_tokens_override():
    task = Existence(answer_tokens={"yes": ["True"], "no": ["False"]})

    scores = task.score_tokens(
        first_token({"true": math.log(0.3), "false": math.log(0.1), "yes": 0.0}),
        expected=True,
    )

    assert scores["p_yes"] == pytest.approx(0.75)


def test_unknown_answer_class_raises():
    with pytest.raises(ValueError, match="Unknown answer classes"):
        Existence(answer_tokens={"maybe": ["perhaps"]})


def test_summarize_auroc_and_disagreement(task):
    rows = pd.DataFrame(
        {
            "expected": [True, False, True, False],
            "output": ["Yes", "Yes", "No", "No"],
            "p_yes": [0.9, 0.7, 0.6, 0.2],
            "correct": [1.0, 0.0, 0.0, 1.0],
        }
    )

    summary = task.summarize(rows, ["correct"])

    assert summary["correct"] == 0.5
    assert summary["auroc"] == pytest.approx(0.75)
    assert summary["disagreement"] == pytest.approx(0.25)


def test_summarize_single_class_auroc_is_nan(task):
    rows = pd.DataFrame(
        {"expected": [True, True], "output": ["Yes", "No"], "p_yes": [0.9, 0.4]}
    )

    assert math.isnan(task.summarize(rows, [])["auroc"])


def test_summarize_without_tokens_is_means(task):
    rows = pd.DataFrame({"expected": [True], "output": ["Yes"], "correct": [1.0]})

    assert task.summarize(rows, ["correct"]).to_dict() == {"correct": 1.0}
