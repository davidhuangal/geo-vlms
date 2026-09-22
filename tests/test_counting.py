import math

import pytest

from geo_vlms.tasks import Category, Counting


@pytest.fixture
def task() -> Counting:
    return Counting()


def test_counting_prompt_uses_plural(task):
    assert task.format_prompt(Category("storage tank", "storage tanks")) == (
        "How many storage tanks are there in this image? Answer with a number only."
    )


def test_counting_parse_int(task):
    assert task.parse_response("50") == 50


def test_counting_int_with_period(task):
    assert task.parse_response("6.") == 6


def test_counting_int_with_spaces(task):
    assert task.parse_response(" 10") == 10


def test_counting_parse_commas(task):
    assert task.parse_response("1,400") == 1400


def test_counting_parse_in_text(task):
    assert task.parse_response("There are 12 planes.") == 12


def test_counting_parse_numeric_word(task):
    assert task.parse_response("fifteen") == 15
    assert task.parse_response("Zero.") == 0
    assert task.parse_response("None") == 0
    assert task.parse_response("There are two ships.") == 2


def test_counting_digits_win_over_words(task):
    assert task.parse_response("one of the 3 tanks") == 3


def test_counting_unparseable_is_none(task):
    assert task.parse_response("I cannot tell.") is None


def test_counting_scoring(task):
    # ----- Scenario 1: pred < expected
    prediction = 4
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert metrics["absolute_error"] == 5.0
    assert metrics["signed_error"] == -5.0
    assert metrics["relative_error"] == pytest.approx(5 / 9)
    assert metrics["within_1"] == 0.0

    # ----- Scenario 2: pred > expected
    prediction = 9
    expected = 4
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert metrics["absolute_error"] == 5.0
    assert metrics["signed_error"] == 5.0
    assert metrics["relative_error"] == 1.25
    assert metrics["within_1"] == 0.0

    # ----- Scenario 3: pred == expected
    prediction = 9
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["absolute_error"] == 0.0
    assert metrics["signed_error"] == 0.0
    assert metrics["relative_error"] == 0.0
    assert metrics["within_1"] == 1.0

    # ----- Scenario 4: pred == None
    prediction = None
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 0.0
    assert metrics["exact_match"] == 0.0
    assert math.isnan(metrics["absolute_error"])
    assert math.isnan(metrics["signed_error"])
    assert math.isnan(metrics["relative_error"])
    assert metrics["within_1"] == 0.0


def test_counting_relative_error_undefined_on_empty(task):
    metrics = task.score(prediction=2, expected=0)
    assert metrics["absolute_error"] == 2.0
    assert math.isnan(metrics["relative_error"])
