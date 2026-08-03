import math

import pytest

from geo_vlms.tasks import Counting


@pytest.fixture
def task() -> Counting:
    return Counting()


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


@pytest.mark.xfail(reason="word-number parsing not yet implemented")
def test_counting_parse_numeric_word(task):
    assert task.parse_response("fifteen") == 15


def test_counting_scoring(task):
    # ----- Scenario 1: pred < expected
    prediction = 4
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert metrics["absolute_error"] == 5.0
    assert metrics["signed_error"] == -5.0
    assert metrics["within_1"] == 0.0

    # ----- Scenario 2: pred > expected
    prediction = 9
    expected = 4
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert metrics["absolute_error"] == 5.0
    assert metrics["signed_error"] == 5.0
    assert metrics["within_1"] == 0.0

    # ----- Scenario 3: pred == expected
    prediction = 9
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["absolute_error"] == 0.0
    assert metrics["signed_error"] == 0.0
    assert metrics["within_1"] == 1.0

    # ----- Scenario 4: pred == None
    prediction = None
    expected = 9
    metrics = task.score(prediction=prediction, expected=expected)
    assert metrics["valid"] == 0.0
    assert metrics["exact_match"] == 0.0
    assert math.isnan(metrics["absolute_error"])
    assert math.isnan(metrics["signed_error"])
    assert metrics["within_1"] == 0.0
