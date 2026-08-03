import math

import pytest

from geo_vlms.tasks import Existence


@pytest.fixture
def task() -> Existence:
    return Existence()


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
