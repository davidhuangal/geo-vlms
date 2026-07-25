import pytest

from geo_vlms.evals.tasks import CountingTask


def test_counting_parse_int():
    task = CountingTask()
    response = "50"
    parsed_response = task.parse_response(response)
    assert parsed_response == 50


def test_counting_parse_commas():
    task = CountingTask()
    response = "1,400"
    parsed_response = task.parse_response(response)
    assert parsed_response == 1400


def test_counting_parse_in_text():
    task = CountingTask()
    response = "There are 12 planes."
    parsed_response = task.parse_response(response)
    assert parsed_response == 12


@pytest.mark.xfail(reason="word-number parsing not yet implemented")
def test_counting_parse_numeric_word():
    task = CountingTask()
    response = "fifteen"
    parsed_response = task.parse_response(response)
    assert parsed_response == 15
