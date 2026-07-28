import pytest

from geo_vlms.counting import parse_response


def test_counting_parse_int():
    assert parse_response("50") == 50


def test_counting_int_with_period():
    assert parse_response("6.") == 6


def test_counting_int_with_spaces():
    assert parse_response(" 10") == 10


def test_counting_parse_commas():
    assert parse_response("1,400") == 1400


def test_counting_parse_in_text():
    assert parse_response("There are 12 planes.") == 12


@pytest.mark.xfail(reason="word-number parsing not yet implemented")
def test_counting_parse_numeric_word():
    assert parse_response("fifteen") == 15
