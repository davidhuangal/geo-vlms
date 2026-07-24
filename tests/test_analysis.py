from geo_vlms.analysis import normalize_text


def test_normalize_text():
    text = "  Python 3!!   Was Released...  "
    normalized = normalize_text(text=text)
    assert normalized == "python 3!! was released..."
