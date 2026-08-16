import math

from amazon_reviews_pipeline.metadata_subset import normalize_json, price_status


def test_price_status_accepts_nonnegative_numbers_and_numeric_strings() -> None:
    assert price_status(0) == "valid"
    assert price_status(19.99) == "valid"
    assert price_status("24.48") == "valid"


def test_price_status_separates_missing_and_invalid_values() -> None:
    assert price_status(None) == "missing"
    assert price_status("None") == "missing"
    assert price_status("—") == "invalid"
    assert price_status("from 24.48") == "invalid"
    assert price_status(-1) == "invalid"


def test_normalize_json_replaces_nonfinite_floats_recursively() -> None:
    result = normalize_json({"values": [math.nan, math.inf, 1.0]})
    assert result == {"values": [None, None, 1.0]}
