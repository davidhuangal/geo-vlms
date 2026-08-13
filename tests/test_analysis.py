import pandas as pd
import pytest

from geo_vlms.analysis import score_records, summarize


def _metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["ship", "ship", "airplane", "airplane"],
            "absolute_error": [1.0, 3.0, 0.0, 2.0],
            "signed_error": [-1.0, 3.0, 0.0, -2.0],
        }
    )


def test_summarize_grouped():
    summary = summarize(
        metrics_df=_metrics_df(),
        group_by=["category"],
        metric_cols=["absolute_error", "signed_error"],
    )
    assert summary.loc["ship", "absolute_error"] == 2.0
    assert summary.loc["ship", "signed_error"] == 1.0
    assert summary.loc["airplane", "absolute_error"] == 1.0
    assert summary.loc["airplane", "signed_error"] == -1.0


def test_summarize_ungrouped():
    summary = summarize(
        metrics_df=_metrics_df(),
        group_by=None,
        metric_cols=["absolute_error", "signed_error"],
    )
    assert isinstance(summary, pd.DataFrame)
    assert summary.loc["mean", "absolute_error"] == 1.5
    assert summary.loc["mean", "signed_error"] == 0.0


def test_score_records():
    def parse(x):
        return int(x)

    def score(predicted, expected):
        return {"absolute_error": abs(predicted - expected)}

    dummy_data = [{"output": str(i), "expected": 10} for i in range(10)]
    records_df = pd.DataFrame(dummy_data)

    # Shuffled index with a duplicate label: label-based alignment would
    # reorder rows or blow up, so only positional alignment passes
    records_df.index = [7, 3, 3, 11, 2, 9, 4, 10, 6, 5]

    metrics_df = score_records(records_df, parse, score)

    expected_df = records_df.copy()
    expected_df["absolute_error"] = range(10, 0, -1)
    pd.testing.assert_frame_equal(metrics_df, expected_df)


def test_score_records_column_collision():
    """A score key that collides with a records column must raise, not shadow."""

    def parse(x):
        return int(x)

    def score(predicted, expected):
        return {"output": float(predicted == expected)}

    records_df = pd.DataFrame([{"output": "1", "expected": 1}])

    with pytest.raises(ValueError):
        score_records(records_df, parse, score)
