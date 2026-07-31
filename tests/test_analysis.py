import pandas as pd

from geo_vlms.analysis import summarize


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
