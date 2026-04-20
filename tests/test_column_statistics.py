"""Tests for data_analytics.column_statistics."""

import pandas as pd
import pytest

from src.data_analytics_mcuig.column_statistics import calculate_column_statistics
from src.data_analytics_mcuig.column_statistics import calculate_max_of_column
from src.data_analytics_mcuig.column_statistics import calculate_min_of_column
from src.data_analytics_mcuig.column_statistics import calculate_span_of_column

EXPECTED_COLUMNS = [
    "mean",
    "median",
    "mode",
    "max",
    "min",
    "kurtosis",
    "skewness",
    "nan_count",
    "non_nan_count",
    "nan_percentage",
]


@pytest.mark.parametrize("func", [calculate_column_statistics])
def test_public_and_module_import_same_result(func):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pd.testing.assert_frame_equal(calculate_column_statistics(df), func(df))


def test_columns_and_index_preserved():
    df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [2.0, 4.0]},
        index=pd.Index(["r0", "r1"], name="row"),
    )
    out = calculate_column_statistics(df)
    assert list(out.columns) == EXPECTED_COLUMNS
    pd.testing.assert_index_equal(out.index, pd.Index(["x", "y"]))


def test_basic_numeric_stats():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 5.0], "c": [3.0, 6.0, 7.0]})
    out = calculate_column_statistics(df)
    assert out["mean"].tolist() == pytest.approx([2.0, 11.0 / 3.0, 16.0 / 3.0])
    assert out["median"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert out["max"].tolist() == pytest.approx([3.0, 5.0, 7.0])
    assert out["min"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert (out["nan_count"] == 0).all()
    assert (out["non_nan_count"] == 3).all()
    assert (out["nan_percentage"] == 0).all()


def test_nan_counts_and_skipna_aggregates():
    df = pd.DataFrame(
        {
            "a": [1.0, float("nan"), 10.0],
            "b": [2.0, 3.0, float("nan")],
            "c": [3.0, float("nan"), float("nan")],
        }
    )
    out = calculate_column_statistics(df)
    assert out["nan_count"].tolist() == [1, 1, 2]
    assert out["non_nan_count"].tolist() == [2, 2, 1]
    assert out["nan_percentage"].tolist() == pytest.approx([1 / 3, 1 / 3, 2 / 3])
    assert out.loc["a", "mean"] == pytest.approx(5.5)
    assert out.loc["b", "mean"] == pytest.approx(2.5)
    assert out.loc["c", "mean"] == pytest.approx(3.0)
    assert out.loc["a", "max"] == pytest.approx(10.0)
    assert out.loc["a", "min"] == pytest.approx(1.0)


def test_numeric_only_ignores_non_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0], "label": ["x", "y"]})
    out = calculate_column_statistics(df, numeric_only=True)
    assert out["mean"].tolist() == pytest.approx([1.5, 3.0])
    assert (out["non_nan_count"] == 2).all()
    assert (out["nan_percentage"] == 0).all()


def test_numeric_only_false_all_numeric():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out_default = calculate_column_statistics(df)
    out_false = calculate_column_statistics(df, numeric_only=False)
    pd.testing.assert_frame_equal(out_default, out_false)


def test_no_numeric_columns_raises():
    df = pd.DataFrame({"label": ["a", "b"], "tag": ["c", "d"]})
    with pytest.raises(ValueError, match="No columns to analyze"):
        calculate_column_statistics(df, numeric_only=True)


def test_empty_columns_raises_when_numeric_only_false():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="no columns"):
        calculate_column_statistics(df, numeric_only=False)


def test_empty_rows_one_row_per_column():
    df = pd.DataFrame({"a": [], "b": []}, dtype=float)
    out = calculate_column_statistics(df)
    assert list(out.columns) == EXPECTED_COLUMNS
    assert list(out.index) == ["a", "b"]
    assert pd.isna(out["mean"]).all()
    assert pd.isna(out["nan_percentage"]).all()


def test_single_observation_kurtosis_skew_nan():
    df = pd.DataFrame([{"a": 5.0, "b": float("nan")}])
    out = calculate_column_statistics(df)
    assert pd.isna(out.loc["a", "kurtosis"])
    assert pd.isna(out.loc["a", "skewness"])
    assert out.loc["a", "nan_percentage"] == pytest.approx(0.0)
    assert out.loc["b", "nan_percentage"] == pytest.approx(1.0)


def test_mode_clear_majority():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 2.0], "c": [1.0, 1.0, 1.0]})
    out = calculate_column_statistics(df)
    assert out.loc["a", "mode"] == pytest.approx(1.0)
    assert out.loc["b", "mode"] == pytest.approx(2.0)
    assert out.loc["c", "mode"] == pytest.approx(1.0)


def test_calculate_max_of_column():
    df = pd.DataFrame({"x": [1.0, 5.0, 3.0], "y": [-2.0, 0.0, 1.0]})
    assert calculate_max_of_column(df, "x") == pytest.approx(5.0)
    assert calculate_max_of_column(df, "y") == pytest.approx(1.0)


def test_calculate_max_of_column_skips_na():
    df = pd.DataFrame({"x": [1.0, float("nan"), 10.0]})
    assert calculate_max_of_column(df, "x") == pytest.approx(10.0)


def test_calculate_max_of_column_all_nan():
    df = pd.DataFrame({"x": [float("nan"), float("nan")]})
    assert pd.isna(calculate_max_of_column(df, "x"))


def test_calculate_min_of_column():
    df = pd.DataFrame({"x": [1.0, 5.0, 3.0], "y": [-2.0, 0.0, 1.0]})
    assert calculate_min_of_column(df, "x") == pytest.approx(1.0)
    assert calculate_min_of_column(df, "y") == pytest.approx(-2.0)


def test_calculate_min_of_column_skips_na():
    df = pd.DataFrame({"x": [1.0, float("nan"), 10.0]})
    assert calculate_min_of_column(df, "x") == pytest.approx(1.0)


def test_calculate_min_of_column_all_nan():
    df = pd.DataFrame({"x": [float("nan"), float("nan")]})
    assert pd.isna(calculate_min_of_column(df, "x"))


def test_calculate_span_of_column():
    df = pd.DataFrame({"x": [1.0, 5.0, 3.0], "y": [-2.0, 0.0, 1.0]})
    assert calculate_span_of_column(df, "x") == pytest.approx(4.0)
    assert calculate_span_of_column(df, "y") == pytest.approx(3.0)


def test_calculate_span_of_column_skips_na():
    df = pd.DataFrame({"x": [1.0, float("nan"), 10.0]})
    assert calculate_span_of_column(df, "x") == pytest.approx(9.0)


def test_calculate_span_of_column_single_value():
    df = pd.DataFrame({"x": [7.0]})
    assert calculate_span_of_column(df, "x") == pytest.approx(0.0)


def test_calculate_span_of_column_all_nan():
    df = pd.DataFrame({"x": [float("nan"), float("nan")]})
    result = calculate_span_of_column(df, "x")
    assert pd.isna(result)
