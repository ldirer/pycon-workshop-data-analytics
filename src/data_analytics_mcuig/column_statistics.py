"""Column-wise descriptive statistics for pandas DataFrames."""

import pandas as pd


def calculate_column_statistics(
    df: pd.DataFrame,
    *,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Return one row of summary statistics per input column.

    For each column, statistics are computed down the column (axis=0). Mean,
    median, mode, max, min, kurtosis, and skewness use pandas' default
    ``skipna=True`` behavior. If several values tie for mode, the first
    reported mode from ``DataFrame.mode`` is used (row ``0`` of the mode
    frame).

    Args:
        df: Input data.
        numeric_only: If True (default), only numeric columns are used. If
            False, all columns are passed through; non-numeric columns affect
            mode and counts but not mean/median/max/min/kurtosis/skewness
            (pandas will coerce or omit as appropriate for mixed types).

    Returns:
        DataFrame indexed by column name with columns ``mean``, ``median``,
        ``mode``, ``max``, ``min``, ``kurtosis``, ``skewness``, ``nan_count``,
        ``non_nan_count``, ``nan_percentage`` (fraction of rows that are NaN,
        in ``[0, 1]``; NaN when the column has zero rows).

    Raises:
        ValueError: If there are no columns to analyze after selection.
    """
    if numeric_only:
        data = df.select_dtypes(include="number")
    else:
        data = df

    if data.shape[1] == 0:
        raise ValueError(
            "No columns to analyze. Use numeric_only=False or add numeric "
            "columns."
            if numeric_only
            else "DataFrame has no columns."
        )

    out = pd.DataFrame(index=data.columns)

    out["mean"] = data.mean(axis=0, skipna=True)
    out["median"] = data.median(axis=0, skipna=True)

    mode_frame = data.mode(axis=0, numeric_only=numeric_only)
    if mode_frame.shape[0] > 0:
        out["mode"] = mode_frame.iloc[0]
    else:
        out["mode"] = pd.Series(pd.NA, index=data.columns)

    out["max"] = data.max(axis=0, skipna=True)
    out["min"] = data.min(axis=0, skipna=True)
    out["kurtosis"] = data.kurtosis(axis=0, skipna=True)
    out["skewness"] = data.skew(axis=0, skipna=True)

    out["nan_count"] = data.isna().sum(axis=0)
    out["non_nan_count"] = data.notna().sum(axis=0)
    n_rows = data.shape[0]
    out["nan_percentage"] = (
        out["nan_count"] / n_rows if n_rows else float("nan")
    )

    return out


def calculate_max_of_column(df: pd.DataFrame, column: str) -> float:
    """Calculate the maximum value of a column in a DataFrame.

    Args:
        df: Input data.
        column: Column name.

    Returns:
        Maximum value of the column.
    """
    return df[column].max()


def calculate_min_of_column(df: pd.DataFrame, column: str) -> float:
    """Calculate the minimum value of a column in a DataFrame.

    Args:
        df: Input data.
        column: Column name.

    Returns:
        Minimum value of the column.
    """
    return df[column].min()


def calculate_span_of_column(df: pd.DataFrame, column: str) -> float:
    """Calculate the span of a column in a DataFrame.

    Args:
        df: Input data.
        column: Column name.

    Returns:
        Span of the column.
    """
    return df[column].max() - df[column].min()