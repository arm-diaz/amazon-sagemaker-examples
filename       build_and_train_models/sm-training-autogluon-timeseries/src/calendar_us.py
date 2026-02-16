"""
US Holiday Calendar Module
===========================
Provides US Federal and state holiday utilities for feature engineering.
Uses the `holidays` library as the base.
"""

from typing import Optional, List, Union

import pandas as pd
import numpy as np
import holidays


def get_us_holidays(
    years: Optional[Union[int, List[int]]] = None,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of US holidays for the given year(s).

    Args:
        years: Year or list of years. Defaults to current year.
        state: Two-letter US state code for state-specific holidays
               (e.g., 'TX', 'CA'). None returns only federal holidays.

    Returns:
        DataFrame with columns ['date', 'holiday_name'].
    """
    if years is None:
        years = [pd.Timestamp.now().year]
    elif isinstance(years, int):
        years = [years]

    us_holidays = holidays.US(years=years, state=state)

    records = [
        {"date": pd.Timestamp(dt), "holiday_name": name}
        for dt, name in sorted(us_holidays.items())
    ]
    return pd.DataFrame(records)


def is_holiday_us(
    date: Union[str, pd.Timestamp],
    state: Optional[str] = None,
) -> bool:
    """Check whether a single date is a US holiday."""
    date = pd.Timestamp(date)
    us_holidays = holidays.US(years=date.year, state=state)
    return date.date() in us_holidays


def days_to_next_holiday(
    date: Union[str, pd.Timestamp],
    holiday_df: pd.DataFrame,
) -> int:
    """
    Compute the number of days from *date* to the next holiday in *holiday_df*.

    Args:
        date: Reference date.
        holiday_df: DataFrame with a 'date' column (output of get_us_holidays).

    Returns:
        Number of days to the next holiday (0 if today is a holiday).
    """
    date = pd.Timestamp(date)
    future = holiday_df.loc[holiday_df["date"] >= date, "date"]
    if future.empty:
        return 365  # sentinel when no future holidays in the frame
    return (future.iloc[0] - date).days


def add_holiday_features(
    df: pd.DataFrame,
    date_column: str = "reading_date",
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add holiday-related feature columns to a DataFrame.

    Adds: is_holiday, holiday_name, days_to_next_holiday.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    years = sorted(dates.dt.year.unique())
    holiday_df = get_us_holidays(years=years, state=state)

    holiday_set = set(holiday_df["date"].dt.date)
    holiday_map = dict(zip(holiday_df["date"].dt.date, holiday_df["holiday_name"]))

    df["is_holiday"] = dates.dt.date.map(lambda d: int(d in holiday_set))
    df["holiday_name"] = dates.dt.date.map(lambda d: holiday_map.get(d))
    df["days_to_next_holiday"] = dates.map(
        lambda d: days_to_next_holiday(d, holiday_df)
    )
    return df
