import numpy as np
import pandas as pd
import pytest

from regime_engine.features import add_features


@pytest.fixture
def raw_df():
    """500 rows of synthetic OHLCV data with realistic price and volume."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)

    close = np.zeros(n)
    close[0] = 4500.0
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.normal(0.0003, 0.012))

    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(2_000_000_000, 5_000_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_ret"] = df["close"].pct_change()
    df.dropna(subset=["log_ret"], inplace=True)
    return df


@pytest.fixture
def featured_df(raw_df):
    """raw_df passed through add_features()."""
    return add_features(raw_df.copy())


@pytest.fixture
def regime_df(featured_df):
    """featured_df with synthetic regime and ewma_vol_daily columns."""
    df = featured_df.copy()
    n = len(df)
    df["regime"] = [i % 3 for i in range(n)]
    df["ewma_vol_daily"] = 0.01
    return df
