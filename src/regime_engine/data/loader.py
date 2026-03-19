"""OHLCV data acquisition via yfinance."""

import logging
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)


def fetch_ohlcv(ticker="^GSPC", lookback_days=1260):
    log.info(f"Fetching {ticker}  ({lookback_days}d)")
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.5))
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.tail(lookback_days).copy()
    df.columns = [c[0].lower() for c in df.columns]
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_ret"] = df["close"].pct_change()
    df.dropna(subset=["log_ret"], inplace=True)
    log.info(f"  {len(df)} bars  [{df.index[0].date()} → {df.index[-1].date()}]")
    return df
