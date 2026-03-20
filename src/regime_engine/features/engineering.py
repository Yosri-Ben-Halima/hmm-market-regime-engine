"""Feature engineering for HMM input."""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_features(df):
    log.info(f"  Input: {len(df)} rows, {len(df.columns)} columns")
    c, h, l_, v = (df[x].values for x in ["close", "high", "low", "volume"])

    for p in [20, 50, 100, 200]:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        df[f"sma_{p}"] = df["close"].rolling(p).mean()

    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_28"] = _rsi(df["close"], 28)

    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    for p in [20, 40]:
        mid = df["close"].rolling(p).mean()
        sd = df["close"].rolling(p).std()
        df[f"bb_upper_{p}"] = mid + 2 * sd
        df[f"bb_lower_{p}"] = mid - 2 * sd
        df[f"bb_pct_{p}"] = (df["close"] - df[f"bb_lower_{p}"]) / (
            df[f"bb_upper_{p}"] - df[f"bb_lower_{p}"] + 1e-9
        )
        df[f"bb_width_{p}"] = (df[f"bb_upper_{p}"] - df[f"bb_lower_{p}"]) / (mid + 1e-9)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    obv = [0]
    for i in range(1, len(df)):
        if c[i] > c[i - 1]:
            obv.append(obv[-1] + v[i])
        elif c[i] < c[i - 1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
    df["obv_signal"] = np.sign(df["obv"] - df["obv_ema"])

    for p in [5, 10, 20, 60, 126]:
        df[f"rvol_{p}"] = df["log_ret"].rolling(p).std() * np.sqrt(252) * 100

    df["vol_ratio"] = df["rvol_5"] / (df["rvol_60"] + 1e-9)

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["vol_sma20"] + 1e-9)

    for p in [5, 21, 63, 126]:
        df[f"mom_{p}"] = df["close"].pct_change(p)

    df["dist_ema50"] = (df["close"] - df["ema_50"]) / (df["ema_50"] + 1e-9)
    df["dist_ema200"] = (df["close"] - df["ema_200"]) / (df["ema_200"] + 1e-9)

    df["adx_14"] = _adx(df, 14)

    df["roll_skew_20"] = df["log_ret"].rolling(20).skew()
    df["roll_kurt_20"] = df["log_ret"].rolling(20).kurt()

    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    log.info(f"  Output: {len(df.columns)} features, {len(df)} rows  (dropped {dropped} NaN rows)")
    return df


def _rsi(s, p):
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def _adx(df, p):
    pdm = (df["high"] - df["high"].shift(1)).clip(lower=0)
    ndm = (df["low"].shift(1) - df["low"]).clip(lower=0)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(com=p - 1, adjust=False).mean()
    pdi = 100 * pdm.ewm(com=p - 1, adjust=False).mean() / (atr + 1e-9)
    ndi = 100 * ndm.ewm(com=p - 1, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(com=p - 1, adjust=False).mean()
