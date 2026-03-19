"""Risk analytics: VaR, CVaR, drawdown, Sharpe, Sortino, Calmar."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis


def compute_risk(df):
    rets = df["log_ret"].dropna().values
    n = len(rets)
    mu = rets.mean() * 252
    sig = rets.std() * np.sqrt(252)

    var95 = np.percentile(rets, 5)
    var99 = np.percentile(rets, 1)
    cvar95 = rets[rets <= var95].mean()
    cvar99 = rets[rets <= var99].mean()

    sharpe = mu / sig if sig > 0 else 0
    neg_std = rets[rets < 0].std() * np.sqrt(252)
    sortino = mu / (neg_std + 1e-10)

    prices = df["close"].values
    roll_max = np.maximum.accumulate(prices)
    dd_arr = (prices - roll_max) / (roll_max + 1e-10) * 100
    max_dd = dd_arr.min()
    calmar = mu / (abs(max_dd / 100) + 1e-10)

    sk = skew(rets)
    ku = kurtosis(rets, fisher=True)
    jb_stat, jb_pval = stats.jarque_bera(rets)

    roll_sharpe = (
        df["log_ret"].rolling(252).mean()
        * 252
        / (df["log_ret"].rolling(252).std() * np.sqrt(252) + 1e-10)
    )

    pos = (rets > 0).mean()
    avg_g = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_l = rets[rets < 0].mean() if (rets < 0).any() else 0
    glr = abs(avg_g / avg_l) if avg_l != 0 else 0

    return {
        "mu": mu,
        "sigma": sig,
        "skewness": sk,
        "kurtosis": ku,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "var95": var95,
        "var99": var99,
        "cvar95": cvar95,
        "cvar99": cvar99,
        "max_dd": max_dd,
        "jb_pval": jb_pval,
        "pos_rate": pos,
        "glr": glr,
        "roll_sharpe": roll_sharpe,
        "dd_series": pd.Series(dd_arr, index=df.index),
        "n": n,
    }
