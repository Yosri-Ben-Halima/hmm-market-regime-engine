"""Per-regime Student-t MLE + RiskMetrics EWMA volatility model."""

import logging

import numpy as np
from scipy import stats

from ..config import REGIME_LABELS, EWMA_LAMBDA

log = logging.getLogger(__name__)


def fit_vol_model_per_regime(df):
    """
    Per-regime Student-t MLE + RiskMetrics EWMA.

    For each regime we:
      1. Fit a Student-t distribution via MLE to get (mu, scale, nu) — honest tail estimation.
      2. Estimate a RiskMetrics EWMA variance series over ALL returns (lambda=0.94),
         then compute the regime-conditional median EWMA vol as a time-varying vol anchor.

    Returns a dict keyed by regime int (0/1/2):
      {
        "nu":         float   – Student-t degrees of freedom (tail thickness)
        "t_loc":      float   – fitted t location (daily, same units as log_ret)
        "t_scale":    float   – fitted t scale   (daily, same units as log_ret)
        "ewma_vol":   float   – median annualised EWMA vol for this regime (used in MC)
        "emp_vol":    float   – empirical std (kept for diagnostics / fallback)
      }
    """
    log.info("Fitting per-regime Student-t MLE + RiskMetrics EWMA vol…")

    LAMBDA = EWMA_LAMBDA
    rets = df["log_ret"].values
    ewma_var = np.zeros(len(rets))
    ewma_var[0] = rets[0] ** 2
    for t in range(1, len(rets)):
        ewma_var[t] = LAMBDA * ewma_var[t - 1] + (1 - LAMBDA) * rets[t - 1] ** 2
    ewma_vol_daily = np.sqrt(ewma_var)
    df = df.copy()
    df["ewma_vol_daily"] = ewma_vol_daily

    results = {}
    for k in range(3):
        mask = df["regime"] == k
        r = df.loc[mask, "log_ret"].values
        ev = df.loc[mask, "ewma_vol_daily"].values

        if len(r) < 30:
            log.warning(
                f"  {REGIME_LABELS[k]}: too few observations ({len(r)}), using fallback"
            )
            results[k] = {
                "nu": 8.0,
                "t_loc": r.mean() if len(r) else 0.0,
                "t_scale": r.std() if len(r) else 0.01,
                "ewma_vol": r.std() * np.sqrt(252) * 100 if len(r) else 15.0,
                "emp_vol": r.std() * np.sqrt(252) * 100 if len(r) else 15.0,
            }
            continue

        try:
            nu, t_loc, t_scale = stats.t.fit(r)
            nu = float(np.clip(nu, 3.0, 50.0))
            t_scale = float(max(t_scale, 1e-6))
        except Exception as e:
            log.warning(
                f"  {REGIME_LABELS[k]}: Student-t MLE failed ({e}), using moments"
            )
            nu = 8.0
            t_loc = float(r.mean())
            t_scale = float(r.std())

        emp_vol_ann = float(r.std() * np.sqrt(252) * 100)

        ewma_vol_ann = float(np.median(ev) * np.sqrt(252) * 100)
        ewma_vol_ann = max(ewma_vol_ann, emp_vol_ann * 0.20)

        results[k] = {
            "nu": nu,
            "t_loc": float(t_loc),
            "t_scale": t_scale,
            "ewma_vol": ewma_vol_ann,
            "emp_vol": emp_vol_ann,
        }

        log.info(
            f"  {REGIME_LABELS[k]:10s}:  ν={nu:.2f}  "
            f"t_loc={t_loc * 252 * 100:+.2f}%/yr  "
            f"t_scale={t_scale * np.sqrt(252) * 100:.2f}%/yr  "
            f"EWMA_vol={ewma_vol_ann:.2f}%/yr  "
            f"emp_vol={emp_vol_ann:.2f}%/yr  "
            f"n={mask.sum()}"
        )

    return results, df
