"""Orchestrates the full regime-detection pipeline."""

import logging
from pathlib import Path

import numpy as np

from .config import REGIME_LABELS
from .data import fetch_ohlcv
from .features import add_features
from .models import fit_hmm, fit_vol_model_per_regime
from .simulation import monte_carlo_forecast
from .analytics import compute_risk
from .visualization import build_tearsheet

log = logging.getLogger(__name__)


def run(ticker="^GSPC", lookback=1260, n_paths=2000, horizon=21, out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  S&P 500 REGIME FORECASTING ENGINE  [Vol: EWMA + Student-t]")
    log.info("═" * 60)

    df = fetch_ohlcv(ticker, lookback)
    df = add_features(df)
    model, df, _, scaler, pca = fit_hmm(df)
    vol_params, df = fit_vol_model_per_regime(df)
    fc = monte_carlo_forecast(
        df, model, n_paths, vol_params=vol_params, horizon=horizon
    )
    rm = compute_risk(df)
    ts = build_tearsheet(
        df, fc, rm, ticker, out=str(out / f"{ticker.replace('^', '')}_tearsheet.png")
    )

    cur, cp = fc["current_regime"], df[["p_bull", "p_bear", "p_crisis"]].iloc[-1].values
    b = fc["bands"]
    P = fc["current_price"]

    entropy_cur = -np.sum(cp * np.log(cp + 1e-300))
    print(f"""
════════════════════════════════════════════════════════════
   REGIME FORECAST · {ticker} · {df.index[-1].date()}
════════════════════════════════════════════════════════════

 Price:                 ${P:>10.2f}
 Regime:                {REGIME_LABELS[cur]}  (conf {cp[cur] * 100:>6.2f}%)

 Probabilities:
   Bull:                {cp[0] * 100:>8.4f}%
   Transition:          {cp[1] * 100:>8.4f}%
   Crisis:              {cp[2] * 100:>8.4f}%

 Uncertainty (Normalized Entropy): {entropy_cur / np.log(3):>8.2%}
   (0 = certain, 100% = uniform)

 21-Day Forecast:
   Median:              ${b[50][-1]:>10.2f}  ({(b[50][-1] / P - 1) * 100:+6.2f}%)
   95th Percentile:     ${b[95][-1]:>10.2f}  ({(b[95][-1] / P - 1) * 100:+6.2f}%)
   5th Percentile:      ${b[5][-1]:>10.2f}  ({(b[5][-1] / P - 1) * 100:+6.2f}%)

 Vol Model:             EWMA (λ=0.94) + Student-t shocks
 Regime ν (tail dof):
   Bull:                {vol_params[0]["nu"]:.2f}
   Transition:          {vol_params[1]["nu"]:.2f}
   Crisis:              {vol_params[2]["nu"]:.2f}

 Risk Metrics:
   Sharpe:              {rm["sharpe"]:>10.2f}
   Sortino:             {rm["sortino"]:>10.2f}
   Max Drawdown:        {rm["max_dd"]:>9.2f}%
   VaR (95%):           {rm["var95"] * 100:>9.2f}%
   Skewness:            {rm["skewness"]:>10.3f}
   Kurtosis:            {rm["kurtosis"]:>10.3f}

════════════════════════════════════════════════════════════
 Tear Sheet → {ts}
""")
    return {"df": df, "model": model, "forecast": fc, "risk": rm, "tearsheet": ts}
