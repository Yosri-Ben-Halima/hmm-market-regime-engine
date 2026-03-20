"""Orchestrates the full regime-detection pipeline."""

import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import REGIME_LABELS, BG, TXT
from .data import fetch_ohlcv
from .features import add_features
from .models import fit_hmm, fit_vol_model_per_regime
from .simulation import monte_carlo_forecast
from .analytics import compute_risk
from .visualization import build_tearsheet

log = logging.getLogger(__name__)


def _step(n, label, t0):
    elapsed = time.time() - t0
    log.info(f"[{n}/7] {label}  ({elapsed:.1f}s elapsed)")


def run(ticker="^GSPC", lookback=1260, n_paths=2000, horizon=21, out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("")
    log.info("═" * 60)
    log.info("  REGIME FORECASTING ENGINE")
    log.info(f"  ticker={ticker}  lookback={lookback}  paths={n_paths}  horizon={horizon}D")
    log.info(f"  vol=EWMA(λ=0.94)+Student-t  output={out}")
    log.info("═" * 60)

    _step(1, "Fetching OHLCV data", t0)
    df = fetch_ohlcv(ticker, lookback)

    _step(2, "Feature engineering", t0)
    df = add_features(df)

    _step(3, "Fitting HMM regime model", t0)
    model, df, _, scaler, pca = fit_hmm(df)

    _step(4, "Fitting per-regime volatility model", t0)
    vol_params, df = fit_vol_model_per_regime(df)

    _step(5, "Running Monte Carlo simulation", t0)
    fc = monte_carlo_forecast(
        df, model, n_paths, vol_params=vol_params, horizon=horizon
    )

    _step(6, "Computing risk analytics", t0)
    rm = compute_risk(df)

    _step(7, "Rendering tearsheet", t0)
    ts = build_tearsheet(
        df, fc, rm, ticker, out=str(out / f"{ticker.replace('^', '')}_tearsheet.png")
    )

    cur, cp = fc["current_regime"], df[["p_bull", "p_bear", "p_crisis"]].iloc[-1].values
    b = fc["bands"]
    P = fc["current_price"]

    entropy_cur = -np.sum(cp * np.log(cp + 1e-300))
    summary_text = (
        f"════════════════════════════════════════════════════════════\n"
        f"   REGIME FORECAST · {ticker} · {df.index[-1].date()}\n"
        f"════════════════════════════════════════════════════════════\n"
        f"\n"
        f" Price:                 ${P:>10.2f}\n"
        f" Regime:                {REGIME_LABELS[cur]}  (conf {cp[cur] * 100:>6.2f}%)\n"
        f"\n"
        f" Probabilities:\n"
        f"   Bull:                {cp[0] * 100:>8.4f}%\n"
        f"   Transition:          {cp[1] * 100:>8.4f}%\n"
        f"   Crisis:              {cp[2] * 100:>8.4f}%\n"
        f"\n"
        f" Uncertainty (Normalized Entropy): {entropy_cur / np.log(3):>8.2%}\n"
        f"   (0 = certain, 100% = uniform)\n"
        f"\n"
        f" 21-Day Forecast:\n"
        f"   Median:              ${b[50][-1]:>10.2f}  ({(b[50][-1] / P - 1) * 100:+6.2f}%)\n"
        f"   95th Percentile:     ${b[95][-1]:>10.2f}  ({(b[95][-1] / P - 1) * 100:+6.2f}%)\n"
        f"   5th Percentile:      ${b[5][-1]:>10.2f}  ({(b[5][-1] / P - 1) * 100:+6.2f}%)\n"
        f"\n"
        f" Vol Model:             EWMA (λ=0.94) + Student-t shocks\n"
        f" Regime ν (tail dof):\n"
        f"   Bull:                {vol_params[0]['nu']:.2f}\n"
        f"   Transition:          {vol_params[1]['nu']:.2f}\n"
        f"   Crisis:              {vol_params[2]['nu']:.2f}\n"
        f"\n"
        f" Risk Metrics:\n"
        f"   Sharpe:              {rm['sharpe']:>10.2f}\n"
        f"   Sortino:             {rm['sortino']:>10.2f}\n"
        f"   Max Drawdown:        {rm['max_dd']:>9.2f}%\n"
        f"   VaR (95%):           {rm['var95'] * 100:>9.2f}%\n"
        f"   Skewness:            {rm['skewness']:>10.3f}\n"
        f"   Kurtosis:            {rm['kurtosis']:>10.3f}\n"
        f"\n"
        f"════════════════════════════════════════════════════════════\n"
        f" Tear Sheet → {ts}\n"
    )

    # Render summary as image
    summary_path = str(out / f"{ticker.replace('^', '')}_summary.png")
    fig_s, ax_s = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax_s.axis("off")
    ax_s.text(
        0.03, 0.97, summary_text,
        transform=ax_s.transAxes,
        fontsize=11,
        fontfamily="monospace",
        color=TXT,
        verticalalignment="top",
    )
    plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig_s)

    total = time.time() - t0
    log.info("")
    log.info("═" * 60)
    log.info(f"  DONE  {ticker}  │  {total:.1f}s total")
    log.info(f"  tearsheet → {ts}")
    log.info(f"  summary   → {summary_path}")
    log.info("═" * 60)

    return {"df": df, "model": model, "forecast": fc, "risk": rm, "tearsheet": ts, "summary": summary_path}
