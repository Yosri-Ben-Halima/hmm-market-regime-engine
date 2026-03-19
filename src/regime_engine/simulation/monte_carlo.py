"""Regime-switching Monte Carlo GBM with EWMA vol and Student-t shocks."""

import logging

import numpy as np

from ..config import REGIME_LABELS, EWMA_LAMBDA, MC_DEFAULT_PATHS, MC_DEFAULT_HORIZON
from ..models.hmm import transition_matrix

log = logging.getLogger(__name__)


def monte_carlo_forecast(
    df, hmm_model, n_paths=MC_DEFAULT_PATHS, vol_params=None, horizon=MC_DEFAULT_HORIZON, seed=42
):
    """
    Regime-switching GBM with:
      - Time-varying vol: per-step EWMA updated each day using λ=0.94
      - Fat-tail shocks:  Student-t(ν) standardised to unit variance
      - Drift:            empirical regime mean (annualised, scaled to daily)

    Vol model (RiskMetrics):
      σ²_t = λ·σ²_{t-1} + (1-λ)·ε²_{t-1}
      ε_t  = σ_t · z_t,   z_t ~ t_ν / √(ν/(ν-2))
    """
    log.info(
        f"Monte Carlo: {n_paths} paths × {horizon}D  [EWMA vol + Student-t shocks]"
    )
    rng = np.random.default_rng(seed)
    LAMBDA = EWMA_LAMBDA

    params = {}
    for k in range(3):
        mask = df["regime"] == k
        mu = df.loc[mask, "log_ret"].mean()
        params[k] = mu
        ann_mu = mu * 252 * 100
        log.info(f"  {REGIME_LABELS[k]}: drift μ={ann_mu:+.2f}%/yr")

    regime_nu = {}
    regime_ewma_var0 = {}
    for k in range(3):
        vp = (vol_params or {}).get(k, {})
        nu = float(vp.get("nu", 8.0))
        regime_nu[k] = float(np.clip(nu, 3.0, 50.0))
        ewma_vol_ann_pct = vp.get("ewma_vol", 15.0)
        ewma_vol_daily = (ewma_vol_ann_pct / 100.0) / np.sqrt(252)
        regime_ewma_var0[k] = ewma_vol_daily**2

    log.info(
        "  Shock dof (ν): "
        + "  ".join(f"{REGIME_LABELS[k]}={regime_nu[k]:.1f}" for k in range(3))
    )
    log.info(
        "  EWMA starting vol (ann %): "
        + "  ".join(
            f"{REGIME_LABELS[k]}={np.sqrt(regime_ewma_var0[k] * 252) * 100:.2f}%"
            for k in range(3)
        )
    )

    emp = transition_matrix(df["regime"].values)
    blended = 0.6 * hmm_model.transmat_ + 0.4 * emp
    blended /= blended.sum(axis=1, keepdims=True)

    cur_reg = int(df["regime"].iloc[-1])
    cur_price = float(df["close"].iloc[-1])
    cur_ewma_var = float(df["ewma_vol_daily"].iloc[-1] ** 2)

    log.info(f"  Starting regime: {REGIME_LABELS[cur_reg]}  price: ${cur_price:.2f}")
    log.info(
        f"  Current EWMA daily vol: {np.sqrt(cur_ewma_var) * np.sqrt(252) * 100:.2f}%/yr"
    )

    paths = np.zeros((n_paths, horizon + 1))
    reg_mat = np.zeros((n_paths, horizon + 1), dtype=np.int8)
    paths[:, 0] = cur_price
    reg_mat[:, 0] = cur_reg

    for s in range(n_paths):
        reg = cur_reg
        price = cur_price
        h = cur_ewma_var

        for d in range(1, horizon + 1):
            new_reg = int(rng.choice(3, p=blended[reg]))
            mu = params[new_reg]
            nu = regime_nu[new_reg]

            if new_reg != reg:
                h = 0.70 * h + 0.30 * regime_ewma_var0[new_reg]
            h = LAMBDA * h

            z = rng.standard_t(df=nu)
            z /= np.sqrt(nu / (nu - 2.0 + 1e-10))

            sig_daily = np.sqrt(max(h, 1e-8))
            eps = sig_daily * z
            ret = mu - 0.5 * sig_daily**2 + eps

            h += (1 - LAMBDA) * eps**2

            h = float(np.clip(h, 1e-8, (0.05) ** 2))

            price *= np.exp(ret)
            paths[s, d] = price
            reg_mat[s, d] = new_reg
            reg = new_reg

    final_prices = paths[:, -1]
    explosion_pct = (final_prices > cur_price * 2.0).mean() * 100
    wipeout_pct = (final_prices < cur_price * 0.5).mean() * 100
    if explosion_pct > 0.5:
        log.warning(f"  ⚠ {explosion_pct:.1f}% of paths >2× start price")
    if wipeout_pct > 0.5:
        log.warning(f"  ⚠ {wipeout_pct:.1f}% of paths <50% start price")
    log.info(
        f"  Path sanity: explosion={explosion_pct:.2f}%  wipeout={wipeout_pct:.2f}%"
    )

    pct = {
        p: np.percentile(paths, p, axis=0) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    pct["mean"] = paths.mean(axis=0)

    term_rets = paths[:, -1] / cur_price - 1
    reg_occ = np.stack([(reg_mat == k).mean(axis=0) for k in range(3)], axis=1)

    log.info(
        f"  Terminal return ({horizon}D):  "
        f"median={np.median(term_rets) * 100:+.2f}%  "
        f"mean={term_rets.mean() * 100:+.2f}%  "
        f"p5={np.percentile(term_rets, 5) * 100:.2f}%  "
        f"p95={np.percentile(term_rets, 95) * 100:.2f}%  "
        f"σ={term_rets.std() * 100:.2f}%"
    )

    return {
        "bands": pct,
        "terminal_rets": term_rets,
        "regime_occ": reg_occ,
        "trans_mat": blended,
        "params": params,
        "current_price": cur_price,
        "current_regime": cur_reg,
        "horizon": horizon,
        "n_paths": n_paths,
        "paths": paths,
        "regime_nu": regime_nu,
    }
