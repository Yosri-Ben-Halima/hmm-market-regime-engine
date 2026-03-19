"""Gaussian HMM fitting, regime labelling, and transition utilities."""

import logging

import numpy as np
import yfinance as yf
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

from ..config import REGIME_LABELS, HMM_FEATURES, HMM_N_STATES, HMM_N_ITER, HMM_N_INIT

log = logging.getLogger(__name__)


def _fetch_risk_free_rate() -> float:
    def _extract_last_close(ticker: str) -> float:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            raise ValueError(f"No history returned for {ticker}")
        val = hist["Close"].dropna().iloc[-1]
        if hasattr(val, "item"):
            val = val.item()
        return float(val)

    sources = [
        ("^IRX", "13-Week T-Bill", 1 / 100),
        ("^FVX", "5-Year T-Note", 1 / 100),
        ("^TNX", "10-Year T-Note", 1 / 100),
    ]

    for ticker, label, scale in sources:
        try:
            raw = _extract_last_close(ticker)
            rate = raw * scale
            if 0.0 < rate < 0.25:
                log.info(
                    f"  Risk-free rate: {rate * 100:.3f}%/yr  (source: {label}  {ticker}  raw={raw:.4f})"
                )
                return rate
        except Exception as e:
            log.warning(f"  RF rate fetch failed for {ticker}: {e}")

    fallback = 0.0425
    log.warning(
        f"  All RF rate sources failed — using hardcoded fallback {fallback * 100:.2f}%/yr"
    )
    return fallback


def fit_hmm(df, n_states=HMM_N_STATES, n_iter=HMM_N_ITER, n_init=HMM_N_INIT):
    log.info(f"Fitting Gaussian HMM  K={n_states}  n_iter={n_iter}  n_init={n_init}")
    X = df[HMM_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    variance_retention = 0.95
    pca = PCA(n_components=variance_retention, whiten=True, random_state=42)
    Xs = pca.fit_transform(Xs)
    log.info(
        f"  PCA whitening: {X.shape[1]} features → {Xs.shape[1]} components ({variance_retention:.0%} variance retained)"
    )

    best, best_score = None, -np.inf
    for seed in range(n_init):
        m = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=1e-6,
            random_state=seed,
            init_params="stmc",
            params="stmc",
            transmat_prior=2.0,
            startprob_prior=2.0,
        )
        try:
            m.fit(Xs)
            s = m.score(Xs)
            if s > best_score:
                best_score, best = s, m
        except Exception:
            continue

    log.info(f"  Best log-likelihood: {best_score:.2f}")
    raw_states = best.predict(Xs)

    try:
        log_posteriors = best._compute_log_likelihood(Xs)
        log_transmat = np.log(best.transmat_ + 1e-300)
        log_startprob = np.log(best.startprob_ + 1e-300)

        T_, K_ = log_posteriors.shape
        log_alpha = np.zeros((T_, K_))
        log_alpha[0] = log_startprob + log_posteriors[0]
        for t in range(1, T_):
            for k in range(K_):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, k])
                    + log_posteriors[t, k]
                )

        log_beta = np.zeros((T_, K_))
        for t in range(T_ - 2, -1, -1):
            for k in range(K_):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_transmat[k, :] + log_posteriors[t + 1, :] + log_beta[t + 1, :]
                )

        log_gamma = log_alpha + log_beta
        log_norm = logsumexp(log_gamma, axis=1, keepdims=True)
        log_gamma_norm = log_gamma - log_norm
        EPS_LOG = np.log(0.02)
        log_gamma_norm = np.maximum(log_gamma_norm, EPS_LOG)
        log_norm2 = logsumexp(log_gamma_norm, axis=1, keepdims=True)
        posteriors = np.exp(log_gamma_norm - log_norm2)

    except Exception as e:
        log.warning(
            f"  Log-space FB failed ({e}), falling back to predict_proba with clip"
        )
        posteriors = best.predict_proba(Xs)
        EPS = 1e-4
        posteriors = np.clip(posteriors, EPS, 1.0 - EPS)
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

    RF_DAILY = _fetch_risk_free_rate() / 252

    def _regime_sort_key(k):
        mask = raw_states == k
        r = df["log_ret"].values[mask]
        if len(r) < 10:
            return (0.0, 0.0)
        mu = r.mean()
        sig = r.std() + 1e-10
        sharpe_daily = (mu - RF_DAILY) / sig
        cum = np.cumsum(r)
        roll_max = np.maximum.accumulate(cum)
        dd = cum - roll_max
        max_dd = dd.min()
        return (sharpe_daily, -abs(max_dd))

    scores = {k: _regime_sort_key(k) for k in range(n_states)}
    order = sorted(scores, key=scores.get, reverse=True)
    remap = {order[i]: i for i in range(n_states)}

    states = np.array([remap[s] for s in raw_states])
    posteriors = posteriors[:, order]

    df = df.copy()
    df["regime"] = states
    df["p_bull"] = posteriors[:, 0]
    df["p_bear"] = posteriors[:, 1]
    df["p_crisis"] = posteriors[:, 2]

    for k in range(n_states):
        mask = df["regime"] == k
        pct = mask.mean() * 100
        r_ = df.loc[mask, "log_ret"].values
        mu = r_.mean() * 252 * 100
        vol = r_.std() * np.sqrt(252) * 100
        sharpe_a = (r_.mean() - RF_DAILY) / (r_.std() + 1e-10) * np.sqrt(252)
        cum_ = np.cumsum(r_)
        roll_max_ = np.maximum.accumulate(cum_)
        max_dd_ = (cum_ - roll_max_).min() * 100
        log.info(
            f"  {REGIME_LABELS[k]:10s}: freq={pct:.1f}%  μ={mu:+.2f}%/yr  σ={vol:.2f}%/yr  "
            f"Sharpe={sharpe_a:+.2f}  MaxDD={max_dd_:.1f}%"
        )

    return best, df, posteriors, scaler, pca


def transition_matrix(states, n=3):
    T = np.zeros((n, n))
    for i in range(len(states) - 1):
        T[states[i], states[i + 1]] += 1
    return T / (T.sum(axis=1, keepdims=True) + 1e-10)


def stationary_dist(T):
    vals, vecs = np.linalg.eig(T.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return np.abs(pi) / np.abs(pi).sum()
