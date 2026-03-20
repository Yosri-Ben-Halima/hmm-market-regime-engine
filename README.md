# HMM Market Regime Engine

## Overview

A quantitative research tool that detects market regimes in equity index data using a Gaussian Hidden Markov Model (HMM) with 3 states (Bull/Trend, Transition, Crisis/Vol), then forecasts forward price paths via regime-switching Monte Carlo simulation.

The volatility model uses **RiskMetrics EWMA (λ = 0.94)** for time-varying volatility combined with **per-regime Student-t MLE** for honest tail estimation.

**Output artifacts:**

- A multi-panel institutional tear sheet (PNG) covering regime classification, volatility, risk analytics, Monte Carlo fan charts, and regime statistics.
- A console summary with current regime probabilities, forecast bands, and risk metrics.

## Architecture

```bash
OHLCV Data → Feature Engineering → HMM Regime Detection & Interpretation
    → Student-t MLE + EWMA Vol → Monte Carlo Forecast
    → Risk Analytics → Tear Sheet
```

## Mathematical Model

### 1. Log Returns

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Where $P_t$ is the closing price at time $t$.

### 2. Hidden Markov Model

The market is modelled as a latent discrete-state process $S_t \in \{0, 1, 2\}$ with:

- **Transition matrix** $A$, where $a_{ij} = P(S_t = j \mid S_{t-1} = i)$
- **Initial state distribution** $\pi_k = P(S_0 = k)$
- **Emission model** $B$: Gaussian with per-state mean $\mu_k$ and covariance $\Sigma_k$:

$$P(\mathbf{x}_t \mid S_t = k) = \mathcal{N}(\mathbf{x}_t;\, \mu_k,\, \Sigma_k)$$

where $\mathbf{x}_t$ is the feature vector at time $t$ (9 features after PCA whitening retaining 95% variance).

The model is fitted via Expectation–Maximisation (Baum–Welch). State decoding uses the Viterbi algorithm. Posterior state probabilities are computed via the forward–backward algorithm in log-space with a 2% floor to prevent overconfident assignments.

### 3. Regime Labelling

Raw HMM states are relabelled 0 (Bull) → 1 (Transition) → 2 (Crisis) by sorting on a composite key:

$$\text{key}(k) = \left(\text{Sharpe}_k^{\text{daily}},\; -|\text{MaxDD}_k|\right)$$

**Daily Sharpe:**

$$\text{Sharpe}_k^{\text{daily}} = \frac{\bar{r}_k - r_f^{\text{daily}}}{\sigma_k}$$

**Max drawdown:**

$$\text{MaxDD}_k = \min_t \left(\sum_{i=1}^{t} r_i - \max_{j \le t} \sum_{i=1}^{j} r_i\right), \quad \forall\, t : S_t = k$$

### 4. Volatility Model — RiskMetrics EWMA + Student-t MLE

**EWMA variance recursion** (applied over the full return series):

$$\sigma^2_t = \lambda\, \sigma^2_{t-1} + (1 - \lambda)\, r_{t-1}^2, \quad \lambda = 0.94$$

**Student-t MLE per regime** (tail estimation only):

$$\hat{\nu}_k,\, \hat{\mu}_k,\, \hat{s}_k = \arg\max_{\nu,\mu,s}\, \sum_{t:\, S_t=k} \log\, t_\nu\!\left(\frac{r_t - \mu}{s}\right)$$

where $\nu_k$ is clipped to $[3, 50]$ for numerical stability. The per-regime EWMA anchor is the median annualised EWMA vol conditional on regime $k$.

### 5. Monte Carlo Regime-Switching GBM

Each simulated path maintains its own EWMA variance state. Per time step:

**EWMA update:**

$$\sigma^2_t = \lambda\,\sigma^2_{t-1} + (1-\lambda)\,\varepsilon^2_{t-1}$$

**Fat-tail shock generation:**

$$z_t \sim t_{\nu_k} \Big/ \sqrt{\nu_k / (\nu_k - 2)}, \quad \varepsilon_t = \sigma_t\, z_t$$

**Price evolution (Itô-corrected GBM):**

$$P_{t+1} = P_t \exp\left(\mu_k - \tfrac{1}{2}\sigma_t^2 + \varepsilon_t\right)$$

**Regime-switch vol blending** (on transition from regime $k$ to $k'$):

$$\sigma^2_t \leftarrow 0.70\,\sigma^2_{t-1} + 0.30\,\bar{\sigma}^2_{k'}$$

where $\bar{\sigma}^2_{k'}$ is the EWMA anchor for the new regime. This prevents discontinuous vol jumps at regime boundaries.

The transition matrix is a 60/40 blend of the HMM-learned matrix and the empirical matrix.

### 6. Risk Metrics

**Value at Risk (95%):**

$$\text{VaR}_{95} = \text{Percentile}(r, 5\%)$$

**Conditional VaR (Expected Shortfall):**

$$\text{CVaR}_{95} = \mathbb{E}[r \mid r \le \text{VaR}_{95}]$$

**Max Drawdown:**

$$\text{MaxDD} = \min_t \frac{P_t - \max_{j \le t} P_j}{\max_{j \le t} P_j}$$

**Sharpe Ratio:**

$$\text{Sharpe} = \frac{\mu_{\text{ann}}}{\sigma_{\text{ann}}}$$

**Sortino Ratio:**

$$\text{Sortino} = \frac{\mu_{\text{ann}}}{\sigma_{\text{downside, ann}}}$$

**Calmar Ratio:**

$$\text{Calmar} = \frac{\mu_{\text{ann}}}{|\text{MaxDD}|}$$

## Repo Walkthrough

| File | Responsibility |
| ---- | ------------- |
| `src/regime_engine/config.py` | All constants: regime labels/colours, HMM hyperparameters, palette |
| `src/regime_engine/data/loader.py` | `fetch_ohlcv()` — yfinance data acquisition, log/simple returns |
| `src/regime_engine/features/engineering.py` | `add_features()`, `_rsi()`, `_adx()` — 20+ OHLCV-derived signals |
| `src/regime_engine/models/hmm.py` | `fit_hmm()`, `_fetch_risk_free_rate()`, `transition_matrix()`, `stationary_dist()` |
| `src/regime_engine/models/vol.py` | `fit_vol_model_per_regime()` — Student-t MLE + EWMA per regime |
| `src/regime_engine/simulation/monte_carlo.py` | `monte_carlo_forecast()` — regime-switching GBM with EWMA vol |
| `src/regime_engine/analytics/risk.py` | `compute_risk()` — VaR, CVaR, Sharpe, Sortino, drawdown, etc. |
| `src/regime_engine/visualization/tearsheet.py` | `build_tearsheet()` — 7-row institutional PNG report |
| `src/regime_engine/pipeline.py` | `run()` — orchestrates all modules end-to-end |
| `src/regime_engine/cli.py` | `main()` — argparse CLI entry point |
| `tests/conftest.py` | Shared fixtures: `raw_df`, `featured_df`, `regime_df` |
| `tests/test_*.py` | Unit tests for each module |

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/Yosri-Ben-Halima/hmm-market-regime-engine.git
cd hmm-market-regime-engine
pip install -e ".[dev]"
```

## Usage

### Quick Start

```bash
task run
```

### CLI

```bash
# Defaults: ^GSPC, 2520-day lookback, 2000 paths, 21-day horizon
regime-engine

# Custom parameters
regime-engine --ticker SPY --lookback 1260 --n_paths 5000 --horizon 63 --out_dir ./outputs
```

### Python API

```python
from regime_engine import run

results = run(ticker="^GSPC", lookback=1260, n_paths=2000, horizon=21, out_dir="outputs")

# results keys: df, model, forecast, risk, tearsheet
```

## Running Tests

```bash
task test
task test-cov

# Or directly
pytest
pytest --cov=regime_engine
pytest tests/test_vol.py -v
```

## Output

The tear sheet PNG contains 7 rows of panels:

1. **KPI Banner** — Last close, annualised return/vol, Sharpe, Sortino, max drawdown, VaR/CVaR, skewness, kurtosis, active regime, confidence.
2. **Price + Regime Overlay** — Close with EMA50/EMA200 and colour-coded regime background shading.
3. **Posterior Probabilities + Volatility** — Stacked regime posteriors from HMM; EWMA vol (λ=0.94) vs 20D/60D realised vol so users can see time-varying vol responsiveness.
4. **Monte Carlo Forecast** — Fan chart with percentile bands; forecast regime occupancy; terminal return distribution with fitted normal overlay.
5. **Distributional Analysis** — Daily return histogram with VaR overlay, Q-Q plot, regime scatter (vol vs return), empirical transition matrix heatmap.
6. **Regime Summary Table** — Per-regime frequency, annualised return/vol, Sharpe, VaR, current posterior.

## Disclaimer

This tool is for **quantitative research and educational purposes only**. It is not financial advice. Past regime classifications and forward simulations do not guarantee future market behaviour. Use at your own risk.
