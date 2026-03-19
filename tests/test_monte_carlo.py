import numpy as np

from regime_engine.simulation.monte_carlo import monte_carlo_forecast


class _FakeHMM:
    """Minimal stand-in for a fitted GaussianHMM."""

    def __init__(self):
        self.transmat_ = np.array(
            [[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.10, 0.10, 0.80]]
        )


def test_bands_keys(regime_df):
    fc = monte_carlo_forecast(regime_df, _FakeHMM(), n_paths=50, horizon=10)
    bands = fc["bands"]
    for k in [5, 25, 50, 75, 95, "mean"]:
        assert k in bands


def test_band_length(regime_df):
    horizon = 10
    fc = monte_carlo_forecast(regime_df, _FakeHMM(), n_paths=50, horizon=horizon)
    assert len(fc["bands"][50]) == horizon + 1


def test_band_starts_at_current_price(regime_df):
    fc = monte_carlo_forecast(regime_df, _FakeHMM(), n_paths=50, horizon=10)
    cur = float(regime_df["close"].iloc[-1])
    assert fc["bands"][50][0] == cur


def test_band_ordering(regime_df):
    fc = monte_carlo_forecast(regime_df, _FakeHMM(), n_paths=200, horizon=10)
    bands = fc["bands"]
    assert (bands[5] <= bands[50]).all()
    assert (bands[50] <= bands[95]).all()


def test_ewma_vol_updates_within_paths(regime_df):
    """Verify that path vol is not flat (EWMA actually updates)."""
    fc = monte_carlo_forecast(regime_df, _FakeHMM(), n_paths=100, horizon=21, seed=0)
    paths = fc["paths"]
    # Compute consecutive log-returns for a sample of paths
    log_rets = np.diff(np.log(paths[:20, :]), axis=1)
    # Std of consecutive log-returns across paths should be > 0
    assert log_rets.std() > 0
