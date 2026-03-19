import numpy as np

from regime_engine.models.vol import fit_vol_model_per_regime


def test_returns_dict_and_df(regime_df):
    results, df = fit_vol_model_per_regime(regime_df)
    assert isinstance(results, dict)
    assert set(results.keys()) == {0, 1, 2}


def test_regime_entry_keys(regime_df):
    results, _ = fit_vol_model_per_regime(regime_df)
    for k in range(3):
        entry = results[k]
        for key in ["nu", "t_loc", "t_scale", "ewma_vol", "emp_vol"]:
            assert key in entry, f"Missing key {key} in regime {k}"


def test_nu_bounds(regime_df):
    results, _ = fit_vol_model_per_regime(regime_df)
    for k in range(3):
        assert 3.0 <= results[k]["nu"] <= 50.0


def test_ewma_vol_daily_column(regime_df):
    _, df = fit_vol_model_per_regime(regime_df)
    assert "ewma_vol_daily" in df.columns
    assert (df["ewma_vol_daily"] > 0).all()


def test_ewma_vol_varies(regime_df):
    """EWMA series should not be constant — it varies with returns."""
    _, df = fit_vol_model_per_regime(regime_df)
    assert df["ewma_vol_daily"].std() > 0
