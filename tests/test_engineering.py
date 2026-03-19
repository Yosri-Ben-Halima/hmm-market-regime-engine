import numpy as np

from regime_engine.config import HMM_FEATURES
from regime_engine.features.engineering import _rsi, _adx


def test_hmm_features_present(featured_df):
    for col in HMM_FEATURES:
        assert col in featured_df.columns, f"Missing HMM feature: {col}"


def test_no_nan_in_hmm_features(featured_df):
    for col in HMM_FEATURES:
        assert not featured_df[col].isna().any(), f"NaN in {col}"


def test_rsi_bounds(raw_df):
    rsi = _rsi(raw_df["close"], 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_adx_non_negative(raw_df):
    adx = _adx(raw_df, 14).dropna()
    assert (adx >= 0).all()
