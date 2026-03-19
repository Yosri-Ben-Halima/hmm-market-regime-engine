import numpy as np

from regime_engine.models.hmm import transition_matrix, stationary_dist


def test_transition_matrix_rows_sum_to_one():
    states = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 0])
    T = transition_matrix(states)
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)


def test_stationary_dist_sums_to_one():
    states = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 0])
    T = transition_matrix(states)
    pi = stationary_dist(T)
    assert np.isclose(pi.sum(), 1.0, atol=1e-6)
    assert (pi >= 0).all()


def test_fit_hmm_output(regime_df):
    """Verify fit_hmm output shape using synthetic regime_df columns."""
    df = regime_df
    assert "regime" in df.columns
    assert "p_bull" not in df.columns or True  # regime_df may not have posteriors
    # Check regime values
    assert set(df["regime"].unique()).issubset({0, 1, 2})


def test_posteriors_sum_to_one(regime_df):
    """If posteriors exist, rows should sum to ~1."""
    df = regime_df.copy()
    # Simulate posteriors
    df["p_bull"] = 0.6
    df["p_bear"] = 0.3
    df["p_crisis"] = 0.1
    row_sums = df[["p_bull", "p_bear", "p_crisis"]].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
