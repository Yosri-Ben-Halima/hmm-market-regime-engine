from regime_engine.analytics.risk import compute_risk


def test_output_keys(featured_df):
    rm = compute_risk(featured_df)
    required = {
        "mu", "sigma", "sharpe", "sortino", "max_dd",
        "var95", "cvar95", "skewness", "kurtosis",
    }
    assert required.issubset(set(rm.keys()))


def test_max_dd_non_positive(featured_df):
    rm = compute_risk(featured_df)
    assert rm["max_dd"] <= 0


def test_var95_negative(featured_df):
    rm = compute_risk(featured_df)
    assert rm["var95"] < 0


def test_sigma_positive(featured_df):
    rm = compute_risk(featured_df)
    assert rm["sigma"] > 0
