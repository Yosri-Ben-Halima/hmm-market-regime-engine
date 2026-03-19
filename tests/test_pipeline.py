import os
import tempfile
from unittest.mock import patch

import numpy as np

from regime_engine.pipeline import run


def test_smoke(regime_df):
    """Mock fetch_ohlcv to return fixture data, call run() end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("regime_engine.pipeline.fetch_ohlcv", return_value=regime_df.drop(columns=["regime", "ewma_vol_daily"])):
            result = run(ticker="TEST", lookback=200, n_paths=50, horizon=5, out_dir=tmpdir)

        assert set(result.keys()) == {"df", "model", "forecast", "risk", "tearsheet"}

        # tearsheet PNG exists on disk
        assert os.path.exists(result["tearsheet"])

        # df has ewma_vol_daily (confirms vol model ran)
        assert "ewma_vol_daily" in result["df"].columns

        # forecast regime_nu is a dict with keys 0, 1, 2
        assert isinstance(result["forecast"]["regime_nu"], dict)
        assert set(result["forecast"]["regime_nu"].keys()) == {0, 1, 2}
