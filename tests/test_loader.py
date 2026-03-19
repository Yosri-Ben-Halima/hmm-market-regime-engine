import numpy as np


def test_columns(raw_df):
    required = {"open", "high", "low", "close", "volume", "log_ret", "simple_ret"}
    assert required.issubset(set(raw_df.columns))


def test_length(raw_df):
    assert len(raw_df) <= 500


def test_log_ret_no_nan_after_first_row(raw_df):
    assert not raw_df["log_ret"].isna().any()
