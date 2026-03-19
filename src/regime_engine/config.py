"""Centralised constants — zero imports from sibling modules."""

REGIME_LABELS = {0: "Bull/Trend", 1: "Transition", 2: "Crisis/Vol"}
REGIME_COLORS = {0: "#22c55e", 1: "#facc15", 2: "#ef4444"}

HMM_FEATURES = [
    "log_ret",
    "rvol_20",
    "vol_ratio",
    "bb_pct_20",
    "rsi_14",
    "adx_14",
    "dist_ema200",
    "roll_skew_20",
    "volume_ratio",
]

EWMA_LAMBDA = 0.94
HMM_N_STATES = 3
HMM_N_ITER = 300
HMM_N_INIT = 15
MC_DEFAULT_PATHS = 2000
MC_DEFAULT_HORIZON = 21
LOOKBACK_DEFAULT = 2520

# Palette
BG = "#0d1117"
GRID = "#1f2937"
TXT = "#e5e7eb"
BLUE = "#3b82f6"
