"""CLI entry point for the regime engine."""

import argparse
import warnings
import logging

from .config import LOOKBACK_DEFAULT, MC_DEFAULT_PATHS, MC_DEFAULT_HORIZON
from .pipeline import run

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    p = argparse.ArgumentParser(description="S&P 500 Regime Engine")
    p.add_argument("--ticker", default="^GSPC")
    p.add_argument("--lookback", default=LOOKBACK_DEFAULT, type=int)
    p.add_argument("--n_paths", default=MC_DEFAULT_PATHS, type=int)
    p.add_argument("--horizon", default=MC_DEFAULT_HORIZON, type=int)
    p.add_argument("--out_dir", default=".", help="Directory for output files")
    args = p.parse_args()
    run(args.ticker, args.lookback, args.n_paths, args.horizon, args.out_dir)


if __name__ == "__main__":
    main()
