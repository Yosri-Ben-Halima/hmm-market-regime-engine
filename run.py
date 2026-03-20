import logging
import warnings
from src.regime_engine import run

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)


results = run(
    ticker="^GSPC", lookback=2520, n_paths=2000, horizon=21, out_dir="outputs"
)
