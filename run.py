from src.regime_engine import run

results = run(ticker="^GSPC", lookback=2520, n_paths=2000, horizon=21, out_dir="outputs")
