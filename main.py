# ======================================================================
# TP FINAL - Single clean cell (ready to submit)
# - Cleaning + Cov estimators (Sample / RMT / OAS / QIS / AO)
# - Optimizers (IVP / GMV / ERC)
# - Backtest daily or every-k-days, with transaction costs + turnover
# - Plots with filtering by strategy names
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
from sklearn.covariance import OAS

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==============================================================================
# 5) MAIN (one organized experiment)
# ==============================================================================
def main():
    # --- Dataset params ---
    CSV_PATH = "WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"
    IN_SAMPLE_DAYS = 1000
    OOS_DAYS = 500
    N_ASSETS = 15

    # --- Backtest params ---
    OPT_WINDOW = 252
    CAL_WINDOW = 1000     # critical for AO
    COST_BPS = 5

    # --- Load data ---
    cleaner = Cleaning(CSV_PATH, in_sample_days=IN_SAMPLE_DAYS, oos_days=OOS_DAYS, N=N_ASSETS)
    returns_df, split_point = cleaner.load_and_prepare()
    if returns_df is None:
        return

    bt = Backtest(
        returns_df,
        split_idx=split_point,
        window_size=OPT_WINDOW,
        cal_window_size=CAL_WINDOW,
        cost_bps=COST_BPS
    )

    # =========================================================
    # A) BASELINES (daily rebalance)
    # =========================================================
    bt.run("IVP (Benchmark)", "ivp", cov_method="sample", use_mu=False)

    # GMV (MVO with mu=0 -> GMV)
    bt.run("GMV (Sample)", "mvo", cov_method="sample", use_mu=False)
    bt.run("GMV (RMT)", "mvo", cov_method="rmt", use_mu=False)
    bt.run("GMV (OAS)", "mvo", cov_method="oas", use_mu=False)
    bt.run("GMV (QIS)", "mvo", cov_method="qis", use_mu=False)
    bt.run("GMV (AO)",  "mvo", cov_method="ao",  use_mu=False)

    # ERC
    bt.run("ERC (Sample)", "erc", cov_method="sample", use_mu=False)
    bt.run("ERC (RMT)",    "erc", cov_method="rmt",    use_mu=False)
    bt.run("ERC (OAS)",    "erc", cov_method="oas",    use_mu=False)
    bt.run("ERC (QIS)",    "erc", cov_method="qis",    use_mu=False)
    bt.run("ERC (AO)",     "erc", cov_method="ao",     use_mu=False)

    # Quick view of core strategies
    bt.plot_separate(names=[
        "IVP (Benchmark)",
        "GMV (Sample)", "GMV (RMT)", "GMV (OAS)", "GMV (QIS)", "GMV (AO)",
        "ERC (Sample)", "ERC (RMT)", "ERC (OAS)", "ERC (QIS)", "ERC (AO)",
    ])

    # =========================================================
    # B) REBALANCING STUDY (example: k=10)
    # =========================================================
    bt.run_rebal_k("GMV (Sample, Rebal 10d)", "mvo", k=10, cov_method="sample", use_mu=False)
    bt.run_rebal_k("ERC (RMT, Rebal 10d)",   "erc", k=10, cov_method="rmt",    use_mu=False)
    bt.run_rebal_k("ERC (AO, Rebal 10d)",    "erc", k=10, cov_method="ao",     use_mu=False)

    bt.plot_separate(names=[
        "GMV (Sample, Rebal 10d)",
        "ERC (RMT, Rebal 10d)",
        "ERC (AO, Rebal 10d)",
    ])

    # =========================================================
    # C) FOCUSED COMPARISON: ERC (RMT) vs ERC (QIS) vs ERC (AO)
    # =========================================================
    bt.plot_separate(names=["ERC (RMT)", "ERC (QIS)", "ERC (AO)"])
