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
# 4) BACKTEST
# ==============================================================================
class Backtest:
    def __init__(self, returns_df, split_idx, window_size=252, cal_window_size=1000, cost_bps=5):
        self.returns = returns_df
        self.start_idx = split_idx
        self.window = window_size            # optimization lookback (e.g., 252)
        self.cal_window = cal_window_size    # calibration lookback for AO (e.g., 1000)
        self.cost = cost_bps / 10000.0
        self.results = {}

    def _compute_sigma(self, t, cov_method, use_mu):
        # optimization window (always)
        data_opt = self.returns.iloc[t - self.window : t]
        mu = data_opt.mean().values * 252 if use_mu else np.zeros(data_opt.shape[1])

        # choose covariance estimator
        if cov_method == "sample":
            sigma = Sigma(data_opt).sample_cov()
        elif cov_method == "rmt":
            sigma = Sigma(data_opt).rmt_filtered_cov()
        elif cov_method == "oas":
            sigma = Sigma(data_opt).oas_cov()
        elif cov_method == "qis":
            sigma = Sigma(data_opt).qis_cov()
        elif cov_method == "ao":
            # IMPORTANT: AO needs long history -> use data_cal
            start_cal = max(0, t - self.cal_window)
            data_cal = self.returns.iloc[start_cal:t]
            sigma = Sigma(data_cal).ao_cov(cal_window=self.cal_window, block=self.window, n_blocks=2)
        else:
            sigma = Sigma(data_opt).sample_cov()

        return mu, sigma

    def _compute_weights(self, method, mu, sigma, w_fallback):
        opt = Optimizer(mu, sigma)
        if method == "ivp":
            return opt.inverse_variance()
        elif method == "mvo":
            return opt.mean_variance()
        elif method == "erc":
            return opt.risk_parity()
        return w_fallback

    def run(self, name, method, cov_method="sample", use_mu=False):
        print(f"> Lancement Backtest : {name}...")
        w_prev = np.zeros(self.returns.shape[1])

        pnl, dates, turnovers = [], [], []

        for i, t in enumerate(range(self.start_idx, len(self.returns))):
            if i % 50 == 0:
                print(f"   Traitement jour {i} / {len(self.returns) - self.start_idx}")

            try:
                mu, sigma = self._compute_sigma(t, cov_method, use_mu)
                w = self._compute_weights(method, mu, sigma, w_prev)
            except:
                w = w_prev

            ret_day = self.returns.iloc[t]
            turnover = float(np.sum(np.abs(w - w_prev)))
            net_ret = float(np.dot(w, ret_day) - turnover * self.cost)

            pnl.append(net_ret)
            turnovers.append(turnover)
            dates.append(self.returns.index[t])
            w_prev = w

        self.results[name] = {"returns": pd.Series(pnl, index=dates), "turnover": np.array(turnovers)}
        print(f"   -> {name} terminé.")

    def run_rebal_k(self, name, method, k=5, cov_method="sample", use_mu=False):
        print(f"> Lancement Backtest : {name} (rebal {k} jours)...")

        w_current = np.zeros(self.returns.shape[1])
        pnl, dates, turnovers = [], [], []

        for i, t in enumerate(range(self.start_idx, len(self.returns))):
            if i % 50 == 0:
                print(f"   Traitement jour {i} / {len(self.returns) - self.start_idx}")

            turnover = 0.0
            if i % k == 0:
                try:
                    mu, sigma = self._compute_sigma(t, cov_method, use_mu)
                    w_new = self._compute_weights(method, mu, sigma, w_current)
                except:
                    w_new = w_current

                turnover = float(np.sum(np.abs(w_new - w_current)))
                w_current = w_new

            ret_day = self.returns.iloc[t]
            net_ret = float(np.dot(w_current, ret_day) - turnover * self.cost)

            pnl.append(net_ret)
            turnovers.append(turnover)
            dates.append(self.returns.index[t])

        self.results[name] = {"returns": pd.Series(pnl, index=dates), "turnover": np.array(turnovers)}
        print(f"   -> {name} terminé.")

    def plot_separate(self, names=None):
        if len(self.results) == 0:
            print("Aucun résultat à afficher.")
            return

        if names is None:
            items = list(self.results.items())
        else:
            missing = [n for n in names if n not in self.results]
            if missing:
                print("Stratégies introuvables:", missing)
            items = [(n, self.results[n]) for n in names if n in self.results]

        if len(items) == 0:
            print("Aucun résultat à afficher (filtre vide).")
            return

        fig, axes = plt.subplots(nrows=len(items), ncols=1, figsize=(12, 4 * len(items)), sharex=True)
        if len(items) == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, (ax, (name, res)) in enumerate(zip(axes, items)):
            s = res["returns"]
            turnover = res["turnover"]

            cum = (1 + s).cumprod()
            cum.plot(ax=ax, color=colors[i % len(colors)], linewidth=2)

            ann_ret = s.mean() * 252
            ann_vol = s.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            avg_turnover = float(np.mean(turnover))

            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Valeur (Base 1)")
            ax.grid(True, linestyle="--", alpha=0.6)

            stats_text = (
                f"Rend: {ann_ret:.1%}\n"
                f"Vol: {ann_vol:.1%}\n"
                f"Sharpe: {sharpe:.2f}\n"
                f"Avg Turnover: {avg_turnover:.3f}"
            )
            ax.text(0.99, 0.05, stats_text, transform=ax.transAxes,
                    horizontalalignment="right", verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()




