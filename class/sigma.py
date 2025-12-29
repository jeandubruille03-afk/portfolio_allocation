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
# 2) SIGMA (Sample / RMT / OAS / QIS / AO)
# ==============================================================================
class Sigma:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.T, self.N = returns.shape

    def sample_cov(self):
        return self.returns.cov().values * 252

    def rmt_filtered_cov(self):
        # sample cov annualized
        S = self.returns.cov().values * 252

        # corr
        std = np.sqrt(np.diag(S))
        outer = np.outer(std, std)
        outer[outer == 0] = 1e-12
        C = S / outer

        # eigen
        evals, evecs = np.linalg.eigh(C)

        # MP threshold
        Q = self.T / self.N
        lambda_max = (1.0) * (1 + 1.0 / np.sqrt(Q)) ** 2

        # clip
        is_signal = evals > lambda_max
        if not np.any(is_signal):
            new_evals = np.full_like(evals, np.mean(evals))
        else:
            avg_noise = np.mean(evals[~is_signal])
            new_evals = evals.copy()
            new_evals[~is_signal] = avg_noise

        C_clean = evecs @ np.diag(new_evals) @ evecs.T
        C_clean = (C_clean + C_clean.T) / 2
        np.fill_diagonal(C_clean, 1.0)

        return C_clean * outer

    def oas_cov(self):
        X = self.returns.values  # (T, N)
        est = OAS(assume_centered=True).fit(X)
        return est.covariance_ * 252

    def qis_cov(self, eps=1e-6):
        """
        Light QIS-style nonlinear eigenvalue shrinkage on correlation matrix.
        (Kernel-based approximation; ok for small/medium N.)
        """
        S = self.returns.cov().values * 252
        std = np.sqrt(np.diag(S))
        outer = np.outer(std, std)
        outer[outer == 0] = 1e-12
        C = S / outer

        evals, evecs = np.linalg.eigh(C)
        l = np.maximum(evals, eps)

        N = len(l)
        T = self.T
        q = N / T  # < 1 typically

        # bandwidth
        h = N ** (-1 / 3)

        L = l.reshape(-1, 1)
        diff = (L - L.T) / (h + eps)
        K = np.exp(-0.5 * diff**2) / (np.sqrt(2 * np.pi) * (h + eps))
        f = np.mean(K, axis=1)

        H = np.mean((L - L.T) / ((L - L.T) ** 2 + (h**2)), axis=1)

        m_re = H
        m_im = -np.pi * f

        denom = (1 - q - q * l * m_re) ** 2 + (q * l * m_im) ** 2
        l_shr = l / np.maximum(denom, eps)

        # trace normalization for corr
        l_shr = l_shr * (N / np.sum(l_shr))

        C_clean = evecs @ np.diag(l_shr) @ evecs.T
        C_clean = (C_clean + C_clean.T) / 2
        np.fill_diagonal(C_clean, 1.0)

        return C_clean * outer

    def ao_cov(self, cal_window=1000, block=252, n_blocks=2, debug=False):
        """
        Average Oracle (AO) covariance estimator.
        IMPORTANT: must be called with a *long* history window (calibration window),
        not just the 252-day optimization window.
        """
        X = self.returns.values
        T, N = X.shape

        min_needed = (n_blocks + 1) * block
        if T < min_needed:
            if debug:
                print(f"AO fallback -> RMT: T={T} < min_needed={min_needed}")
            return self.rmt_filtered_cov()

        cal_T = min(cal_window, T)
        Xcal = X[-cal_T:, :]

        lambdas = []
        for b in range(n_blocks):
            end_next = cal_T - b * block
            start_next = end_next - block
            end_prev = start_next
            start_prev = end_prev - block

            X_prev = Xcal[start_prev:end_prev, :]
            X_next = Xcal[start_next:end_next, :]

            S_next = np.cov(X_next, rowvar=False) * 252
            S_prev = np.cov(X_prev, rowvar=False) * 252

            std = np.sqrt(np.diag(S_prev))
            outer = np.outer(std, std)
            outer[outer == 0] = 1e-12
            C_prev = S_prev / outer

            _, V_prev = np.linalg.eigh(C_prev)

            # oracle eigenvalues on corr-scale
            LamO = np.diag(V_prev.T @ (S_next / outer) @ V_prev)
            lambdas.append(LamO)

        lam_ao = np.mean(np.stack(lambdas, axis=0), axis=0)
        lam_ao = np.maximum(lam_ao, 1e-8)
        lam_ao = lam_ao * (N / np.sum(lam_ao))  # trace = N

        # eigenvectors from current train corr (last block)
        S_train = np.cov(X[-block:, :], rowvar=False) * 252
        stdt = np.sqrt(np.diag(S_train))
        outert = np.outer(stdt, stdt)
        outert[outert == 0] = 1e-12
        C_train = S_train / outert

        _, V_t = np.linalg.eigh(C_train)

        C_clean = V_t @ np.diag(lam_ao) @ V_t.T
        C_clean = (C_clean + C_clean.T) / 2
        np.fill_diagonal(C_clean, 1.0)

        return C_clean * outert

