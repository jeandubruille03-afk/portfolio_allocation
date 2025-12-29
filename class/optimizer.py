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
# 3) OPTIMIZER (IVP / GMV / ERC)
# ==============================================================================
class Optimizer:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.N = len(mu)

    def _constraints(self):
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = tuple((0.0, 1.0) for _ in range(self.N))
        return cons, bounds

    def inverse_variance(self):
        v = np.diag(self.sigma)
        w = 1.0 / (v + 1e-12)
        return w / np.sum(w)

    def mean_variance(self, risk_aversion=2.0):
        cons, bounds = self._constraints()
        # minimize: (λ/2) wΣw - μw
        func = lambda w: (risk_aversion / 2.0) * (w @ self.sigma @ w) - (w @ self.mu)
        try:
            res = minimize(func, np.ones(self.N) / self.N, method="SLSQP", bounds=bounds, constraints=cons)
            return res.x
        except:
            return np.ones(self.N) / self.N

    def risk_parity(self):
        cons, bounds = self._constraints()

        def func(w):
            p_vol = np.sqrt(w @ self.sigma @ w)
            mrc = (self.sigma @ w) / (p_vol + 1e-12)      # marginal risk
            rc = w * mrc                                   # risk contrib
            target = p_vol / self.N
            return np.sum((rc - target) ** 2)

        try:
            res = minimize(func, np.ones(self.N) / self.N, method="SLSQP", bounds=bounds, constraints=cons, tol=1e-4)
            return res.x
        except:
            return np.ones(self.N) / self.N

