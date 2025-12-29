import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
from sklearn.covariance import OAS

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

class Cleaning:
    def __init__(self, csv_path, in_sample_days=1000, oos_days=252, N=100):
        self.csv_path = csv_path
        self.in_sample_days = in_sample_days
        self.oos_days = oos_days
        self.N = N

    def load_and_prepare(self):
        print(f"Chargement de '{self.csv_path}'...")

        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Erreur: Le fichier '{self.csv_path}' est introuvable.")
            return None, None

        # Cas 1 : format matrice (date index, tickers colonnes)
        if pd.to_datetime(df.iloc[:, 0], errors="coerce").notna().mean() > 0.5:
            df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
            prices = df.apply(pd.to_numeric, errors="coerce")

        # Cas 2 : format DB (ticker, date, close)
        elif {"ticker", "date", "close"}.issubset(set(c.lower() for c in df.columns)):
            df.columns = df.columns.str.lower()
            prices = df.pivot(index="date", columns="ticker", values="close")
            prices.index = pd.to_datetime(prices.index)

        else:
            raise ValueError("Format CSV inconnu")

        prices = prices.sort_index()

        # Slice global
        total_days_needed = self.in_sample_days + self.oos_days
        if prices.shape[0] < total_days_needed:
            print(f"Warning: Pas assez de données ({prices.shape[0]} jours). Utilisation du max dispo.")
            self.in_sample_days = int(prices.shape[0] * 0.8)
            total_days_needed = prices.shape[0]

        prices_slice = prices.iloc[-total_days_needed:]

        # Sélection N actifs sur la période in-sample
        prices_in = prices_slice.iloc[: self.in_sample_days]
        valid_counts = prices_in.count()

        if len(valid_counts) < self.N:
            selected_tickers = valid_counts.index.tolist()
        else:
            selected_tickers = valid_counts.nlargest(self.N).index.tolist()

        print(f"Actifs sélectionnés (Top {len(selected_tickers)}) sur base In-Sample.")

        # Prix clean + log-returns
        final_prices = prices_slice[selected_tickers].ffill()
        returns = np.log(final_prices).diff().fillna(0.0)

        return returns, self.in_sample_days



