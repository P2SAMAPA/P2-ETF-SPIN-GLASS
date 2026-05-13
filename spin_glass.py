import numpy as np
import pandas as pd

class SpinGlass:
    def __init__(self, returns_df, window=60):
        self.returns = returns_df
        self.window = window
        self.n_assets = returns_df.shape[1]
        self.assets = returns_df.columns.tolist()

    def compute_spins(self, returns_series):
        """Map returns to +1 (positive) and -1 (negative or zero)."""
        return np.sign(returns_series)

    def compute_correlation_matrix(self):
        """Compute rolling correlation matrix over last `window` days."""
        recent_returns = self.returns.iloc[-self.window:]
        corr = recent_returns.corr().values
        return corr

    def compute_local_fields(self, spins, J):
        """h_i = sum_j J_ij s_j"""
        return J @ spins

    def compute_magnetisation(self, spins):
        """m = (1/N) sum s_i"""
        return np.mean(spins)

    def compute_energy(self, spins, J):
        """Hamiltonian H = -sum_{i<j} J_ij s_i s_j"""
        # Sum over all pairs (i<j)
        H = 0.0
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                H -= J[i, j] * spins[i] * spins[j]
        return H

    def run(self):
        """Compute spins, J, local fields, magnetisation, energy."""
        # Use the last day's returns to determine current spin configuration
        last_returns = self.returns.iloc[-1].values
        spins = self.compute_spins(last_returns)
        J = self.compute_correlation_matrix()
        local_fields = self.compute_local_fields(spins, J)
        magnetisation = self.compute_magnetisation(spins)
        energy = self.compute_energy(spins, J)

        # For diagnostics, compute effective temperature = cross-sectional vol of recent returns
        recent_returns = self.returns.iloc[-self.window:]
        cross_vol = recent_returns.std(axis=0).mean()   # average std across assets
        effective_temperature = cross_vol * config.TEMPERATURE_SCALE

        return {
            "spins": spins.tolist(),
            "local_fields": local_fields.tolist(),
            "magnetisation": float(magnetisation),
            "energy": float(energy),
            "effective_temperature": float(effective_temperature),
            "assets": self.assets
        }
