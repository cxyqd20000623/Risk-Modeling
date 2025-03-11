#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
from copulas.multivariate.gaussian import GaussianMultivariate
from scipy.stats import rankdata
import numpy as np


# In[2]:


class StressScenario:
    """
    Stress scenario generator using PCA for factor selection, GaussianMultivariate Copula modeling,
    and Monte Carlo simulation. Suitable for more than 2 PCA factors.
    """
        
    def __init__(self, stress_horizon="10d", market_factors=None):
        """
        Initializes the StressScenario class with manually provided risk factors.

        :param stress_horizon: Time horizon for stress return (e.g., "1d", "10d", "1m", "1y").
        :param market_factors: NumPy array or list of shape (N, M), where:
                               - N = number of observations (historical data points)
                               - M = number of market risk factors (equity, fx, interest rates, etc.)
        """
        self.stress_horizon = stress_horizon

        if market_factors is None:
            raise ValueError("Market factors must be provided manually.")

        self.market_factors = np.array(market_factors)

        if self.market_factors.shape[1] < 2:
            raise ValueError("Market factors must have at least TWO independent variables for PCA & Copula.")

        self.selected_factors = None
        self.copula_model = None
        self.simulated_factors = None

    def pca_factor_selection(self, explained_variance_threshold=0.95, min_factors=2):
        """
        Runs PCA on manually provided market factors and applies ECDF transformation to ensure Copula compatibility.

        :param explained_variance_threshold: Cumulative variance threshold for factor selection.
        :param min_factors: Minimum number of factors to ensure Copula modeling works.
        :return: (selected_factors, num_factors)
                 - selected_factors: PCA-transformed factors, then ECDF-transformed to [0,1].
                 - num_factors: number of PCA factors selected.
        """
        pca = PCA()
        transformed_factors = pca.fit_transform(self.market_factors)

        # Compute cumulative variance and determine number of factors
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_factors = max(np.searchsorted(cumulative_variance, explained_variance_threshold) + 1,
                          min_factors)

        # Ensure we don't exceed available factors
        num_factors = min(num_factors, transformed_factors.shape[1])

        if num_factors < 2:
            raise ValueError("PCA resulted in fewer than 2 factors. A multivariate copula requires >=2 dims.")

        # Select the required number of factors
        self.selected_factors = transformed_factors[:, :num_factors]

        # Apply ECDF transformation (rank-based) so factors lie in [0,1] for copula fitting
        ecdf_factors = np.zeros_like(self.selected_factors)
        for i in range(self.selected_factors.shape[1]):
            # rankdata => ranks from 1..N; we scale them to (0,1) by subtracting 0.5 and dividing by N
            ecdf_factors[:, i] = (rankdata(self.selected_factors[:, i]) - 0.5) / len(self.selected_factors)

        self.selected_factors = ecdf_factors

        print(f"Selected {num_factors} PCA factors for Copula (ECDF in [0,1]).")
        print(f"   Value range after ECDF: min={np.min(self.selected_factors):.6f}, "
              f"max={np.max(self.selected_factors):.6f}")

        return self.selected_factors, num_factors

    def fit_copula(self):
        """
        Fits a multivariate Gaussian copula model to the selected PCA factors.
        Can handle 2 or more factors without dimension mismatch.
        """
        if self.selected_factors is None:
            raise ValueError("Must run pca_factor_selection() first.")

        # Ensure values are strictly within (0,1)
        epsilon = 1e-10
        adjusted_factors = np.clip(self.selected_factors, epsilon, 1 - epsilon)

        # Use a GaussianMultivariate (handles 2D+)
        self.copula_model = GaussianMultivariate()
        self.copula_model.fit(adjusted_factors)

        print(f"Successfully fitted GaussianMultivariate copula "
              f"with dimension={adjusted_factors.shape[1]}")
        return self.copula_model

    def monte_carlo_simulation(self, num_simulations=10000, num_periods=10, dt=1/252):
        if self.selected_factors is None:
            raise ValueError("PCA must be run first.")
        if self.copula_model is None:
            raise ValueError("Copula model must be fitted first.")

        # 1) Sample from the multivariate copula => shape (num_simulations * num_periods, n_factors)
        correlated_df = self.copula_model.sample(num_simulations * num_periods)

        # 2) Convert to NumPy and reshape => [num_simulations, num_periods, n_factors]
        correlated_shocks = correlated_df.to_numpy().reshape(num_simulations, num_periods, -1)

        # 3) Compute factor volatilities from the selected_factors
        factor_volatilities = np.std(self.selected_factors, axis=0)  # shape: (n_factors,)

        # 4) Build multi-period factor paths (zero drift)
        simulated_paths = np.zeros_like(correlated_shocks)
        for t in range(1, num_periods):
            diffusion = factor_volatilities * correlated_shocks[:, t, :]
            simulated_paths[:, t, :] = simulated_paths[:, t-1, :] + diffusion * np.sqrt(dt)

        self.simulated_factors = simulated_paths
        print(
            f"Generated {num_simulations} Monte Carlo simulations over {num_periods} periods "
            f"(dimension={factor_volatilities.shape[0]})."
        )
        return self.simulated_factors


    def apply_stress_to_assets(self, asset_sensitivities, cumulative=True):
        """
        Applies Monte Carlo simulated multi-period stress factors to asset factor exposures.

        :param asset_sensitivities: 2D NumPy array of shape (n_assets, n_factors).
                                    Each row is an asset's factor exposure.
        :param cumulative: If True, return final cumulative impact over all periods. 
                           If False, return entire path (every period).
        :return: If cumulative=True, shape => [num_simulations, n_assets].
                 If cumulative=False, shape => [num_simulations, num_periods, n_assets].
        """
        if self.simulated_factors is None:
            raise ValueError("Must run monte_carlo_simulation() first.")

        # self.simulated_factors => [num_sim, num_periods, n_factors]
        # asset_sensitivities  => [n_assets, n_factors]

        # stress_matrix => [num_sim, num_periods, n_assets]
        stress_matrix = np.einsum('ijk,kl->ijl', self.simulated_factors, asset_sensitivities.T)

        if cumulative:
            # Sum across the time dimension => shape [num_sim, n_assets]
            return np.sum(stress_matrix, axis=1)

        # Otherwise, return the full (num_period) path => [num_sim, num_periods, n_assets]
        return stress_matrix


# In[ ]:




