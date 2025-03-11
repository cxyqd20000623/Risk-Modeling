#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from stress_scenario import StressScenario
from simple_calculator import SimpleCalculator


# In[23]:


class AbstractRiskCalculator(ABC):
    def __init__(self, returns, weights=None, confidence_level=0.95, lambda_factor=0.94):
        """
        :param returns: Asset return series (single asset or portfolio returns).
        :param weights: Portfolio weights (None if single asset).
        :param confidence_level: Confidence level for VaR (default 95%).
        :param lambda_factor: Decay factor for weighted VaR (1.0 = Rolling Mean, <1.0 = EWMA).
        """
        self.returns = np.array(returns) if returns is not None else None
        self.weights = np.array(weights) if weights is not None else None
        self.confidence_level = confidence_level
        self.lambda_factor = lambda_factor

    @abstractmethod
    def compute_risk(self, method="historical"):
        raise NotImplementedError("Subclasses must implement compute_risk method.")


# In[31]:


class VaRCalculator(AbstractRiskCalculator):
    """
    Measure market risk in multiple ways, including historical, EWMA, Monte Carlo,
    and a 'Stressed VaR' approach that uses our StressScenario for PCA+Copula simulation.
    """

    def __init__(self, returns, weights=None, confidence_level=0.95,
                 lambda_factor=0.94, market_factors=None, stress_horizon="10d"):
        super().__init__(returns, weights, confidence_level, lambda_factor)
        
        if market_factors is not None:
            self.scenario_generator = StressScenario(stress_horizon, market_factors)
        else:
            self.scenario_generator = None
        
    def compute_var(self):
        """
        Computes Historical or EWMA VaR depending on lambda_factor (<1 => Weighted/EWMA).
        """
        sorted_returns = np.sort(self.returns)
        index = int((1 - self.confidence_level) * len(sorted_returns))

        # If lambda_factor < 1.0 => Weighted VaR
        if self.lambda_factor < 1.0:
            weights = self._calculate_weights(len(sorted_returns))
            var_value = np.average(sorted_returns, weights=weights)
            return abs(var_value)

        # Otherwise, standard historical VaR
        return abs(sorted_returns[index])

    def compute_cvar(self):
        """
        Computes Conditional VaR (Expected Shortfall).
        If portfolio weights are provided, we apply them to the returns first.
        """
        var_value = self.compute_var()
        if self.weights is not None:
            total_returns = self.returns * self.weights
        else:
            total_returns = self.returns

        sorted_returns = np.sort(total_returns)
        cvar = np.mean(sorted_returns[sorted_returns <= -var_value])
        return abs(cvar)

    def compute_svar(self,
                     asset_sensitivities=None,
                     explained_variance_threshold=0.95,
                     min_factors=2,
                     num_simulations=10000,
                     num_periods=10,
                     dt=1/252):
        """
        Computes Stressed VaR (SVaR)
        :param asset_sensitivities: 2D array of shape (n_assets, n_factors). Each row is an asset's factor exposures.
        :param explained_variance_threshold: PCA variance threshold for factor selection.
        :param min_factors: Minimum # of factors to keep for copula modeling.
        :param num_simulations: # of Monte Carlo simulations to run.
        :param num_periods: # of periods (e.g., 10 days).
        :param dt: Single time-step fraction (1/252 for daily).
        :return: The computed Stressed VaR value.
        """
        # Validate user-provided asset_sensitivities
        if asset_sensitivities is None:
            raise ValueError("'asset_sensitivities' must be provided to compute SVaR.")

        # 1) PCA factor selection
        self.scenario_generator.pca_factor_selection(explained_variance_threshold, min_factors)

        # 2) Copula fitting
        self.scenario_generator.fit_copula()

        # 3) Monte Carlo simulation
        self.scenario_generator.monte_carlo_simulation(num_simulations, num_periods, dt)

        # 4) Apply stress to the assets -> get a single final return per simulation
        #    shape of 'stress_returns' => [num_sim, n_assets], if cumulative=True
        stress_returns = self.scenario_generator.apply_stress_to_assets(asset_sensitivities, cumulative=True)

        # Combine all assets into a single portfolio PnL, or just pick the sum of them.
        # E.g., if the user wants to see portfolio-level PnL:
        portfolio_stress_returns = np.sum(stress_returns, axis=1)

        # Now compute VaR on these simulated portfolio returns
        # Sort in ascending order
        sorted_stress = np.sort(portfolio_stress_returns)
        index = int((1 - self.confidence_level) * len(sorted_stress))

        # Weighted if lambda_factor < 1.0
        if self.lambda_factor < 1.0:
            weights = self._calculate_weights(len(sorted_stress))
            svar_value = np.average(sorted_stress, weights=weights)
            return abs(svar_value)

        return abs(sorted_stress[index])

    def compute_var_on_returns(self, returns):
        """
        Helper function to compute VaR on any specified return series.
        """
        sorted_returns = np.sort(returns)
        index = int((1 - self.confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])

    def compute_risk(self, method="historical", **kwargs):
            """
            Unified method to compute various VaR measures:
              - 'historical'  => compute_var()
              - 'weighted'    => compute_var() (with lambda_factor < 1)
              - 'cvar'        => compute_cvar()
              - 'svar'        => compute_svar(**kwargs)
              - 'monte_carlo' => compute_monte_carlo_var(**kwargs)

            :param method: Which VaR measure to compute.
            :param kwargs: Extra parameters for specific methods (e.g., 'num_simulations' for Monte Carlo).
            """
            if method in ("historical", "weighted"):
                return self.compute_var()
            elif method == "cvar":
                return self.compute_cvar()
            elif method == "svar":
                # We'll forward extra kwargs (like 'asset_sensitivities', 'num_simulations', etc.)
                return self.compute_svar(**kwargs)
            elif method == "monte_carlo":
                return self.compute_monte_carlo_var(
                    num_simulations=kwargs.get("num_simulations", 10000),
                    distribution=kwargs.get("distribution", "normal")
                )
            else:
                raise ValueError(f"Unknown method: {method}")

    def compute_monte_carlo_var(self, num_simulations=10000, distribution="normal"):
        """
        Simple Monte Carlo VaR using normal or t-distribution (separate from Copula approach).
        """
        mean, std = np.mean(self.returns), np.std(self.returns)

        if distribution == "normal":
            simulated = np.random.normal(mean, std, num_simulations)
        elif distribution == "t":
            df = len(self.returns) - 1  # degrees of freedom
            simulated = np.random.standard_t(df, size=num_simulations) * std + mean
        else:
            raise ValueError("Invalid distribution. Use 'normal' or 't'.")

        var_threshold = np.percentile(simulated, 100 * (1 - self.confidence_level))
        return abs(var_threshold)

    def _calculate_weights(self, length):
        """
        Exponential decay weights for Weighted/EWMA VaR if lambda_factor < 1.0,
        else uniform weights if lambda_factor=1.0.
        """
        if self.lambda_factor == 1.0:
            return np.ones(length) / length

        weights = [(1 - self.lambda_factor) * (self.lambda_factor ** i) for i in range(length)]
        weights = np.array(weights)[::-1]  # so most recent gets highest weight
        return weights / np.sum(weights)


