#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:





# In[2]:


class SimpleCalculator:
    @staticmethod
    def compute_covariance_matrix(returns):
        """
        Computes the covariance matrix of asset returns.
        
        :param returns: 2D NumPy array where each column represents an asset's return series.
        :return: Covariance matrix.
        """
        returns = np.array(returns)  # Ensure NumPy array
        return np.cov(returns, rowvar=False)  # Compute covariance between assets

    @staticmethod
    def cholesky_decomposition(cov_matrix):
        """
        Performs Cholesky decomposition for Monte Carlo simulations.
        
        :param cov_matrix: Covariance matrix of asset returns.
        :return: Lower triangular matrix L such that L @ L.T = cov_matrix.
        """
        return np.linalg.cholesky(cov_matrix)

    @staticmethod
    def calculate_volatility(returns, annualize=True, periods_per_year=252):
        """
        Computes annualized volatility (standard deviation) of returns.
        
        :param returns: 1D NumPy array of asset returns.
        :param annualize: If True, annualizes volatility.
        :param periods_per_year: Number of periods per year (252 for daily).
        :return: Annualized or raw standard deviation.
        """
        volatility = np.std(returns, ddof=1)  # Use sample standard deviation
        return volatility * np.sqrt(periods_per_year) if annualize else volatility

    @staticmethod
    def calculate_beta(asset_returns, market_returns):
        """
        Computes beta of an asset relative to the market.
        
        :param asset_returns: 1D NumPy array of asset returns.
        :param market_returns: 1D NumPy array of market returns.
        :return: Beta value.
        """
        cov_matrix = np.cov(asset_returns, market_returns)
        return cov_matrix[0, 1] / cov_matrix[1, 1]  # Cov(asset, market) / Var(market)

    @staticmethod
    def calculate_sharpe(returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Computes the Sharpe ratio.
        
        :param returns: 1D NumPy array of asset returns.
        :param risk_free_rate: Risk-free rate (annualized).
        :param periods_per_year: Number of periods per year.
        :return: Sharpe ratio.
        """
        mean_return = np.mean(returns) * periods_per_year  # Annualized return
        volatility = SimpleCalculator.calculate_volatility(returns, annualize=True, periods_per_year=periods_per_year)
        return (mean_return - risk_free_rate) / volatility if volatility > 0 else np.nan

