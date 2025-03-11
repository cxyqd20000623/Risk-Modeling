#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from var_calculator import VaRCalculator
from credit_risk_model import CreditRiskModel


# In[3]:


if __name__ == "__main__":
    # Credit Risk Model (ECL & CVaR)
    # Define sample credit portfolio parameters
    ead = [1_000_000, 2_000_000, 1_500_000]
    
    # PD Matrix: Markov Transition Matrix for PD evolution
    pd_matrix = [
        [0.85, 0.10, 0.05],
        [0.30, 0.60, 0.10],
        [0.15, 0.25, 0.60]
    ]

    # LGD: Loss Given Default for each loan
    lgd = [0.40, 0.30, 0.35]  

    # Portfolio weights
    weights = [0.2, 0.5, 0.3]

    # Create the CreditRiskModel
    credit_model = CreditRiskModel(
        ead=ead, 
        pd_matrix=pd_matrix,
        lgd=lgd,
        weights=weights,
        num_periods=5,
        num_simulations=1000,
        confidence_level=0.95
    )

    # Compute risk using compute_risk()
    ecl_portfolio = credit_model.compute_risk(method="ECL", portfolio_level=True)
    print("Portfolio ECL (per period):", ecl_portfolio)

    ecl_individual = credit_model.compute_risk(method="ECL", portfolio_level=False)
    print("Individual ECL (shape = periods x loans):\n", ecl_individual)

    cvar_portfolio = credit_model.compute_risk(method="CVaR", portfolio_level=True)
    print("Portfolio CVaR (final period):", cvar_portfolio)

    cvar_per_loan = credit_model.compute_risk(method="CVaR", portfolio_level=False)
    print("CVaR Per Loan (final period):\n", cvar_per_loan)


    # VaR Calculator (Historical, EWMA, CVaR, Monte Carlo VaR)

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 1000)

    # Create VaRCalculator without 'market_factors' => scenario_generator=None
    var_calc = VaRCalculator(
        returns=returns, 
        weights=None, 
        confidence_level=0.95, 
        lambda_factor=0.94
    )

    # Compute different types of VaR
    ewma_var = var_calc.compute_risk(method="historical")
    print(f"EWMA VaR: {ewma_var:.6f}")

    cvar_value = var_calc.compute_risk(method="cvar")
    print(f"CVaR: {cvar_value:.6f}")

    mc_var = var_calc.compute_risk(method="monte_carlo", num_simulations=5000, distribution="normal")
    print(f"Monte Carlo VaR (normal): {mc_var:.6f}")


    # Stressed VaR (SVaR)
    # Create new returns for historical/EWMA part
    returns = np.random.normal(0, 0.01, 1000)

    # Create "market_factors" for StressScenario (needed for SVaR)
    market_factors = np.random.normal(0, 1, (1000, 5))

    # Create VaRCalculator WITH market_factors => scenario_generator is created
    var_calc = VaRCalculator(
        returns=returns,
        weights=None,
        confidence_level=0.95,
        lambda_factor=0.94,
        market_factors=market_factors
    )

    # Example asset sensitivities for SVaR (1 asset, 5 factor loadings)
    asset_sens = np.array([[0.8, 0.2, 0.4, -0.3, 0.9]])  # shape (1,5)

    # Compute Stressed VaR (SVaR)
    svar_value = var_calc.compute_risk(method="svar", asset_sensitivities=asset_sens)
    print(f"\nStressed VaR: {svar_value:.6f}")

