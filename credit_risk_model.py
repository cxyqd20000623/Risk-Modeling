#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from var_calculator import AbstractRiskCalculator


# In[3]:


class CreditRiskModel(AbstractRiskCalculator):
    """
    Industry-level Credit Risk Model using Monte Carlo simulation for CVaR and Expected Credit Loss (ECL).
    Supports both **individual loan risk** and **portfolio credit risk**.
    """

    def __init__(self, ead, pd_matrix, lgd, weights=None, num_periods=10, num_simulations=10000, confidence_level=0.95):
        """
        Initializes the Credit Risk Model.

        :param ead: Exposure at Default (EAD), array of loan exposures.
        :param pd_matrix: Markov Transition Matrix for PD evolution.
        :param lgd: Loss Given Default (LGD), array of loss percentages per loan.
        :param weights: Portfolio weights (if None, assumes equal weighting).
        :param num_periods: Number of periods for simulation (default 10).
        :param num_simulations: Number of Monte Carlo simulations (default 10,000).
        :param confidence_level: VaR confidence level (default 95%).
        """
        super().__init__(returns=None, weights=weights, confidence_level=confidence_level)  
        self.ead = np.array(ead)  # Loan exposures
        self.pd_matrix = np.array(pd_matrix)  # Transition matrix for PD evolution
        self.lgd = np.array(lgd)  # Loss given default rates
        self.num_periods = num_periods
        self.num_simulations = num_simulations
        self.weights = np.array(weights) if weights is not None else np.ones(len(ead)) / len(ead)  # Default: Equal weighting

    def simulate_pd_paths(self):
        """
        Uses a Markov transition matrix to model PD evolution over multiple periods.
        :return: Simulated PD paths (num_simulations, num_periods, num_loans).
        """
        num_loans = len(self.ead)
        simulated_pd = np.zeros((self.num_simulations, self.num_periods, num_loans))

        # Initialize PDs from first row of transition matrix
        current_pd = self.pd_matrix[0, :]

        for i in range(self.num_simulations):
            for t in range(self.num_periods):
                # PD evolves based on Markov Chain
                current_pd = np.dot(current_pd, self.pd_matrix)  
                simulated_pd[i, t, :] = current_pd

        return simulated_pd

    def compute_expected_credit_loss(self, portfolio_level=True):
        """
        Computes Expected Credit Loss (ECL) using Monte Carlo simulation.

        :param portfolio_level: If True, computes **portfolio** expected loss (weighted sum).
        :return: Expected Credit Loss (ECL) per period (individual or portfolio).
        """
        simulated_pd = self.simulate_pd_paths()
        ecl = np.mean(simulated_pd * self.ead * self.lgd, axis=0)  # Average loss over simulations
        
        if portfolio_level:
            return np.dot(ecl, self.weights)  # Weighted portfolio ECL
        
        return ecl  # Shape: (num_periods, num_loans)

    def compute_cvar(self, portfolio_level=True, use_final_period=True):
        """
        Computes Credit Value at Risk (CVaR) using Monte Carlo simulation,
        focusing on either final-period or full-horizon losses.
        """
        # 1) Simulate PD paths => shape [num_sim, num_periods, num_loans]
        simulated_pd = self.simulate_pd_paths()

        # 2) Convert PD to credit losses
        credit_losses = simulated_pd * self.ead * self.lgd  # shape [num_sim, num_periods, num_loans]

        if portfolio_level:
            # Weighted sum across loans => shape [num_sim, num_periods]
            portfolio_losses = np.einsum('ijk,k->ij', credit_losses, self.weights)
            # Optionally, pick final period's losses or sum across all periods
            if use_final_period:
                portfolio_losses = portfolio_losses[:, -1]  # shape [num_sim]
            else:
                # e.g. sum across time to get total horizon loss
                portfolio_losses = np.sum(portfolio_losses, axis=1)  # shape [num_sim]
            # Sort the 1D distribution
            sorted_losses = np.sort(portfolio_losses)
        else:
            # Not portfolio-level => shape is [num_sim, num_periods, num_loans]
            # We'll illustrate final-period approach for each loan
            if use_final_period:
                # shape => [num_sim, num_loans]
                final_period_losses = credit_losses[:, -1, :]
                sorted_losses = np.sort(final_period_losses, axis=0)  # Sort each loan's distribution
            else:
                # sum or max across periods if you want multi-period approach per loan
                # for demonstration, let's sum across periods => shape [num_sim, num_loans]
                sum_periods_losses = np.sum(credit_losses, axis=1)
                sorted_losses = np.sort(sum_periods_losses, axis=0)

        # 3) Locate the tail index
        index = int((1 - self.confidence_level) * self.num_simulations)
        index = max(0, min(index, self.num_simulations - 1))  # boundary check

        # 4) CVaR is the mean loss in the worst (1 - confidence_level)% scenarios
        #    i.e. the average of sorted losses above 'index'
        if portfolio_level:
            tail_losses = sorted_losses[index:]
            cvar_value = np.mean(tail_losses)  # average in the tail
            return cvar_value
        else:
            # For each loan, we gather the tail portion across simulations:
            tail_losses = sorted_losses[index:, :]
            cvar_per_loan = np.mean(tail_losses, axis=0)
            return cvar_per_loan


    def compute_risk(self, method="ECL", portfolio_level=True):
        """
        Implements risk computation based on selected method.

        :param method: Risk measure type ("ECL" for Expected Credit Loss, "CVaR" for Credit VaR).
        :param portfolio_level: If True, computes portfolio risk.
        :return: Computed risk measure.
        """
        if method == "ECL":
            return self.compute_expected_credit_loss(portfolio_level=portfolio_level)
        elif method == "CVaR":
            return self.compute_cvar(portfolio_level=portfolio_level)
        else:
            raise ValueError("Invalid method. Choose 'ECL' or 'CVaR'.")

