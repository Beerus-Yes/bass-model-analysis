"""
Core Bass Diffusion Model implementation.

This module contains the main BassModel class that implements the Bass Diffusion Model
for forecasting product adoption over time.

Author: Bass Model Analysis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class BassModel:
    """
    Enhanced Bass Diffusion Model implementation with forecasting capabilities.
    
    The Bass model predicts the adoption of new products or technologies based on:
    - Innovation coefficient (p): External influence (advertising, media)
    - Imitation coefficient (q): Internal influence (word-of-mouth)
    
    Attributes:
        M (int): Total addressable market size
        p (float): Innovation coefficient (external influence)
        q (float): Imitation coefficient (word-of-mouth influence) 
        results (pd.DataFrame): Forecast results after running forecast()
    """
    
    def __init__(self, market_size: int, innovation_coef: float, imitation_coef: float):
        """
        Initialize Bass Model parameters.
        
        Args:
            market_size: Total addressable market (M)
            innovation_coef: Innovation coefficient (p) - external influence rate
            imitation_coef: Imitation coefficient (q) - word-of-mouth influence rate
            
        Example:
            >>> model = BassModel(market_size=100000, innovation_coef=0.02, imitation_coef=0.4)
        """
        if market_size <= 0:
            raise ValueError("Market size must be positive")
        if innovation_coef < 0 or innovation_coef > 1:
            raise ValueError("Innovation coefficient must be between 0 and 1")
        if imitation_coef < 0:
            raise ValueError("Imitation coefficient must be non-negative")
            
        self.M = market_size
        self.p = innovation_coef
        self.q = imitation_coef
        self.results = None
        
    def forecast(self, periods: int = 24, time_unit: str = "months") -> pd.DataFrame:
        """
        Generate Bass Model forecast.
        
        Args:
            periods: Number of periods to forecast
            time_unit: Time unit for periods (e.g., "months", "quarters")
            
        Returns:
            DataFrame with forecast results
            
        Example:
            >>> model = BassModel(100000, 0.02, 0.4)
            >>> forecast_df = model.forecast(periods=24)
            >>> print(forecast_df.head())
        """
        results = []
        cumulative_adopters = 0  # Start from zero
        
        for t in range(1, periods + 1):
            # Calculate new adopters using Bass equation
            remaining_market = self.M - cumulative_adopters
            
            if remaining_market <= 0:
                new_adopters = 0
            else:
                # Bass Model equation: f(t) = [p + (q * Y(t-1)/m)] * [m - Y(t-1)]
                # where Y(t-1) is cumulative adopters at time t-1
                adoption_rate = self.p + (self.q * cumulative_adopters / self.M)
                new_adopters = adoption_rate * remaining_market
                
                # Ensure we don't exceed market size
                if cumulative_adopters + new_adopters > self.M:
                    new_adopters = self.M - cumulative_adopters
            
            # Update cumulative adopters
            cumulative_adopters += new_adopters
            
            # Calculate metrics
            market_penetration = (cumulative_adopters / self.M) * 100
            adoption_rate_hazard = new_adopters / remaining_market if remaining_market > 0 else 0
            
            # Store results
            results.append({
                time_unit.capitalize().rstrip('s'): t,  # "Month", "Quarter", etc.
                "New Adopters": round(new_adopters),
                "Cumulative Adopters": round(cumulative_adopters),
                "Market Penetration (%)": round(market_penetration, 2),
                "Adoption Rate": round(adoption_rate_hazard, 4),
                "Remaining Market": round(self.M - cumulative_adopters)
            })
        
        self.results = pd.DataFrame(results)
        return self.results

    def get_peak_period(self) -> Dict:
            """
            Find the period with maximum new adopters (peak of adoption curve).
            
            Returns:
                Dictionary containing:
                - period: Period number when peak occurs
                - new_adopters: Number of new adopters at peak
                - cumulative_penetration: Market penetration at peak
                
            Raises:
                ValueError: If forecast() hasn't been run yet
                
            Example:
                >>> peak_info = model.get_peak_period()
                >>> print(f"Peak occurs in period {peak_info['period']}")
            """
            if self.results is None:
                raise ValueError("Must run forecast() first")
                
            peak_idx = self.results["New Adopters"].idxmax()
            peak_period = self.results.iloc[peak_idx]
            
            return {
                "period": peak_period.iloc[0],
                "new_adopters": peak_period["New Adopters"],
                "cumulative_penetration": peak_period["Market Penetration (%)"]
            }
    
    def get_time_to_peak(self) -> int:
        """
        Calculate time to peak adoption using Bass model formula.
        
        Formula: t* = (1/p+q) * ln((p+q)/p)
        
        Returns:
            Time periods to reach peak adoption
        """
        if self.p <= 0:
            return float('inf')
        return int((1 / (self.p + self.q)) * np.log((self.p + self.q) / self.p))
    
    def get_market_potential_at_peak(self) -> float:
        """
        Calculate market penetration percentage at peak period.
        
        Returns:
            Market penetration percentage at peak
        """
        peak_info = self.get_peak_period()
        return peak_info["cumulative_penetration"]
    
    def sensitivity_analysis(self, param_ranges: Dict, periods: int = 24) -> pd.DataFrame:
        """
        Perform sensitivity analysis on model parameters.
        
        Tests how changes in parameters (p, q, M) affect model outcomes.
        
        Args:
            param_ranges: Dictionary with parameter ranges to test
                         Format: {'p': [0.01, 0.02, 0.03], 'q': [0.3, 0.4, 0.5]}
            periods: Number of periods for analysis
            
        Returns:
            DataFrame with sensitivity analysis results showing how different
            parameter combinations affect total adopters, peak period, etc.
            
        Example:
            >>> sensitivity_ranges = {'p': [0.01, 0.02, 0.03], 'q': [0.3, 0.4, 0.5]}
            >>> sensitivity_df = model.sensitivity_analysis(sensitivity_ranges)
        """
        results = []
        
        for p_val in param_ranges.get('p', [self.p]):
            for q_val in param_ranges.get('q', [self.q]):
                for m_val in param_ranges.get('M', [self.M]):
                    # Create temporary model with new parameters
                    temp_model = BassModel(m_val, p_val, q_val)
                    temp_forecast = temp_model.forecast(periods)
                    
                    # Calculate key metrics
                    total_adopters = temp_forecast["Cumulative Adopters"].iloc[-1]
                    peak_period = temp_forecast["New Adopters"].idxmax() + 1
                    peak_adopters = temp_forecast["New Adopters"].max()
                    final_penetration = (total_adopters / m_val) * 100
                    
                    results.append({
                        'p (Innovation)': p_val,
                        'q (Imitation)': q_val,
                        'M (Market Size)': m_val,
                        'Total Adopters': total_adopters,
                        'Peak Period': peak_period,
                        'Peak Adopters': peak_adopters,
                        'Final Penetration (%)': round(final_penetration, 2)
                    })
        
        return pd.DataFrame(results)
    
    def get_adoption_summary(self) -> Dict:
        """
        Get a summary of key adoption metrics.
        
        Returns:
            Dictionary with key metrics including peak period, final penetration,
            total adopters, and adoption efficiency measures.
            
        Raises:
            ValueError: If forecast() hasn't been run yet
        """
        if self.results is None:
            raise ValueError("Must run forecast() first")
            
        peak_info = self.get_peak_period()
        final_adopters = self.results["Cumulative Adopters"].iloc[-1]
        final_penetration = self.results["Market Penetration (%)"].iloc[-1]
        
        return {
            "model_parameters": {
                "market_size": self.M,
                "innovation_coef": self.p,
                "imitation_coef": self.q
            },
            "adoption_metrics": {
                "total_adopters": final_adopters,
                "final_penetration_pct": final_penetration,
                "peak_period": peak_info["period"],
                "peak_adopters": peak_info["new_adopters"],
                "theoretical_time_to_peak": self.get_time_to_peak()
            },
            "efficiency_metrics": {
                "avg_adopters_per_period": final_adopters / len(self.results),
                "adoption_acceleration": self.q / self.p if self.p > 0 else float('inf')
            }
        }
    
    def __str__(self) -> str:
        """String representation of the Bass Model."""
        return f"BassModel(M={self.M:,}, p={self.p}, q={self.q})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Bass Model."""
        return f"BassModel(market_size={self.M}, innovation_coef={self.p}, imitation_coef={self.q})"