"""
Advanced analysis tools and utility functions for Bass Model analysis.

This module provides utility functions for:
- Break-even point analysis between pricing models
- ROI calculations and cost-per-user metrics
- Scenario analysis and what-if modeling
- Statistical analysis of adoption patterns
- Business intelligence and decision support

Updated with corrected usage patterns:
- ONECI: 1 request per user (registration only)
- SmileID: 2 one-time + 1 monthly recurring per user
- DKB: 1 signature per user (signing only)

Author: Bass Model Analysis Team
Version: 1.1.0 - FIXED ONECI usage pattern
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from bass_model import BassModel
from pricing_models import financial_analysis, compare_pricing_models, get_pricing_summary
import warnings


def analyze_break_even_points(model: BassModel, requests_per_user: int = None) -> Dict:
    """
    Analyze break-even points between different pricing models.
    
    Identifies when each provider becomes the most cost-effective option
    and calculates crossover periods.
    
    CORRECTED USAGE PATTERNS:
    - ONECI: 1 request per user (registration only)
    - SmileID: 2 one-time + 1 monthly recurring per user
    - DKB: 1 signature per user (signing only)
    
    Args:
        model: BassModel instance with forecast results
        requests_per_user: DEPRECATED - usage patterns are now fixed per provider
        
    Returns:
        Dictionary with break-even analysis including:
        - Periods when each provider is cheapest
        - Crossover points between providers
        - Cost advantage analysis
        
    Example:
        >>> model = BassModel(100000, 0.02, 0.4)
        >>> model.forecast(24)
        >>> breakeven = analyze_break_even_points(model)
        >>> print(breakeven['summary']['best_overall'])
    """
    if requests_per_user is not None:
        warnings.warn("requests_per_user parameter is deprecated. Usage patterns are now fixed per provider.", 
                     DeprecationWarning, stacklevel=2)
    
    comparison_df = compare_pricing_models(model)
    
    # Find when each provider is cheapest
    oneci_cheapest = comparison_df[comparison_df['Best Option'] == 'ONECI']
    smileid_cheapest = comparison_df[comparison_df['Best Option'] == 'SmileID']
    dkb_cheapest = comparison_df[comparison_df['Best Option'] == 'DKB']
    
    # Calculate dominance periods
    periods_analysis = {}
    
    for provider, cheapest_periods in [("ONECI", oneci_cheapest), ("SmileID", smileid_cheapest), ("DKB", dkb_cheapest)]:
        if not cheapest_periods.empty:
            periods_analysis[provider] = {
                "periods_cheapest": len(cheapest_periods),
                "first_cheapest_month": cheapest_periods['Month'].min(),
                "last_cheapest_month": cheapest_periods['Month'].max(),
                "consecutive_periods": _find_consecutive_periods(cheapest_periods['Month'].tolist()),
                "dominance_percentage": (len(cheapest_periods) / len(comparison_df)) * 100
            }
        else:
            periods_analysis[provider] = {
                "periods_cheapest": 0,
                "first_cheapest_month": None,
                "last_cheapest_month": None,
                "consecutive_periods": [],
                "dominance_percentage": 0
            }
    
    # Find crossover points
    crossovers = []
    for i in range(len(comparison_df) - 1):
        current_best = comparison_df.iloc[i]['Best Option']
        next_best = comparison_df.iloc[i + 1]['Best Option']
        
        if current_best != next_best:
            crossovers.append({
                "month": i + 1,
                "from_provider": current_best,
                "to_provider": next_best,
                "users_at_crossover": comparison_df.iloc[i]['Cumulative Users']
            })
    
    # Overall best option
    best_counts = comparison_df['Best Option'].value_counts()
    overall_best = best_counts.index[0] if not best_counts.empty else "No clear winner"
    
    return {
        "periods_analysis": periods_analysis,
        "crossover_points": crossovers,
        "summary": {
            "total_periods": len(comparison_df),
            "best_overall": overall_best,
            "number_of_crossovers": len(crossovers),
            "most_stable_provider": _find_most_stable_provider(comparison_df)
        }
    }


def calculate_roi_metrics(model: BassModel, requests_per_user: int = None) -> Dict:
    """
    Calculate comprehensive ROI metrics for each pricing model.
    
    CORRECTED USAGE PATTERNS:
    - ONECI: 1 request per user (registration only)
    - SmileID: 2 one-time + 1 monthly recurring per user
    - DKB: 1 signature per user (signing only)
    
    Args:
        model: BassModel instance with forecast results
        requests_per_user: DEPRECATED - usage patterns are now fixed per provider
        
    Returns:
        Dictionary with ROI analysis including:
        - Total costs and cost per user
        - Monthly cost trends
        - Efficiency metrics
        - Investment analysis
        
    Example:
        >>> roi_metrics = calculate_roi_metrics(model)
        >>> print(f"DKB setup ROI: {roi_metrics['dkb']['setup_roi_months']} months")
    """
    if requests_per_user is not None:
        warnings.warn("requests_per_user parameter is deprecated. Usage patterns are now fixed per provider.", 
                     DeprecationWarning, stacklevel=2)
    
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid")
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    periods = len(model.results)
    
    roi_analysis = {}
    
    # Updated model types to reflect corrected usage patterns
    provider_info = [
        ("oneci", oneci_df, "one-time (registration only)"), 
        ("smileid", smileid_df, "mixed (one-time + recurring)"), 
        ("dkb", dkb_df, "front-loaded (signing only)")
    ]
    
    for name, df, model_type in provider_info:
        total_cost = df["Cumulative Cost (FCFA)"].iloc[-1]
        monthly_costs = df["Monthly Cost (FCFA)"].astype(float)
        
        # Basic metrics
        cost_per_user = total_cost / final_users if final_users > 0 else 0
        avg_monthly_cost = monthly_costs.mean()
        
        # Trend analysis
        cost_trend = "increasing" if monthly_costs.iloc[-1] > monthly_costs.iloc[0] else "decreasing"
        cost_volatility = monthly_costs.std() / monthly_costs.mean() if monthly_costs.mean() > 0 else 0
        
        # Efficiency metrics
        peak_cost = monthly_costs.max()
        peak_month = monthly_costs.idxmax() + 1
        min_cost = monthly_costs.min()
        cost_efficiency = min_cost / peak_cost if peak_cost > 0 else 0
        
        roi_analysis[name] = {
            "basic_metrics": {
                "total_cost_fcfa": total_cost,
                "cost_per_user_fcfa": cost_per_user,
                "avg_monthly_cost_fcfa": avg_monthly_cost,
                "final_monthly_cost_fcfa": float(monthly_costs.iloc[-1])
            },
            "trend_analysis": {
                "cost_trend": cost_trend,
                "cost_volatility": cost_volatility,
                "peak_cost_fcfa": peak_cost,
                "peak_month": peak_month,
                "min_cost_fcfa": min_cost,
                "cost_range_fcfa": peak_cost - min_cost
            },
            "efficiency_metrics": {
                "cost_efficiency_ratio": cost_efficiency,
                "cost_per_user_per_period": cost_per_user / periods,
                "model_type": model_type
            }
        }
        
        # Special metrics for DKB (front-loaded costs)
        if name == "dkb":
            setup_cost = float(dkb_df["Setup Cost (FCFA)"].iloc[0]) if len(dkb_df) > 0 else 0
            ongoing_costs = total_cost - setup_cost
            
            # Calculate setup ROI (how many months to break even vs cheapest alternative)
            comparison = compare_pricing_models(model)
            dkb_costs = [pd.to_numeric(comparison.iloc[i]["DKB Cost (FCFA)"].replace(",", "")) for i in range(len(comparison))]
            
            # Find cheapest alternative each month
            alt_costs = []
            for i in range(len(comparison)):
                oneci_cost = pd.to_numeric(comparison.iloc[i]["ONECI Cost (FCFA)"].replace(",", ""))
                smileid_cost = pd.to_numeric(comparison.iloc[i]["SmileID Cost (FCFA)"].replace(",", ""))
                alt_costs.append(min(oneci_cost, smileid_cost))
            
            # Calculate break-even
            cumulative_savings = 0
            breakeven_month = None
            for i, (dkb_cost, alt_cost) in enumerate(zip(dkb_costs, alt_costs)):
                monthly_savings = alt_cost - dkb_cost
                cumulative_savings += monthly_savings
                if cumulative_savings >= setup_cost and breakeven_month is None:
                    breakeven_month = i + 1
            
            roi_analysis[name]["dkb_specific"] = {
                "setup_cost_fcfa": setup_cost,
                "ongoing_costs_fcfa": ongoing_costs,
                "setup_percentage": (setup_cost / total_cost * 100) if total_cost > 0 else 0,
                "setup_roi_months": breakeven_month,
                "total_savings_vs_alternatives": cumulative_savings
            }
    
    # Comparative analysis
    costs = {name: roi_analysis[name]["basic_metrics"]["total_cost_fcfa"] for name in roi_analysis}
    cheapest_provider = min(costs, key=costs.get)
    most_expensive_provider = max(costs, key=costs.get)
    
    roi_analysis["comparative_analysis"] = {
        "cheapest_provider": cheapest_provider,
        "most_expensive_provider": most_expensive_provider,
        "cost_spread_fcfa": costs[most_expensive_provider] - costs[cheapest_provider],
        "cost_spread_percentage": ((costs[most_expensive_provider] - costs[cheapest_provider]) / costs[most_expensive_provider] * 100) if costs[most_expensive_provider] > 0 else 0,
        "all_costs": costs
    }
    
    return roi_analysis


def scenario_analysis(base_model: BassModel, scenarios: Dict[str, Dict]) -> pd.DataFrame:
    """
    Perform scenario analysis with different parameter combinations.
    
    Uses corrected usage patterns:
    - ONECI: 1 request per user (registration only)
    - SmileID: 2 one-time + 1 monthly recurring per user
    - DKB: 1 signature per user (signing only)
    
    Args:
        base_model: Base BassModel instance (used for reference parameters)
        scenarios: Dictionary of scenarios with parameter variations
                  Format: {"scenario_name": {"p": 0.02, "q": 0.4, "M": 100000}}
        
    Returns:
        DataFrame comparing outcomes across scenarios
        
    Example:
        >>> scenarios = {
        ...     "Conservative": {"p": 0.01, "q": 0.3, "M": 50000},
        ...     "Optimistic": {"p": 0.03, "q": 0.5, "M": 200000},
        ...     "Base Case": {"p": 0.02, "q": 0.4, "M": 100000}
        ... }
        >>> results = scenario_analysis(base_model, scenarios)
    """
    results = []
    
    for scenario_name, params in scenarios.items():
        # Create model with scenario parameters
        scenario_model = BassModel(
            market_size=params.get('M', base_model.M),
            innovation_coef=params.get('p', base_model.p),
            imitation_coef=params.get('q', base_model.q)
        )
        
        # Run forecast
        scenario_model.forecast(periods=24)
        
        # Get basic metrics
        peak_info = scenario_model.get_peak_period()
        final_users = scenario_model.results["Cumulative Adopters"].iloc[-1]
        final_penetration = scenario_model.results["Market Penetration (%)"].iloc[-1]
        
        # Get pricing analysis
        pricing_summary = get_pricing_summary(scenario_model)
        
        results.append({
            "Scenario": scenario_name,
            "Market Size": params.get('M', base_model.M),
            "Innovation Coef (p)": params.get('p', base_model.p),
            "Imitation Coef (q)": params.get('q', base_model.q),
            "Peak Period": peak_info["period"],
            "Peak Adopters": peak_info["new_adopters"],
            "Final Users": final_users,
            "Final Penetration (%)": round(final_penetration, 1),
            "ONECI Total Cost (1 req/user)": f"{pricing_summary['oneci']['total_cost_fcfa']:,.0f}",
            "SmileID Total Cost (2+1 req/user)": f"{pricing_summary['smileid']['total_cost_fcfa']:,.0f}",
            "DKB Total Cost (1 sig/user)": f"{pricing_summary['dkb']['total_cost_fcfa']:,.0f}",
            "Best Option": pricing_summary['comparison']['cheapest_provider']
        })
    
    return pd.DataFrame(results)


def calculate_market_timing_metrics(model: BassModel) -> Dict:
    """
    Calculate market timing and adoption velocity metrics.
    
    Args:
        model: BassModel instance with forecast results
        
    Returns:
        Dictionary with timing analysis including:
        - Time to reach penetration milestones
        - Adoption velocity metrics
        - Market saturation analysis
        
    Example:
        >>> timing = calculate_market_timing_metrics(model)
        >>> print(f"Time to 50% penetration: {timing['milestones']['50_percent']} months")
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    penetration = model.results["Market Penetration (%)"]
    new_adopters = model.results["New Adopters"]
    cumulative = model.results["Cumulative Adopters"]
    
    # Find milestone periods
    milestones = {}
    for milestone in [10, 25, 50, 75, 90]:
        milestone_periods = model.results[penetration >= milestone]
        if not milestone_periods.empty:
            milestones[f"{milestone}_percent"] = milestone_periods.iloc[0, 0]  # First period
        else:
            milestones[f"{milestone}_percent"] = None
    
    # Adoption velocity metrics
    velocity_metrics = {}
    
    # Peak velocity period and magnitude
    peak_period_idx = new_adopters.idxmax()
    velocity_metrics["peak_velocity_period"] = model.results.iloc[peak_period_idx, 0]
    velocity_metrics["peak_velocity_adopters"] = new_adopters.iloc[peak_period_idx]
    
    # Average velocity in different phases
    periods = len(model.results)
    early_phase = new_adopters[:periods//3].mean() if periods >= 3 else new_adopters.mean()
    middle_phase = new_adopters[periods//3:2*periods//3].mean() if periods >= 3 else new_adopters.mean()
    late_phase = new_adopters[2*periods//3:].mean() if periods >= 3 else new_adopters.mean()
    
    velocity_metrics["early_phase_avg"] = early_phase
    velocity_metrics["middle_phase_avg"] = middle_phase
    velocity_metrics["late_phase_avg"] = late_phase
    
    # Velocity acceleration/deceleration
    if periods >= 2:
        velocity_changes = new_adopters.diff().dropna()
        velocity_metrics["avg_acceleration"] = velocity_changes.mean()
        velocity_metrics["max_acceleration"] = velocity_changes.max()
        velocity_metrics["max_deceleration"] = velocity_changes.min()
    
    # Market saturation analysis
    final_penetration = penetration.iloc[-1]
    theoretical_saturation = 100.0
    
    saturation_metrics = {
        "final_penetration_pct": final_penetration,
        "market_saturation_ratio": final_penetration / theoretical_saturation,
        "remaining_market_pct": theoretical_saturation - final_penetration,
        "saturation_velocity": penetration.diff().mean()  # Average penetration increase per period
    }
    
    # Time-based efficiency metrics
    efficiency_metrics = {}
    if milestones.get("50_percent"):
        efficiency_metrics["time_to_majority"] = milestones["50_percent"]
        efficiency_metrics["users_at_majority"] = cumulative[cumulative.index[milestones["50_percent"]-1]]
    
    # Adoption curve shape analysis
    shape_metrics = {
        "peak_period_normalized": (velocity_metrics["peak_velocity_period"] / periods) * 100,
        "curve_skewness": _calculate_curve_skewness(new_adopters),
        "adoption_concentration": _calculate_adoption_concentration(new_adopters)
    }
    
    return {
        "milestones": milestones,
        "velocity_metrics": velocity_metrics,
        "saturation_metrics": saturation_metrics,
        "efficiency_metrics": efficiency_metrics,
        "shape_metrics": shape_metrics
    }


def what_if_analysis(model: BassModel, parameter_changes: Dict[str, List[float]], 
                     metric: str = "Final Users") -> pd.DataFrame:
    """
    Perform what-if analysis by varying single parameters.
    
    Args:
        model: Base BassModel instance
        parameter_changes: Dictionary of parameter variations
                          Format: {"p": [0.01, 0.02, 0.03], "q": [0.3, 0.4, 0.5]}
        metric: Target metric to analyze
        
    Returns:
        DataFrame showing impact of parameter changes on target metric
        
    Example:
        >>> changes = {"p": [0.01, 0.02, 0.03], "M": [50000, 100000, 150000]}
        >>> results = what_if_analysis(model, changes, "Final Users")
    """
    results = []
    
    for param_name, values in parameter_changes.items():
        for value in values:
            # Create modified parameters
            params = {"M": model.M, "p": model.p, "q": model.q}
            params[param_name] = value
            
            # Create and run modified model
            test_model = BassModel(params["M"], params["p"], params["q"])
            test_model.forecast(periods=24)
            
            # Calculate target metric
            if metric == "Final Users":
                result_value = test_model.results["Cumulative Adopters"].iloc[-1]
            elif metric == "Final Penetration (%)":
                result_value = test_model.results["Market Penetration (%)"].iloc[-1]
            elif metric == "Peak Period":
                result_value = test_model.get_peak_period()["period"]
            elif metric == "Peak Adopters":
                result_value = test_model.get_peak_period()["new_adopters"]
            else:
                result_value = None
            
            results.append({
                "Parameter": param_name,
                "Value": value,
                "Base Value": getattr(model, param_name.lower()) if param_name.lower() in ["p", "q"] else model.M,
                "Change (%)": ((value - (getattr(model, param_name.lower()) if param_name.lower() in ["p", "q"] else model.M)) / 
                              (getattr(model, param_name.lower()) if param_name.lower() in ["p", "q"] else model.M)) * 100,
                metric: result_value,
                "Impact": "Positive" if result_value > (getattr(model, param_name.lower()) if param_name.lower() in ["p", "q"] else model.M) else "Negative"
            })
    
    return pd.DataFrame(results)


def generate_executive_summary(model: BassModel, requests_per_user: int = None) -> Dict:
    """
    Generate a comprehensive executive summary of the analysis.
    
    CORRECTED USAGE PATTERNS REFLECTED IN ANALYSIS:
    - ONECI: 1 request per user (registration only)
    - SmileID: 2 one-time + 1 monthly recurring per user
    - DKB: 1 signature per user (signing only)
    
    Args:
        model: BassModel instance with forecast results
        requests_per_user: DEPRECATED - usage patterns are now fixed per provider
        
    Returns:
        Dictionary with executive summary including key findings and recommendations
        
    Example:
        >>> summary = generate_executive_summary(model)
        >>> print(summary['key_findings']['market_opportunity'])
    """
    if requests_per_user is not None:
        warnings.warn("requests_per_user parameter is deprecated. Usage patterns are now fixed per provider.", 
                     DeprecationWarning, stacklevel=2)
    
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    # Get all analysis components
    pricing_summary = get_pricing_summary(model)
    roi_metrics = calculate_roi_metrics(model)
    breakeven_analysis = analyze_break_even_points(model)
    timing_metrics = calculate_market_timing_metrics(model)
    
    # Key findings
    peak_info = model.get_peak_period()
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    final_penetration = model.results["Market Penetration (%)"].iloc[-1]
    
    key_findings = {
        "market_opportunity": {
            "total_addressable_market": model.M,
            "projected_adoption": final_users,
            "market_penetration": f"{final_penetration:.1f}%",
            "peak_growth_period": f"Month {peak_info['period']}",
            "peak_monthly_adoption": peak_info['new_adopters']
        },
        "cost_analysis": {
            "most_cost_effective": pricing_summary['comparison']['cheapest_provider'],
            "potential_savings": f"{pricing_summary['comparison']['max_potential_savings_fcfa']:,.0f} FCFA",
            "cost_range": f"{min(pricing_summary['oneci']['total_cost_fcfa'], pricing_summary['smileid']['total_cost_fcfa'], pricing_summary['dkb']['total_cost_fcfa']):,.0f} - {max(pricing_summary['oneci']['total_cost_fcfa'], pricing_summary['smileid']['total_cost_fcfa'], pricing_summary['dkb']['total_cost_fcfa']):,.0f} FCFA",
            "oneci_advantage": "ONECI costs reduced due to registration-only usage pattern",
            "usage_patterns": {
                "oneci": "1 request/user (registration only)",
                "smileid": "2 one-time + 1 monthly/user",
                "dkb": "1 signature/user (signing only)"
            }
        },
        "timing_insights": {
            "time_to_majority": timing_metrics['milestones'].get('50_percent', 'Not reached'),
            "peak_adoption_period": timing_metrics['velocity_metrics']['peak_velocity_period'],
            "market_saturation_level": f"{timing_metrics['saturation_metrics']['final_penetration_pct']:.1f}%"
        }
    }
    
    # Strategic recommendations
    recommendations = []
    
    # Provider recommendation with updated context
    best_provider = pricing_summary['comparison']['cheapest_provider']
    provider_details = {
        "ONECI": "registration-only usage (1 request/user)",
        "SmileID": "comprehensive solution (2 one-time + 1 monthly/user)",
        "DKB": "signing-only usage (1 signature/user)"
    }
    
    recommendations.append({
        "category": "Vendor Selection",
        "recommendation": f"Choose {best_provider} as the primary digital signature provider",
        "rationale": f"Offers lowest total cost of ownership at {pricing_summary[best_provider.lower()]['total_cost_fcfa']:,.0f} FCFA over 24 months with {provider_details.get(best_provider, 'optimized')} pattern",
        "priority": "High"
    })
    
    # ONECI-specific recommendation if it's competitive
    if best_provider == "ONECI":
        recommendations.append({
            "category": "Implementation Strategy",
            "recommendation": "Leverage ONECI's cost advantage from registration-only usage",
            "rationale": "ONECI's corrected usage pattern (1 request/user) provides significant cost savings compared to full-service alternatives",
            "priority": "High"
        })
    
    # Timing recommendation
    if peak_info['period'] <= 6:
        recommendations.append({
            "category": "Implementation Timing",
            "recommendation": "Accelerate implementation - peak adoption occurs early",
            "rationale": f"Peak adoption in month {peak_info['period']} suggests rapid early growth",
            "priority": "High"
        })
    elif peak_info['period'] >= 18:
        recommendations.append({
            "category": "Implementation Timing", 
            "recommendation": "Gradual rollout strategy recommended",
            "rationale": f"Peak adoption in month {peak_info['period']} allows for phased implementation",
            "priority": "Medium"
        })
    
    # Budget recommendation
    if best_provider == "DKB":
        dkb_setup = roi_metrics['dkb']['dkb_specific']['setup_cost_fcfa']
        recommendations.append({
            "category": "Budget Planning",
            "recommendation": f"Prepare for upfront investment of {dkb_setup:,.0f} FCFA",
            "rationale": "DKB requires significant setup costs but offers long-term savings with signing-only usage pattern",
            "priority": "High"
        })
    
    # Market penetration recommendation
    if final_penetration < 50:
        recommendations.append({
            "category": "Market Strategy",
            "recommendation": "Implement adoption acceleration programs",
            "rationale": f"Current projection shows only {final_penetration:.1f}% market penetration",
            "priority": "Medium"
        })
    
    # Risk assessment
    risks = []
    
    # Cost volatility risk
    for provider in ['oneci', 'smileid', 'dkb']:
        volatility = roi_metrics[provider]['trend_analysis']['cost_volatility']
        if volatility > 0.5:  # High volatility threshold
            risks.append({
                "risk": f"{provider.upper()} cost volatility",
                "description": f"High cost variation (volatility: {volatility:.2f}) in {roi_metrics[provider]['efficiency_metrics']['model_type']} model",
                "mitigation": "Monitor monthly costs and consider alternative providers if costs spike",
                "severity": "Medium"
            })
    
    # Market penetration risk
    if final_penetration < 30:
        risks.append({
            "risk": "Low market penetration",
            "description": f"Projected penetration of {final_penetration:.1f}% may indicate adoption challenges",
            "mitigation": "Invest in user education and adoption incentives",
            "severity": "High"
        })
    
    # ONECI-specific risk assessment
    if best_provider == "ONECI":
        risks.append({
            "risk": "Limited functionality scope",
            "description": "ONECI only handles registration; additional solutions needed for contract signing",
            "mitigation": "Plan for complementary signing solution or hybrid approach",
            "severity": "Medium"
        })
    
    return {
        "executive_overview": {
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "model_parameters": f"Innovation: {model.p}, Imitation: {model.q}, Market: {model.M:,}",
            "analysis_period": "24 months",
            "key_metric": f"{final_users:,} projected users ({final_penetration:.1f}% penetration)",
            "usage_patterns_note": "Analysis reflects corrected usage patterns: ONECI (1 req/user), SmileID (2+1 req/user), DKB (1 sig/user)"
        },
        "key_findings": key_findings,
        "recommendations": recommendations,
        "risk_assessment": risks,
        "next_steps": [
            "Finalize vendor selection based on updated cost analysis",
            "Prepare implementation timeline aligned with adoption curve",
            "Establish budget allocation for selected provider",
            "Develop user adoption and training programs",
            "Set up monitoring and evaluation framework",
            "Plan for complementary services if using ONECI (registration-only)"
        ]
    }


# Helper functions
def _find_consecutive_periods(periods: List[int]) -> List[Tuple[int, int]]:
    """Find consecutive period ranges in a list of periods."""
    if not periods:
        return []
    
    periods = sorted(periods)
    ranges = []
    start = periods[0]
    prev = periods[0]
    
    for period in periods[1:]:
        if period != prev + 1:
            ranges.append((start, prev))
            start = period
        prev = period
    
    ranges.append((start, prev))
    return ranges


def _find_most_stable_provider(comparison_df: pd.DataFrame) -> str:
    """Find the provider with the most consecutive periods as best option."""
    provider_stability = {}
    
    for provider in ['ONECI', 'SmileID', 'DKB']:
        provider_periods = comparison_df[comparison_df['Best Option'] == provider]['Month'].tolist()
        consecutive_ranges = _find_consecutive_periods(provider_periods)
        max_consecutive = max([end - start + 1 for start, end in consecutive_ranges]) if consecutive_ranges else 0
        provider_stability[provider] = max_consecutive
    
    return max(provider_stability, key=provider_stability.get)


def _calculate_curve_skewness(adoption_data: pd.Series) -> float:
    """Calculate skewness of the adoption curve."""
    try:
        from scipy import stats
        return float(stats.skew(adoption_data.dropna()))
    except ImportError:
        # Fallback calculation if scipy not available
        mean_val = adoption_data.mean()
        std_val = adoption_data.std()
        n = len(adoption_data)
        
        if std_val == 0:
            return 0
        
        skew = ((adoption_data - mean_val) ** 3).sum() / (n * std_val ** 3)
        return float(skew)


def _calculate_adoption_concentration(adoption_data: pd.Series) -> float:
    """Calculate concentration of adoption (Gini-like coefficient)."""
    values = adoption_data.sort_values().values
    n = len(values)
    
    if n == 0 or values.sum() == 0:
        return 0
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(values)
    total = cumsum[-1]
    
    if total == 0:
        return 0
    
    # Gini coefficient calculation
    gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * values) / total) / n
    return float(gini)


def validate_analysis_inputs(model: BassModel, **kwargs) -> None:
    """
    Validate inputs for analysis functions.
    
    Note: requests_per_user parameter is deprecated since usage patterns
    are now fixed per provider (ONECI: 1 req/user, SmileID: 2+1 req/user, DKB: 1 sig/user).
    
    Args:
        model: BassModel instance
        **kwargs: Additional parameters to validate
        
    Raises:
        ValueError: If any inputs are invalid
    """
    if not isinstance(model, BassModel):
        raise ValueError("model must be a BassModel instance")
    
    if model.results is None:
        raise ValueError("Model must have forecast results. Run model.forecast() first.")
    
    for key, value in kwargs.items():
        if key == 'requests_per_user':
            warnings.warn("requests_per_user parameter is deprecated. Usage patterns are now fixed per provider.", 
                         DeprecationWarning, stacklevel=2)
            if value < 0:
                raise ValueError("requests_per_user must be non-negative (though this parameter is deprecated)")
        elif key == 'periods' and value <= 0:
            raise ValueError("periods must be positive")


def get_corrected_usage_patterns_summary() -> Dict:
    """
    Get a summary of the corrected usage patterns implemented in this version.
    
    Returns:
        Dictionary with corrected usage patterns and their business impact
    """
    return {
        "corrected_patterns": {
            "ONECI": {
                "old_pattern": "2 requests per user (registration + contract signing)",
                "new_pattern": "1 request per user (registration only)",
                "change": "Removed contract signing functionality",
                "cost_impact": "Reduced costs by ~50%"
            },
            "SmileID": {
                "pattern": "2 one-time + 1 monthly recurring per user",
                "details": "Registration + signing + payment verification",
                "change": "No change",
                "cost_impact": "Unchanged"
            },
            "DKB": {
                "pattern": "1 signature per user (contract signing only)",
                "details": "Digital signature for contract signing",
                "change": "No change", 
                "cost_impact": "Unchanged"
            }
        },
        "business_impact": {
            "oneci_competitiveness": "Significantly improved due to lower usage requirements",
            "cost_comparison": "ONECI now more likely to be cheapest option",
            "implementation_strategy": "ONECI suitable for registration-only workflows",
            "hybrid_approach": "May need complementary solution for contract signing if using ONECI"
        },
        "version_info": {
            "correction_date": "2024",
            "affected_modules": ["pricing_models.py", "analysis_tools.py", "streamlit_app.py"],
            "key_change": "ONECI usage pattern corrected from 2 requests to 1 request per user"
        }
    }