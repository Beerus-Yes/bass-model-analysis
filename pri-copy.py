"""
Pricing models for different service providers.

This module contains pricing functions and financial analysis for:
- ONECI: 2 one-time requests per user (registration + contract signing)
- SmileID: 2 one-time requests per user + 1 monthly recurring (payment verification)  
- DKB Solutions: 1 one-time signature per user (contract signing only)

Author: Bass Model Analysis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from bass_model import BassModel


# Pricing tier configurations
ONECI_TIERS = [
    (999, 120),
    (9999, 110), 
    (49999, 100),
    (99999, 90),
    (499999, 80),
    (999999, 70),
    (float('inf'), 60)
]

SMILEID_AUTH_TIERS = [
    (501000, 0.25, "Pay-per-use", 0),
    (2000000, 0.10, "Tier 1", 300000),
    (12000000, 0.07, "Tier 2", 600000),
    (float('inf'), 0.05, "Tier 3", 900000)
]

SMILEID_DOC_TIERS = [
    (501000, 0.25, "Pay-per-use", 0),
    (2000000, 0.25, "Tier 1", 300000),
    (12000000, 0.20, "Tier 2", 600000),
    (float('inf'), 0.15, "Tier 3", 900000)
]

DKB_SIGNATURE_TIERS = [
    (100, 550, "Custom Quote"),
    (3001, 550, "PACK STARTER"),
    (10001, 450, "PACK PRO"),
    (50001, 300, "PACK PREMIUM"),
    (100000, 205, "PACK PREMIUM PLUS"),
    (float('inf'), 150, "PACK ENTREPRISE")
]

# DKB Setup costs (in FCFA)
DKB_SETUP_COSTS = {
    'certificate': 463000,        # Electronic certificate (one-time)
    'api_integration': 4000000,   # API integration (one-time)
    'web_license': 200000,        # Annual web platform license
    'training': 150000            # Training (one-time)
}

# Currency conversion rate
USD_TO_FCFA = 600  # Approximate conversion rate


def get_oneci_tarif(volume: int) -> int:
    """
    Get ONECI unit price based on volume tiers.
    
    Volume-based pricing with decreasing rates for higher volumes.
    
    Args:
        volume: Monthly request volume
        
    Returns:
        Unit price in FCFA per request
        
    Example:
        >>> get_oneci_tarif(5000)  # Returns 110 FCFA
        >>> get_oneci_tarif(500)   # Returns 120 FCFA
    """
    for max_volume, price in ONECI_TIERS:
        if volume <= max_volume:
            return price
    return ONECI_TIERS[-1][1]  # Default to highest tier


def get_smileid_pricing(annual_volume: int, service_type: str = "authentication") -> Tuple[float, str, int]:
    """
    Get SmileID pricing based on annual volume and service type.
    
    Args:
        annual_volume: Annual request volume
        service_type: "authentication" or "document_verification"
        
    Returns:
        Tuple of (unit_price_usd, tier_name, annual_contract_fee_usd)
        
    Example:
        >>> price, tier, contract = get_smileid_pricing(1000000, "authentication")
        >>> print(f"${price} per request, {tier} tier, ${contract} annual fee")
    """
    tiers = SMILEID_AUTH_TIERS if service_type == "authentication" else SMILEID_DOC_TIERS
    
    for min_volume, unit_price, tier_name, annual_fee in tiers:
        if annual_volume >= min_volume:
            return unit_price, tier_name, annual_fee
            
    # Default to pay-per-use if below minimum
    return tiers[0][1], tiers[0][2], tiers[0][3]


def get_dkb_pricing(annual_volume: int) -> Tuple[int, str, Dict[str, int]]:
    """
    Get DKB Solutions pricing based on annual volume estimation.
    
    Note: DKB charges per signature creation (one-time per person),
    but tier determination uses annual volume estimate.
    
    Args:
        annual_volume: Estimated annual signature volume (for tier calculation)
        
    Returns:
        Tuple of (unit_price_fcfa, tier_name, setup_costs_dict)
        
    Example:
        >>> price, tier, costs = get_dkb_pricing(50000)
        >>> print(f"{price} FCFA per signature, {tier} tier")
    """
    for min_volume, unit_price, tier_name in DKB_SIGNATURE_TIERS:
        if annual_volume >= min_volume:
            return unit_price, tier_name, DKB_SETUP_COSTS.copy()
            
    # Default to custom quote
    return DKB_SIGNATURE_TIERS[0][1], DKB_SIGNATURE_TIERS[0][2], DKB_SETUP_COSTS.copy()


def financial_analysis(model: BassModel, requests_per_user: int = 3, pricing_model: str = "oneci") -> pd.DataFrame:
    """
    Calculate monthly cost based on cumulative adopters and pricing model.
    
    CORRECTED USAGE PATTERNS:
    - ONECI: 2 one-time requests per user (registration + contract signing)
    - SmileID: 2 one-time requests per user + 1 monthly recurring (payment verification)  
    - DKB: 1 one-time signature per user (contract signing only)
    
    Args:
        model: BassModel instance with forecast results
        requests_per_user: Ignored - usage patterns are now fixed per provider
        pricing_model: "oneci", "smileid", or "dkb"
        
    Returns:
        DataFrame with monthly financial analysis including:
        - Monthly costs and volume
        - Pricing tiers and unit prices
        - Cumulative costs
        - Setup costs (DKB only)
        
    Example:
        >>> model = BassModel(100000, 0.02, 0.4)
        >>> model.forecast(24)
        >>> oneci_analysis = financial_analysis(model, pricing_model="oneci")
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    if pricing_model not in ["oneci", "smileid", "dkb"]:
        raise ValueError("pricing_model must be 'oneci', 'smileid', or 'dkb'")
    
    cumulative_users = model.results["Cumulative Adopters"]
    data = []
    
    # Track one-time costs for DKB
    dkb_setup_applied = False
    
    for idx, n_cumul in enumerate(cumulative_users, start=1):
        new_adopters_this_month = model.results["New Adopters"].iloc[idx-1]
        
        if pricing_model == "oneci":
            # ONECI: 2 requests per user (registration + contract signing) - ONE TIME ONLY
            # Only NEW users generate requests (2 requests each)
            monthly_requests = new_adopters_this_month * 2  # 2 one-time requests per new user
            
            unit_price_fcfa = get_oneci_tarif(monthly_requests)
            monthly_cost = monthly_requests * unit_price_fcfa
            tier_info = "Volume tier"
            annual_contract = 0
            setup_cost = 0
            volume_display = monthly_requests
            requests_display = monthly_requests
            
        elif pricing_model == "smileid":
            # SmileID: 2 one-time requests + 1 monthly recurring per user
            # NEW users: 2 requests (registration + signing)
            # ALL users: 1 monthly payment verification
            new_user_requests = new_adopters_this_month * 2  # Registration + signing
            recurring_requests = n_cumul * 1  # Monthly payment verification for all users
            monthly_requests = new_user_requests + recurring_requests
            annual_requests = monthly_requests * 12
            
            unit_price_usd, tier_name, annual_contract_usd = get_smileid_pricing(annual_requests)
            unit_price_fcfa = unit_price_usd * USD_TO_FCFA
            monthly_cost = monthly_requests * unit_price_fcfa
            
            # Add prorated annual contract fee
            monthly_contract_fee = (annual_contract_usd * USD_TO_FCFA) / 12
            monthly_cost += monthly_contract_fee
            
            tier_info = tier_name
            annual_contract = annual_contract_usd * USD_TO_FCFA
            setup_cost = 0
            volume_display = annual_requests
            requests_display = monthly_requests
            
        else:  # DKB pricing - Only NEW adopters sign contracts
            # Get NEW adopters for this specific month
            dkb_monthly_signatures = new_adopters_this_month  # Each new person signs ONCE
            dkb_annual_estimate = dkb_monthly_signatures * 12  # For tier calculation only
            
            unit_price_fcfa, tier_name, setup_costs = get_dkb_pricing(dkb_annual_estimate)
            monthly_cost = dkb_monthly_signatures * unit_price_fcfa  # Only new adopters pay
            
            # Add setup costs only in FIRST month
            setup_cost = 0
            if idx == 1 and not dkb_setup_applied:
                setup_cost = (setup_costs['certificate'] + 
                            setup_costs['api_integration'] + 
                            setup_costs['training'])
                monthly_cost += setup_cost
                dkb_setup_applied = True
            
            # Add monthly web platform license (prorated)
            monthly_license = setup_costs['web_license'] / 12
            monthly_cost += monthly_license
            
            tier_info = tier_name
            annual_contract = setup_costs['web_license']  # Annual license fee
            volume_display = dkb_monthly_signatures
            requests_display = dkb_monthly_signatures
        
        # Append row data
        data.append([
            idx,                                    # Month
            n_cumul,                               # Cumulative Users
            requests_display,                      # Monthly Requests/Signatures
            volume_display,                        # Volume Base (for tier calculation)
            f"{unit_price_fcfa:.0f}",             # Unit Price
            tier_info,                             # Tier
            f"{monthly_cost:.0f}",                # Monthly Cost
            annual_contract if pricing_model in ["smileid", "dkb"] else 0,  # Annual Contract
            setup_cost if pricing_model == "dkb" else 0    # Setup Cost
        ])
    
    # Define columns based on pricing model
    if pricing_model == "oneci":
        columns = [
            "Month", "Cumulative Users", "New User Requests", "Volume Base",
            "Unit Price (FCFA)", "Tier", "Monthly Cost (FCFA)", "Annual Contract (FCFA)", "Setup Cost (FCFA)"
        ]
    elif pricing_model == "smileid":
        columns = [
            "Month", "Cumulative Users", "Total Requests", "Annual Requests", 
            "Unit Price (FCFA)", "Tier", "Monthly Cost (FCFA)", "Annual Contract (FCFA)", "Setup Cost (FCFA)"
        ]
    else:  # DKB
        columns = [
            "Month", "Cumulative Users", "New Signatures", "New Signatures",
            "Unit Price (FCFA)", "Tier", "Monthly Cost (FCFA)", "Annual License (FCFA)", "Setup Cost (FCFA)"
        ]
    
    df_finance = pd.DataFrame(data, columns=columns)
    
    # Convert cost columns to numeric for calculations
    df_finance["Monthly Cost (FCFA)"] = pd.to_numeric(df_finance["Monthly Cost (FCFA)"])
    df_finance["Cumulative Cost (FCFA)"] = df_finance["Monthly Cost (FCFA)"].cumsum()
    
    return df_finance


def compare_pricing_models(model: BassModel, requests_per_user: int = 3, periods: int = 24) -> pd.DataFrame:
    """
    Compare ONECI vs SmileID vs DKB pricing models side by side.
    
    CORRECTED USAGE PATTERNS:
    - ONECI: 2 one-time requests per NEW user (registration + contract signing)
    - SmileID: 2 one-time + 1 monthly recurring per user (registration + signing + payment verification)
    - DKB: 1 one-time signature per NEW user (contract signing only)
    
    Args:
        model: BassModel instance
        requests_per_user: Ignored - usage patterns are now fixed per provider
        periods: Number of periods to compare
        
    Returns:
        DataFrame comparing all three pricing models with:
        - Monthly costs for each provider
        - Volume requirements (different for each provider)
        - Best option identification
        - Potential savings calculations
        
    Example:
        >>> model = BassModel(100000, 0.02, 0.4)
        >>> model.forecast(24) 
        >>> comparison = compare_pricing_models(model)
    """
    if model.results is None:
        model.forecast(periods)
    
    # Get financial analysis for all three models (requests_per_user ignored)
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid") 
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    comparison_data = []
    
    for i in range(len(oneci_df)):
        # Extract monthly costs
        oneci_cost = pd.to_numeric(oneci_df.iloc[i]["Monthly Cost (FCFA)"])
        smileid_cost = pd.to_numeric(smileid_df.iloc[i]["Monthly Cost (FCFA)"])
        dkb_cost = pd.to_numeric(dkb_df.iloc[i]["Monthly Cost (FCFA)"])
        
        # Extract volume data (different meanings for each provider)
        oneci_volume = oneci_df.iloc[i].iloc[2]      # New user requests (2 per new user)
        smileid_volume = smileid_df.iloc[i].iloc[2]  # Total requests (new + recurring)
        dkb_volume = dkb_df.iloc[i]["New Signatures"] # NEW signatures only
        
        # Find the cheapest option this month
        costs = {"ONECI": oneci_cost, "SmileID": smileid_cost, "DKB": dkb_cost}
        cheapest = min(costs, key=costs.get)
        cheapest_cost = costs[cheapest]
        
        # Calculate potential savings
        most_expensive_cost = max(costs.values())
        max_savings = most_expensive_cost - cheapest_cost
        savings_pct = (max_savings / most_expensive_cost * 100) if most_expensive_cost > 0 else 0
        
        comparison_data.append({
            "Month": i + 1,
            "Cumulative Users": oneci_df.iloc[i]["Cumulative Users"],
            "ONECI New User Requests": oneci_volume,
            "SmileID Total Requests": smileid_volume,
            "DKB New Signatures": dkb_volume,
            "ONECI Cost (FCFA)": f"{oneci_cost:,.0f}",
            "ONECI Tier": oneci_df.iloc[i]["Tier"],
            "SmileID Cost (FCFA)": f"{smileid_cost:,.0f}",
            "SmileID Tier": smileid_df.iloc[i]["Tier"],
            "DKB Cost (FCFA)": f"{dkb_cost:,.0f}",
            "DKB Tier": dkb_df.iloc[i]["Tier"],
            "Best Option": cheapest,
            "Max Savings (FCFA)": f"{max_savings:,.0f}",
            "Max Savings %": f"{savings_pct:.1f}%"
        })
    
    return pd.DataFrame(comparison_data)


def get_pricing_summary(model: BassModel, requests_per_user: int = 3) -> Dict:
    """
    Get a comprehensive summary of pricing analysis for all models.
    
    Note: requests_per_user parameter is ignored as usage patterns are now fixed:
    - ONECI: 2 requests per new user (one-time)
    - SmileID: 2 requests per new user + 1 per user monthly (mixed)
    - DKB: 1 signature per new user (one-time)
    
    Args:
        model: BassModel instance with forecast results
        requests_per_user: Ignored - usage patterns are fixed per provider
        
    Returns:
        Dictionary with total costs, average monthly costs, and key insights
        for all three pricing models
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    # Generate financial analysis for all models (requests_per_user ignored)
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid")
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    # Calculate totals
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    
    summary = {
        "usage_patterns": {
            "oneci": "2 one-time requests per new user (registration + contract signing)",
            "smileid": "2 one-time requests per new user + 1 monthly recurring per user (payment verification)",
            "dkb": "1 one-time signature per new user (contract signing only)"
        },
        "forecast_summary": {
            "periods": len(model.results),
            "final_users": final_users,
            "final_penetration_pct": model.results["Market Penetration (%)"].iloc[-1]
        },
        "oneci": {
            "total_cost_fcfa": oneci_df["Cumulative Cost (FCFA)"].iloc[-1],
            "avg_monthly_cost_fcfa": oneci_df["Monthly Cost (FCFA)"].astype(float).mean(),
            "cost_per_user_fcfa": oneci_df["Cumulative Cost (FCFA)"].iloc[-1] / final_users,
            "final_monthly_cost_fcfa": float(oneci_df["Monthly Cost (FCFA)"].iloc[-1])
        },
        "smileid": {
            "total_cost_fcfa": smileid_df["Cumulative Cost (FCFA)"].iloc[-1], 
            "avg_monthly_cost_fcfa": smileid_df["Monthly Cost (FCFA)"].astype(float).mean(),
            "cost_per_user_fcfa": smileid_df["Cumulative Cost (FCFA)"].iloc[-1] / final_users,
            "final_monthly_cost_fcfa": float(smileid_df["Monthly Cost (FCFA)"].iloc[-1])
        },
        "dkb": {
            "total_cost_fcfa": dkb_df["Cumulative Cost (FCFA)"].iloc[-1],
            "avg_monthly_cost_fcfa": dkb_df["Monthly Cost (FCFA)"].astype(float).mean(),
            "cost_per_user_fcfa": dkb_df["Cumulative Cost (FCFA)"].iloc[-1] / final_users,
            "final_monthly_cost_fcfa": float(dkb_df["Monthly Cost (FCFA)"].iloc[-1]),
            "peak_monthly_cost_fcfa": dkb_df["Monthly Cost (FCFA)"].astype(float).max(),
            "setup_cost_fcfa": float(dkb_df["Setup Cost (FCFA)"].iloc[0]) if len(dkb_df) > 0 else 0
        }
    }
    
    # Determine best option
    costs = {
        "ONECI": summary["oneci"]["total_cost_fcfa"],
        "SmileID": summary["smileid"]["total_cost_fcfa"], 
        "DKB": summary["dkb"]["total_cost_fcfa"]
    }
    
    cheapest = min(costs, key=costs.get)
    most_expensive = max(costs, key=costs.get)
    max_savings = costs[most_expensive] - costs[cheapest]
    
    summary["comparison"] = {
        "cheapest_provider": cheapest,
        "most_expensive_provider": most_expensive,
        "max_potential_savings_fcfa": max_savings,
        "savings_percentage": (max_savings / costs[most_expensive] * 100) if costs[most_expensive] > 0 else 0
    }
    
    return summary


def validate_pricing_inputs(model: BassModel, requests_per_user: int, pricing_model: str) -> None:
    """
    Validate inputs for pricing analysis.
    
    Args:
        model: BassModel instance
        requests_per_user: Monthly requests per user (ignored but kept for compatibility)
        pricing_model: Pricing model name
        
    Raises:
        ValueError: If any inputs are invalid
    """
    if model.results is None:
        raise ValueError("Model must have forecast results. Run model.forecast() first.")
    
    if requests_per_user < 0:
        raise ValueError("requests_per_user must be non-negative")
    
    if pricing_model not in ["oneci", "smileid", "dkb"]:
        raise ValueError("pricing_model must be one of: 'oneci', 'smileid', 'dkb'")


# Utility functions for pricing tier information
def get_oneci_tier_info() -> pd.DataFrame:
    """Get ONECI pricing tier information as DataFrame."""
    return pd.DataFrame([
        {"Volume Tier": f"≤ {vol:,}" if vol != float('inf') else f"> {ONECI_TIERS[-2][0]:,}", 
         "Unit Price (FCFA)": price}
        for vol, price in ONECI_TIERS
    ])


def get_smileid_tier_info(service_type: str = "authentication") -> pd.DataFrame:
    """Get SmileID pricing tier information as DataFrame."""
    tiers = SMILEID_AUTH_TIERS if service_type == "authentication" else SMILEID_DOC_TIERS
    
    return pd.DataFrame([
        {
            "Annual Volume": f"≥ {vol:,}" if vol != float('inf') else "All volumes",
            "Unit Price (USD)": f"${price:.3f}",
            "Tier Name": tier,
            "Annual Contract (USD)": f"${contract:,}" if contract > 0 else "None"
        }
        for vol, price, tier, contract in reversed(tiers)
    ])


def get_dkb_tier_info() -> pd.DataFrame:
    """Get DKB Solutions pricing tier information as DataFrame."""
    return pd.DataFrame([
        {
            "Annual Volume": f"≥ {vol:,}" if vol != float('inf') else "All volumes",
            "Unit Price (FCFA)": f"{price:,}",
            "Package Name": tier
        }
        for vol, price, tier in reversed(DKB_SIGNATURE_TIERS)
    ])


def get_usage_pattern_summary() -> Dict:
    """
    Get a summary of the corrected usage patterns for all providers.
    
    Returns:
        Dictionary explaining how each provider charges
    """
    return {
        "corrected_usage_patterns": {
            "ONECI": {
                "description": "2 one-time requests per user",
                "breakdown": [
                    "1 request for user registration/verification",
                    "1 request for contract signing verification"
                ],
                "billing_model": "One-time charges for new users only",
                "monthly_cost_pattern": "Decreases as adoption curve flattens"
            },
            "SmileID": {
                "description": "2 one-time requests + 1 monthly recurring per user",
                "breakdown": [
                    "1 request for user registration/verification",
                    "1 request for contract signing verification", 
                    "1 monthly request for payment verification (recurring)"
                ],
                "billing_model": "Mixed: one-time setup + ongoing monthly charges",
                "monthly_cost_pattern": "Increases with user base growth"
            },
            "DKB": {
                "description": "1 one-time signature per user",
                "breakdown": [
                    "1 digital signature for contract signing only"
                ],
                "billing_model": "One-time charges for new users only + setup costs",
                "monthly_cost_pattern": "High initial setup, then decreases as adoption curve flattens"
            }
        },
        "key_differences": {
            "cost_scaling": {
                "ONECI": "Peaks with adoption peak, then declines",
                "SmileID": "Continuously grows with user base",
                "DKB": "Front-loaded costs, then declining"
            },
            "volume_impact": {
                "ONECI": "Volume discounts apply to monthly new user batches",
                "SmileID": "Volume discounts apply to total monthly requests",
                "DKB": "Volume discounts apply to new signature batches"
            }
        }
    }