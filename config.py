"""
Configuration settings for Bass Model Analysis.

This module contains all configuration parameters, constants, and settings
used throughout the Bass Model analysis system. Centralized configuration
allows for easy maintenance and customization.

UPDATED with corrected usage patterns:
- ONECI: 1 request per user (registration only)
- SmileID: 2 one-time + 1 monthly recurring per user
- DKB: 1 signature per user (signing only)

Author: Bass Model Analysis Team
Version: 1.1.0 - FIXED ONECI usage pattern
"""

import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# =============================================================================
# GENERAL SETTINGS
# =============================================================================

# Application metadata
APP_NAME = "Bass Model Analysis"
APP_VERSION = "1.1.0"  # Updated version for ONECI correction
APP_AUTHOR = "Bass Model Analysis Team"
APP_DESCRIPTION = "Comprehensive Bass Diffusion Model with pricing comparison - CORRECTED ONECI usage pattern"

# Default analysis parameters
DEFAULT_PERIODS = 24
DEFAULT_TIME_UNIT = "months"
DEFAULT_REQUESTS_PER_USER = None  # DEPRECATED - usage patterns now fixed per provider

# File paths and directories
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CHARTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "charts")
DEFAULT_REPORTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "reports")
DEFAULT_DATA_DIR = "data"

# Ensure output directories exist
for directory in [DEFAULT_OUTPUT_DIR, DEFAULT_CHARTS_DIR, DEFAULT_REPORTS_DIR, DEFAULT_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


# =============================================================================
# BASS MODEL PARAMETERS
# =============================================================================

# Default Bass Model parameters
DEFAULT_BASS_PARAMS = {
    "market_size": 500_000,
    "innovation_coef": 0.02,  # External influence (p)
    "imitation_coef": 0.4     # Word-of-mouth influence (q)
}

# Parameter validation ranges
BASS_PARAM_RANGES = {
    "market_size": {"min": 1000, "max": 10_000_000, "type": int},
    "innovation_coef": {"min": 0.001, "max": 0.1, "type": float},
    "imitation_coef": {"min": 0.1, "max": 2.0, "type": float}
}

# Common Bass Model scenarios for quick testing
BASS_SCENARIOS = {
    "Conservative": {
        "market_size": 250_000,
        "innovation_coef": 0.01,
        "imitation_coef": 0.3,
        "description": "Slow adoption, limited word-of-mouth - ONECI likely most cost-effective"
    },
    "Balanced": {
        "market_size": 500_000,
        "innovation_coef": 0.02,
        "imitation_coef": 0.4,
        "description": "Moderate adoption with balanced influences - Compare all providers"
    },
    "Aggressive": {
        "market_size": 1_000_000,
        "innovation_coef": 0.03,
        "imitation_coef": 0.5,
        "description": "Rapid adoption, strong viral effect - SmileID recurring costs may dominate"
    },
    "Tech_Startup": {
        "market_size": 100_000,
        "innovation_coef": 0.05,
        "imitation_coef": 0.8,
        "description": "Small market, heavy word-of-mouth - ONECI registration-only ideal"
    },
    "Enterprise": {
        "market_size": 10_000,
        "innovation_coef": 0.01,
        "imitation_coef": 0.2,
        "description": "Small enterprise market, slow adoption - DKB setup costs may be prohibitive"
    }
}


# =============================================================================
# PRICING MODEL CONFIGURATIONS
# =============================================================================

# Currency conversion rates
CURRENCY_RATES = {
    "USD_TO_FCFA": 600,  # Approximate conversion rate
    "EUR_TO_FCFA": 650,  # If needed for future expansion
    "FCFA": 1            # Base currency
}

# CORRECTED Usage patterns for each provider
USAGE_PATTERNS = {
    "oneci": {
        "description": "1 one-time request per user (registration only)",
        "requests_per_new_user": 1,
        "monthly_recurring_per_user": 0,
        "billing_type": "one_time",
        "cost_scaling": "decreases_with_adoption_curve",
        "breakdown": [
            "1 request for user registration/verification only"
        ],
        "use_cases": [
            "User identity verification",
            "Account creation and validation",
            "Basic KYC compliance"
        ],
        "limitations": [
            "Does not handle contract signing",
            "May need complementary solution for document workflows"
        ]
    },
    "smileid": {
        "description": "2 one-time requests + 1 monthly recurring per user",
        "requests_per_new_user": 2,
        "monthly_recurring_per_user": 1,
        "billing_type": "mixed",
        "cost_scaling": "increases_with_user_base",
        "breakdown": [
            "1 request for user registration/verification",
            "1 request for contract signing verification",
            "1 monthly request for payment verification (recurring)"
        ],
        "use_cases": [
            "Complete identity verification",
            "Document signing workflows",
            "Ongoing payment verification",
            "Comprehensive compliance"
        ],
        "advantages": [
            "Full-service solution",
            "Ongoing verification capabilities",
            "Comprehensive audit trail"
        ]
    },
    "dkb": {
        "description": "1 one-time signature per user (signing only)",
        "requests_per_new_user": 1,
        "monthly_recurring_per_user": 0,
        "billing_type": "one_time_with_setup",
        "cost_scaling": "front_loaded_then_decreases",
        "breakdown": [
            "1 digital signature for contract signing only"
        ],
        "use_cases": [
            "Legal document signing",
            "Contract execution",
            "High-value transaction authentication"
        ],
        "characteristics": [
            "High initial setup costs",
            "Specialized for signing workflows",
            "Enterprise-grade security"
        ]
    }
}

# ONECI pricing tiers (volume-based monthly pricing in FCFA)
ONECI_PRICING_TIERS = [
    {"max_volume": 999, "unit_price": 120, "tier_name": "Starter"},
    {"max_volume": 9999, "unit_price": 110, "tier_name": "Professional"}, 
    {"max_volume": 49999, "unit_price": 100, "tier_name": "Business"},
    {"max_volume": 99999, "unit_price": 90, "tier_name": "Enterprise"},
    {"max_volume": 499999, "unit_price": 80, "tier_name": "Corporate"},
    {"max_volume": 999999, "unit_price": 70, "tier_name": "Enterprise Plus"},
    {"max_volume": float('inf'), "unit_price": 60, "tier_name": "Volume"}
]

# SmileID pricing tiers (annual volume-based pricing in USD)
SMILEID_AUTHENTICATION_TIERS = [
    {"min_volume": 0, "unit_price_usd": 0.25, "tier_name": "Pay-per-use", "annual_contract_usd": 0},
    {"min_volume": 501_000, "unit_price_usd": 0.10, "tier_name": "Tier 1", "annual_contract_usd": 300_000},
    {"min_volume": 2_000_000, "unit_price_usd": 0.07, "tier_name": "Tier 2", "annual_contract_usd": 600_000},
    {"min_volume": 12_000_000, "unit_price_usd": 0.05, "tier_name": "Tier 3", "annual_contract_usd": 900_000}
]

SMILEID_DOCUMENT_TIERS = [
    {"min_volume": 0, "unit_price_usd": 0.25, "tier_name": "Pay-per-use", "annual_contract_usd": 0},
    {"min_volume": 501_000, "unit_price_usd": 0.25, "tier_name": "Tier 1", "annual_contract_usd": 300_000},
    {"min_volume": 2_000_000, "unit_price_usd": 0.20, "tier_name": "Tier 2", "annual_contract_usd": 600_000},
    {"min_volume": 12_000_000, "unit_price_usd": 0.15, "tier_name": "Tier 3", "annual_contract_usd": 900_000}
]

# DKB Solutions pricing tiers (annual volume-based pricing in FCFA)
DKB_SIGNATURE_TIERS = [
    {"min_volume": 0, "unit_price": 550, "tier_name": "Custom Quote"},
    {"min_volume": 100, "unit_price": 550, "tier_name": "PACK STARTER"},
    {"min_volume": 3_001, "unit_price": 450, "tier_name": "PACK PRO"},
    {"min_volume": 10_001, "unit_price": 300, "tier_name": "PACK PREMIUM"},
    {"min_volume": 50_001, "unit_price": 205, "tier_name": "PACK PREMIUM PLUS"},
    {"min_volume": 100_000, "unit_price": 150, "tier_name": "PACK ENTREPRISE"}
]

# DKB Setup and recurring costs (in FCFA)
DKB_SETUP_COSTS = {
    "electronic_certificate": {
        "cost": 463_000,
        "type": "one_time",
        "description": "Electronic certificate for digital signatures"
    },
    "api_integration": {
        "cost": 4_000_000,
        "type": "one_time", 
        "description": "API integration and customization"
    },
    "web_platform_license": {
        "cost": 200_000,
        "type": "annual",
        "description": "Annual web platform license"
    },
    "training": {
        "cost": 150_000,
        "type": "one_time",
        "description": "User training and onboarding"
    },
    "support_maintenance": {
        "cost_percentage": 15,
        "type": "annual_percentage",
        "description": "Support and maintenance (15% of total cost)"
    }
}


# =============================================================================
# CORRECTED BUSINESS IMPACT ANALYSIS
# =============================================================================

# Impact of ONECI correction on business scenarios
ONECI_CORRECTION_IMPACT = {
    "cost_reduction": {
        "description": "ONECI costs reduced by ~50% due to registration-only usage",
        "impact_on_comparison": "ONECI now significantly more competitive",
        "scenarios_most_affected": ["Conservative", "Tech_Startup", "Enterprise"]
    },
    "competitive_positioning": {
        "before_correction": "ONECI often more expensive than expected",
        "after_correction": "ONECI likely cheapest option in many scenarios",
        "market_implications": "Better cost-effectiveness for registration-only workflows"
    },
    "use_case_alignment": {
        "optimal_for": [
            "User onboarding workflows",
            "Identity verification processes", 
            "Account creation and KYC",
            "Applications not requiring document signing"
        ],
        "not_suitable_for": [
            "Contract signing workflows",
            "Document execution processes",
            "Complete digital signature solutions"
        ]
    }
}

# Provider recommendation matrix based on corrected patterns
PROVIDER_RECOMMENDATION_MATRIX = {
    "registration_only": {
        "primary": "ONECI",
        "rationale": "Optimized for registration workflows with lowest cost",
        "considerations": "May need additional solution for contract signing"
    },
    "full_service": {
        "primary": "SmileID", 
        "rationale": "Comprehensive solution with ongoing verification",
        "considerations": "Higher costs due to recurring monthly charges"
    },
    "signing_focused": {
        "primary": "DKB",
        "rationale": "Specialized for contract signing and legal documents",
        "considerations": "High setup costs, suitable for high-value transactions"
    },
    "hybrid_approach": {
        "primary": "ONECI + DKB",
        "rationale": "ONECI for registration, DKB for signing when needed",
        "considerations": "Requires integration between two systems"
    }
}


# =============================================================================
# VISUALIZATION SETTINGS  
# =============================================================================

# Color scheme for consistent branding
COLOR_PALETTE = {
    "primary": {
        "oneci": "#1f77b4",      # Professional blue - now more prominent due to cost advantage
        "smileid": "#2ca02c",    # Growth green
        "dkb": "#d62728",        # Strong red
        "peak": "#ff7f0e",       # Highlight orange
        "market": "#9467bd",     # Market purple
        "neutral": "#7f7f7f"     # Neutral gray
    },
    "secondary": {
        "grid": "#cccccc",       # Light grid
        "background": "#f8f9fa", # Light background
        "text": "#333333",       # Dark text
        "accent": "#17a2b8"      # Accent blue
    },
    "gradients": {
        "success": ["#28a745", "#20c997"],  # Often ONECI after correction
        "warning": ["#ffc107", "#fd7e14"],  # Mixed scenarios
        "danger": ["#dc3545", "#e83e8c"]    # High-cost scenarios
    },
    "correction_highlight": {
        "oneci_improved": "#28a745",        # Green for improvement
        "correction_badge": "#17a2b8"       # Blue for correction indicators
    }
}

# Chart default settings with ONECI correction annotations
CHART_DEFAULTS = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "font_size": 10,
    "title_size": 14,
    "label_size": 12,
    "legend_size": 10,
    "line_width": 2,
    "marker_size": 6,
    "grid_alpha": 0.3,
    "correction_annotations": {
        "show_oneci_correction": True,
        "correction_color": "#17a2b8",
        "correction_style": "dashed"
    }
}

# Dashboard layout settings
DASHBOARD_CONFIG = {
    "figsize": (20, 12),
    "layout": (2, 3),  # 2 rows, 3 columns
    "hspace": 0.3,
    "wspace": 0.3,
    "suptitle_size": 18,
    "subplot_title_size": 12,
    "show_correction_banner": True
}

# Export settings
EXPORT_SETTINGS = {
    "excel_engine": "openpyxl",
    "chart_format": "png",
    "chart_dpi": 300,
    "chart_bbox": "tight",
    "include_timestamp": True,
    "auto_open": False,
    "include_correction_notes": True
}


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Sensitivity analysis default ranges (updated for ONECI correction impact)
SENSITIVITY_DEFAULTS = {
    "innovation_coef_range": [0.01, 0.015, 0.02, 0.025, 0.03],
    "imitation_coef_range": [0.3, 0.35, 0.4, 0.45, 0.5],
    "market_size_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "oneci_correction_impact": "High - expect ONECI to dominate in more scenarios"
}

# Break-even analysis settings
BREAKEVEN_CONFIG = {
    "volatility_threshold": 0.5,  # High volatility threshold
    "stability_weight": 0.3,      # Weight for stability in recommendations
    "cost_weight": 0.7,           # Weight for cost in recommendations
    "oneci_advantage_threshold": 0.15  # 15% cost advantage threshold for ONECI recommendations
}

# Market timing milestones (penetration percentages)
MARKET_MILESTONES = [10, 25, 50, 75, 90, 95]

# ROI calculation settings
ROI_CONFIG = {
    "discount_rate": 0.1,         # 10% annual discount rate
    "analysis_period_years": 2,   # 2-year analysis period
    "high_setup_threshold": 1_000_000,  # Threshold for high setup costs (FCFA)
    "volatility_periods": 6,      # Periods to analyze for cost volatility
    "oneci_efficiency_bonus": 1.1 # 10% efficiency bonus for simplified workflow
}


# =============================================================================
# BUSINESS RULES & VALIDATION
# =============================================================================

# Provider comparison rules (updated for ONECI correction)
COMPARISON_RULES = {
    "min_cost_difference_fcfa": 10_000,     # Minimum difference to declare a winner
    "min_cost_difference_percent": 5,        # Minimum percentage difference
    "crossover_significance_threshold": 0.1, # Minimum significance for crossovers
    "stability_minimum_periods": 3,          # Minimum periods to be considered stable
    "oneci_bias_correction": True,           # Account for ONECI's corrected lower costs
    "usage_pattern_weight": 0.2              # Weight functional suitability in recommendations
}

# Recommendation logic (updated for corrected patterns)
RECOMMENDATION_LOGIC = {
    "high_setup_cost_threshold": 2_000_000,  # FCFA - triggers budget recommendation
    "early_peak_threshold": 6,               # Months - triggers acceleration recommendation
    "late_peak_threshold": 18,               # Months - triggers gradual rollout
    "low_penetration_threshold": 30,         # Percent - triggers market strategy recommendation
    "high_volatility_threshold": 0.5,        # Volatility ratio - triggers risk warning
    "oneci_registration_only_scenarios": [   # When to strongly recommend ONECI
        "user_onboarding_focused",
        "identity_verification_only", 
        "account_creation_workflows",
        "kyc_compliance_basic"
    ],
    "hybrid_approach_scenarios": [           # When to suggest ONECI + complementary solution
        "mixed_workflows",
        "registration_plus_occasional_signing",
        "cost_optimization_focus"
    ]
}

# Validation rules
VALIDATION_RULES = {
    "min_market_size": 1_000,
    "max_market_size": 50_000_000,
    "min_periods": 6,
    "max_periods": 120,
    "min_innovation_coef": 0.001,
    "max_innovation_coef": 0.2,
    "min_imitation_coef": 0.05,
    "max_imitation_coef": 3.0,
    "usage_pattern_consistency": True  # Ensure corrected patterns are used consistently
}


# =============================================================================
# LOGGING & DEBUG SETTINGS
# =============================================================================

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": os.path.join(DEFAULT_OUTPUT_DIR, "bass_model_analysis.log"),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "include_correction_logs": True
}

# Debug settings
DEBUG_CONFIG = {
    "enable_debug_mode": False,
    "verbose_output": False,
    "save_intermediate_results": False,
    "debug_charts": False,
    "timing_analysis": False,
    "track_usage_pattern_corrections": True
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    "enable_profiling": False,
    "memory_tracking": False,
    "execution_time_warnings": True,
    "slow_operation_threshold": 5.0,  # seconds
    "correction_impact_monitoring": True
}


# =============================================================================
# ENVIRONMENT & DEPLOYMENT
# =============================================================================

# Environment detection
ENVIRONMENT = os.getenv("BASS_MODEL_ENV", "development")  # development, testing, production

# Environment-specific settings
ENV_SETTINGS = {
    "development": {
        "debug": True,
        "auto_save": True,
        "show_warnings": True,
        "detailed_logging": True,
        "show_correction_details": True
    },
    "testing": {
        "debug": False,
        "auto_save": False,
        "show_warnings": False,
        "detailed_logging": False,
        "show_correction_details": False
    },
    "production": {
        "debug": False,
        "auto_save": True,
        "show_warnings": False,
        "detailed_logging": True,
        "show_correction_details": True
    }
}

# Get current environment settings
CURRENT_ENV = ENV_SETTINGS.get(ENVIRONMENT, ENV_SETTINGS["development"])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_default_filename(base_name: str = "bass_analysis") -> str:
    """
    Generate a default filename with timestamp.
    
    Args:
        base_name: Base name for the file
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def get_output_path(filename: str, subdirectory: str = "") -> str:
    """
    Get full output path for a file.
    
    Args:
        filename: Name of the file
        subdirectory: Optional subdirectory (charts, reports, etc.)
        
    Returns:
        Full file path
    """
    if subdirectory:
        directory = os.path.join(DEFAULT_OUTPUT_DIR, subdirectory)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    else:
        return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def validate_bass_parameters(market_size: int, innovation_coef: float, imitation_coef: float) -> Dict[str, bool]:
    """
    Validate Bass Model parameters against defined rules.
    
    Args:
        market_size: Total addressable market
        innovation_coef: Innovation coefficient (p)
        imitation_coef: Imitation coefficient (q)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "market_size": VALIDATION_RULES["min_market_size"] <= market_size <= VALIDATION_RULES["max_market_size"],
        "innovation_coef": VALIDATION_RULES["min_innovation_coef"] <= innovation_coef <= VALIDATION_RULES["max_innovation_coef"],
        "imitation_coef": VALIDATION_RULES["min_imitation_coef"] <= imitation_coef <= VALIDATION_RULES["max_imitation_coef"],
        "all_valid": True
    }
    
    validation_results["all_valid"] = all(validation_results[key] for key in validation_results if key != "all_valid")
    
    return validation_results


def get_pricing_config(provider: str) -> Dict:
    """
    Get pricing configuration for a specific provider.
    
    UPDATED to reflect corrected usage patterns:
    - ONECI: 1 request per user (registration only)
    - SmileID: 2 one-time + 1 monthly recurring per user
    - DKB: 1 signature per user (signing only)
    
    Args:
        provider: Provider name ("oneci", "smileid", "dkb")
        
    Returns:
        Provider-specific configuration dictionary
        
    Raises:
        ValueError: If provider is not recognized
    """
    configs = {
        "oneci": {
            "tiers": ONECI_PRICING_TIERS,
            "currency": "FCFA",
            "billing_type": USAGE_PATTERNS["oneci"]["billing_type"],
            "usage_pattern": USAGE_PATTERNS["oneci"],
            "correction_notes": "CORRECTED: Reduced from 2 to 1 request per user (registration only)"
        },
        "smileid": {
            "authentication_tiers": SMILEID_AUTHENTICATION_TIERS,
            "document_tiers": SMILEID_DOCUMENT_TIERS,
            "currency": "USD",
            "conversion_rate": CURRENCY_RATES["USD_TO_FCFA"],
            "billing_type": USAGE_PATTERNS["smileid"]["billing_type"],
            "usage_pattern": USAGE_PATTERNS["smileid"],
            "correction_notes": "No change - pattern confirmed as 2 one-time + 1 monthly recurring"
        },
        "dkb": {
            "signature_tiers": DKB_SIGNATURE_TIERS,
            "setup_costs": DKB_SETUP_COSTS,
            "currency": "FCFA",
            "billing_type": USAGE_PATTERNS["dkb"]["billing_type"],
            "usage_pattern": USAGE_PATTERNS["dkb"],
            "correction_notes": "No change - pattern confirmed as 1 signature per user"
        }
    }
    
    if provider.lower() not in configs:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: {list(configs.keys())}")
    
    return configs[provider.lower()]


def get_corrected_usage_summary() -> Dict:
    """
    Get a summary of the corrected usage patterns and their business impact.
    
    Returns:
        Dictionary with correction details and impact analysis
    """
    return {
        "correction_summary": {
            "date": "2024",
            "primary_change": "ONECI usage pattern corrected from 2 to 1 request per user",
            "affected_modules": ["pricing_models.py", "analysis_tools.py", "streamlit_app.py", "config.py"],
            "business_impact": "Significant - ONECI now much more cost-competitive"
        },
        "corrected_patterns": USAGE_PATTERNS,
        "recommendation_changes": {
            "before": "ONECI often not cost-competitive due to 2x request volume",
            "after": "ONECI likely cheapest option for registration-only workflows",
            "scenarios_most_affected": BASS_SCENARIOS
        },
        "implementation_guidance": PROVIDER_RECOMMENDATION_MATRIX,
        "oneci_specific_impact": ONECI_CORRECTION_IMPACT
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Print configuration summary if run directly
if __name__ == "__main__":
    print(f"=== {APP_NAME} v{APP_VERSION} Configuration ===")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Output Directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Default Market Size: {DEFAULT_BASS_PARAMS['market_size']:,}")
    print(f"Available Providers: ONECI (CORRECTED), SmileID, DKB Solutions")
    print(f"Available Scenarios: {', '.join(BASS_SCENARIOS.keys())}")
    print()
    print("🔧 CORRECTED USAGE PATTERNS:")
    for provider, pattern in USAGE_PATTERNS.items():
        print(f"  • {provider.upper()}: {pattern['description']}")
    print()
    print("✅ Configuration loaded successfully with ONECI correction!")
    print("📊 ONECI costs reduced ~50% due to registration-only usage pattern")
    print("🏆 ONECI now significantly more competitive in cost analysis")