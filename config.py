"""
Configuration settings for Bass Model Analysis.

This module contains all configuration parameters, constants, and settings
used throughout the Bass Model analysis system. Centralized configuration
allows for easy maintenance and customization.

Author: Bass Model Analysis Team
Version: 1.0.0
"""

import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# =============================================================================
# GENERAL SETTINGS
# =============================================================================

# Application metadata
APP_NAME = "Bass Model Analysis"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Bass Model Analysis Team"
APP_DESCRIPTION = "Comprehensive Bass Diffusion Model with pricing comparison"

# Default analysis parameters
DEFAULT_PERIODS = 24
DEFAULT_TIME_UNIT = "months"
DEFAULT_REQUESTS_PER_USER = 3  # Legacy parameter (now ignored but kept for compatibility)

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
    "market_size": 1_000_000,
    "innovation_coef": 0.01,  # External influence (p)
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
        "description": "Slow adoption, limited word-of-mouth"
    },
    "Balanced": {
        "market_size": 1_000_000,
        "innovation_coef": 0.01,
        "imitation_coef": 0.4,
        "description": "Moderate adoption with balanced influences"
    },
    "Aggressive": {
        "market_size": 1_000_000,
        "innovation_coef": 0.03,
        "imitation_coef": 0.5,
        "description": "Rapid adoption, strong viral effect"
    },
    "Tech_Startup": {
        "market_size": 100_000,
        "innovation_coef": 0.05,
        "imitation_coef": 0.8,
        "description": "Small market, heavy word-of-mouth"
    },
    "Enterprise": {
        "market_size": 10_000,
        "innovation_coef": 0.01,
        "imitation_coef": 0.2,
        "description": "Small enterprise market, slow adoption"
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

# Usage patterns for each provider (corrected patterns)
USAGE_PATTERNS = {
    "oneci": {
        "description": "2 one-time requests per user",
        "requests_per_new_user": 2,
        "monthly_recurring_per_user": 0,
        "billing_type": "one_time",
        "breakdown": [
            "1 request for user registration/verification",
            "1 request for contract signing verification"
        ]
    },
    "smileid": {
        "description": "2 one-time requests + 1 monthly recurring per user", 
        "requests_per_new_user": 2,
        "monthly_recurring_per_user": 1,
        "billing_type": "mixed",
        "breakdown": [
            "1 request for user registration/verification",
            "1 request for contract signing verification",
            "1 monthly request for payment verification (recurring)"
        ]
    },
    "dkb": {
        "description": "1 one-time signature per user",
        "requests_per_new_user": 1, 
        "monthly_recurring_per_user": 0,
        "billing_type": "one_time_with_setup",
        "breakdown": [
            "1 digital signature for contract signing only"
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
# VISUALIZATION SETTINGS  
# =============================================================================

# Color scheme for consistent branding
COLOR_PALETTE = {
    "primary": {
        "oneci": "#1f77b4",      # Professional blue
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
        "success": ["#28a745", "#20c997"],
        "warning": ["#ffc107", "#fd7e14"],
        "danger": ["#dc3545", "#e83e8c"]
    }
}

# Chart default settings
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
    "grid_alpha": 0.3
}

# Dashboard layout settings
DASHBOARD_CONFIG = {
    "figsize": (20, 12),
    "layout": (2, 3),  # 2 rows, 3 columns
    "hspace": 0.3,
    "wspace": 0.3,
    "suptitle_size": 18,
    "subplot_title_size": 12
}

# Export settings
EXPORT_SETTINGS = {
    "excel_engine": "openpyxl",
    "chart_format": "png",
    "chart_dpi": 300,
    "chart_bbox": "tight",
    "include_timestamp": True,
    "auto_open": False
}


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Sensitivity analysis default ranges
SENSITIVITY_DEFAULTS = {
    "innovation_coef_range": [0.01, 0.015, 0.02, 0.025, 0.03],
    "imitation_coef_range": [0.3, 0.35, 0.4, 0.45, 0.5],
    "market_size_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}

# Break-even analysis settings
BREAKEVEN_CONFIG = {
    "volatility_threshold": 0.5,  # High volatility threshold
    "stability_weight": 0.3,      # Weight for stability in recommendations
    "cost_weight": 0.7            # Weight for cost in recommendations
}

# Market timing milestones (penetration percentages)
MARKET_MILESTONES = [10, 25, 50, 75, 90, 95]

# ROI calculation settings
ROI_CONFIG = {
    "discount_rate": 0.1,         # 10% annual discount rate
    "analysis_period_years": 2,   # 2-year analysis period
    "high_setup_threshold": 1_000_000,  # Threshold for high setup costs (FCFA)
    "volatility_periods": 6       # Periods to analyze for cost volatility
}


# =============================================================================
# BUSINESS RULES & VALIDATION
# =============================================================================

# Provider comparison rules
COMPARISON_RULES = {
    "min_cost_difference_fcfa": 10_000,     # Minimum difference to declare a winner
    "min_cost_difference_percent": 5,        # Minimum percentage difference
    "crossover_significance_threshold": 0.1, # Minimum significance for crossovers
    "stability_minimum_periods": 3           # Minimum periods to be considered stable
}

# Recommendation logic
RECOMMENDATION_LOGIC = {
    "high_setup_cost_threshold": 2_000_000,  # FCFA - triggers budget recommendation
    "early_peak_threshold": 6,               # Months - triggers acceleration recommendation
    "late_peak_threshold": 18,               # Months - triggers gradual rollout
    "low_penetration_threshold": 30,         # Percent - triggers market strategy recommendation
    "high_volatility_threshold": 0.5         # Volatility ratio - triggers risk warning
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
    "max_imitation_coef": 3.0
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
    "backup_count": 5
}

# Debug settings
DEBUG_CONFIG = {
    "enable_debug_mode": False,
    "verbose_output": False,
    "save_intermediate_results": False,
    "debug_charts": False,
    "timing_analysis": False
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    "enable_profiling": False,
    "memory_tracking": False,
    "execution_time_warnings": True,
    "slow_operation_threshold": 5.0  # seconds
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
        "detailed_logging": True
    },
    "testing": {
        "debug": False,
        "auto_save": False,
        "show_warnings": False,
        "detailed_logging": False
    },
    "production": {
        "debug": False,
        "auto_save": True,
        "show_warnings": False,
        "detailed_logging": True
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
            "usage_pattern": USAGE_PATTERNS["oneci"]
        },
        "smileid": {
            "authentication_tiers": SMILEID_AUTHENTICATION_TIERS,
            "document_tiers": SMILEID_DOCUMENT_TIERS,
            "currency": "USD",
            "conversion_rate": CURRENCY_RATES["USD_TO_FCFA"],
            "billing_type": USAGE_PATTERNS["smileid"]["billing_type"],
            "usage_pattern": USAGE_PATTERNS["smileid"]
        },
        "dkb": {
            "signature_tiers": DKB_SIGNATURE_TIERS,
            "setup_costs": DKB_SETUP_COSTS,
            "currency": "FCFA",
            "billing_type": USAGE_PATTERNS["dkb"]["billing_type"],
            "usage_pattern": USAGE_PATTERNS["dkb"]
        }
    }
    
    if provider.lower() not in configs:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: {list(configs.keys())}")
    
    return configs[provider.lower()]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Print configuration summary if run directly
if __name__ == "__main__":
    print(f"=== {APP_NAME} v{APP_VERSION} Configuration ===")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Output Directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Default Market Size: {DEFAULT_BASS_PARAMS['market_size']:,}")
    print(f"Available Providers: ONECI, SmileID, DKB Solutions")
    print(f"Available Scenarios: {', '.join(BASS_SCENARIOS.keys())}")
    print("Configuration loaded successfully!")