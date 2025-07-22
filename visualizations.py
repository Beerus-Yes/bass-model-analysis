"""
Visualization functions for Bass Model analysis.

This module provides comprehensive visualization capabilities for:
- Bass Model adoption curves and forecasts
- Financial analysis for all pricing models
- Comparative cost analysis between providers
- Export functionality for charts and reports

Author: Bass Model Analysis Team
Version: 1.0.0
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import seaborn as sns
from bass_model import BassModel
from pricing_models import financial_analysis, compare_pricing_models


# Set default style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Color scheme for consistent branding
COLORS = {
    'oneci': '#1f77b4',      # Blue
    'smileid': '#2ca02c',    # Green  
    'dkb': '#d62728',        # Red
    'peak': '#ff7f0e',       # Orange
    'market': '#9467bd',     # Purple
    'grid': '#cccccc'        # Light gray
}


def plot_adoption_curve(model: BassModel, figsize: Tuple[int, int] = (12, 8), 
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of the Bass Model results.
    
    Args:
        model: BassModel instance with results
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> model = BassModel(100000, 0.02, 0.4)
        >>> model.forecast(24)
        >>> fig = plot_adoption_curve(model)
        >>> plt.show()
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    periods = model.results.iloc[:, 0]
    
    # 1. New Adopters per Period
    ax1.plot(periods, model.results["New Adopters"], 'b-', linewidth=2, 
             marker='o', color=COLORS['oneci'], markersize=4)
    ax1.set_title("New Adopters per Period", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Period")
    ax1.set_ylabel("New Adopters")
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Highlight peak
    peak_info = model.get_peak_period()
    ax1.axvline(x=peak_info["period"], color=COLORS['peak'], linestyle='--', alpha=0.8)
    ax1.text(peak_info["period"], peak_info["new_adopters"] * 1.05, 
            f'Peak: {peak_info["new_adopters"]:,}', 
            horizontalalignment='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 2. Cumulative Adoption (S-Curve)
    ax2.plot(periods, model.results["Cumulative Adopters"], 'g-', linewidth=2, 
             marker='s', color=COLORS['smileid'], markersize=4)
    ax2.set_title("Cumulative Adoption (S-Curve)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Cumulative Adopters")
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Add market size line
    ax2.axhline(y=model.M, color=COLORS['dkb'], linestyle='--', alpha=0.8, 
               label=f'Market Size: {model.M:,}')
    ax2.legend()
    
    # Format y-axis for large numbers
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 3. Market Penetration
    ax3.plot(periods, model.results["Market Penetration (%)"], 'purple', 
             linewidth=2, marker='^', color=COLORS['market'], markersize=4)
    ax3.set_title("Market Penetration Over Time", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Market Penetration (%)")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Add penetration milestones
    for milestone in [25, 50, 75]:
        if model.results["Market Penetration (%)"].max() >= milestone:
            first_milestone = model.results[model.results["Market Penetration (%)"] >= milestone].iloc[0]
            ax3.axhline(y=milestone, color='gray', linestyle=':', alpha=0.5)
    
    # 4. Adoption Rate (Hazard Function)
    ax4.plot(periods, model.results["Adoption Rate"], 'orange', linewidth=2, 
             marker='d', color=COLORS['peak'], markersize=4)
    ax4.set_title("Adoption Rate (Hazard Function)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Adoption Rate")
    ax4.grid(True, alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.suptitle(f"Bass Model Analysis (p={model.p}, q={model.q}, M={model.M:,})", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Adoption curve saved to {save_path}")
    
    return fig


def plot_financial_analysis(model: BassModel, requests_per_user: int = 3, 
                           pricing_model: str = "oneci", figsize: Tuple[int, int] = (14, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create financial analysis visualization for a specific pricing model.
    
    Args:
        model: BassModel instance
        requests_per_user: Ignored - usage patterns are fixed per provider
        pricing_model: "oneci", "smileid", or "dkb"
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> fig = plot_financial_analysis(model, pricing_model="smileid")
        >>> plt.show()
    """
    financial_df = financial_analysis(model, pricing_model=pricing_model)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    months = financial_df["Month"]
    model_color = COLORS[pricing_model]
    
    # 1. Monthly Cost Evolution
    monthly_costs = pd.to_numeric(financial_df["Monthly Cost (FCFA)"])
    ax1.plot(months, monthly_costs, 'b-', linewidth=2, marker='o', 
             color=model_color, markersize=4)
    ax1.set_title("Monthly Cost Evolution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Monthly Cost (FCFA)")
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Highlight peak/min costs
    peak_cost_idx = monthly_costs.idxmax()
    min_cost_idx = monthly_costs.idxmin()
    ax1.scatter(months.iloc[peak_cost_idx], monthly_costs.iloc[peak_cost_idx], 
               color=COLORS['peak'], s=100, zorder=5)
    ax1.annotate(f'Peak: {monthly_costs.iloc[peak_cost_idx]:,.0f}', 
                xy=(months.iloc[peak_cost_idx], monthly_costs.iloc[peak_cost_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 2. Cumulative Cost
    ax2.plot(months, financial_df["Cumulative Cost (FCFA)"], 'g-', linewidth=2, 
             marker='s', color=COLORS['smileid'], markersize=4)
    ax2.set_title("Cumulative Cost", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Cumulative Cost (FCFA)")
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 3. Volume Evolution (specific to each pricing model)
    if pricing_model == "dkb":
        volume_col = "New Signatures"
        volume_title = "New Signatures per Month (DKB)"
        volume_label = "New Signatures"
    elif pricing_model == "smileid":
        volume_col = "Total Requests"
        volume_title = "Total Monthly Requests (SmileID)"
        volume_label = "Total Requests"
    else:  # oneci
        volume_col = "New User Requests"
        volume_title = "New User Requests per Month (ONECI)"
        volume_label = "New User Requests"
    
    ax3.plot(months, financial_df[volume_col], 'purple', linewidth=2, 
             marker='^', color=COLORS['market'], markersize=4)
    ax3.set_title(volume_title, fontsize=14, fontweight='bold')
    ax3.set_xlabel("Month")
    ax3.set_ylabel(volume_label)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])
    ax3.ticklabel_format(style='plain', axis='y')
    
    # 4. Tier Progression
    tier_counts = financial_df['Tier'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(tier_counts)))
    wedges, texts, autotexts = ax4.pie(tier_counts.values, labels=tier_counts.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax4.set_title("Distribution of Pricing Tiers", fontsize=14, fontweight='bold')
    
    # Improve pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    model_name = {"oneci": "ONECI", "smileid": "SmileID", "dkb": "DKB Solutions"}[pricing_model]
    plt.suptitle(f"{model_name} Financial Analysis", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{model_name} financial analysis saved to {save_path}")
    
    return fig


def create_cost_comparison_chart(model: BassModel, requests_per_user: int = 3, 
                                figsize: Tuple[int, int] = (14, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comparison chart of all three pricing models.
    
    Args:
        model: BassModel instance
        requests_per_user: Ignored - usage patterns are fixed per provider
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> fig = create_cost_comparison_chart(model)
        >>> plt.show()
    """
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid")
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    months = range(1, len(oneci_df) + 1)
    oneci_costs = [pd.to_numeric(oneci_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(oneci_df))]
    smileid_costs = [pd.to_numeric(smileid_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(smileid_df))]
    dkb_costs = [pd.to_numeric(dkb_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(dkb_df))]
    
    fig = plt.figure(figsize=figsize)
    
    plt.plot(months, oneci_costs, '-', linewidth=3, marker='o', 
             label='ONECI (2 one-time requests per new user)', 
             color=COLORS['oneci'], markersize=6)
    plt.plot(months, smileid_costs, '-', linewidth=3, marker='s', 
             label='SmileID (2 one-time + 1 monthly per user)', 
             color=COLORS['smileid'], markersize=6)
    plt.plot(months, dkb_costs, '-', linewidth=3, marker='^', 
             label='DKB Solutions (1 one-time signature per new user)', 
             color=COLORS['dkb'], markersize=6)
    
    plt.title('Monthly Cost Comparison - All Providers', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Monthly Cost (FCFA)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, color=COLORS['grid'])
    plt.ticklabel_format(style='plain', axis='y')
    
    # Find and mark crossover points
    for i in range(len(months)-1):
        month = months[i]
        
        # Check for crossovers between any two providers
        costs_this_month = [oneci_costs[i], smileid_costs[i], dkb_costs[i]]
        costs_next_month = [oneci_costs[i+1], smileid_costs[i+1], dkb_costs[i+1]]
        
        # Find if ranking changes
        current_cheapest = np.argmin(costs_this_month)
        next_cheapest = np.argmin(costs_next_month)
        
        if current_cheapest != next_cheapest:
            plt.axvline(x=month+0.5, color='gray', linestyle=':', alpha=0.7)
            plt.text(month+0.5, max(max(oneci_costs), max(smileid_costs), max(dkb_costs))*0.8,
                    'Crossover', rotation=90, ha='center', va='bottom', 
                    fontsize=9, alpha=0.7)
    
    # Add usage pattern explanation
    plt.figtext(0.02, 0.02, 
                "Note: ONECI & DKB decline as adoption curve flattens (one-time costs)\n" +
                "SmileID grows with user base (includes monthly recurring costs)", 
                fontsize=10, style='italic', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost comparison chart saved to {save_path}")
    
    return fig


def create_dashboard(model: BassModel, requests_per_user: int = 3,
                    figsize: Tuple[int, int] = (20, 12),
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive dashboard with all key visualizations.
    
    Args:
        model: BassModel instance
        requests_per_user: Ignored - usage patterns are fixed per provider
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object with 6-panel dashboard
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Adoption curve (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    periods = model.results.iloc[:, 0]
    peak_info = model.get_peak_period()
    
    ax1.plot(periods, model.results["New Adopters"], 'b-', linewidth=2, 
             marker='o', color=COLORS['oneci'], markersize=4)
    ax1.axvline(x=peak_info["period"], color=COLORS['peak'], linestyle='--', alpha=0.8)
    ax1.set_title("New Adopters per Period", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("New Adopters")
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative adoption (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(periods, model.results["Cumulative Adopters"], 'g-', linewidth=2, 
             marker='s', color=COLORS['smileid'], markersize=4)
    ax2.axhline(y=model.M, color=COLORS['dkb'], linestyle='--', alpha=0.8)
    ax2.set_title("Cumulative Adoption (S-Curve)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Cumulative Adopters")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 3. Market penetration (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(periods, model.results["Market Penetration (%)"], 'purple', 
             linewidth=2, marker='^', color=COLORS['market'], markersize=4)
    ax3.set_title("Market Penetration", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Penetration (%)")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cost comparison (bottom, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid")
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    months = range(1, len(oneci_df) + 1)
    oneci_costs = [pd.to_numeric(oneci_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(oneci_df))]
    smileid_costs = [pd.to_numeric(smileid_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(smileid_df))]
    dkb_costs = [pd.to_numeric(dkb_df.iloc[i]["Monthly Cost (FCFA)"]) for i in range(len(dkb_df))]
    
    ax4.plot(months, oneci_costs, '-', linewidth=2, marker='o', 
             label='ONECI', color=COLORS['oneci'], markersize=4)
    ax4.plot(months, smileid_costs, '-', linewidth=2, marker='s', 
             label='SmileID', color=COLORS['smileid'], markersize=4)
    ax4.plot(months, dkb_costs, '-', linewidth=2, marker='^', 
             label='DKB Solutions', color=COLORS['dkb'], markersize=4)
    
    ax4.set_title("Monthly Cost Comparison", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Monthly Cost (FCFA)")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='plain', axis='y')
    
    # 5. Key metrics summary (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')  # Hide axes for text display
    
    # Calculate key metrics
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    oneci_total = oneci_df["Cumulative Cost (FCFA)"].iloc[-1]
    smileid_total = smileid_df["Cumulative Cost (FCFA)"].iloc[-1]
    dkb_total = dkb_df["Cumulative Cost (FCFA)"].iloc[-1]
    
    costs = {"ONECI": oneci_total, "SmileID": smileid_total, "DKB": dkb_total}
    cheapest = min(costs, key=costs.get)
    
    metrics_text = f"""KEY METRICS
    
Market Size: {model.M:,}
Final Users: {final_users:,}
Penetration: {model.results["Market Penetration (%)"].iloc[-1]:.1f}%
Peak Period: Month {peak_info['period']}

TOTAL COSTS (24 months):
ONECI: {oneci_total:,.0f} FCFA
SmileID: {smileid_total:,.0f} FCFA  
DKB: {dkb_total:,.0f} FCFA

BEST OPTION: {cheapest}
"""
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.suptitle("Bass Model Analysis Dashboard", fontsize=18, fontweight='bold', y=0.98)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig


def export_results(model: BassModel, filename: str, include_charts: bool = True, 
                  requests_per_user: int = 3, pricing_model: str = "oneci"):
    """
    Export results to Excel with optional charts.
    
    Args:
        model: BassModel instance
        filename: Output filename (should end with .xlsx)
        include_charts: Whether to save charts as separate images
        requests_per_user: Ignored - usage patterns are fixed per provider
        pricing_model: Primary pricing model for detailed analysis
    """
    if model.results is None:
        raise ValueError("Must run forecast() first")
    
    # Generate financial analysis for all models
    financial_df = financial_analysis(model, pricing_model=pricing_model)
    comparison_df = compare_pricing_models(model)
    
    # Export to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Basic forecast
        model.results.to_excel(writer, sheet_name='Forecast', index=False)
        
        # Primary pricing model analysis
        financial_df.to_excel(writer, sheet_name=f'{pricing_model.upper()} Financial', index=False)
        
        # 3-way comparison
        comparison_df.to_excel(writer, sheet_name='3-Way Comparison', index=False)
        
        # Individual model sheets for complete analysis
        for pmodel in ["oneci", "smileid", "dkb"]:
            if pmodel != pricing_model:
                pmodel_df = financial_analysis(model, pricing_model=pmodel)
                pmodel_df.to_excel(writer, sheet_name=f'{pmodel.upper()} Financial', index=False)
        
        # Summary sheet
        from pricing_models import get_pricing_summary
        summary = get_pricing_summary(model)
        
        summary_data = {
            'Parameter': [
                'Market Size (M)', 'Innovation Coef (p)', 'Imitation Coef (q)', 
                'Forecast Periods', 'Total Adopters', 'Final Penetration (%)',
                'Peak Period', 'Peak New Adopters',
                'ONECI Total Cost', 'SmileID Total Cost', 'DKB Total Cost',
                'Best Overall Option'
            ],
            'Value': [
                f"{model.M:,}",
                model.p,
                model.q,
                len(model.results),
                f"{model.results['Cumulative Adopters'].iloc[-1]:,}",
                f"{model.results['Market Penetration (%)'].iloc[-1]:.1f}%",
                model.get_peak_period()['period'],
                f"{model.get_peak_period()['new_adopters']:,}",
                f"{summary['oneci']['total_cost_fcfa']:,.0f} FCFA",
                f"{summary['smileid']['total_cost_fcfa']:,.0f} FCFA", 
                f"{summary['dkb']['total_cost_fcfa']:,.0f} FCFA",
                summary['comparison']['cheapest_provider']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Results exported to {filename}")
    
    # Save charts if requested
    if include_charts:
        base_name = filename.replace('.xlsx', '')
        
        # Adoption charts
        adoption_fig = plot_adoption_curve(model, save_path=f"{base_name}_adoption.png")
        plt.close(adoption_fig)
        
        # Financial charts for each model
        for pmodel in ["oneci", "smileid", "dkb"]:
            financial_fig = plot_financial_analysis(model, pricing_model=pmodel, 
                                                   save_path=f"{base_name}_{pmodel}_financial.png")
            plt.close(financial_fig)
        
        # Comparison chart
        comparison_fig = create_cost_comparison_chart(model, 
                                                     save_path=f"{base_name}_comparison.png")
        plt.close(comparison_fig)
        
        # Dashboard
        dashboard_fig = create_dashboard(model, save_path=f"{base_name}_dashboard.png")
        plt.close(dashboard_fig)
        
        print(f"Charts saved with base name: {base_name}")


def plot_sensitivity_heatmap(model: BassModel, param_ranges: dict, 
                            metric: str = 'Total Adopters',
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing sensitivity of outcomes to parameter changes.
    
    Args:
        model: BassModel instance
        param_ranges: Dictionary with parameter ranges (e.g., {'p': [0.01, 0.02], 'q': [0.3, 0.4]})
        metric: Metric to display ('Total Adopters', 'Peak Period', 'Final Penetration (%)')
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    sensitivity_df = model.sensitivity_analysis(param_ranges)
    
    if 'p' in param_ranges and 'q' in param_ranges:
        # Create pivot table for heatmap
        pivot_df = sensitivity_df.pivot(index='q (Imitation)', 
                                       columns='p (Innovation)', 
                                       values=metric)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_df, annot=True, fmt='.0f' if 'Adopters' in metric else '.2f',
                   cmap='viridis', ax=ax)
        ax.set_title(f'Sensitivity Analysis: {metric}', fontsize=14, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity heatmap saved to {save_path}")
        
        return fig
    else:
        raise ValueError("Heatmap requires both 'p' and 'q' in param_ranges")


# Utility function to set up plotting style
def setup_plot_style():
    """Set up consistent plotting style for all visualizations."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10


# Initialize plot style when module is imported
setup_plot_style()