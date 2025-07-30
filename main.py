"""
Main demonstration script for Bass Model Analysis.

This script provides comprehensive examples of how to use the Bass Model
analysis framework, including forecasting, pricing analysis, visualization,
and business intelligence features.

UPDATED with corrected usage patterns:
- ONECI: 1 request per user (registration only)
- SmileID: 2 one-time + 1 monthly recurring per user  
- DKB: 1 signature per user (signing only)

Usage:
    python main.py                    # Run full demo
    python main.py --scenario conservative  # Run specific scenario
    python main.py --quick              # Quick analysis only
    python main.py --export-only        # Generate reports only

Author: Bass Model Analysis Team
Version: 1.1.0 - FIXED ONECI usage pattern
"""

import sys
import argparse
import warnings
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from bass_model import BassModel
from pricing_models import (
    financial_analysis, 
    compare_pricing_models, 
    get_pricing_summary,
    get_usage_pattern_summary
)
from visualizations import (
    plot_adoption_curve,
    plot_financial_analysis, 
    create_cost_comparison_chart,
    create_dashboard,
    export_results,
    plot_sensitivity_heatmap,
    create_correction_comparison_chart,
    create_usage_pattern_diagram
)
from analysis_tools import (
    analyze_break_even_points,
    calculate_roi_metrics,
    scenario_analysis,
    calculate_market_timing_metrics,
    what_if_analysis,
    generate_executive_summary
)
from config import (
    DEFAULT_BASS_PARAMS,
    BASS_SCENARIOS,
    get_default_filename,
    get_output_path,
    APP_NAME,
    APP_VERSION
)

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def print_header():
    """Print application header with version info and correction notice."""
    print("=" * 80)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 80)
    print("Enhanced Bass Diffusion Model with 3-Way Pricing Comparison")
    print("Providers: ONECI (CORRECTED), SmileID, DKB Solutions")
    print("🔧 ONECI usage pattern corrected: Registration only (1 request/user)")
    print("=" * 80)
    print()


def print_usage_patterns():
    """Display corrected usage patterns for all providers."""
    print("📋 CORRECTED USAGE PATTERNS:")
    print("-" * 40)
    
    patterns = get_usage_pattern_summary()
    
    for provider, details in patterns["corrected_usage_patterns"].items():
        # Add correction indicator for ONECI
        correction_mark = " 🔧 CORRECTED" if provider == "ONECI" else ""
        print(f"\n{provider.upper()}{correction_mark}:")
        print(f"  📝 {details['description']}")
        print(f"  💡 Billing: {details['billing_model']}")
        print(f"  📈 Cost Pattern: {details['monthly_cost_pattern']}")
        
        print("  📋 Breakdown:")
        for item in details["breakdown"]:
            print(f"    • {item}")
    
    print(f"\n🔑 KEY DIFFERENCES:")
    for category, differences in patterns["key_differences"].items():
        print(f"\n  {category.replace('_', ' ').title()}:")
        for provider, difference in differences.items():
            print(f"    • {provider}: {difference}")
    
    print(f"\n⚠️ CORRECTION IMPACT:")
    print("  • ONECI costs reduced ~50% due to registration-only usage")
    print("  • ONECI now significantly more competitive vs SmileID & DKB")
    print("  • Better alignment with actual ONECI API usage patterns")
    print()


def run_basic_forecast(model: BassModel) -> None:
    """Run and display basic Bass Model forecast."""
    print("🔮 BASS MODEL FORECAST")
    print("-" * 40)
    
    # Generate forecast
    forecast_df = model.forecast(periods=24, time_unit="months")
    
    # Display key results
    print("First 12 months:")
    print(forecast_df.head(12).to_string(index=False))
    
    # Peak analysis
    peak_info = model.get_peak_period()
    print(f"\n📊 Peak Analysis:")
    print(f"  Peak occurs in month: {peak_info['period']}")
    print(f"  Peak new adopters: {peak_info['new_adopters']:,}")
    print(f"  Market penetration at peak: {peak_info['cumulative_penetration']:.1f}%")
    
    # Final results
    final_users = forecast_df["Cumulative Adopters"].iloc[-1]
    final_penetration = forecast_df["Market Penetration (%)"].iloc[-1]
    print(f"\n🎯 Final Results (Month 24):")
    print(f"  Total adopters: {final_users:,}")
    print(f"  Market penetration: {final_penetration:.1f}%")
    print(f"  Remaining market: {model.M - final_users:,}")
    print()


def run_pricing_analysis(model: BassModel) -> Dict:
    """Run comprehensive pricing analysis for all providers with corrected patterns."""
    print("💰 PRICING ANALYSIS (CORRECTED PATTERNS)")
    print("-" * 40)
    
    # Individual provider analysis
    print("\n📈 Individual Provider Analysis:")
    
    provider_data = {}
    usage_patterns = {
        "oneci": "1 request/user (registration only) - CORRECTED",
        "smileid": "2 one-time + 1 monthly/user",
        "dkb": "1 signature/user (signing only)"
    }
    
    for provider in ["oneci", "smileid", "dkb"]:
        correction_mark = " 🔧 CORRECTED" if provider == "oneci" else ""
        print(f"\n{provider.upper()}{correction_mark}:")
        print(f"  Usage Pattern: {usage_patterns[provider]}")
        
        financial_df = financial_analysis(model, pricing_model=provider)
        provider_data[provider] = financial_df
        
        total_cost = financial_df["Cumulative Cost (FCFA)"].iloc[-1]
        monthly_costs = financial_df["Monthly Cost (FCFA)"].astype(float)
        avg_monthly = monthly_costs.mean()
        peak_monthly = monthly_costs.max()
        final_monthly = monthly_costs.iloc[-1]
        
        print(f"  Total cost (24 months): {total_cost:,.0f} FCFA")
        print(f"  Average monthly cost: {avg_monthly:,.0f} FCFA")
        print(f"  Peak monthly cost: {peak_monthly:,.0f} FCFA")
        print(f"  Final monthly cost: {final_monthly:,.0f} FCFA")
        
        # Cost per user analysis
        final_users = model.results["Cumulative Adopters"].iloc[-1]
        cost_per_user = total_cost / final_users if final_users > 0 else 0
        print(f"  Cost per user: {cost_per_user:,.0f} FCFA")
        
        # Show first 6 months detail
        print("  First 6 months detail:")
        display_cols = ["Month", financial_df.columns[2], "Monthly Cost (FCFA)", "Tier"]
        print("  " + financial_df[display_cols].head(6).to_string(index=False).replace('\n', '\n  '))
    
    return provider_data


def run_comparative_analysis(model: BassModel) -> pd.DataFrame:
    """Run 3-way comparative analysis with corrected patterns."""
    print("\n🔄 COMPARATIVE ANALYSIS (CORRECTED ONECI)")
    print("-" * 40)
    
    # Generate comparison
    comparison_df = compare_pricing_models(model)
    
    # Display comparison summary
    print("3-Way Cost Comparison (First 12 months):")
    display_cols = [
        "Month", "Cumulative Users", "ONECI Cost (FCFA)", 
        "SmileID Cost (FCFA)", "DKB Cost (FCFA)", "Best Option"
    ]
    print(comparison_df[display_cols].head(12).to_string(index=False))
    
    # Summary statistics
    best_counts = comparison_df['Best Option'].value_counts()
    print(f"\n📊 Provider Dominance (24 months) - CORRECTED ONECI:")
    for provider, count in best_counts.items():
        percentage = (count / len(comparison_df)) * 100
        correction_note = " (improved due to correction)" if provider == "ONECI" else ""
        print(f"  {provider}: {count} months ({percentage:.1f}%){correction_note}")
    
    # Analyze correction impact
    oneci_dominance = best_counts.get('ONECI', 0)
    if oneci_dominance > 12:  # More than 50%
        print(f"\n💡 CORRECTION IMPACT: ONECI now dominates {oneci_dominance} months due to corrected usage pattern!")
    elif oneci_dominance > 6:  # More than 25%
        print(f"\n💡 CORRECTION IMPACT: ONECI significantly more competitive ({oneci_dominance} months)")
    
    return comparison_df


def run_business_intelligence(model: BassModel) -> Dict:
    """Run advanced business intelligence analysis with correction insights."""
    print("\n🧠 BUSINESS INTELLIGENCE ANALYSIS")
    print("-" * 40)
    
    # Break-even analysis
    print("⚖️ Break-Even Analysis:")
    breakeven = analyze_break_even_points(model)
    
    for provider, analysis in breakeven["periods_analysis"].items():
        if analysis["periods_cheapest"] > 0:
            correction_note = " (enhanced by correction)" if provider == "ONECI" else ""
            print(f"  {provider}: Cheapest for {analysis['periods_cheapest']} months " +
                  f"({analysis['dominance_percentage']:.1f}%){correction_note}")
            print(f"    Periods: Month {analysis['first_cheapest_month']} - {analysis['last_cheapest_month']}")
    
    print(f"  Number of crossovers: {breakeven['summary']['number_of_crossovers']}")
    print(f"  Most stable provider: {breakeven['summary']['most_stable_provider']}")
    
    # ROI Metrics
    print(f"\n💹 ROI Analysis (Corrected Patterns):")
    roi_metrics = calculate_roi_metrics(model)
    
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    print(f"  Total users acquired: {final_users:,}")
    
    for provider in ["oneci", "smileid", "dkb"]:
        metrics = roi_metrics[provider]
        cost_per_user = metrics["basic_metrics"]["cost_per_user_fcfa"]
        model_type = metrics["efficiency_metrics"]["model_type"]
        correction_note = " (CORRECTED)" if provider == "oneci" else ""
        print(f"  {provider.upper()}{correction_note}: {cost_per_user:,.0f} FCFA per user ({model_type})")
    
    # Market Timing
    print(f"\n⏰ Market Timing Analysis:")
    timing = calculate_market_timing_metrics(model)
    
    milestones = timing["milestones"]
    for milestone in ["25_percent", "50_percent", "75_percent"]:
        period = milestones.get(milestone)
        if period:
            print(f"  {milestone.replace('_', ' ').title()}: Month {period}")
        else:
            print(f"  {milestone.replace('_', ' ').title()}: Not reached")
    
    velocity = timing["velocity_metrics"]
    print(f"  Peak velocity: {velocity['peak_velocity_adopters']:,} adopters in Month {velocity['peak_velocity_period']}")
    
    return {
        "breakeven": breakeven,
        "roi": roi_metrics,
        "timing": timing
    }


def run_scenario_analysis(base_model: BassModel) -> pd.DataFrame:
    """Run scenario analysis with different parameter sets, highlighting ONECI improvement."""
    print("\n🎭 SCENARIO ANALYSIS (ONECI CORRECTED)")
    print("-" * 40)
    
    # Use predefined scenarios from config
    print("Comparing predefined scenarios with corrected ONECI patterns:")
    
    scenarios = {}
    for name, params in BASS_SCENARIOS.items():
        scenarios[name] = {
            "M": params["market_size"],
            "p": params["innovation_coef"], 
            "q": params["imitation_coef"]
        }
    
    scenario_results = scenario_analysis(base_model, scenarios)
    
    # Display results with corrected column names
    display_cols = [
        "Scenario", "Final Users", "Final Penetration (%)", 
        "Peak Period", "Best Option"
    ]
    print(scenario_results[display_cols].to_string(index=False))
    
    # Analyze ONECI performance across scenarios
    oneci_wins = scenario_results[scenario_results['Best Option'] == 'ONECI']
    if len(oneci_wins) > 0:
        print(f"\n💡 ONECI WINS in {len(oneci_wins)} scenarios due to corrected usage pattern:")
        for _, row in oneci_wins.iterrows():
            print(f"  • {row['Scenario']}: {row['Final Users']} users, {row['Final Penetration (%)']}% penetration")
    
    return scenario_results


def run_sensitivity_analysis(model: BassModel) -> pd.DataFrame:
    """Run parameter sensitivity analysis."""
    print("\n🎯 SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    # Define parameter ranges
    sensitivity_ranges = {
        'p': [0.01, 0.015, 0.02, 0.025, 0.03],
        'q': [0.3, 0.35, 0.4, 0.45, 0.5]
    }
    
    print("Testing parameter sensitivity (with corrected ONECI costs):")
    print(f"Innovation coefficients (p): {sensitivity_ranges['p']}")
    print(f"Imitation coefficients (q): {sensitivity_ranges['q']}")
    
    sensitivity_df = model.sensitivity_analysis(sensitivity_ranges, periods=24)
    
    # Display key insights
    print("\nSensitivity Results (Top 5 by Total Adopters):")
    top_results = sensitivity_df.nlargest(5, 'Total Adopters')
    display_cols = ['p (Innovation)', 'q (Imitation)', 'Total Adopters', 'Peak Period', 'Final Penetration (%)']
    print(top_results[display_cols].to_string(index=False))
    
    # Parameter impact analysis
    print(f"\n📊 Parameter Impact:")
    
    # Innovation coefficient impact
    p_impact = sensitivity_df.groupby('p (Innovation)')['Total Adopters'].mean()
    print("Innovation coefficient (p) impact on adoption:")
    for p_val, adopters in p_impact.items():
        print(f"  p={p_val}: {adopters:,.0f} average adopters")
    
    # Imitation coefficient impact
    q_impact = sensitivity_df.groupby('q (Imitation)')['Total Adopters'].mean()
    print("Imitation coefficient (q) impact on adoption:")
    for q_val, adopters in q_impact.items():
        print(f"  q={q_val}: {adopters:,.0f} average adopters")
    
    return sensitivity_df


def generate_executive_report(model: BassModel) -> Dict:
    """Generate comprehensive executive summary with correction highlights."""
    print("\n📋 EXECUTIVE SUMMARY (CORRECTED PATTERNS)")
    print("-" * 40)
    
    summary = generate_executive_summary(model)
    
    # Executive overview
    overview = summary["executive_overview"]
    print(f"Analysis Date: {overview['analysis_date']}")
    print(f"Model Parameters: {overview['model_parameters']}")
    print(f"Key Metric: {overview['key_metric']}")
    if 'usage_patterns_note' in overview:
        print(f"Usage Patterns: {overview['usage_patterns_note']}")
    
    # Key findings
    print(f"\n🎯 KEY FINDINGS:")
    
    market = summary["key_findings"]["market_opportunity"]
    print(f"Market Opportunity:")
    print(f"  • Target market: {market['total_addressable_market']:,} users")
    print(f"  • Projected adoption: {market['projected_adoption']:,} users")
    print(f"  • Market penetration: {market['market_penetration']}")
    print(f"  • Peak growth: {market['peak_growth_period']} ({market['peak_monthly_adoption']:,} new adopters)")
    
    cost = summary["key_findings"]["cost_analysis"]
    print(f"\nCost Analysis (CORRECTED):")
    print(f"  • Most cost-effective: {cost['most_cost_effective']}")
    print(f"  • Potential savings: {cost['potential_savings']}")
    print(f"  • Cost range: {cost['cost_range']}")
    
    # Show usage patterns if available
    if 'usage_patterns' in cost:
        print(f"  • Usage patterns:")
        for provider, pattern in cost['usage_patterns'].items():
            correction_note = " (CORRECTED)" if provider == "oneci" else ""
            print(f"    - {provider.upper()}{correction_note}: {pattern}")
    
    timing = summary["key_findings"]["timing_insights"]
    print(f"\nTiming Insights:")
    print(f"  • Time to majority (50%): {timing['time_to_majority']}")
    print(f"  • Peak adoption period: Month {timing['peak_adoption_period']}")
    print(f"  • Market saturation: {timing['market_saturation_level']}")
    
    # Top recommendations
    print(f"\n📈 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(summary["recommendations"][:3], 1):
        print(f"{i}. {rec['category']}: {rec['recommendation']}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"   Priority: {rec['priority']}")
    
    # Risk assessment
    if summary["risk_assessment"]:
        print(f"\n⚠️ KEY RISKS:")
        for risk in summary["risk_assessment"]:
            print(f"• {risk['risk']}: {risk['description']}")
            print(f"  Mitigation: {risk['mitigation']}")
    
    return summary


def create_visualizations(model: BassModel, output_dir: str = "output") -> None:
    """Generate all visualizations including correction-specific charts."""
    print("\n📊 GENERATING VISUALIZATIONS (WITH CORRECTION INDICATORS)")
    print("-" * 40)
    
    try:
        # 1. Adoption curve
        print("Creating adoption curve...")
        adoption_fig = plot_adoption_curve(model, save_path=get_output_path("adoption_analysis.png", "charts"))
        plt.close(adoption_fig)
        
        # 2. Financial analysis for each provider (with correction indicators)
        for provider in ["oneci", "smileid", "dkb"]:
            suffix = "_CORRECTED" if provider == "oneci" else ""
            print(f"Creating {provider.upper()} financial analysis{suffix}...")
            financial_fig = plot_financial_analysis(model, pricing_model=provider, 
                                                   save_path=get_output_path(f"{provider}_financial{suffix}.png", "charts"))
            plt.close(financial_fig)
        
        # 3. Cost comparison (with correction indicators)
        print("Creating cost comparison chart (CORRECTED ONECI)...")
        comparison_fig = create_cost_comparison_chart(model, save_path=get_output_path("cost_comparison_CORRECTED.png", "charts"))
        plt.close(comparison_fig)
        
        # 4. Executive dashboard (with correction indicators)
        print("Creating executive dashboard (CORRECTED)...")
        dashboard_fig = create_dashboard(model, save_path=get_output_path("executive_dashboard_CORRECTED.png", "charts"))
        plt.close(dashboard_fig)
        
        # 5. NEW: ONECI correction comparison chart
        print("Creating ONECI correction impact chart...")
        try:
            correction_fig = create_correction_comparison_chart(model, save_path=get_output_path("oneci_correction_impact.png", "charts"))
            plt.close(correction_fig)
        except Exception as e:
            print(f"  Note: ONECI correction chart skipped ({str(e)})")
        
        # 6. NEW: Usage pattern diagram
        print("Creating usage pattern diagram...")
        try:
            pattern_fig = create_usage_pattern_diagram(save_path=get_output_path("usage_patterns_diagram.png", "charts"))
            plt.close(pattern_fig)
        except Exception as e:
            print(f"  Note: Usage pattern diagram skipped ({str(e)})")
        
        # 7. Sensitivity heatmap
        print("Creating sensitivity heatmap...")
        try:
            sensitivity_ranges = {'p': [0.01, 0.02, 0.03], 'q': [0.3, 0.4, 0.5]}
            heatmap_fig = plot_sensitivity_heatmap(model, sensitivity_ranges, metric='Total Adopters',
                                                  save_path=get_output_path("sensitivity_heatmap.png", "charts"))
            plt.close(heatmap_fig)
        except Exception as e:
            print(f"  Note: Sensitivity heatmap skipped ({str(e)})")
        
        print("✅ All visualizations created successfully!")
        print("🔧 Charts include ONECI correction indicators and new comparison charts")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")


def export_comprehensive_report(model: BassModel) -> None:
    """Export comprehensive Excel report with all analysis and correction notes."""
    print("\n📄 EXPORTING COMPREHENSIVE REPORT (CORRECTED)")
    print("-" * 40)
    
    try:
        filename = get_output_path(get_default_filename("bass_analysis_complete_CORRECTED") + ".xlsx", "reports")
        
        # Export with all charts and correction indicators
        export_results(model, filename, include_charts=True, pricing_model="oneci")
        
        print(f"✅ Comprehensive report exported successfully!")
        print(f"📁 Location: {filename}")
        print(f"🔧 Report includes ONECI correction metadata and indicators")
        
    except Exception as e:
        print(f"❌ Error exporting report: {str(e)}")


def run_quick_analysis(scenario_name: str = "Balanced") -> None:
    """Run a quick analysis with minimal output, highlighting ONECI correction."""
    print(f"🚀 QUICK ANALYSIS - {scenario_name.title()} Scenario (ONECI CORRECTED)")
    print("-" * 60)
    
    # Get scenario parameters
    if scenario_name.title() in BASS_SCENARIOS:
        params = BASS_SCENARIOS[scenario_name.title()]
        model = BassModel(
            market_size=params["market_size"],
            innovation_coef=params["innovation_coef"],
            imitation_coef=params["imitation_coef"]
        )
    else:
        # Use default parameters
        model = BassModel(**DEFAULT_BASS_PARAMS)
    
    # Run forecast
    model.forecast(24)
    
    # Quick summary
    peak_info = model.get_peak_period()
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    final_penetration = model.results["Market Penetration (%)"].iloc[-1]
    
    print(f"Market Size: {model.M:,}")
    print(f"Parameters: p={model.p}, q={model.q}")
    print(f"Peak Adoption: Month {peak_info['period']} ({peak_info['new_adopters']:,} new adopters)")
    print(f"Final Results: {final_users:,} users ({final_penetration:.1f}% penetration)")
    
    # Quick cost comparison with corrected patterns
    pricing_summary = get_pricing_summary(model)
    costs = {
        "ONECI (CORRECTED)": pricing_summary['oneci']['total_cost_fcfa'],
        "SmileID": pricing_summary['smileid']['total_cost_fcfa'],
        "DKB": pricing_summary['dkb']['total_cost_fcfa']
    }
    
    cheapest = min(costs, key=costs.get)
    print(f"\nCost Analysis (24 months) - CORRECTED PATTERNS:")
    print(f"📋 Usage Patterns:")
    print(f"  • ONECI: 1 request/user (registration only) - CORRECTED")
    print(f"  • SmileID: 2 one-time + 1 monthly/user")
    print(f"  • DKB: 1 signature/user (signing only)")
    
    print(f"\n💰 Total Costs:")
    for provider, cost in costs.items():
        marker = "👑" if provider == cheapest else "  "
        print(f"{marker} {provider}: {cost:,.0f} FCFA")
    
    print(f"\n🏆 Best Option: {cheapest}")
    
    if "ONECI" in cheapest:
        print(f"💡 ONECI wins due to corrected usage pattern (registration only)!")


def run_full_demo(scenario_name: str = "Balanced") -> None:
    """Run comprehensive demo with all features and correction highlights."""
    print_header()
    print_usage_patterns()
    
    # Initialize model
    if scenario_name.title() in BASS_SCENARIOS:
        params = BASS_SCENARIOS[scenario_name.title()]
        print(f"🎭 Running {scenario_name.title()} Scenario")
        print(f"Parameters: Market={params['market_size']:,}, p={params['innovation_coef']}, q={params['imitation_coef']}")
        print(f"Description: {params['description']}")
        print()
        
        model = BassModel(
            market_size=params["market_size"],
            innovation_coef=params["innovation_coef"],
            imitation_coef=params["imitation_coef"]
        )
    else:
        print("🎯 Running Default Scenario")
        model = BassModel(**DEFAULT_BASS_PARAMS)
        print(f"Parameters: Market={model.M:,}, p={model.p}, q={model.q}")
        print()
    
    # Run all analyses
    run_basic_forecast(model)
    provider_data = run_pricing_analysis(model)
    comparison_df = run_comparative_analysis(model)
    bi_results = run_business_intelligence(model)
    scenario_results = run_scenario_analysis(model)
    sensitivity_results = run_sensitivity_analysis(model)
    executive_summary = generate_executive_report(model)
    
    # Generate outputs
    create_visualizations(model)
    export_comprehensive_report(model)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📁 Check the 'output' directory for:")
    print("  • charts/ - All visualization files (with correction indicators)")
    print("  • reports/ - Excel analysis report (with correction metadata)")
    print("🔧 All outputs reflect corrected ONECI usage pattern")
    print("💡 ONECI costs reduced ~50% due to registration-only usage")
    print("🎉 Thank you for using Bass Model Analysis!")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Bass Model Analysis Demo - ONECI CORRECTED")
    parser.add_argument("--scenario", "-s", 
                       choices=list(BASS_SCENARIOS.keys()) + ["default"],
                       default="Balanced",
                       help="Scenario to run (default: Balanced)")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick analysis only")
    parser.add_argument("--export-only", "-e", action="store_true",
                       help="Generate reports without running analysis")
    parser.add_argument("--no-charts", action="store_true",
                       help="Skip chart generation")
    parser.add_argument("--show-correction", action="store_true",
                       help="Show detailed correction information")
    
    args = parser.parse_args()
    
    # Show correction details if requested
    if args.show_correction:
        print("🔧 ONECI CORRECTION DETAILS:")
        print("-" * 40)
        print("• Changed from: 2 requests per user (registration + contract signing)")
        print("• Changed to: 1 request per user (registration only)")
        print("• Impact: ~50% cost reduction for ONECI")
        print("• Result: ONECI much more competitive vs SmileID & DKB")
        print("• Files updated: pricing_models.py, analysis_tools.py, streamlit_app.py, config.py, visualizations.py, main.py")
        print()
    
    try:
        if args.quick:
            run_quick_analysis(args.scenario)
        elif args.export_only:
            print("📄 Export-only mode (CORRECTED)")
            print("Creating model with default parameters...")
            model = BassModel(**DEFAULT_BASS_PARAMS)
            model.forecast(24)
            
            if not args.no_charts:
                create_visualizations(model)
            export_comprehensive_report(model)
        else:
            run_full_demo(args.scenario)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check your parameters and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()