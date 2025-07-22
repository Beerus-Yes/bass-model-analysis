"""
Streamlit Executive Dashboard for Bass Model Analysis

A professional web application for analyzing Bass Diffusion Models with
3-way pricing comparison (ONECI, SmileID, DKB Solutions).

Run with: streamlit run streamlit_app.py

Author: Bass Model Analysis Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from io import BytesIO

# Import our Bass Model modules
from bass_model import BassModel
from pricing_models import (
    financial_analysis, 
    compare_pricing_models, 
    get_pricing_summary,
    get_usage_pattern_summary
)
from analysis_tools import (
    analyze_break_even_points,
    calculate_roi_metrics,
    calculate_market_timing_metrics,
    generate_executive_summary
)
from config import BASS_SCENARIOS, COLOR_PALETTE

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bass Model Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .provider-card {
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .provider-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scenario_data():
    """Load predefined scenarios for quick selection."""
    return BASS_SCENARIOS


@st.cache_data
def run_bass_analysis(market_size, innovation_coef, imitation_coef, periods=24):
    """Run Bass Model analysis with caching for performance."""
    model = BassModel(market_size, innovation_coef, imitation_coef)
    forecast_df = model.forecast(periods)
    
    # Get all analysis results
    pricing_summary = get_pricing_summary(model)
    comparison_df = compare_pricing_models(model)
    breakeven_analysis = analyze_break_even_points(model)
    roi_metrics = calculate_roi_metrics(model)
    timing_metrics = calculate_market_timing_metrics(model)
    executive_summary = generate_executive_summary(model)
    
    return {
        'model': model,
        'forecast': forecast_df,
        'pricing_summary': pricing_summary,
        'comparison': comparison_df,
        'breakeven': breakeven_analysis,
        'roi': roi_metrics,
        'timing': timing_metrics,
        'executive_summary': executive_summary
    }


def create_adoption_chart(forecast_df):
    """Create interactive adoption curve chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('New Adopters per Period', 'Cumulative Adoption (S-Curve)', 
                       'Market Penetration (%)', 'Adoption Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    periods = forecast_df.iloc[:, 0]
    
    # New adopters
    fig.add_trace(
        go.Scatter(x=periods, y=forecast_df["New Adopters"],
                  mode='lines+markers', name='New Adopters',
                  line=dict(color=COLOR_PALETTE['primary']['oneci'], width=3)),
        row=1, col=1
    )
    
    # Cumulative adoption
    fig.add_trace(
        go.Scatter(x=periods, y=forecast_df["Cumulative Adopters"],
                  mode='lines+markers', name='Cumulative',
                  line=dict(color=COLOR_PALETTE['primary']['smileid'], width=3)),
        row=1, col=2
    )
    
    # Market penetration
    fig.add_trace(
        go.Scatter(x=periods, y=forecast_df["Market Penetration (%)"],
                  mode='lines+markers', name='Penetration',
                  line=dict(color=COLOR_PALETTE['primary']['market'], width=3)),
        row=2, col=1
    )
    
    # Adoption rate
    fig.add_trace(
        go.Scatter(x=periods, y=forecast_df["Adoption Rate"],
                  mode='lines+markers', name='Adoption Rate',
                  line=dict(color=COLOR_PALETTE['primary']['peak'], width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Bass Model Adoption Analysis",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig


def create_cost_comparison_chart(analysis_results):
    """Create interactive cost comparison chart."""
    # Get financial data for all providers
    model = analysis_results['model']
    oneci_df = financial_analysis(model, pricing_model="oneci")
    smileid_df = financial_analysis(model, pricing_model="smileid") 
    dkb_df = financial_analysis(model, pricing_model="dkb")
    
    months = oneci_df["Month"]
    
    fig = go.Figure()
    
    # Add cost lines for each provider
    fig.add_trace(go.Scatter(
        x=months,
        y=pd.to_numeric(oneci_df["Monthly Cost (FCFA)"]),
        mode='lines+markers',
        name='ONECI',
        line=dict(color=COLOR_PALETTE['primary']['oneci'], width=3),
        hovertemplate='<b>ONECI</b><br>Month: %{x}<br>Cost: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=months,
        y=pd.to_numeric(smileid_df["Monthly Cost (FCFA)"]),
        mode='lines+markers',
        name='SmileID',
        line=dict(color=COLOR_PALETTE['primary']['smileid'], width=3),
        hovertemplate='<b>SmileID</b><br>Month: %{x}<br>Cost: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=months,
        y=pd.to_numeric(dkb_df["Monthly Cost (FCFA)"]),
        mode='lines+markers',
        name='DKB Solutions',
        line=dict(color=COLOR_PALETTE['primary']['dkb'], width=3),
        hovertemplate='<b>DKB Solutions</b><br>Month: %{x}<br>Cost: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Cost Comparison - All Providers',
        xaxis_title='Month',
        yaxis_title='Monthly Cost (FCFA)',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_provider_breakdown_chart(analysis_results):
    """Create provider breakdown with volume information."""
    comparison_df = analysis_results['comparison']
    
    # Count best options
    best_counts = comparison_df['Best Option'].value_counts()
    
    # Create color mapping based on available providers
    color_map = {
        'ONECI': COLOR_PALETTE['primary']['oneci'],
        'SmileID': COLOR_PALETTE['primary']['smileid'], 
        'DKB': COLOR_PALETTE['primary']['dkb']
    }
    
    # Get colors for the providers that appear in results
    chart_colors = [color_map.get(provider, '#cccccc') for provider in best_counts.index]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=best_counts.index,
            values=best_counts.values,
            hole=0.4,
            marker_colors=chart_colors,  # Use marker_colors instead of colors
            textinfo='label+percent',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Months as best option: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Provider Dominance Over 24 Months",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def display_kpi_cards(analysis_results):
    """Display key performance indicators as cards."""
    model = analysis_results['model']
    pricing_summary = analysis_results['pricing_summary']
    peak_info = model.get_peak_period()
    final_users = model.results["Cumulative Adopters"].iloc[-1]
    final_penetration = model.results["Market Penetration (%)"].iloc[-1]
    
    # Create 4 columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📊 Final Users",
            value=f"{final_users:,}",
            delta=f"{final_penetration:.1f}% market penetration"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🚀 Peak Period",
            value=f"Month {peak_info['period']}",
            delta=f"{peak_info['new_adopters']:,} new adopters"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        best_provider = pricing_summary['comparison']['cheapest_provider']
        best_cost = pricing_summary[best_provider.lower()]['total_cost_fcfa']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🏆 Best Provider",
            value=best_provider,
            delta=f"{best_cost:,.0f} FCFA total cost"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        max_savings = pricing_summary['comparison']['max_potential_savings_fcfa']
        savings_pct = pricing_summary['comparison']['savings_percentage']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="💰 Potential Savings",
            value=f"{max_savings:,.0f} FCFA",
            delta=f"{savings_pct:.1f}% cost reduction"
        )
        st.markdown('</div>', unsafe_allow_html=True)


def display_provider_details(analysis_results):
    """Display detailed provider comparison."""
    pricing_summary = analysis_results['pricing_summary']
    
    st.subheader("🏢 Provider Detailed Analysis")
    
    # Create columns for each provider
    col1, col2, col3 = st.columns(3)
    
    providers = [
        ("ONECI", "oneci", col1, COLOR_PALETTE['primary']['oneci']),
        ("SmileID", "smileid", col2, COLOR_PALETTE['primary']['smileid']),
        ("DKB Solutions", "dkb", col3, COLOR_PALETTE['primary']['dkb'])
    ]
    
    # Provider name mapping for usage patterns
    pattern_key_mapping = {
        "ONECI": "ONECI",
        "SmileID": "SMILEID",  # Fix: SmileID maps to SMILEID
        "DKB Solutions": "DKB"
    }
    
    for name, key, col, color in providers:
        with col:
            data = pricing_summary[key]
            is_best = pricing_summary['comparison']['cheapest_provider'] == name
            
            # Provider card styling
            border_color = "#28a745" if is_best else "#dee2e6"
            crown = "👑 " if is_best else ""
            
            st.markdown(f'''
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="color: {color}; margin-bottom: 1rem;">{crown}{name}</h4>
                <p><strong>Total Cost:</strong> {data['total_cost_fcfa']:,.0f} FCFA</p>
                <p><strong>Cost per User:</strong> {data['cost_per_user_fcfa']:,.0f} FCFA</p>
                <p><strong>Avg Monthly:</strong> {data['avg_monthly_cost_fcfa']:,.0f} FCFA</p>
                <p><strong>Final Monthly:</strong> {data['final_monthly_cost_fcfa']:,.0f} FCFA</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Usage pattern - using correct key mapping
            try:
                patterns = get_usage_pattern_summary()
                pattern_key = pattern_key_mapping.get(name, name.upper())
                pattern_info = patterns['corrected_usage_patterns'][pattern_key]
                
                with st.expander(f"📋 {name} Usage Pattern"):
                    st.write(f"**Description:** {pattern_info['description']}")
                    st.write("**Breakdown:**")
                    for item in pattern_info['breakdown']:
                        st.write(f"• {item}")
                    st.write(f"**Billing Model:** {pattern_info['billing_model']}")
                    
            except (KeyError, AttributeError) as e:
                # Fallback if usage patterns not available
                with st.expander(f"📋 {name} Usage Pattern"):
                    if name == "ONECI":
                        st.write("**Description:** 2 one-time requests per user")
                        st.write("**Breakdown:**")
                        st.write("• 1 request for user registration/verification")
                        st.write("• 1 request for contract signing verification")
                        st.write("**Billing Model:** One-time costs")
                    elif name == "SmileID":
                        st.write("**Description:** 2 one-time + 1 monthly recurring per user")
                        st.write("**Breakdown:**")
                        st.write("• 1 request for user registration/verification")
                        st.write("• 1 request for contract signing verification")
                        st.write("• 1 monthly request for payment verification (recurring)")
                        st.write("**Billing Model:** Mixed (one-time + recurring)")
                    else:  # DKB Solutions
                        st.write("**Description:** 1 one-time signature per user")
                        st.write("**Breakdown:**")
                        st.write("• 1 digital signature for contract signing only")
                        st.write("**Billing Model:** One-time with high setup costs")


def display_executive_insights(analysis_results):
    """Display executive summary and insights."""
    executive_summary = analysis_results['executive_summary']
    
    st.subheader("💡 Executive Insights")
    
    # Key findings
    findings = executive_summary['key_findings']
    
    # Market opportunity
    market = findings['market_opportunity']
    st.markdown(f'''
    <div class="insight-box">
        <h4>🎯 Market Opportunity</h4>
        <ul>
            <li><strong>Target Market:</strong> {market['total_addressable_market']:,} users</li>
            <li><strong>Projected Adoption:</strong> {market['projected_adoption']:,} users ({market['market_penetration']})</li>
            <li><strong>Peak Growth:</strong> {market['peak_growth_period']} with {market['peak_monthly_adoption']:,} new adopters</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Cost analysis
    cost = findings['cost_analysis']
    st.markdown(f'''
    <div class="recommendation-box">
        <h4>💰 Cost Analysis</h4>
        <ul>
            <li><strong>Most Cost-Effective:</strong> {cost['most_cost_effective']}</li>
            <li><strong>Potential Savings:</strong> {cost['potential_savings']}</li>
            <li><strong>Cost Range:</strong> {cost['cost_range']}</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Strategic recommendations
    if executive_summary.get('recommendations'):
        st.markdown("### 📈 Strategic Recommendations")
        
        for i, rec in enumerate(executive_summary['recommendations'][:3], 1):
            priority_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}.get(rec['priority'], "#6c757d")
            
            st.markdown(f'''
            <div style="border-left: 4px solid {priority_color}; padding-left: 1rem; margin: 1rem 0;">
                <h5>{i}. {rec['category']}</h5>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <p><strong>Rationale:</strong> {rec['rationale']}</p>
                <p><strong>Priority:</strong> <span style="color: {priority_color}; font-weight: bold;">{rec['priority']}</span></p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Risk assessment
    if executive_summary.get('risk_assessment'):
        st.markdown("### ⚠️ Risk Assessment")
        
        for risk in executive_summary['risk_assessment']:
            st.markdown(f'''
            <div class="warning-box">
                <h5>🚨 {risk['risk']}</h5>
                <p><strong>Description:</strong> {risk['description']}</p>
                <p><strong>Mitigation:</strong> {risk['mitigation']}</p>
                <p><strong>Severity:</strong> {risk['severity']}</p>
            </div>
            ''', unsafe_allow_html=True)


def create_export_data(analysis_results):
    """Create Excel file for download."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Basic forecast
        analysis_results['forecast'].to_excel(writer, sheet_name='Forecast', index=False)
        
        # Pricing comparison
        analysis_results['comparison'].to_excel(writer, sheet_name='Pricing Comparison', index=False)
        
        # Individual provider analyses
        model = analysis_results['model']
        for provider in ['oneci', 'smileid', 'dkb']:
            provider_df = financial_analysis(model, pricing_model=provider)
            provider_df.to_excel(writer, sheet_name=f'{provider.upper()} Analysis', index=False)
        
        # Executive summary data
        exec_data = []
        summary = analysis_results['executive_summary']
        
        # Key findings
        findings = summary['key_findings']
        exec_data.extend([
            ['Market Opportunity', ''],
            ['Target Market', f"{findings['market_opportunity']['total_addressable_market']:,} users"],
            ['Projected Adoption', f"{findings['market_opportunity']['projected_adoption']:,} users"],
            ['Market Penetration', findings['market_opportunity']['market_penetration']],
            ['Peak Growth Period', findings['market_opportunity']['peak_growth_period']],
            ['', ''],
            ['Cost Analysis', ''],
            ['Most Cost-Effective', findings['cost_analysis']['most_cost_effective']],
            ['Potential Savings', findings['cost_analysis']['potential_savings']],
            ['Cost Range', findings['cost_analysis']['cost_range']],
        ])
        
        exec_df = pd.DataFrame(exec_data, columns=['Metric', 'Value'])
        exec_df.to_excel(writer, sheet_name='Executive Summary', index=False)
    
    output.seek(0)
    return output


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Bass Model Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Model Parameters")
    
    # Scenario selection
    scenarios = load_scenario_data()
    scenario_names = ["Custom"] + list(scenarios.keys())
    selected_scenario = st.sidebar.selectbox("📋 Select Scenario", scenario_names)
    
    # Parameter inputs
    if selected_scenario == "Custom":
        st.sidebar.markdown("### 🔧 Custom Parameters")
        market_size = st.sidebar.slider(
            "🎯 Market Size",
            min_value=50_000,
            max_value=2_000_000,
            value=1_000_000,
            step=50_000,
            help="Total addressable market size"
        )
        
        innovation_coef = st.sidebar.slider(
            "📢 Innovation Coefficient (p)",
            min_value=0.005,
            max_value=0.05,
            value=0.01,
            step=0.005,
            help="External influence rate (advertising, media)"
        )
        
        imitation_coef = st.sidebar.slider(
            "👥 Imitation Coefficient (q)",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Word-of-mouth influence rate"
        )
    else:
        # Use predefined scenario
        scenario_data = scenarios[selected_scenario]
        market_size = scenario_data["market_size"]
        innovation_coef = scenario_data["innovation_coef"]
        imitation_coef = scenario_data["imitation_coef"]
        
        st.sidebar.markdown(f"### 📊 {selected_scenario} Scenario")
        st.sidebar.markdown(f"**Market Size:** {market_size:,}")
        st.sidebar.markdown(f"**Innovation (p):** {innovation_coef}")
        st.sidebar.markdown(f"**Imitation (q):** {imitation_coef}")
        st.sidebar.markdown(f"**Description:** {scenario_data['description']}")
    
    # Analysis period
    periods = st.sidebar.slider("📅 Analysis Period (months)", 12, 36, 24, 3)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    # Initialize session state
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    # Auto-run on parameter change
    if not st.session_state.run_analysis:
        st.session_state.run_analysis = True
    
    if st.session_state.run_analysis:
        # Show loading spinner
        with st.spinner('🔄 Running Bass Model Analysis...'):
            # Run analysis
            analysis_results = run_bass_analysis(market_size, innovation_coef, imitation_coef, periods)
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        # KPI Cards
        display_kpi_cards(analysis_results)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Adoption Analysis", 
            "💰 Cost Comparison", 
            "🏢 Provider Details",
            "💡 Executive Insights",
            "📊 Data Tables"
        ])
        
        with tab1:
            st.plotly_chart(
                create_adoption_chart(analysis_results['forecast']), 
                use_container_width=True
            )
            
            # Peak analysis
            model = analysis_results['model']
            peak_info = model.get_peak_period()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Peak Period:** Month {peak_info['period']}")
            with col2:
                st.info(f"**Peak Adopters:** {peak_info['new_adopters']:,}")
            with col3:
                st.info(f"**Penetration at Peak:** {peak_info['cumulative_penetration']:.1f}%")
        
        with tab2:
            st.plotly_chart(
                create_cost_comparison_chart(analysis_results), 
                use_container_width=True
            )
            
            # Provider dominance chart
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(
                    create_provider_breakdown_chart(analysis_results),
                    use_container_width=True
                )
            with col2:
                # Break-even analysis
                breakeven = analysis_results['breakeven']
                st.markdown("### ⚖️ Break-Even Analysis")
                
                for provider, analysis in breakeven["periods_analysis"].items():
                    if analysis["periods_cheapest"] > 0:
                        st.write(f"**{provider}:** {analysis['periods_cheapest']} months ({analysis['dominance_percentage']:.1f}%)")
                
                st.write(f"**Crossovers:** {breakeven['summary']['number_of_crossovers']}")
                st.write(f"**Most Stable:** {breakeven['summary']['most_stable_provider']}")
        
        with tab3:
            display_provider_details(analysis_results)
        
        with tab4:
            display_executive_insights(analysis_results)
        
        with tab5:
            st.subheader("📊 Analysis Data Tables")
            
            # Forecast table
            st.markdown("#### 📈 Forecast Data")
            st.dataframe(analysis_results['forecast'], use_container_width=True)
            
            # Comparison table  
            st.markdown("#### 💰 Cost Comparison")
            st.dataframe(analysis_results['comparison'], use_container_width=True)
        
        # Export functionality
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Results")
        
        # Generate Excel file
        excel_data = create_export_data(analysis_results)
        
        st.sidebar.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"bass_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Display current parameters for reference
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Current Analysis")
        st.sidebar.markdown(f"**Market Size:** {market_size:,}")
        st.sidebar.markdown(f"**Innovation (p):** {innovation_coef}")
        st.sidebar.markdown(f"**Imitation (q):** {imitation_coef}")
        st.sidebar.markdown(f"**Periods:** {periods} months")
        
    else:
        # Welcome screen
        st.markdown("""
        <div class="insight-box" style="text-align: center;">
            <h2>🎯 Welcome to Bass Model Analysis</h2>
            <p>Analyze market adoption patterns and compare pricing strategies for digital signature providers.</p>
            <p><strong>Providers:</strong> ONECI • SmileID • DKB Solutions</p>
            <p>👈 Adjust parameters in the sidebar and click "Run Analysis" to begin!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage patterns explanation
        st.markdown("### 📋 Provider Usage Patterns")
        
        patterns = get_usage_pattern_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔵 ONECI**
            - 2 one-time requests per user
            - Registration + Contract signing
            - Declining costs as growth slows
            """)
        
        with col2:
            st.markdown("""
            **🟢 SmileID**
            - 2 one-time + 1 monthly per user
            - Registration + Signing + Payment verification
            - Growing costs with user base
            """)
        
        with col3:
            st.markdown("""
            **🔴 DKB Solutions**
            - 1 one-time signature per user
            - Contract signing only
            - High setup, then declining costs
            """)


if __name__ == "__main__":
    main()