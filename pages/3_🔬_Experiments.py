"""
Model Comparison and A/B Testing Dashboard
Interactive interface for comparing model performance and analyzing A/B test results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append("src")

try:
    from business.business_metrics import BusinessMetricsTracker, MLMetricsConverter
    EXPERIMENTS_AVAILABLE = True
except ImportError as e:
    EXPERIMENTS_AVAILABLE = False
    st.warning(f"Some experiment components not available: {e}. Using demo mode.")
    
    # Create simple demo MLMetricsConverter for demo mode
    class MockBusinessMetrics:
        def __init__(self, **kwargs):
            self.session_duration_minutes = 26.3
            self.items_per_session = 8.5
            self.return_rate_7day = 0.47
            self.conversion_rate = 0.34
            self.average_order_value = 15.20
            self.customer_lifetime_value = 185
            self.user_satisfaction_score = 0.742
            self.churn_risk_reduction = 0.12
    
    class MLMetricsConverter:
        @staticmethod
        def convert_ml_to_business_metrics(ml_metrics):
            return MockBusinessMetrics()

# Page configuration
st.set_page_config(
    page_title="Model Experiments",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .experiment-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .metric-improvement {
        color: #28a745;
        font-weight: bold;
    }
    .metric-decline {
        color: #dc3545;
        font-weight: bold;
    }
    .winner-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .comparison-table {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔬 Model Experiments & A/B Testing")
    st.markdown("Compare model performance and analyze business impact of recommendation algorithms")
    
    if not EXPERIMENTS_AVAILABLE:
        st.info("🧪 Running in demo mode - showing example experiment results and capabilities")
        st.markdown("*In a production environment, this would connect to live experiment tracking systems.*")
    
    # Sidebar navigation
    experiment_type = st.sidebar.radio(
        "Experiment Type",
        ["Model Comparison", "A/B Testing", "Business Impact Analysis", "Performance Monitoring"]
    )
    
    if experiment_type == "Model Comparison":
        render_model_comparison()
    elif experiment_type == "A/B Testing":
        render_ab_testing()
    elif experiment_type == "Business Impact Analysis":
        render_business_impact()
    else:
        render_performance_monitoring()

def render_model_comparison():
    """Render model comparison interface"""
    st.header("🏁 Model Performance Comparison")
    
    # Mock experiment results (in production, load from actual experiments)
    model_results = create_mock_model_results()
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Performance Summary")
        
        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Model': result['name'],
                'RMSE': f"{result['metrics']['rmse']:.4f}",
                'Precision@10': f"{result['metrics']['precision_at_10']:.3f}",
                'NDCG@10': f"{result['metrics']['ndcg_at_10']:.3f}",
                'Diversity': f"{result['metrics']['diversity']:.3f}",
                'Training Time': f"{result['training_time']:.1f}s",
                'Parameters': f"{result['param_count']:,}"
            }
            for result in model_results
        ])
        
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Winner Analysis")
        
        # Find best model by business value
        best_model = max(model_results, key=lambda x: x['business_score'])
        
        st.markdown(f"""
        <div class="winner-card">
            <h3>🥇 {best_model['name']}</h3>
            <h4>Business Value Score: {best_model['business_score']:.3f}</h4>
            <p>Projected Annual Revenue Impact: <strong>${best_model['revenue_impact']:,.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key improvements
        baseline = min(model_results, key=lambda x: x['business_score'])
        
        improvements = {
            'Precision@10': ((best_model['metrics']['precision_at_10'] - baseline['metrics']['precision_at_10']) / baseline['metrics']['precision_at_10'] * 100),
            'NDCG@10': ((best_model['metrics']['ndcg_at_10'] - baseline['metrics']['ndcg_at_10']) / baseline['metrics']['ndcg_at_10'] * 100),
            'Diversity': ((best_model['metrics']['diversity'] - baseline['metrics']['diversity']) / baseline['metrics']['diversity'] * 100)
        }
        
        st.write("**Key Improvements over Baseline:**")
        for metric, improvement in improvements.items():
            color = "metric-improvement" if improvement > 0 else "metric-decline"
            st.markdown(f"- {metric}: <span class='{color}'>+{improvement:.1f}%</span>", unsafe_allow_html=True)
    
    # Detailed performance charts
    st.subheader("📈 Detailed Performance Analysis")
    
    # Create performance radar chart
    metrics_for_radar = ['precision_at_10', 'ndcg_at_10', 'diversity', 'coverage']
    
    fig = go.Figure()
    
    for result in model_results:
        values = [result['metrics'][metric] for metric in metrics_for_radar]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_for_radar + [metrics_for_radar[0]],
            fill='toself',
            name=result['name']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training efficiency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Training time vs performance
        fig_efficiency = px.scatter(
            x=[result['training_time'] for result in model_results],
            y=[result['metrics']['precision_at_10'] for result in model_results],
            size=[result['param_count']/1000 for result in model_results],
            color=[result['name'] for result in model_results],
            title="Training Efficiency: Time vs Performance",
            labels={'x': 'Training Time (seconds)', 'y': 'Precision@10'},
            hover_data={'size': True}
        )
        fig_efficiency.update_traces(hovertemplate='<b>%{color}</b><br>Training Time: %{x:.1f}s<br>Precision@10: %{y:.3f}<br>Parameters: %{marker.size:.0f}K')
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        # Business impact comparison
        fig_business = px.bar(
            x=[result['name'] for result in model_results],
            y=[result['revenue_impact'] for result in model_results],
            title="Projected Annual Revenue Impact",
            labels={'x': 'Model', 'y': 'Revenue Impact ($)'},
            color=[result['revenue_impact'] for result in model_results],
            color_continuous_scale='Greens'
        )
        fig_business.update_traces(hovertemplate='<b>%{x}</b><br>Revenue Impact: $%{y:,.0f}')
        st.plotly_chart(fig_business, use_container_width=True)

def render_ab_testing():
    """Render A/B testing interface"""
    st.header("🧪 A/B Testing Dashboard")
    
    # A/B test configuration
    with st.expander("⚙️ Configure New A/B Test", expanded=False):
        test_name = st.text_input("Test Name", value="Model Comparison V2")
        test_description = st.text_area("Description", value="Compare Hybrid VAE vs Neural CF with business metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            control_model = st.selectbox("Control Model", ["Matrix Factorization", "Neural CF", "Hybrid VAE"])
            control_traffic = st.slider("Control Traffic %", 10, 80, 50)
        
        with col2:
            treatment_model = st.selectbox("Treatment Model", ["Hybrid VAE", "Two Tower", "Neural CF"])
            treatment_traffic = st.slider("Treatment Traffic %", 10, 80, 50)
        
        duration_days = st.slider("Test Duration (days)", 7, 30, 14)
        
        if st.button("🚀 Launch A/B Test"):
            st.success(f"A/B Test '{test_name}' launched successfully!")
            st.info(f"Control: {control_model} ({control_traffic}%) vs Treatment: {treatment_model} ({treatment_traffic}%)")
    
    # Active A/B tests
    st.subheader("📊 Active A/B Tests")
    
    # Mock active tests
    active_tests = create_mock_ab_tests()
    
    for i, test in enumerate(active_tests):
        with st.container():
            st.markdown(f"""
            <div class="experiment-card">
                <h4>{test['name']}</h4>
                <p><strong>Status:</strong> {test['status']} | <strong>Duration:</strong> {test['duration_days']} days | <strong>Sample Size:</strong> {test['sample_size']:,}</p>
                <p>{test['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Test results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Control CTR",
                    f"{test['control_ctr']:.3f}",
                    delta=None
                )
                st.metric(
                    "Treatment CTR", 
                    f"{test['treatment_ctr']:.3f}",
                    delta=f"+{((test['treatment_ctr'] - test['control_ctr'])/test['control_ctr']*100):+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Control Rating",
                    f"{test['control_rating']:.2f}",
                    delta=None
                )
                st.metric(
                    "Treatment Rating",
                    f"{test['treatment_rating']:.2f}", 
                    delta=f"{test['treatment_rating'] - test['control_rating']:+.2f}"
                )
            
            with col3:
                # Statistical significance
                significance = "✅ Significant" if test['p_value'] < 0.05 else "⏳ Not Yet Significant"
                st.metric(
                    "P-Value",
                    f"{test['p_value']:.4f}",
                    delta=significance
                )
                
                # Recommended action
                if test['p_value'] < 0.05 and test['treatment_ctr'] > test['control_ctr']:
                    action = "🚀 Implement Treatment"
                    st.success(action)
                elif test['p_value'] < 0.05:
                    action = "🔄 Keep Control"  
                    st.info(action)
                else:
                    action = "⏳ Continue Test"
                    st.warning(action)
    
    # A/B test history and trends
    st.subheader("📈 A/B Test History")
    
    # Historical results chart
    history_data = create_mock_ab_history()
    
    fig_history = px.line(
        history_data,
        x='date',
        y='ctr_lift',
        color='test_name',
        title="A/B Test CTR Lift Over Time",
        labels={'ctr_lift': 'CTR Lift (%)', 'date': 'Date'}
    )
    fig_history.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")
    
    st.plotly_chart(fig_history, use_container_width=True)

def render_business_impact():
    """Render business impact analysis"""
    st.header("💰 Business Impact Analysis")
    
    # Revenue attribution model
    st.subheader("📊 Revenue Attribution Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Base Business Parameters:**")
        mau = st.number_input("Monthly Active Users", value=200000, format="%d")
        sessions_per_user = st.number_input("Avg Sessions/User/Month", value=8.0, format="%.1f")
        revenue_per_session = st.number_input("Baseline Revenue/Session ($)", value=12.50, format="%.2f")
        
        st.write("**Model Performance:**")
        current_ctr = st.slider("Current CTR", 0.20, 0.50, 0.35, 0.01)
        current_ndcg = st.slider("Current NDCG@10", 0.30, 0.70, 0.42, 0.01) 
        current_diversity = st.slider("Current Diversity", 0.30, 0.80, 0.52, 0.01)
    
    with col2:
        # Calculate business impact
        baseline_ctr = 0.30
        baseline_ndcg = 0.40
        baseline_diversity = 0.50
        
        # Revenue impact calculations
        ctr_lift = (current_ctr - baseline_ctr) / baseline_ctr
        engagement_lift = (current_ndcg - baseline_ndcg) / baseline_ndcg
        satisfaction_lift = (current_diversity - baseline_diversity) / baseline_diversity
        
        # Annual revenue impact
        annual_sessions = mau * sessions_per_user * 12
        ctr_revenue_impact = annual_sessions * revenue_per_session * ctr_lift * 2.8
        engagement_revenue_impact = annual_sessions * revenue_per_session * engagement_lift * 1.5
        satisfaction_revenue_impact = annual_sessions * revenue_per_session * satisfaction_lift * 0.5
        
        total_revenue_impact = ctr_revenue_impact + engagement_revenue_impact + satisfaction_revenue_impact
        
        st.metric("CTR Lift", f"{ctr_lift:+.1%}", f"${ctr_revenue_impact:,.0f}/year")
        st.metric("Engagement Lift", f"{engagement_lift:+.1%}", f"${engagement_revenue_impact:,.0f}/year")
        st.metric("Satisfaction Lift", f"{satisfaction_lift:+.1%}", f"${satisfaction_revenue_impact:,.0f}/year")
        
        st.markdown(f"""
        ### 💰 Total Annual Impact
        **${total_revenue_impact:,.0f}**
        
        **Revenue Per User:** ${total_revenue_impact/mau:.2f}
        **ROI Multiple:** {total_revenue_impact/(mau*0.15*12):.1f}x
        """)
    
    # Business metrics breakdown
    st.subheader("📈 Business Metrics Breakdown")
    
    # Convert current performance to business metrics
    ml_metrics = {
        'precision_at_10': current_ctr,
        'ndcg_at_10': current_ndcg,
        'diversity': current_diversity,
        'rmse': 1.21,
        'coverage': 0.45
    }
    
    business_metrics = MLMetricsConverter.convert_ml_to_business_metrics(ml_metrics)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Session Duration", f"{business_metrics.session_duration_minutes:.1f} min")
        st.metric("Items/Session", f"{business_metrics.items_per_session:.1f}")
    
    with col2:
        st.metric("Return Rate (7d)", f"{business_metrics.return_rate_7day:.1%}")
        st.metric("Conversion Rate", f"{business_metrics.conversion_rate:.1%}")
    
    with col3:
        st.metric("Avg Order Value", f"${business_metrics.average_order_value:.2f}")
        st.metric("Customer LTV", f"${business_metrics.customer_lifetime_value:.0f}")
    
    with col4:
        st.metric("User Satisfaction", f"{business_metrics.user_satisfaction_score:.3f}")
        st.metric("Churn Reduction", f"{business_metrics.churn_risk_reduction:.1%}")
    
    # ROI sensitivity analysis
    st.subheader("🎯 ROI Sensitivity Analysis")
    
    # Create sensitivity analysis
    ctr_range = np.arange(0.25, 0.45, 0.02)
    roi_values = []
    
    for ctr in ctr_range:
        lift = (ctr - baseline_ctr) / baseline_ctr
        revenue_impact = annual_sessions * revenue_per_session * lift * 2.8
        roi_values.append(revenue_impact / 1000000)  # In millions
    
    fig_sensitivity = px.line(
        x=ctr_range,
        y=roi_values,
        title="Revenue Impact Sensitivity to CTR Changes",
        labels={'x': 'Click-Through Rate', 'y': 'Annual Revenue Impact ($M)'}
    )
    
    fig_sensitivity.add_vline(x=current_ctr, line_dash="dash", annotation_text="Current CTR")
    fig_sensitivity.add_vline(x=baseline_ctr, line_dash="dash", line_color="red", annotation_text="Baseline")
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)

def render_performance_monitoring():
    """Render performance monitoring interface"""
    st.header("📊 Performance Monitoring")
    
    # Real-time performance metrics
    st.subheader("⚡ Real-time Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock real-time metrics
    with col1:
        st.metric(
            "Current CTR",
            "0.347",
            delta="+0.012 vs yesterday"
        )
    
    with col2:
        st.metric(
            "Avg Rating",
            "4.12",
            delta="-0.05 vs yesterday"
        )
    
    with col3:
        st.metric(
            "Session Duration",
            "26.3 min",
            delta="+1.2 min vs yesterday"
        )
    
    with col4:
        st.metric(
            "Revenue/Session",
            "$13.45",
            delta="+$0.95 vs yesterday"
        )
    
    # Performance trend monitoring
    st.subheader("📈 Performance Trends (Last 30 Days)")
    
    # Generate mock trend data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    trend_data = pd.DataFrame({
        'date': dates,
        'ctr': 0.35 + np.random.normal(0, 0.01, 30),
        'avg_rating': 4.1 + np.random.normal(0, 0.05, 30),
        'revenue_per_session': 13.0 + np.random.normal(0, 0.5, 30)
    })
    
    # Add slight upward trend
    trend_data['ctr'] += np.linspace(-0.01, 0.01, 30)
    trend_data['revenue_per_session'] += np.linspace(-0.5, 1.0, 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ctr = px.line(trend_data, x='date', y='ctr', title='CTR Trend')
        fig_ctr.add_hline(y=0.30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        st.plotly_chart(fig_ctr, use_container_width=True)
    
    with col2:
        fig_revenue = px.line(trend_data, x='date', y='revenue_per_session', title='Revenue per Session')
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Alert system
    st.subheader("🚨 Performance Alerts")
    
    alerts = [
        {
            'severity': 'warning',
            'message': 'Average rating declined 3.2% over last 3 days',
            'timestamp': '2 hours ago',
            'metric': 'avg_rating'
        },
        {
            'severity': 'info',
            'message': 'CTR improvement trending upward (+1.8% week over week)',
            'timestamp': '6 hours ago', 
            'metric': 'ctr'
        },
        {
            'severity': 'success',
            'message': 'Revenue per session exceeded target ($13+ for 5 consecutive days)',
            'timestamp': '1 day ago',
            'metric': 'revenue'
        }
    ]
    
    for alert in alerts:
        if alert['severity'] == 'warning':
            st.warning(f"⚠️ {alert['message']} - {alert['timestamp']}")
        elif alert['severity'] == 'success':
            st.success(f"✅ {alert['message']} - {alert['timestamp']}")
        else:
            st.info(f"ℹ️ {alert['message']} - {alert['timestamp']}")

def create_mock_model_results():
    """Create mock model comparison results"""
    return [
        {
            'name': 'Matrix Factorization',
            'metrics': {'rmse': 1.25, 'precision_at_10': 0.28, 'ndcg_at_10': 0.38, 'diversity': 0.45, 'coverage': 0.35},
            'training_time': 45.2,
            'param_count': 125000,
            'business_score': 0.672,
            'revenue_impact': 1200000
        },
        {
            'name': 'Hybrid VAE',
            'metrics': {'rmse': 1.21, 'precision_at_10': 0.35, 'ndcg_at_10': 0.42, 'diversity': 0.52, 'coverage': 0.47},
            'training_time': 128.5,
            'param_count': 2400000,
            'business_score': 0.785,
            'revenue_impact': 2300000
        },
        {
            'name': 'Neural CF',
            'metrics': {'rmse': 1.23, 'precision_at_10': 0.32, 'ndcg_at_10': 0.40, 'diversity': 0.48, 'coverage': 0.42},
            'training_time': 89.3,
            'param_count': 890000,
            'business_score': 0.718,
            'revenue_impact': 1750000
        },
        {
            'name': 'Two Tower',
            'metrics': {'rmse': 1.28, 'precision_at_10': 0.30, 'ndcg_at_10': 0.39, 'diversity': 0.51, 'coverage': 0.44},
            'training_time': 67.8,
            'param_count': 1200000,
            'business_score': 0.695,
            'revenue_impact': 1450000
        }
    ]

def create_mock_ab_tests():
    """Create mock A/B test data"""
    return [
        {
            'name': 'Hybrid VAE vs Matrix Factorization',
            'description': 'Testing advanced VAE against simple baseline',
            'status': '🟢 Active',
            'duration_days': 12,
            'sample_size': 45000,
            'control_ctr': 0.287,
            'treatment_ctr': 0.346,
            'control_rating': 4.05,
            'treatment_rating': 4.18,
            'p_value': 0.0023
        },
        {
            'name': 'Diversity Optimization Test',
            'description': 'Testing higher diversity weight in business logic',
            'status': '🟡 Analyzing',
            'duration_days': 14,
            'sample_size': 52000,
            'control_ctr': 0.334,
            'treatment_ctr': 0.329,
            'control_rating': 4.12,
            'treatment_rating': 4.15,
            'p_value': 0.1420
        }
    ]

def create_mock_ab_history():
    """Create mock A/B test history data"""
    dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
    return pd.DataFrame({
        'date': list(dates) + list(dates),
        'test_name': ['Test A']*14 + ['Test B']*14,
        'ctr_lift': np.concatenate([
            np.cumsum(np.random.normal(0.5, 0.2, 14)),
            np.cumsum(np.random.normal(0.2, 0.15, 14))
        ])
    })

if __name__ == "__main__":
    main()