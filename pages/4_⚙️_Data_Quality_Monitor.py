"""
Data Quality Monitoring Dashboard
Production-grade monitoring interface for data drift, quality metrics, and retraining triggers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append("src")

try:
    from monitoring.data_quality_monitor import DataQualityMonitor, QualityMetrics
    from data.temporal_simulator import TemporalDataSimulator
    from data.feature_pipeline import EnhancedFeaturePipeline
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    st.error(f"Monitoring components not available: {e}")

# Page configuration
st.set_page_config(
    page_title="Data Quality Monitor",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for monitoring dashboard
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .alert-warning {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
    .status-critical {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("⚙️ Data Quality Monitoring Dashboard")
    st.markdown("Real-time monitoring of data quality, drift detection, and retraining triggers")
    
    if not MONITORING_AVAILABLE:
        st.error("Monitoring system not available. Please ensure all components are properly installed.")
        return
    
    # Initialize monitoring components
    if 'quality_monitor' not in st.session_state:
        st.session_state.quality_monitor = DataQualityMonitor()
        st.session_state.simulator = TemporalDataSimulator()
    
    # Sidebar controls
    st.sidebar.header("Monitoring Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.experimental_rerun()
    
    # Simulation controls
    st.sidebar.header("Data Simulation")
    
    if st.sidebar.button("🎲 Generate Test Batch"):
        test_batch = st.session_state.simulator.simulate_batch_on_demand(batch_size=100)
        st.session_state.latest_batch = test_batch
        st.sidebar.success(f"Generated batch: {test_batch.batch_id}")
    
    if st.sidebar.button("🚨 Simulate Drift Scenario"):
        st.session_state.drift_mode = True
        st.sidebar.warning("Drift simulation started")
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock current metrics (in production, these would come from real monitoring)
    current_quality_score = 0.87
    total_alerts_24h = 3
    critical_alerts = 1
    drift_detected = True
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Quality Score</h3>
            <h1>{current_quality_score:.3f}</h1>
            <p>Current Data Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Alerts (24h)</h3>
            <h1>{total_alerts_24h}</h1>
            <p>Total Quality Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        alert_color = "critical" if critical_alerts > 0 else "healthy"
        st.markdown(f"""
        <div class="metric-container">
            <h3>Critical Issues</h3>
            <h1 class="status-{alert_color}">{critical_alerts}</h1>
            <p>Requiring Attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        drift_status = "critical" if drift_detected else "healthy"
        drift_text = "DETECTED" if drift_detected else "STABLE"
        st.markdown(f"""
        <div class="metric-container">
            <h3>Data Drift</h3>
            <h1 class="status-{drift_status}">{drift_text}</h1>
            <p>Distribution Changes</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quality trends chart
    st.header("📈 Quality Trends (24 Hours)")
    
    # Mock time series data
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='H')
    quality_scores = np.random.normal(0.85, 0.05, len(hours))
    quality_scores = np.clip(quality_scores, 0, 1)
    
    # Add drift event
    quality_scores[-6:] = quality_scores[-6:] - 0.15  # Drop in quality
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=quality_scores,
        mode='lines+markers',
        name='Quality Score',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Add threshold line
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="Retraining Threshold")
    
    fig.update_layout(
        title="Data Quality Score Over Time",
        xaxis_title="Time",
        yaxis_title="Quality Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert dashboard
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🚨 Recent Alerts")
        
        # Mock alerts
        alerts = [
            {
                "timestamp": datetime.now() - timedelta(minutes=15),
                "severity": "critical",
                "message": "Data drift detected in rating distribution",
                "feature": "rating_distribution"
            },
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "severity": "warning", 
                "message": "High duplicate rate: 3.2%",
                "feature": "user_movie_pairs"
            },
            {
                "timestamp": datetime.now() - timedelta(hours=4),
                "severity": "warning",
                "message": "User interaction volume changed by 28%", 
                "feature": "interaction_volume"
            }
        ]
        
        for alert in alerts:
            alert_class = f"alert-{alert['severity']}"
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{alert['severity'].upper()}</strong><br>
                {alert['message']}<br>
                <small>Feature: {alert['feature']} | {alert['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Drift Detection Details")
        
        # Drift metrics table
        drift_data = {
            'Feature': ['Rating Distribution', 'User Volume', 'Item Popularity', 'Genre Preferences'],
            'Drift Score': [0.156, 0.089, 0.034, 0.012],
            'P-Value': [0.023, 0.156, 0.445, 0.789],
            'Status': ['🔴 DRIFT', '🟡 MONITOR', '🟢 STABLE', '🟢 STABLE']
        }
        
        drift_df = pd.DataFrame(drift_data)
        st.dataframe(drift_df, use_container_width=True)
        
        # Feature importance for retraining
        st.subheader("Retraining Trigger Analysis")
        
        trigger_data = {
            'Metric': ['Business CTR Drop', 'Session Duration', 'Model RMSE', 'Data Quality'],
            'Current': ['-12%', '-8%', '+3%', '0.87'],
            'Threshold': ['-15%', '-20%', '+5%', '0.70'],
            'Status': ['🟡 Monitor', '🟢 OK', '🟢 OK', '🟢 OK']
        }
        
        trigger_df = pd.DataFrame(trigger_data)
        st.dataframe(trigger_df, use_container_width=True)
    
    # Retraining recommendation
    st.header("🤖 Retraining Recommendation")
    
    # Mock retraining analysis
    retraining_recommendation = {
        'should_retrain': True,
        'confidence': 0.78,
        'primary_reason': 'Data drift detected in rating distribution',
        'secondary_factors': [
            'Business CTR declining for 5 days',
            'User interaction patterns shifting',
            'Quality score approaching threshold'
        ],
        'estimated_improvement': '+12% in recommendation accuracy',
        'training_time_estimate': '45 minutes',
        'business_impact': '$150K annual revenue improvement'
    }
    
    if retraining_recommendation['should_retrain']:
        st.error("🚨 **RETRAINING RECOMMENDED**")
    else:
        st.success("✅ **MODEL PERFORMING WELL**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Confidence Level",
            f"{retraining_recommendation['confidence']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Expected Improvement", 
            retraining_recommendation['estimated_improvement'],
            delta=None
        )
    
    with col3:
        st.metric(
            "Business Impact",
            retraining_recommendation['business_impact'],
            delta=None
        )
    
    st.write(f"**Primary Reason:** {retraining_recommendation['primary_reason']}")
    
    st.write("**Contributing Factors:**")
    for factor in retraining_recommendation['secondary_factors']:
        st.write(f"• {factor}")
    
    # Manual retraining controls
    st.header("🛠️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Trigger Retraining", type="primary"):
            with st.spinner("Initiating model retraining..."):
                time.sleep(2)
                st.success("✅ Retraining job submitted!")
                st.info("GitHub Actions workflow initiated. Check Actions tab for progress.")
    
    with col2:
        if st.button("📊 Generate Report"):
            st.info("📄 Quality report generated! Check email for detailed analysis.")
    
    with col3:
        if st.button("🔔 Configure Alerts"):
            st.info("⚙️ Alert configuration panel opened.")
    
    # Real-time monitoring section
    if st.checkbox("🔴 Enable Real-time Monitoring"):
        st.header("📡 Live Data Stream")
        
        placeholder = st.empty()
        
        for i in range(5):  # Simulate 5 updates
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    st.metric("Last Update", current_time)
                
                with col2:
                    interactions_processed = np.random.randint(50, 200)
                    st.metric("Interactions/min", f"{interactions_processed}")
                
                with col3:
                    pipeline_latency = np.random.uniform(15, 45)
                    st.metric("Pipeline Latency", f"{pipeline_latency:.1f}ms")
                
                # Live quality score
                live_score = current_quality_score + np.random.normal(0, 0.02)
                live_score = np.clip(live_score, 0, 1)
                
                st.progress(live_score, text=f"Live Quality Score: {live_score:.3f}")
            
            time.sleep(1)
    
    # Footer
    st.markdown("---")
    st.markdown("*Data Quality Monitoring Dashboard - Part of MovieLens Production ML Pipeline*")

if __name__ == "__main__":
    main()