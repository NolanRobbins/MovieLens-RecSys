"""
MovieLens RecSys: SS4Rec vs Neural CF Demo
Streamlit application for comparing 2025 SOTA research results
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append("src")

# Page configuration
st.set_page_config(
    page_title="SS4Rec vs Neural CF Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .sota-highlight {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .baseline-card {
        border-left: 4px solid #ffa726;
    }
    .ss4rec-card {
        border-left: 4px solid #42a5f5;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 MovieLens RecSys: SS4Rec Implementation</h1>
        <p>2025 SOTA Sequential Recommendation vs Neural Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Research overview
    st.markdown("## 🔬 Research Focus")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card baseline-card">
            <h3>📊 Neural CF (Baseline)</h3>
            <p><strong>Paper:</strong> Neural Collaborative Filtering (WWW 2017)</p>
            <p><strong>Architecture:</strong> MF + MLP paths</p>
            <p><strong>Expected RMSE:</strong> ~0.85-0.90</p>
            <p><strong>Purpose:</strong> Solid baseline for comparison</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card ss4rec-card">
            <h3>🚀 SS4Rec (SOTA 2025)</h3>
            <p><strong>Paper:</strong> Continuous-Time Sequential Recommendation (Feb 2025)</p>
            <p><strong>Architecture:</strong> State Space Models + Temporal</p>
            <p><strong>Expected RMSE:</strong> ~0.60-0.70</p>
            <p><strong>Innovation:</strong> Time-aware + Relation-aware SSM</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("## 🎯 Expected Performance")
    
    # Create comparison metrics
    models = ["Previous VAE", "Neural CF", "SS4Rec"]
    val_rmse = [1.25, 0.875, 0.65]
    improvements = [0, 30, 48]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Previous VAE",
            value="1.25 RMSE",
            delta=None,
            help="Previous overfitted result"
        )
    
    with col2:
        st.metric(
            label="Neural CF (Baseline)",
            value="0.88 RMSE",
            delta="-30%",
            delta_color="normal",
            help="Expected baseline performance"
        )
    
    with col3:
        st.metric(
            label="SS4Rec (SOTA)",
            value="0.65 RMSE",
            delta="-48%",
            delta_color="normal",
            help="Target SOTA performance"
        )
    
    # Key features
    st.markdown("## ✨ SS4Rec Key Innovations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⏰ Time-Aware SSM**
        - Handles irregular time intervals
        - Models temporal user preference decay
        - Utilizes MovieLens timestamps
        """)
    
    with col2:
        st.markdown("""
        **🔗 Relation-Aware SSM**
        - Item-item contextual dependencies
        - Sequential pattern modeling
        - Better than attention mechanisms
        """)
    
    with col3:
        st.markdown("""
        **📈 Continuous-Time Modeling**
        - Real user interest evolution
        - Better than discrete transformers
        - Proven on dense datasets
        """)
    
    # Dataset information
    st.markdown("## 📊 MovieLens Dataset")
    
    # Mock dataset stats
    st.markdown("""
    - **Size**: 25M ratings, 162K movies, 280K users
    - **Temporal**: Rich timestamp data (currently underutilized)
    - **Sparsity**: High sparsity typical of collaborative filtering
    - **Characteristics**: Dense dataset optimal for SS4Rec
    """)
    
    # Training status
    st.markdown("## 🚀 Training Status")
    
    # Create training status (placeholder)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Neural CF Status")
        st.info("📋 Ready for training on A6000 GPU")
        st.markdown("""
        - Configuration: `configs/ncf_baseline.yaml`
        - Training script: `training/train_ncf.py`
        - Expected time: ~2-3 hours
        """)
    
    with col2:
        st.markdown("### SS4Rec Status")
        st.warning("🔧 Implementation in progress")
        st.markdown("""
        - Configuration: `configs/ss4rec.yaml`
        - Training script: `training/train_ss4rec.py`
        - Expected time: ~4-5 hours
        """)
    
    # Research links
    st.markdown("## 📚 Research References")
    
    st.markdown("""
    1. **SS4Rec Paper**: [Continuous-Time Sequential Recommendation with State Space Models](https://arxiv.org/abs/2502.08132)
    2. **SS4Rec GitHub**: [Official Implementation](https://github.com/XiaoWei-i/SS4Rec)
    3. **Neural CF Paper**: [Neural Collaborative Filtering (WWW 2017)](https://arxiv.org/abs/1708.05031)
    4. **MovieLens Dataset**: [GroupLens Research](https://grouplens.org/datasets/movielens/)
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🎯 <strong>Goal:</strong> Demonstrate SS4Rec's superiority over traditional collaborative filtering<br>
        📈 <strong>Target:</strong> Validation RMSE < 0.70 (vs current ~1.25)<br>
        🔬 <strong>Focus:</strong> Real generalization performance using 2025 SOTA research
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()