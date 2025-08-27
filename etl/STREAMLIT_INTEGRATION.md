# 🖥️ Streamlit Integration Guide for ETL Pipeline

## Overview
This guide shows how to integrate the ETL pipeline with your existing Streamlit app to create a comprehensive ML engineering dashboard.

---

## 🚀 **Quick Integration Steps**

### **1. Import ETL Components in Streamlit**

Add these imports to your Streamlit app:

```python
# Add to your streamlit imports
import sys
sys.path.append('.')  # If running from root directory

from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
from etl.utils.logging_config import setup_etl_logging
```

### **2. Add ETL Dashboard Section**

```python
def create_etl_dashboard():
    """Create ETL pipeline dashboard section"""
    
    st.header("🏗️ ETL Pipeline Dashboard")
    st.markdown("Real-time monitoring of batch data processing and A/B testing")
    
    # Initialize ETL framework
    ab_framework = ETL_AB_TestingFramework()
    
    # Pipeline Status Section
    st.subheader("📊 Pipeline Status")
    
    status = ab_framework.get_etl_pipeline_status()
    
    if status['pipeline_active']:
        # Progress bar
        progress = status['completion_percentage'] / 100
        st.progress(progress)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Day", f"{status['current_day']}/20")
        with col2:
            st.metric("Progress", f"{status['completion_percentage']:.1f}%")
        with col3:
            st.metric("Records Processed", f"{status['processed_records']:,}")
        with col4:
            remaining = status['total_records'] - status['processed_records']
            st.metric("Remaining", f"{remaining:,}")
        
        # Last updated
        if status.get('last_updated'):
            st.caption(f"Last updated: {status['last_updated']}")
    else:
        st.info("ETL pipeline not started yet. Run your first batch to see progress.")
    
    return ab_framework
```

### **3. Add A/B Testing Results Section**

```python
def show_ab_testing_results(ab_framework):
    """Display A/B testing results"""
    
    st.subheader("🧪 A/B Testing Results")
    
    # Get latest results
    results = ab_framework.run_ab_comparison_for_streamlit()
    
    if results['status'] == 'success':
        # Winner announcement
        winner = results['comparison']['winner']
        confidence = results['comparison']['confidence']
        
        if winner == 'SS4Rec':
            st.success(f"🏆 **Winner: SS4Rec** (Confidence: {confidence})")
        else:
            st.info(f"📊 **Winner: NCF Baseline** (Confidence: {confidence})")
        
        # Model comparison table
        st.subheader("📈 Model Performance Comparison")
        
        ncf_results = results['model_results']['ncf_baseline']
        ss4rec_results = results['model_results']['ss4rec']
        
        comparison_data = {
            'Metric': ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10'],
            'NCF Baseline': [
                f"{ncf_results['rmse']:.3f}",
                f"{ncf_results['mae']:.3f}",
                f"{ncf_results['precision_at_10']:.3f}",
                f"{ncf_results['recall_at_10']:.3f}",
                f"{ncf_results['ndcg_at_10']:.3f}"
            ],
            'SS4Rec': [
                f"{ss4rec_results['rmse']:.3f}",
                f"{ss4rec_results['mae']:.3f}",
                f"{ss4rec_results['precision_at_10']:.3f}",
                f"{ss4rec_results['recall_at_10']:.3f}",
                f"{ss4rec_results['ndcg_at_10']:.3f}"
            ],
            'Improvement': [
                f"{results['comparison']['improvements']['rmse_improvement']:+.1f}%",
                f"{results['comparison']['improvements']['mae_improvement']:+.1f}%",
                f"{results['comparison']['improvements']['precision_at_10_improvement']:+.1f}%",
                f"{results['comparison']['improvements']['recall_at_10_improvement']:+.1f}%",
                f"{results['comparison']['improvements']['ndcg_at_10_improvement']:+.1f}%"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Test data info
        st.subheader("📊 Test Data Information")
        test_info = results['test_data_info']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Records Evaluated", f"{test_info['records']:,}")
        with col2:
            st.write(f"**Date Range**: {test_info['date_range']['start']} to {test_info['date_range']['end']}")
    
    else:
        st.warning(results.get('message', 'No A/B testing data available'))
```

### **4. Add Data Quality Monitoring**

```python
def show_data_quality_dashboard(ab_framework):
    """Display data quality monitoring dashboard"""
    
    st.subheader("🔍 Data Quality Monitoring")
    
    # Get data quality trends
    trends = ab_framework.get_data_quality_trends()
    
    if trends['status'] == 'success':
        # Quality averages
        st.subheader("📊 Overall Data Quality")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completeness = trends['averages']['completeness']
            st.metric("Completeness", f"{completeness:.1%}", 
                     delta=f"{'✅' if completeness > 0.95 else '⚠️'}")
        
        with col2:
            validity = trends['averages']['validity']
            st.metric("Validity", f"{validity:.1%}",
                     delta=f"{'✅' if validity > 0.99 else '⚠️'}")
        
        with col3:
            uniqueness = trends['averages']['uniqueness']
            st.metric("Uniqueness", f"{uniqueness:.1%}",
                     delta=f"{'✅' if uniqueness > 0.90 else '⚠️'}")
        
        with col4:
            consistency = trends['averages']['consistency']
            st.metric("Consistency", f"{consistency:.1%}",
                     delta=f"{'✅' if consistency > 0.95 else '⚠️'}")
        
        # Quality trends chart
        if len(trends['days']) > 1:
            st.subheader("📈 Data Quality Trends")
            
            chart_data = pd.DataFrame({
                'Day': trends['days'],
                'Completeness': trends['metrics']['completeness'],
                'Validity': trends['metrics']['validity'],
                'Uniqueness': trends['metrics']['uniqueness'],
                'Consistency': trends['metrics']['consistency']
            })
            
            st.line_chart(chart_data.set_index('Day'))
    
    else:
        st.info("No data quality trends available yet. Process some batches to see trends.")
```

### **5. Main ETL Dashboard Page**

```python
def etl_dashboard_page():
    """Main ETL dashboard page"""
    
    st.title("🏗️ ETL Pipeline & A/B Testing Dashboard")
    st.markdown("Monitor your production ML pipeline in real-time")
    
    # Initialize ETL framework
    ab_framework = create_etl_dashboard()
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pipeline Status", "🧪 A/B Testing", "🔍 Data Quality", "⚙️ Controls"])
    
    with tab1:
        # Already handled in create_etl_dashboard()
        pass
    
    with tab2:
        show_ab_testing_results(ab_framework)
    
    with tab3:
        show_data_quality_dashboard(ab_framework)
    
    with tab4:
        st.subheader("⚙️ Pipeline Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Process Next Day"):
                with st.spinner("Processing next day..."):
                    # This would trigger the ETL pipeline
                    st.success("Next day processing initiated!")
        
        with col2:
            if st.button("🧪 Run A/B Test"):
                with st.spinner("Running A/B testing..."):
                    results = ab_framework.run_ab_comparison_for_streamlit()
                    if results['status'] == 'success':
                        st.success("A/B testing completed!")
                    else:
                        st.error("A/B testing failed")
        
        with col3:
            if st.button("📊 Refresh Data"):
                st.rerun()
```

---

## 🎯 **Integration with Existing Streamlit App**

### **Option 1: Add as New Page**

```python
# In your main streamlit app
def main():
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Home": home_page,
        "🤖 Model Inference": inference_page,
        "📊 Model Evaluation": evaluation_page,
        "🏗️ ETL Dashboard": etl_dashboard_page,  # Add this line
        "🧪 A/B Testing": ab_testing_page
    }
    
    page = st.sidebar.selectbox("Choose a page", list(pages.keys()))
    pages[page]()
```

### **Option 2: Add as Sidebar Widget**

```python
# Add ETL status to sidebar
def add_etl_sidebar_status():
    """Add ETL pipeline status to sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ ETL Status")
    
    ab_framework = ETL_AB_TestingFramework()
    status = ab_framework.get_etl_pipeline_status()
    
    if status['pipeline_active']:
        progress = status['completion_percentage'] / 100
        st.sidebar.progress(progress)
        st.sidebar.write(f"Day {status['current_day']}/20")
        st.sidebar.write(f"{status['processed_records']:,} records")
    else:
        st.sidebar.info("ETL not started")
```

### **Option 3: Integrate with Existing Model Comparison**

```python
# Enhance your existing model comparison with ETL A/B results
def enhanced_model_comparison():
    """Enhanced model comparison with ETL A/B testing"""
    
    st.subheader("🤖 Model Performance Comparison")
    
    # Your existing model comparison code
    # ...
    
    # Add ETL A/B testing results
    st.markdown("---")
    st.subheader("🏗️ Real-time A/B Testing (from ETL Pipeline)")
    
    ab_framework = ETL_AB_TestingFramework()
    results = ab_framework.load_latest_ab_results()
    
    if results.get('status') == 'success' or 'model_results' in results:
        show_ab_testing_results(ab_framework)
    else:
        st.info("No real-time A/B testing results available yet.")
```

---

## 📊 **Sample Dashboard Layout**

```python
def complete_etl_integration():
    """Complete ETL integration example"""
    
    # Header
    st.title("🎬 MovieLens RecSys - Production ML Pipeline")
    
    # Key metrics at the top
    ab_framework = ETL_AB_TestingFramework()
    status = ab_framework.get_etl_pipeline_status()
    
    if status['pipeline_active']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ETL Progress", f"{status['completion_percentage']:.1f}%")
        with col2:
            st.metric("Current Day", f"{status['current_day']}/20")
        with col3:
            results = ab_framework.load_latest_ab_results()
            winner = results.get('comparison', {}).get('winner', 'N/A')
            st.metric("A/B Winner", winner)
        with col4:
            st.metric("Records Processed", f"{status['processed_records']:,}")
    
    # Main content sections
    st.markdown("---")
    
    # Your existing content (model inference, evaluation, etc.)
    # Plus new ETL sections
    
    # Real-time updates
    if st.checkbox("Auto-refresh (30s)", value=False):
        import time
        time.sleep(30)
        st.rerun()
```

---

## 🔧 **Development Tips**

### **Testing Integration**
```bash
# Test ETL components work with Streamlit
cd etl
make streamlit-check

# Run a quick simulation for demo
make run-simulation

# Check that data is available for Streamlit
make progress
```

### **Error Handling**
```python
def safe_etl_integration():
    """ETL integration with error handling"""
    
    try:
        ab_framework = ETL_AB_TestingFramework()
        # Your ETL integration code
    except ImportError:
        st.error("ETL components not available. Please check installation.")
    except Exception as e:
        st.error(f"ETL integration error: {e}")
        st.info("ETL features temporarily unavailable.")
```

### **Performance Optimization**
```python
# Cache ETL data for better performance
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_etl_status():
    ab_framework = ETL_AB_TestingFramework()
    return ab_framework.get_etl_pipeline_status()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_ab_results():
    ab_framework = ETL_AB_TestingFramework()
    return ab_framework.run_ab_comparison_for_streamlit()
```

---

## 🎉 **Result: Complete ML Engineering Portfolio**

With this integration, your Streamlit app will showcase:

✅ **End-to-End Pipeline**: Data → ETL → Training → Inference → Monitoring  
✅ **Real-time Monitoring**: Live ETL progress and data quality  
✅ **A/B Testing**: Automated model comparison with business metrics  
✅ **Production Readiness**: Proper logging, error handling, CI/CD  
✅ **Business Impact**: Revenue calculations and KPI tracking  

**This demonstrates enterprise-level ML engineering skills that will significantly enhance your portfolio!** 🚀
