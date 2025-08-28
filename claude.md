# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MovieLens-RecSys is a state-of-the-art recommender system implementation comparing **SS4Rec** (Continuous-Time Sequential Recommendation with State Space Models, 2025 SOTA) against Neural Collaborative Filtering baseline. The project focuses on achieving best possible validation RMSE on MovieLens 25M dataset.

**🚨 CRITICAL UPDATE (2025-08-28)**: Current SS4Rec implementation has **MAJOR DEVIATIONS** from SOTA paper causing gradient explosion. **OPTION A RESET** in progress to official RecBole implementation.

**Updated Goal**: Faithful SS4Rec paper replication with HR@10 > 0.30, NDCG@10 > 0.25 vs NCF baseline using standard RecSys evaluation

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (base project)
uv pip install -r requirements.txt

# Install SS4Rec dependencies (advanced model)
uv pip install -r requirements_ss4rec.txt
```

### Training Commands
```bash
# ⚠️ DEPRECATED: Current SS4Rec training (gradient explosion issues)
# python training/train_ss4rec.py --config configs/ss4rec.yaml

# 🚀 NEW: Official SS4Rec implementation (Option A reset)
# python training/official/train_ss4rec_official.py --config configs/official/ss4rec_official.yaml

# Train Neural CF baseline (for comparison)
python training/train_ncf.py --config configs/ncf_baseline.yaml

# ⚠️ RunPod training currently disabled pending Option A implementation
# ./runpod_entrypoint.sh --model ss4rec --debug
```

### Data Pipeline
```bash
# Run ETL pipeline for new data
python src/data/etl_pipeline.py

# Validate processed data
python src/data/feature_pipeline.py --validate

# Check data quality metrics
python src/monitoring/data_quality_monitor.py
```

### Model Evaluation
```bash
# Evaluate trained model
python src/evaluation/evaluate_model.py --model-path results/ss4rec/best_model.pt

# Advanced evaluation with business metrics
python src/evaluation/advanced_evaluation.py --config configs/evaluation.yaml

# Compare multiple models
python experiments/run_experiment.py --compare-models
```

### Testing & Quality
```bash
# Run all tests
pytest tests/ -v

# Test specific model components
pytest tests/test_system_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov=models --cov-report=html
```

### Streamlit Demo
```bash
# Launch interactive demo
streamlit run streamlit_app.py

# Launch with specific port
streamlit run streamlit_app.py --server.port 8501
```

### Deployment
```bash
# Start FastAPI inference server
python src/api/inference_api.py

# Production deployment (Docker)
docker-compose -f deployment/docker-compose.prod.yml up

# Local development API
uvicorn src.api.inference_api:app --reload --port 8000
```

## High-Level Architecture

### Model Architecture Comparison
1. **Neural CF (Baseline)**: `models/baseline/neural_cf.py`
   - Matrix Factorization + MLP paths
   - Standard collaborative filtering approach
   - Target: Competitive baseline for fair comparison

2. **SS4Rec (SOTA 2025)**: 
   - ⚠️ **DEPRECATED**: `models/sota_2025/ss4rec.py` (custom implementation - gradient explosion)
   - 🚀 **NEW**: `models/official_ss4rec/` (official RecBole implementation)
   - Uses official `mamba-ssm==2.2.2` and `s5-pytorch==0.2.1` libraries
   - Components: Time-Aware SSM, Relation-Aware SSM, Sequential Recommendation
   - Target: HR@10 > 0.30, NDCG@10 > 0.25 (paper benchmarks)

### Data Flow Architecture
```
MovieLens Raw Data → ETL Pipeline → Feature Engineering → Model Training
                                 ↓
                              Feature Store → Sequential Datasets → SS4Rec/NCF
                                 ↓
                            Evaluation → Business Logic → API Serving
```

### Key Components
- **ETL Pipeline**: `src/data/etl_pipeline.py` - Production data ingestion with validation
- **Feature Pipeline**: `src/data/feature_pipeline.py` - Temporal feature engineering for SS4Rec
- **Sequential Dataset**: `training/utils/data_loaders.py` - Time-aware data loading
- **Evaluation Framework**: `src/evaluation/` - Multi-metric validation system
- **Business Logic**: `src/business/` - Real-world recommendation constraints
- **API Layer**: `src/api/` - FastAPI serving with caching

### Experiment Management
- **Experiment Runner**: `experiments/experiment_manager.py` - Streamlined experiment tracking
- **Config Management**: `configs/` - YAML-based model and training configurations
- **Results Tracking**: Integration with Weights & Biases for experiment logging

## SS4Rec Implementation Status

### 🚨 CRITICAL: Implementation Reset Required
- **Current Status**: Custom implementation causing gradient explosion
- **Issue**: Major deviations from SOTA paper (arXiv:2502.08132)
- **Solution**: Complete reset to official RecBole + mamba-ssm + s5-pytorch

### ⚠️ DEPRECATED Implementation Details (Archive Only)
- **Custom SSM**: `models/sota_2025/` (archived due to numerical instability)
- **Gradient Issues**: Exploding gradients after epoch 3 
- **Root Cause**: Custom Mamba selective scan causing exponential overflow

### 🚀 NEW Official Implementation (In Progress)
```bash
# Official dependencies for numerical stability
uv pip install recbole==1.0
uv pip install mamba-ssm==2.2.2  
uv pip install s5-pytorch==0.2.1
uv pip install causal-conv1d>=1.2.0
```

### State Space Model Components (Official)
- **RecBole Framework**: Standard sequential recommendation evaluation
- **Official Mamba**: Battle-tested selective state space implementation
- **Official S5**: Time-aware state space models
- **Standard Evaluation**: HR@K, NDCG@K, MRR@K metrics

## RunPod Training Workflow

The project is optimized for A6000 GPU training on RunPod:

1. **Automated Setup**: `runpod_entrypoint.sh` handles environment setup, dependency installation
2. **Data Management**: Smart caching to avoid re-downloading processed data
3. **Discord Integration**: Automated training notifications via `auto_train_ss4rec.py`
4. **Model Checkpointing**: Automatic saving of best models and training metadata

### RunPod Command Examples
```bash
# Full SS4Rec training with notifications
./runpod_entrypoint.sh --model ss4rec --debug

# Production training (optimal performance)  
./runpod_entrypoint.sh --model ss4rec --production

# NCF baseline training
./runpod_entrypoint.sh --model ncf
```

## Data Requirements

### Input Data Format
- **Train/Val/Test**: CSV files with columns: `user_idx`, `movie_idx`, `rating`, `timestamp`, `rating_date`, `rating_year`, `rating_month`, `rating_weekday`
- **Data Mappings**: `data/processed/data_mappings.pkl` contains user/item ID mappings
- **Sequential Format**: SS4Rec requires time-ordered user interaction sequences

### Data Pipeline Validation
- **Quality Checks**: Automated validation of data integrity in ETL pipeline
- **Temporal Consistency**: Ensures proper timestamp ordering for sequential models
- **Cold Start Handling**: Built-in filtering for users/items with insufficient interactions

## Performance Targets & Evaluation

### 🚀 Updated Performance Targets (Official Implementation)
- **SS4Rec HR@10**: > 0.30 (paper benchmark)
- **SS4Rec NDCG@10**: > 0.25 (paper benchmark)  
- **SS4Rec MRR@10**: Competitive performance vs NCF
- **Training Stability**: Zero gradient explosion (NaN/Inf errors)

### Evaluation Protocol (Updated)
- **Leave-One-Out**: Standard RecBole evaluation methodology
- **Ranking Metrics**: HR@K, NDCG@K, MRR@K (industry standard)
- **Fair Comparison**: Both SS4Rec and NCF using same RecBole framework
- **Temporal Consistency**: Proper sequential recommendation evaluation

## Production Deployment

### API Architecture
- **FastAPI Server**: `src/api/inference_api.py` with Pydantic validation
- **Business Logic Integration**: Real-time filtering and recommendation rules
- **Caching Strategy**: Redis integration for performance optimization
- **Model Versioning**: Support for A/B testing different model versions

### Monitoring & Observability
- **Data Quality**: `src/monitoring/data_quality_monitor.py` tracks data drift
- **Model Performance**: Continuous validation RMSE tracking
- **System Health**: API response times, error rates, throughput metrics

## Development Guidelines

### Code Quality Standards
- **Type Annotations**: All functions must have complete type hints
- **Testing**: Maintain 80%+ coverage for business logic, 100% for data transformations
- **Logging**: Use structured logging with appropriate levels (DEBUG for NaN detection)
- **Error Handling**: Comprehensive exception handling with informative messages

### Model Development Workflow
1. **Research Implementation**: Start with paper reproduction in `models/sota_2025/`
2. **Validation**: Use debug mode for NaN detection and gradient debugging
3. **Optimization**: A6000-specific optimizations for production training
4. **Evaluation**: Multi-metric validation against established baselines
5. **Integration**: Business logic integration and API development

### Experimental Protocol
- **Reproducibility**: All experiments use fixed random seeds
- **Version Control**: Model checkpoints and configurations tracked
- **Documentation**: Results documented in `results/` directory structure
- **Comparison**: Always benchmark against NCF baseline

This architecture enables rapid experimentation with SOTA recommendation models while maintaining production-ready deployment capabilities and comprehensive evaluation frameworks.