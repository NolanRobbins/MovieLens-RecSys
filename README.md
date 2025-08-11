# MovieLens Hybrid VAE Recommender System

A production-ready recommendation system built with hybrid VAE architecture, featuring advanced business logic, comprehensive evaluation metrics, and real-time inference capabilities.

## 🏗️ Architecture Overview

```
MovieLens-RecSys/
├── src/                          # Source code modules
│   ├── api/                      # API and web services
│   │   ├── inference_api.py      # FastAPI inference service
│   │   ├── streamlit_demo.py     # Interactive demo interface
│   │   └── demo_deployment.py    # Deployment utilities
│   ├── evaluation/               # Model evaluation and metrics
│   │   ├── evaluate_model.py     # Base evaluation framework
│   │   ├── advanced_evaluation.py # Advanced metrics with business impact
│   │   └── ranking_metrics_evaluation.py # Ranking-specific metrics
│   ├── models/                   # Model architectures and training
│   │   ├── cloud_training.py     # Training pipeline
│   │   ├── ranking_loss_functions.py # Advanced loss functions
│   │   └── ranking_optimized_training.py # Ranking-optimized training
│   ├── data/                     # Data processing and ETL
│   │   ├── etl_pipeline.py       # Data processing pipeline
│   │   └── etl_monitor.py        # ETL monitoring and validation
│   └── business/                 # Business logic and rules
│       └── business_logic_system.py # Business rules engine
├── data/                         # Data storage
│   ├── raw/                      # Raw MovieLens data
│   ├── processed/                # Processed datasets
│   └── models/                   # Trained model files
├── config/                       # Configuration files
├── scripts/                      # Deployment and utility scripts
├── deployment/                   # Docker and deployment configs
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
└── tests/                        # Test files
```

## 🚀 Quick Start

### Using the Main Entry Point

```bash
# Run the inference API
python3 main.py api

# Launch the Streamlit demo
python3 main.py demo

# Run model evaluation
python3 main.py eval

# Start model training
python3 main.py train
```

### Using Individual Components

```bash
# API Service
cd src/api && uvicorn inference_api:app --host 0.0.0.0 --port 8000

# Streamlit Demo
streamlit run src/api/streamlit_demo.py

# Advanced Evaluation
python3 src/evaluation/advanced_evaluation.py --model_path data/models/experiment_2_epoch086_val0.5996.pt

# Training
python3 src/models/ranking_optimized_training.py
```

### Using Scripts

```bash
# Quick demo launcher
./scripts/run_demo.sh

# Full deployment
./scripts/deploy.sh
```

## 🎯 Key Features

### Advanced ML Architecture
- **Hybrid VAE**: Combines collaborative filtering with deep learning
- **150D embeddings** → **[512,256,128] hidden layers** → **64D latent space**
- **β-VAE regularization** for improved recommendation diversity
- **Cold-start handling** via latent space generation

### Production-Ready ETL Pipeline
- **Temporal data splitting** (64% train, 16% val, 20% test)
- **Automated quality validation** with comprehensive metrics
- **Incremental learning** capabilities
- **32M+ rating interactions** from MovieLens-32M dataset

### Comprehensive Evaluation
- **Basic Metrics**: RMSE, MAE, R² for prediction accuracy
- **Ranking Metrics**: mAP@10, NDCG@10, MRR@10 for recommendation quality
- **Business Impact**: Diversity, catalog coverage, popularity bias analysis
- **A/B Testing Framework**: Compare different recommendation strategies

### Business Logic Integration
- **Hard/Soft Filtering**: Genre preferences, year ranges, content ratings
- **Diversity Optimization**: Balance between accuracy and variety
- **Recency Bias Control**: Adjustable preference for recent movies
- **Cold-Start Solutions**: Handle new users and movies gracefully

### Real-Time Inference
- **FastAPI Service**: Async request handling with <200ms p95 latency
- **Intelligent Caching**: Reduces response times and computational load
- **Monitoring & Alerting**: Comprehensive health checks and metrics
- **Horizontal Scaling**: Docker containerization for production deployment

## 📊 Performance Metrics

### Model Performance
- **RMSE**: 1.2547 (validation), 1.2135 (test)
- **mAP@10**: 0.75+ for ranking quality
- **NDCG@10**: 0.74+ for recommendation relevance
- **R² Score**: Competitive with state-of-the-art approaches

### Business Impact
- **Diversity Score**: 51% genre spread in recommendations
- **Catalog Coverage**: 0.4% of total catalog recommended
- **Response Time**: <1.5s average inference time
- **Scalability**: Handles 200K+ users, 84K+ movies

## 🛠️ Development

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For API-only deployment
pip install -r requirements_api.txt
```

### Running Tests

```bash
# Run evaluation tests
python3 src/evaluation/advanced_evaluation.py

# Test API health
curl http://localhost:8000/health

# Test recommendation endpoint
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 10}'
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t movielens-recsys .
docker run -p 8000:8000 movielens-recsys
```

## 📈 API Endpoints

### Health Check
- `GET /health` - System health and metrics
- `GET /` - API information and status

### Recommendations
- `POST /recommend` - Generate personalized recommendations
- `POST /recommend/batch` - Batch recommendation processing
- `POST /user-profile` - Update user preferences

### Model Information
- `GET /model/info` - Model architecture and performance
- `GET /model/stats` - Usage statistics and metrics

## 🔧 Configuration

### ETL Pipeline
- **config/etl_config.json**: Data processing parameters
- **Environment Variables**: `MOVIELENS_DATA_PATH`, `MODEL_PATH`

### API Settings
- **Host/Port**: Configurable via environment
- **Caching**: Redis integration for production
- **Logging**: Structured logging with configurable levels

### Business Rules
- **Genre Preferences**: User-specific boosts/penalties
- **Content Filtering**: Age ratings, languages, release years
- **Diversity Controls**: Exploration vs exploitation balance

## 📚 Documentation

- **docs/README.md**: General overview and setup
- **docs/PROJECT_SHOWCASE.md**: Technical deep-dive
- **docs/ADVANCED_INTEGRATION_GUIDE.md**: Integration patterns
- **docs/integration_plan.md**: Implementation roadmap

## 🤝 Contributing

1. **Follow the modular structure**: Place new features in appropriate src/ subdirectories
2. **Update imports**: Use relative imports within modules
3. **Add tests**: Include test coverage for new functionality
4. **Document changes**: Update relevant documentation

## 📄 License

This project is built for educational and demonstration purposes using the MovieLens dataset. Please refer to MovieLens licensing terms for commercial usage.

## 🎉 Acknowledgments

- **MovieLens Dataset**: GroupLens Research Project
- **PyTorch Framework**: Deep learning foundation
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Interactive demo interface