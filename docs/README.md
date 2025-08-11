# MovieLens Hybrid VAE Recommender System 🎬

A hybrid recommendation system combining collaborative filtering embeddings with Variational Autoencoders (VAE) for the MovieLens 32M dataset.

## 🚀 Features

- **Hybrid Architecture**: Combines EmbeddingNet + VAE for superior recommendations
- **Temporal Split**: Time-based data splitting to simulate real-world ETL pipelines  
- **Cloud Training**: Optimized for GPU cloud providers (RunPod, Lambda Labs)
- **No Data Leakage**: Proper train/val/test splits with temporal considerations
- **Production Ready**: Includes inference pipeline and recommendation generation

## 📊 Dataset

- **MovieLens 32M**: 32 million ratings from 200K+ users on 84K+ movies
- **Temporal Split**: 
  - Training: 64% (older data)
  - Validation: 16% (older data) 
  - ETL Test: 20% (newest data - simulates incoming ratings)

## 🏗️ Architecture

### Hybrid VAE Model
```
User/Movie Embeddings → Encoder → Latent Space (μ, σ) → Decoder → Rating Prediction
                                     ↓
                               VAE Loss: MSE + KL Divergence
```

**Key Components:**
- Collaborative filtering embeddings (150D)
- Deep encoder/decoder networks (512→256 hidden layers)  
- 64D latent space with reparameterization trick
- Batch normalization + dropout for regularization

## 🛠️ Setup & Training

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd MovieLens-RecSys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
jupyter notebook data_loading.ipynb
```

### Cloud Training (RunPod Recommended)

1. **Prepare Data Locally:**
   ```python
   # Run export cell in data_loading.ipynb to create:
   # - train_data.csv
   # - val_data.csv  
   # - data_mappings.pkl
   ```

2. **Deploy to RunPod:**
   ```bash
   # Upload files to RunPod workspace
   # Run setup script
   bash setup_runpod.sh
   
   # Start training
   python cloud_training.py --data_path ./data --save_path ./models/hybrid_vae_best.pt --use_wandb
   ```

### Training Parameters
```python
Config = {
    'n_factors': 150,      # Embedding dimension
    'hidden_dims': [512, 256],  # Encoder/decoder layers
    'latent_dim': 64,      # VAE latent space
    'dropout_rate': 0.3,   # Regularization
    'lr': 1e-4,           # Learning rate
    'batch_size': 512,    # Batch size
    'kl_weight': 1.0      # β-VAE parameter
}
```

## 💰 Cost Estimates

| Provider | GPU | Cost/Hour | Training Time | Total Cost |
|----------|-----|-----------|---------------|------------|
| **RunPod** | RTX 3080 | $0.34 | ~3 hours | **~$1.00** |
| Lambda Labs | A10 | $0.60 | ~2 hours | ~$1.20 |
| Paperspace | P4000 | $0.51 | ~4 hours | ~$2.04 |

## 📈 Performance

The Hybrid VAE model provides significant improvements over traditional collaborative filtering:

- **Cold Start**: Handles new users/movies via latent space generation
- **Diversity**: VAE generates diverse recommendations beyond top-popular items
- **Scalability**: Efficient training on 32M ratings dataset
- **ETL Ready**: Temporal splits simulate real-world deployment scenarios

## 📁 Project Structure

```
MovieLens-RecSys/
├── data_loading.ipynb          # Data preprocessing & model definition
├── cloud_training.py           # Production training script
├── requirements.txt            # Local dependencies
├── requirements_cloud.txt      # Cloud dependencies  
├── setup_runpod.sh            # Cloud setup script
├── .gitignore                 # Git ignore rules
└── README.md                  # This file

# Generated files (ignored by git):
├── train_data.csv             # Training data export
├── val_data.csv              # Validation data export
├── data_mappings.pkl         # User/movie mappings
└── models/                   # Trained model checkpoints
```

## 🚀 Production Deployment

### Quick Start
```bash
# Clone the repository
git clone <your-repo-url>
cd MovieLens-RecSys

# Run complete deployment
./deploy.sh

# Access services
# API: http://localhost:8000
# Demo: http://localhost:8501
```

### Full System Components

#### 🔄 **ETL Pipeline** (`etl_pipeline.py`)
- **Automated data ingestion** with quality validation
- **Temporal splitting** to simulate real-world scenarios  
- **Incremental updates** and data freshness monitoring
- **Comprehensive logging** and error handling

```bash
python etl_pipeline.py --data_source ./ml-32m --output ./processed_data
```

#### 🌐 **Inference API** (`inference_api.py`)
- **FastAPI service** with async request handling
- **Real-time recommendations** with business logic filtering
- **Caching and performance optimization**
- **Health monitoring** and metrics collection

```bash
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

#### 🎯 **Interactive Demo** (`streamlit_demo.py`)
- **Professional interface** for recruiters/hiring managers
- **Live recommendation generation** with customizable parameters
- **Business impact calculator** and ROI demonstration
- **System metrics dashboard** and architecture overview

```bash
streamlit run streamlit_demo.py
```

#### 🐳 **Production Deployment**
- **Docker containerization** with multi-service orchestration
- **Health checks** and auto-restart capabilities
- **Monitoring stack** (Prometheus + Grafana)
- **Scalable architecture** with load balancing

```bash
# Full deployment with monitoring
./deploy.sh --monitoring

# Quick deployment
./deploy.sh --quick
```

### 📊 Business Impact Dashboard

The system includes a comprehensive business impact demonstration:

- **ROI Calculator**: Quantify revenue impact based on engagement lift
- **Performance Metrics**: Real-time API response times and throughput
- **Data Quality Monitoring**: ETL pipeline health and data freshness
- **A/B Testing Framework**: Compare recommendation strategies

### 🎯 Key Differentiators for Technical Interviews

1. **Production-Ready Architecture**
   - Proper error handling and logging
   - Health checks and graceful degradation
   - Horizontal scaling capabilities

2. **Advanced ML Engineering**
   - Hybrid VAE with collaborative filtering
   - Cold-start problem solutions
   - Business logic integration

3. **End-to-End Data Pipeline**
   - Real ETL with quality validation
   - Incremental learning capabilities  
   - Monitoring and alerting

4. **Business Value Focus**
   - ROI calculations and impact metrics
   - User engagement optimization
   - Recommendation diversity vs accuracy trade-offs

## 🔮 Advanced Features

### Implemented ✅
- [x] **Hybrid VAE Model** with temporal splitting
- [x] **Production ETL Pipeline** with quality validation
- [x] **FastAPI Inference Service** with caching
- [x] **Interactive Demo Interface** for stakeholders
- [x] **Docker Deployment** with monitoring
- [x] **Business Logic Engine** with filtering rules
- [x] **Comprehensive Documentation** and showcase

### Future Enhancements 🚧
- [ ] **A/B Testing Framework** with statistical analysis
- [ ] **Real-time Model Updates** with online learning
- [ ] **Multi-armed Bandits** for exploration vs exploitation
- [ ] **Graph Neural Networks** for enhanced recommendations
- [ ] **Federated Learning** for privacy-preserving training

## 📚 References

- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License - see LICENSE file for details.