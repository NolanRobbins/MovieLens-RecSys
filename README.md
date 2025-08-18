# 🎬 MovieLens-RecSys: SS4Rec Implementation

## 🎯 Project Overview

Implementation and evaluation of **SS4Rec** (Continuous-Time Sequential Recommendation with State Space Models) on the MovieLens dataset, comparing against Neural Collaborative Filtering baseline.

### **Research Focus**
- **Paper**: "SS4Rec: Continuous-Time Sequential Recommendation with State Space Models" (Feb 2025)
- **Baseline**: Neural Collaborative Filtering (NCF)
- **Dataset**: MovieLens 25M
- **Goal**: Achieve best possible validation RMSE using 2025 SOTA research

## 🏗️ Architecture

### **Models**
1. **Neural CF (Baseline)**: Standard collaborative filtering with neural networks
2. **SS4Rec (SOTA)**: State space models for continuous-time sequential recommendation

### **Key Features**
- ⏰ **Temporal Modeling**: Utilizes MovieLens timestamp data
- 🔄 **Sequential Patterns**: Models user behavior sequences  
- 📊 **Dense Dataset Optimization**: Optimized for MovieLens characteristics
- 🎯 **Validation Focus**: Real generalization performance tracking

## 📊 Expected Results

| Model | Expected Validation RMSE | Improvement |
|-------|-------------------------|-------------|
| Neural CF | ~0.85-0.90 | Baseline |
| **SS4Rec** | **~0.60-0.70** | **25-30%** |

## 🚀 Quick Start

### **Training on A6000 GPU (RunPod)**

```bash
# Clone repository
git clone <repo-url>
cd MovieLens-RecSys

# Install dependencies
pip install -r requirements.txt

# Train Neural CF baseline
python training/train_ncf.py --config configs/ncf_baseline.yaml

# Train SS4Rec model
python training/train_ss4rec.py --config configs/ss4rec.yaml

# Compare results
python evaluation/compare_models.py
```

### **Local Development**

```bash
# Run Streamlit demo
streamlit run streamlit_app.py

# Data preprocessing
python src/data/etl_pipeline.py

# Model evaluation
python src/evaluation/evaluate_model.py
```

## 📁 Project Structure

```
MovieLens-RecSys/
├── models/
│   ├── baseline/
│   │   └── neural_cf.py          # Neural CF implementation
│   └── sota_2025/
│       ├── ss4rec.py             # SS4Rec implementation
│       └── components/           # SSM components
├── training/
│   ├── train_ncf.py              # NCF training
│   ├── train_ss4rec.py           # SS4Rec training
│   └── utils/
├── configs/
│   ├── ncf_baseline.yaml         # NCF configuration
│   └── ss4rec.yaml               # SS4Rec configuration
├── src/
│   ├── data/                     # ETL pipeline
│   ├── evaluation/               # Model evaluation
│   ├── api/                      # Inference API
│   └── serving/                  # Production serving
├── streamlit_app.py              # Web interface
└── requirements.txt
```

## 🔬 Research Implementation

### **SS4Rec Key Components**
- **Time-Aware SSM**: Handles irregular time intervals
- **Relation-Aware SSM**: Models contextual dependencies
- **Continuous-Time Encoder**: User interest evolution modeling

### **Evaluation Metrics**
- **Primary**: Validation RMSE
- **Secondary**: HR@10, NDCG@10, MRR@10
- **Protocol**: Leave-one-out evaluation

## 🎯 Performance Targets

### **Primary Goal**
- **Validation RMSE < 0.70** (vs current ~1.25)

### **Secondary Goals**
- **HR@10 > 0.30**
- **NDCG@10 > 0.25**
- **Training efficiency on A6000**

## 🛠️ Development Setup

### **Requirements**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- A6000 GPU (recommended for training)

### **Installation**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 📈 Results & Evaluation

Results will be documented in:
- `results/ncf_baseline/` - Neural CF performance
- `results/ss4rec/` - SS4Rec performance  
- `evaluation/comparison_report.md` - Model comparison

## 🚀 Production Deployment

- **API**: FastAPI inference server
- **Serving**: Real-time recommendation endpoint
- **Monitoring**: Performance and drift detection
- **Streamlit**: Interactive demo interface

## 📚 References

1. **SS4Rec**: "Continuous-Time Sequential Recommendation with State Space Models" (2025)
2. **Neural CF**: "Neural Collaborative Filtering" (WWW 2017)
3. **MovieLens**: GroupLens Research Dataset

## 🤝 Contributing

Focus areas:
- SS4Rec implementation improvements
- Evaluation metric enhancements
- A6000 training optimizations
- Documentation updates

---

**Research Goal**: Demonstrate SS4Rec's superiority over traditional collaborative filtering on MovieLens dataset using proper validation methodology.