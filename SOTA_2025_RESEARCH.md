# 🚀 MovieLens-RecSys: 2025 SOTA Research Implementation

## 🎯 Project Objective
Implement and compare cutting-edge 2025 recommender system architectures on MovieLens dataset to achieve best possible **validation RMSE**.

## 📊 Current Status

### ❌ **Previous Approach Issues**
- **Hybrid VAE**: Based on 2018-2019 research (outdated)
- **Validation RMSE**: ~1.25 (actual performance)  
- **Training RMSE**: 0.5996 (misleading - overfitted)
- **Problem**: 2-3 years behind SOTA

### ✅ **New 2025 SOTA Approach**
- **Baseline**: Neural Collaborative Filtering (NCF)
- **SOTA Options**: HydraRec (Jan 2025) vs SS4Rec (2025)
- **Target**: Best possible validation RMSE
- **Focus**: Real generalization performance

## 🏗️ Repository Structure (2025 SOTA)

```
MovieLens-RecSys/
├── models/
│   ├── baseline/
│   │   └── neural_cf.py           # NCF baseline implementation
│   ├── sota_2025/
│   │   ├── hydra_rec.py          # HydraRec (Jan 2025)
│   │   ├── ss4rec.py             # SS4Rec (2025)
│   │   └── components/
│   │       ├── hydra_attention.py
│   │       ├── state_space.py
│   │       └── linear_attention.py
│   └── evaluation/
│       └── validation_tracker.py  # Proper train/val tracking
├── training/
│   ├── train_ncf.py              # Baseline training
│   ├── train_hydra_rec.py        # HydraRec training
│   ├── train_ss4rec.py           # SS4Rec training
│   └── utils/
│       ├── data_loaders.py
│       └── metrics.py
├── experiments/
│   ├── configs/
│   │   ├── ncf_baseline.yaml
│   │   ├── hydra_rec.yaml
│   │   └── ss4rec.yaml
│   └── results/
│       ├── ncf_baseline/
│       ├── hydra_rec/
│       └── ss4rec/
└── README_2025.md
```

## 🎯 **Models to Implement**

### 1. **Neural CF (Baseline)**
- **Paper**: Neural Collaborative Filtering (2017)
- **Purpose**: Solid baseline for comparison
- **Expected Val RMSE**: ~0.85-0.90
- **Implementation**: Already exists in `src/models/baseline_models.py`

### 2. **HydraRec (SOTA Candidate #1)**
- **Paper**: "An Efficient Attention Mechanism for Sequential Recommendation Tasks" (Jan 2025)
- **Innovation**: Linear attention complexity O(n) vs O(n²)
- **Key Features**:
  - Hydra attention mechanism
  - Efficient for long sequences
  - Comparable to BERT4Rec with better runtime
- **Expected Val RMSE**: ~0.65-0.75

### 3. **SS4Rec (SOTA Candidate #2)**  
- **Paper**: "Sequential Recommendation with State Space Models" (2025)
- **Innovation**: Continuous-time user interest modeling
- **Key Features**:
  - State Space Models for sequential data
  - SOTA on 5 datasets (HR@10, NDCG@10, MRR@10)
  - Dynamic user preference evolution
- **Expected Val RMSE**: ~0.60-0.70

## 📈 **Experimental Plan**

### Phase 1: Baseline Establishment (Week 1)
1. **Clean NCF Implementation**
   - Remove VAE-specific code
   - Focus on clean Neural CF architecture
   - Proper train/validation split tracking

2. **Baseline Training**
   - Train NCF on MovieLens dataset
   - Establish validation RMSE baseline
   - Document performance metrics

### Phase 2: SOTA Implementation (Week 2-3)
1. **Model Selection**: Choose between HydraRec vs SS4Rec
2. **Implementation**: Code chosen SOTA model
3. **Training**: Train and compare to NCF baseline
4. **Analysis**: Validate performance improvements

### Phase 3: Optimization (Week 4)
1. **Hyperparameter Tuning**
2. **Performance Analysis**
3. **Final Comparison Report**

## 🤔 **HydraRec vs SS4Rec Analysis**

### **MovieLens Dataset Characteristics**
- **Type**: Implicit feedback (ratings)
- **Size**: 25M ratings, 162K movies, 280K users
- **Temporal**: Timestamps available
- **Sparsity**: High sparsity typical of collaborative filtering

### **Model Suitability Analysis**

#### **HydraRec Advantages for MovieLens**
✅ **Efficient Attention**: Better for large user/item vocabulary
✅ **Proven on RecSys**: Directly designed for recommendation tasks
✅ **Linear Complexity**: Scales better with MovieLens size
✅ **BERT4Rec Compatible**: Can leverage transformer knowledge

#### **SS4Rec Advantages for MovieLens** 
✅ **Temporal Modeling**: MovieLens has rich timestamp data
✅ **SOTA Results**: Proven best on multiple RecSys datasets
✅ **Continuous-Time**: Better models user interest evolution
✅ **Newer Research**: Most recent advances (2025)

## 🏆 **Recommendation: SS4Rec**

**Why SS4Rec is better for MovieLens:**

1. **Temporal Data Utilization**: MovieLens timestamps are underutilized in current implementation
2. **Proven SOTA Results**: Documented SOTA on 5 RecSys datasets
3. **Continuous Modeling**: Better matches real user behavior patterns
4. **Research Recency**: Most cutting-edge 2025 research

**Expected Performance:**
- **NCF Baseline**: ~0.85 validation RMSE
- **SS4Rec SOTA**: ~0.60-0.65 validation RMSE (**25-30% improvement**)

## 📝 **Next Steps**

1. ✅ Clean repository structure for SOTA focus
2. ✅ Set up Neural CF baseline training
3. ✅ Implement SS4Rec architecture
4. ✅ Compare validation RMSE results
5. ✅ Document performance improvements

---

**Target**: Achieve best possible validation RMSE using 2025 SOTA research  
**Timeline**: 3-4 weeks for complete implementation and evaluation