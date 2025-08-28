# RecBole ETL Pipeline Integration Guide

## 🔄 **CRITICAL: Understanding RecBole Data Splitting**

RecBole uses a **fundamentally different approach** from traditional train/val/test splits:

### **Traditional Approach (What We Had)**:
```
train_data.csv  →  80% interactions (older)
val_data.csv    →  10% interactions (middle)  
test_data.csv   →  10% interactions (newer)
```

### **RecBole Approach (What We Have Now)**:
```
movielens.inter  →  100% interactions (ALL data, temporally sorted)
                    ├── Training: All but last 2 interactions per user
                    ├── Validation: Second-to-last interaction per user
                    └── Test: Last interaction per user
```

## 🎯 **ETL Pipeline Integration Benefits**

This single-file approach is **PERFECT** for your ETL pipeline because:

### **1. Future Data Evaluation**
```python
# Day 1-20: Historical data (20M interactions)
original_data = load_movielens_data()  # 2020-2023 data

# Day 21: New ETL batch arrives  
new_batch = etl_pipeline.get_batch(day=21)  # 2024 data

# Combine: RecBole automatically uses newest data as test set
combined_data = append_future_data_for_etl(new_batch, 'movielens.inter')

# Result: Model trained on 2020-2023, tested on 2024 (true temporal evaluation)
```

### **2. Realistic Performance Measurement**
- ✅ **True temporal evaluation**: Model never sees future data during training
- ✅ **No data leakage**: Strict temporal ordering maintained
- ✅ **Realistic metrics**: Performance measured on genuinely unseen future interactions

### **3. ETL Integration Workflow**
```bash
# Step 1: Train initial model on historical data
./runpod_entrypoint.sh --model ss4rec-official

# Step 2: ETL pipeline processes new batch
python etl/batch_etl_pipeline.py --day 21

# Step 3: Append new data and re-evaluate
python data/recbole_format/movielens_adapter.py --append-etl-batch

# Step 4: Retrain model with new data as test set
./runpod_entrypoint.sh --model ss4rec-official --eval-only

# Result: Performance metrics on truly unseen future data
```

## 📊 **Data Flow Example**

### **Original Training Data (Days 1-20)**:
```
movielens.inter:
user_1,item_100,1577836800.0  # 2020-01-01
user_1,item_200,1609459200.0  # 2021-01-01  
user_1,item_300,1640995200.0  # 2022-01-01
...
```

### **After ETL Batch (Day 21)**:
```
movielens.inter:
user_1,item_100,1577836800.0  # 2020-01-01 (training)
user_1,item_200,1609459200.0  # 2021-01-01 (training)
user_1,item_300,1640995200.0  # 2022-01-01 (validation)
user_1,item_400,1672531200.0  # 2023-01-01 (test - NEW!)
```

### **RecBole Automatic Splitting**:
- **Training**: Items 100, 200 (older interactions)
- **Validation**: Item 300 (second-to-last)  
- **Test**: Item 400 (newest - from ETL batch)

## 🎯 **Key Benefits for Your Project**

### **1. Business Realism**
- Model performance measured on **actual future user behavior**
- No artificial random splits that don't reflect real-world deployment

### **2. A/B Testing Integration**  
- Compare SS4Rec vs NCF on **same future data**
- Fair comparison using identical temporal evaluation

### **3. Portfolio Demonstration**
- Shows understanding of **temporal data science**
- Demonstrates **production-ready** evaluation methodology
- Enables **realistic performance claims**

## 🚀 **Implementation Ready**

Your ETL pipeline can now:
1. ✅ **Process new batches** (already implemented)
2. ✅ **Append to RecBole format** (function ready)
3. ✅ **Trigger model re-evaluation** (RunPod integration ready)
4. ✅ **Generate realistic performance metrics** (temporal evaluation)

This approach ensures your SS4Rec vs NCF comparison reflects **real-world performance** on genuinely unseen future data! 🏆