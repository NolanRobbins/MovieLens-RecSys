# Runpod A100 Deployment Guide for Hybrid VAE Training

This guide walks you through deploying and training your fixed Hybrid VAE model on Runpod's A100 GPUs.

## 🚀 Quick Start

1. **Launch Runpod Instance**
2. **Upload Files** 
3. **Run Setup Script**
4. **Start Training**

---

## Step 1: Launch Runpod Instance

### Recommended Configuration:
- **GPU**: NVIDIA A100 (40GB or 80GB)
- **Template**: PyTorch 2.0+ 
- **Storage**: 50GB+ persistent volume
- **Container Disk**: 20GB+

### Launch Command:
```bash
# Use the PyTorch template or custom image
# Ensure you have enough persistent storage for data and models
```

---

## Step 2: Upload Required Files

Upload these files to your Runpod instance:

### Core Training Files:
```
/workspace/
├── runpod_training.py          # Main training script
├── runpod_setup.sh             # Environment setup
├── runpod_data_prep.py         # Data preparation
├── requirements_runpod.txt     # Dependencies
└── data/                       # Your data files
    ├── train_data.csv
    ├── val_data.csv
    └── data_mappings.pkl
```

### Upload Methods:
1. **Web Interface**: Use Runpod's file upload
2. **SCP**: `scp -r files/ user@runpod-ip:/workspace/`
3. **Git Clone**: Clone your repository directly
4. **Cloud Storage**: Download from S3/GCS/etc.

---

## Step 3: Environment Setup

Connect to your instance and run:

```bash
# Make setup script executable
chmod +x runpod_setup.sh

# Run setup script
./runpod_setup.sh

# Verify GPU access
nvidia-smi
```

Expected output:
```
NVIDIA A100-SXM4-40GB
GPU Memory: 40.0 GB
✅ Environment setup complete!
```

---

## Step 4: Data Preparation

Validate and optimize your data:

```bash
# Run data preparation
python runpod_data_prep.py

# Expected output:
# ✅ train_data.csv: 319.1 MB
# ✅ val_data.csv: 79.8 MB  
# ✅ data_mappings.pkl: 6.6 MB
# ✅ No data quality issues found
# 🎯 Data preparation complete!
```

---

## Step 5: Start Training

### Basic Training:
```bash
python runpod_training.py
```

### Advanced Training with Monitoring:
```bash
# Install wandb for experiment tracking
pip install wandb
wandb login

# Run with monitoring
nohup python runpod_training.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## Training Configuration

The training script uses optimized settings for A100:

```python
config = {
    'batch_size': 16384,        # Large batch for A100
    'n_epochs': 100,
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'kl_weight_start': 0.0,     # KL annealing
    'kl_weight_end': 0.1,
    'kl_warmup_epochs': 20,
    'patience': 15,
}
```

### Key A100 Optimizations:
- **Mixed Precision Training**: Uses AMP for 2x speedup
- **TF32 Enabled**: Automatic precision optimization
- **Large Batches**: 16,384 samples per batch
- **Optimized Data Loading**: 8 workers, prefetching
- **GELU Activation**: Optimized for A100 architecture
- **LayerNorm**: Better than BatchNorm for transformers

---

## Expected Training Results

### Performance Targets:
- **Training Speed**: ~2-3 minutes per epoch
- **Memory Usage**: ~15-20GB GPU memory
- **Target RMSE**: 0.75-0.85 (down from 1.21)
- **Training Time**: 2-4 hours total

### Sample Training Output:
```
🚀 Starting A100-Optimized Hybrid VAE Training
============================================================
GPU: NVIDIA A100-SXM4-40GB
Model parameters: 43,318,873
Epoch 001/100 | Train RMSE: 0.8234 | Val RMSE: 0.8456 | Time: 2.3min
Epoch 005/100 | Train RMSE: 0.7892 | Val RMSE: 0.8123 | Time: 11.5min
Epoch 010/100 | Train RMSE: 0.7654 | Val RMSE: 0.7987 | Time: 23.1min
...
✅ Training completed in 156.7 minutes
🏆 Best validation RMSE: 0.7743
```

---

## Monitoring and Troubleshooting

### Real-time Monitoring:
```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f training.log

# System resources
htop
```

### Common Issues:

#### 1. Out of Memory Error:
```bash
# Reduce batch size in runpod_training.py
config['batch_size'] = 8192  # Reduce from 16384
```

#### 2. NaN Values:
The fixed model includes extensive NaN prevention:
- Gradient clipping
- KL divergence clamping  
- Proper weight initialization
- Numerical stability checks

#### 3. Slow Training:
```bash
# Verify A100 optimizations are enabled
python -c "import torch; print(f'TF32: {torch.backends.cuda.matmul.allow_tf32}')"
```

#### 4. Data Loading Issues:
```bash
# Check data files
ls -la /workspace/data/
python runpod_data_prep.py
```

---

## Model Saving and Export

### Automatic Saving:
- Best model saved to `/workspace/models/hybrid_vae_fixed.pt`
- Training summary saved to `/workspace/models/training_summary.json`

### Manual Export:
```bash
# Download trained model
# Use Runpod's download interface or:
scp user@runpod-ip:/workspace/models/hybrid_vae_fixed.pt ./
```

---

## Cost Optimization

### A100 Pricing Tips:
1. **Use Spot Instances**: 50-70% cheaper
2. **Monitor Training**: Stop early if converged
3. **Batch Jobs**: Train multiple experiments
4. **Persistent Storage**: Keep data between sessions

### Expected Costs (A100 40GB):
- **On-Demand**: ~$2-4 for full training
- **Spot Instance**: ~$1-2 for full training
- **Storage**: ~$0.10/month for 50GB

---

## Advanced Features

### Experiment Tracking with W&B:
```python
# In runpod_training.py, add:
import wandb

wandb.init(project="movielens-hybrid-vae")
# Automatic logging included
```

### Multi-GPU Training:
```bash
# For multiple A100s (if available)
python -m torch.distributed.launch --nproc_per_node=2 runpod_training.py
```

### Hyperparameter Sweeps:
```bash
# Create sweep configuration
# Run multiple experiments automatically
```

---

## Deployment Checklist

- [ ] Runpod A100 instance launched
- [ ] Files uploaded to `/workspace/`
- [ ] Environment setup completed (`./runpod_setup.sh`)
- [ ] Data validation passed (`python runpod_data_prep.py`)  
- [ ] GPU access verified (`nvidia-smi`)
- [ ] Training started (`python runpod_training.py`)
- [ ] Monitoring active (`tail -f training.log`)

---

## Support and Next Steps

### If Training Succeeds:
1. Download the trained model
2. Integrate into production pipeline
3. Run A/B tests against baseline
4. Monitor real-world performance

### If Issues Persist:
1. Check the troubleshooting section
2. Review training logs for specific errors
3. Validate data quality
4. Consider reducing model complexity

### Contact Points:
- Runpod Discord: Technical support
- GitHub Issues: Code-related problems
- Documentation: Extended guides and examples

---

## Files Summary

| File | Purpose |
|------|---------|
| `runpod_training.py` | Main A100-optimized training script |
| `runpod_setup.sh` | Environment setup and dependencies |
| `runpod_data_prep.py` | Data validation and optimization |
| `requirements_runpod.txt` | Python package requirements |
| `RUNPOD_DEPLOYMENT_GUIDE.md` | This comprehensive guide |

**Total estimated setup time: 15-30 minutes**
**Total training time: 2-4 hours**
**Expected RMSE improvement: 1.21 → 0.75-0.85**