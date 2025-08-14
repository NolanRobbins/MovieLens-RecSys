# 🚀 Runpod A100 Setup Guide - Ready to Deploy!

## 🎯 Quick Deploy (5 minutes)

Your repository is now **fully ready** for A100 deployment with comprehensive W&B monitoring!

---

## Step 1: Launch Runpod Instance

### Recommended Configuration:
```yaml
GPU: NVIDIA A100 (40GB or 80GB)
Template: PyTorch 2.0+ (or RunPod PyTorch)
CPU: 8+ vCPUs  
RAM: 32GB+
Storage: 50GB+ persistent volume
Container Disk: 20GB+
```

### Launch Steps:
1. Go to [console.runpod.io](https://console.runpod.io)
2. Click "Deploy" → "GPU Pods"
3. Select **A100 40GB** (or 80GB if budget allows)
4. Choose **PyTorch** template
5. Set storage to **50GB persistent**
6. Launch pod

**Expected cost**: ~$0.50-1.00/hour

---

## Step 2: One-Command Deployment

Once your pod is running, connect via SSH or web terminal and run:

```bash
# Single command deployment - handles everything!
curl -sSL https://raw.githubusercontent.com/NolanRobbins/MovieLens-RecSys/main/deploy_to_runpod.sh | bash
```

This script automatically:
- ✅ Clones your repository
- ✅ Installs all dependencies  
- ✅ Sets up GPU optimizations
- ✅ Configures W&B integration
- ✅ Creates monitoring tools
- ✅ Validates system requirements

---

## Step 3: Upload Your Data

The script will prompt you to upload these files to `/workspace/data/processed/`:
- `train_data.csv` (319MB)
- `val_data.csv` (80MB) 
- `data_mappings.pkl` (6.6MB)

### Upload Methods:

**Option A: Web Interface**
1. Use Runpod's file manager
2. Upload to `/workspace/data/processed/`

**Option B: SCP** (if you have SSH access)
```bash
scp -r data/processed/* runpod-ip:/workspace/data/processed/
```

**Option C: From your local machine**
```bash
# If you have the data locally
rsync -av data/processed/ runpod-user@runpod-ip:/workspace/data/processed/
```

---

## Step 4: Configure W&B

You'll need your Weights & Biases API key:

1. Go to [wandb.ai/settings](https://wandb.ai/settings)
2. Copy your API key
3. The deployment script will prompt for it
4. Or set as environment variable: `export WANDB_API_KEY="your-key-here"`

---

## Step 5: Start Training

After deployment completes:

```bash
# Start training with full monitoring
./run_training.sh

# Choose option 2 for full W&B integration
```

**Expected Output:**
```
🚀 Starting A100 Training with W&B Monitoring
==============================================

Training started with PID: 12345
📈 W&B dashboard: https://wandb.ai/your-username/movielens-hybrid-vae-a100
📊 GPU monitoring: nvidia-smi -l 1
```

---

## Step 6: Monitor Progress

### Real-time Monitoring:
```bash
./monitor.sh      # Comprehensive system monitor
tail -f training.log   # Training progress
nvidia-smi -l 1        # GPU utilization
```

### W&B Dashboard:
- **URL**: https://wandb.ai/your-username/movielens-hybrid-vae-a100
- **Real-time metrics**: Loss, RMSE, GPU usage, training speed
- **Model artifacts**: Best model automatically saved

---

## 🎯 Expected Results

### Performance Targets:
- **Current baseline**: 0.5996 RMSE (experiment_2)
- **Target**: 0.52-0.55 RMSE (8-13% improvement)
- **Training time**: 2-3 hours
- **GPU memory**: 15-20GB usage
- **Total cost**: ~$2-4

### Training Progress:
```
Epoch  05/100 | Train Loss: 0.4234 | Val RMSE: 0.7891 | Time: 2.1min 🏆
Epoch  15/100 | Train Loss: 0.3876 | Val RMSE: 0.6543 | Time: 2.3min 
Epoch  30/100 | Train Loss: 0.3654 | Val RMSE: 0.5789 | Time: 2.2min 
Epoch  45/100 | Train Loss: 0.3523 | Val RMSE: 0.5432 | Time: 2.4min 🏆
✅ Training completed! Best RMSE: 0.5287
```

---

## 🔧 Troubleshooting

### Common Issues:

**Out of Memory (OOM)**
```bash
# Edit runpod_training_wandb.py and reduce batch_size
'batch_size': 3072,  # Reduce from 6144
```

**Slow Training**
```bash
# Verify TF32 is enabled
python -c "import torch; print(f'TF32: {torch.backends.cuda.matmul.allow_tf32}')"
```

**W&B Not Logging**
```bash
# Check API key
wandb status
wandb login
```

**Training Crashes**
```bash
# Check logs
tail -100 training.log
# Look for NaN values or GPU errors
```

---

## 📊 W&B Metrics You'll See

### Training Metrics:
- `train/batch_loss`, `train/batch_rmse`
- `train/grad_norm`, `train/learning_rate`
- `val/rmse`, `val/loss`, `val/mae`

### GPU Metrics:
- `gpu_0_memory_used_gb`, `gpu_0_util_percent`
- `gpu_0_temp_c`, `gpu_0_power_watts`
- `gpu_0_graphics_clock_mhz`

### System Metrics:
- `system/cpu_count`, `system/memory_gb`
- `system/gpu_name`, `system/is_a100`

---

## ✅ Success Indicators

You'll know it's working when you see:

1. **GPU Utilization**: 80-95% consistently
2. **Memory Usage**: 15-20GB/40GB
3. **Training Speed**: ~2-3 minutes per epoch
4. **RMSE Improving**: Starts ~0.80, targets 0.52-0.55
5. **W&B Logging**: Real-time charts updating
6. **No NaN Values**: Stable loss curves

---

## 🎉 When Training Completes

### Outputs:
- **Best Model**: `/workspace/models/hybrid_vae_a100_best.pt`
- **Summary**: `/workspace/models/training_summary.json`
- **W&B Artifacts**: Automatically uploaded

### Download Results:
```bash
# Download best model
scp runpod-ip:/workspace/models/hybrid_vae_a100_best.pt ./
```

---

## 💰 Cost Optimization

### Estimated Costs (A100 40GB):
- **Spot Instance**: $0.50-0.70/hour → **$1.50-2.50 total**
- **On-Demand**: $0.80-1.20/hour → **$2.50-4.00 total**

### Tips:
- Use **spot instances** for 50%+ savings
- Stop pod immediately when done
- Monitor progress to stop early if target achieved

---

## 🚀 You're Ready!

Your comprehensive A100 training setup includes:
- ✅ **Validated data pipeline** (fixed mapping issues)
- ✅ **A100-optimized model** (based on experiment_2 success)
- ✅ **Full W&B integration** (comprehensive monitoring)
- ✅ **Automated deployment** (one command setup)
- ✅ **Real-time monitoring** (GPU + training metrics)
- ✅ **Error prevention** (NaN handling, gradient clipping)

**Go launch that Runpod instance and run the deployment script!** 🚀

Target: **0.52-0.55 RMSE in 2-3 hours for ~$2-4** 🎯