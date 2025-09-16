# Google Cloud SS4Rec Training Guide

## 🚀 Quick Start Commands

### 1. Create GPU Instance
```bash
# Create a VM with GPU (adjust zone as needed)
gcloud compute instances create ss4rec-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

### 2. Connect to Instance
```bash
# SSH into the instance
gcloud compute ssh ss4rec-training --zone=us-central1-a

# Or use the web SSH from Cloud Console
```

### 3. Setup Environment
```bash
# Once connected, run setup script
curl -fsSL https://raw.githubusercontent.com/your-repo/MovieLens-RecSys/main/gcp_setup.sh | bash

# Or manual setup (if script not available)
sudo apt update && sudo apt install -y git python3-pip

# Clone repository
git clone https://github.com/your-username/MovieLens-RecSys.git
cd MovieLens-RecSys

# Run setup
chmod +x gcp_setup.sh
./gcp_setup.sh
```

### 4. Run Training
```bash
# Activate environment
source activate_ss4rec.sh

# Run validation first
python pre_training_check.py --fix-issues

# Start debug training
python gcp_training.py --log-level DEBUG
```

## 📋 Step-by-Step Instructions

### Phase 1: Instance Creation

1. **Open Google Cloud Console**
   - Go to Compute Engine > VM Instances
   - Click "Create Instance"

2. **Configure Instance**
   - **Name**: `ss4rec-training`
   - **Region**: `us-central1` (or closest to you)
   - **Zone**: `us-central1-a`
   - **Machine type**: `n1-standard-4` (4 vCPUs, 15GB RAM)
   - **GPU**: Add 1x NVIDIA Tesla T4

3. **Boot Disk**
   - **Image**: Deep Learning VM (PyTorch 1.13 with CUDA 11.8)
   - **Size**: 100GB SSD

4. **Advanced Options**
   - **Management**: Check "Enable deletion protection" if desired
   - **Networking**: Default (or configure firewall for specific access)

### Phase 2: Environment Setup

1. **SSH Connection**
```bash
# From your local machine
gcloud compute ssh ss4rec-training --zone=us-central1-a
```

2. **Verify GPU**
```bash
# Check GPU is available
nvidia-smi

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

3. **Get Codebase**
```bash
# Clone repository (replace with your fork/repo)
git clone https://github.com/your-username/MovieLens-RecSys.git
cd MovieLens-RecSys

# Run automated setup
chmod +x gcp_setup.sh
./gcp_setup.sh
```

### Phase 3: Pre-Training Validation

1. **Activate Environment**
```bash
source activate_ss4rec.sh
```

2. **Run Complete Validation**
```bash
# Run comprehensive validation with auto-fix
python pre_training_check.py --fix-issues

# Should output:
# ✅ ALL VALIDATIONS PASSED!
# 🚀 Ready for SS4Rec training!
```

3. **Manual Validation (if needed)**
```bash
# Test individual components
python validate_dependencies.py --fix-missing
python validate_data_pipeline.py --dataset ml-25m
```

### Phase 4: Debug Training

1. **Start Training with Full Monitoring**
```bash
# Debug mode with comprehensive logging
python gcp_training.py --log-level DEBUG 2>&1 | tee training_debug.log
```

2. **Monitor in Real-Time** (separate terminal)
```bash
# Open another SSH session
gcloud compute ssh ss4rec-training --zone=us-central1-a

# Monitor GPU usage
watch -n 2 nvidia-smi

# Monitor training logs
tail -f training_debug.log

# Monitor stability logs
tail -f logs/stability/stability_*.log
```

### Phase 5: Expected Output

**Successful Training Start:**
```
🚀 SS4Rec Google Cloud Training Script
🔍 Running pre-training validations...
✅ Dependencies validated
✅ Data pipeline validated
🛡️ Setting up training stability monitoring...
✅ GPU: Tesla T4
✅ Memory: 14.0GB free / 15.1GB total
✅ CUDA operations test passed
🏃 Starting RecBole training pipeline...
```

**Training Progress:**
```
Train epoch 1:
Loss: 0.6234 | HR@10: 0.1234 | NDCG@10: 0.0856
Gradient norm: 2.4567 (mean: 1.2345)
Memory: GPU 3.2GB/15.1GB, System 45%

Train epoch 2:
Loss: 0.5891 | HR@10: 0.1456 | NDCG@10: 0.0923
Gradient norm: 2.1234 (mean: 1.1567)
Memory: GPU 3.2GB/15.1GB, System 46%
```

## 🚨 Troubleshooting

### GPU Not Available
```bash
# Check if GPU is attached
lspci | grep -i nvidia

# Reinstall drivers if needed
sudo apt install nvidia-driver-525
sudo reboot
```

### Dependency Issues
```bash
# Force reinstall problematic packages
pip uninstall mamba-ssm causal-conv1d -y
pip install --no-cache-dir mamba-ssm causal-conv1d

# Check s5-pytorch
pip install s5-pytorch
```

### Memory Issues
```bash
# Reduce batch size in config
# Edit: configs/official/ss4rec_official.yaml
train_batch_size: 1024  # Reduce from 4096
```

### Data Not Found
```bash
# Check data directory
ls -la data/recbole_format/ml-25m/

# Download data manually if needed
python -c "
import gdown
gdown.download('https://drive.google.com/uc?id=1tGY6F_2nEeSWwAXJ_4F832p0BzEbAGfv', 'data/recbole_format/ml-25m/ml-25m.inter')
"
```

## 💰 Cost Management

### Estimated Costs (T4 GPU)
- **Training Time**: ~2-4 hours for debug run
- **Instance Cost**: ~$0.35/hour
- **Storage**: ~$10/month for 100GB
- **Total Debug Run**: ~$1.40-$2.80

### Cost Optimization
```bash
# Stop instance when not training
gcloud compute instances stop ss4rec-training --zone=us-central1-a

# Start when needed
gcloud compute instances start ss4rec-training --zone=us-central1-a

# Delete when completely done
gcloud compute instances delete ss4rec-training --zone=us-central1-a
```

### Automatic Shutdown
```bash
# Set auto-shutdown after training (add to training script)
sudo shutdown -h +180  # Shutdown in 3 hours
```

## 📊 Monitoring Training

### Key Metrics to Watch
1. **Loss**: Should decrease steadily
2. **HR@10**: Should increase (target >0.30)
3. **NDCG@10**: Should increase (target >0.25)
4. **Gradient Norm**: Should stay 0.1-10.0 range
5. **GPU Memory**: Should be stable <80%

### Success Indicators
- ✅ No NaN/Inf values in 10+ epochs
- ✅ Loss decreasing over time
- ✅ Metrics improving
- ✅ Stable gradient norms
- ✅ No memory leaks

### Failure Indicators
- ❌ Loss becomes NaN
- ❌ Gradient explosion (norm >10)
- ❌ Memory keeps growing
- ❌ Training crashes
- ❌ No metric improvement after 20+ epochs

## 🎯 Next Steps After Debug Success

1. **Full Training**: Switch to production config with larger batch sizes
2. **Hyperparameter Tuning**: Experiment with learning rates, model sizes
3. **Comparison**: Train NCF baseline for fair comparison
4. **Production**: Deploy best model using FastAPI inference server

Ready to start? Let's go! 🚀