#!/bin/bash
# 🚀 Complete Runpod A100 Deployment Script
# Automates the entire deployment process with W&B integration

set -e  # Exit on any error

echo "🚀 MovieLens RecSys A100 Deployment Script"
echo "==========================================="
echo ""

# Configuration
REPO_URL="https://github.com/NolanRobbins/MovieLens-RecSys.git"
WORKSPACE_DIR="/workspace/MovieLens-RecSys"
DATA_DIR="/workspace/data"
MODELS_DIR="/workspace/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: System Information
log_info "Step 1: Checking system information..."
echo "🖥️  System Info:"
echo "   OS: $(uname -a)"
echo "   CPU: $(nproc) cores"
echo "   RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "   Python: $(python3 --version)"

if command -v nvidia-smi &> /dev/null; then
    echo "   GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu; do
        echo "     $gpu MB"
    done
    log_success "GPU detected"
else
    log_error "NVIDIA GPU not detected!"
    exit 1
fi
echo ""

# Step 2: Clone Repository
log_info "Step 2: Setting up repository..."
if [ -d "$WORKSPACE_DIR" ]; then
    log_warning "Directory exists, updating..."
    cd "$WORKSPACE_DIR"
    git pull origin main
else
    log_info "Cloning repository..."
    git clone "$REPO_URL" "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
fi
log_success "Repository ready"
echo ""

# Step 3: Setup Python Environment
log_info "Step 3: Setting up Python environment..."
python3 -m pip install --upgrade pip
pip install -r requirements_runpod.txt
log_success "Python packages installed"
echo ""

# Step 4: Create Directory Structure
log_info "Step 4: Creating directory structure..."
mkdir -p "$DATA_DIR" "$MODELS_DIR"
mkdir -p "$DATA_DIR/processed" "$DATA_DIR/raw"
mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/artifacts"
log_success "Directories created"
echo ""

# Step 5: Data Download from Google Drive
log_info "Step 5: Downloading data files from Google Drive..."
pip install gdown --quiet

mkdir -p "$DATA_DIR/processed"

if [ ! -f "$DATA_DIR/processed/train_data.csv" ]; then
    log_info "Downloading training data files from Google Drive..."
    
    # Download train_data.csv (291MB)
    log_info "Downloading train_data.csv..."
    gdown "https://drive.google.com/uc?id=1cNyAUmJ81erOW_YSAWT1_xNhsj7JG361" -O "$DATA_DIR/processed/train_data.csv"
    
    # Download val_data.csv (73MB)  
    log_info "Downloading val_data.csv..."
    gdown "https://drive.google.com/uc?id=1Yar-2nQRHuW7srknw48I8cT8AoQotSAb" -O "$DATA_DIR/processed/val_data.csv"
    
    # Download data_mappings.pkl (3.4MB)
    log_info "Downloading data_mappings.pkl..."
    gdown "https://drive.google.com/uc?id=1ngS42tO5U37p09Mw_L6TEZ7w9mR0FsKB" -O "$DATA_DIR/processed/data_mappings.pkl"
    
    log_success "All data files downloaded from Google Drive"
else
    log_info "Data files already exist, skipping download"
fi

# Run data preparation if available
if [ -f "runpod_data_prep.py" ]; then
    log_info "Running data validation..."
    python3 runpod_data_prep.py
    if [ $? -eq 0 ]; then
        log_success "Data validation passed"
    else
        log_error "Data validation failed"
        exit 1
    fi
else
    log_warning "Data prep script not found, skipping validation"
fi
echo ""

# Step 6: W&B Setup
log_info "Step 6: Setting up Weights & Biases..."
if [ -z "$WANDB_API_KEY" ]; then
    log_warning "WANDB_API_KEY not set as environment variable"
    echo "Please provide your W&B API key for experiment tracking:"
    echo "Find it at: https://wandb.ai/settings"
    echo ""
    read -p "Enter W&B API Key: " wandb_key
    export WANDB_API_KEY="$wandb_key"
    echo "export WANDB_API_KEY=\"$wandb_key\"" >> ~/.bashrc
fi

# Test W&B login
python3 -c "import wandb; wandb.login()" 2>/dev/null
if [ $? -eq 0 ]; then
    log_success "W&B authentication successful"
else
    log_error "W&B authentication failed"
    echo "Please check your API key and try again"
    exit 1
fi
echo ""

# Step 7: GPU Optimization
log_info "Step 7: Configuring GPU optimizations..."

# Enable TF32 (A100 optimization)
export NVIDIA_TF32_OVERRIDE=1
echo "export NVIDIA_TF32_OVERRIDE=1" >> ~/.bashrc

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> ~/.bashrc

# Optimize CUDA cache
export CUDA_CACHE_PATH=/workspace/.nv/ComputeCache
mkdir -p $CUDA_CACHE_PATH
echo "export CUDA_CACHE_PATH=$CUDA_CACHE_PATH" >> ~/.bashrc

log_success "GPU optimizations applied"
echo ""

# Step 8: Pre-training Checks
log_info "Step 8: Running pre-training checks..."

# Check GPU memory
gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$gpu_memory" -lt 30000 ]; then
    log_warning "Low GPU memory: ${gpu_memory}MB available"
    echo "Consider reducing batch size if training fails"
fi

# Check CUDA and PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}')
"

log_success "Pre-training checks completed"
echo ""

# Step 9: Training Setup
log_info "Step 9: Setting up training environment..."

# Create training script selector
cat > run_training.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting A100 Training with W&B Monitoring"
echo "=============================================="

# Choose training script
echo "Available training scripts:"
echo "1. runpod_training.py (Basic A100 training)"
echo "2. runpod_training_wandb.py (Full W&B integration)"
echo ""

read -p "Select training script [1-2]: " script_choice

case $script_choice in
    1)
        training_script="runpod_training.py"
        ;;
    2)
        training_script="runpod_training_wandb.py"
        ;;
    *)
        echo "Invalid choice, using full W&B version"
        training_script="runpod_training_wandb.py"
        ;;
esac

echo "🏋️ Starting training with $training_script..."
echo "📊 Monitor progress at: https://wandb.ai/$(whoami)/movielens-hybrid-vae-a100"
echo ""

# Start training with logging
nohup python3 $training_script > training.log 2>&1 &
training_pid=$!

echo "Training started with PID: $training_pid"
echo "Log file: training.log"
echo ""
echo "Monitoring commands:"
echo "  tail -f training.log          # View training progress"
echo "  nvidia-smi -l 1              # Monitor GPU usage"
echo "  kill $training_pid           # Stop training if needed"
echo ""

# Monitor training start
sleep 5
if ps -p $training_pid > /dev/null; then
    echo "✅ Training is running successfully!"
    echo "📈 W&B dashboard: https://wandb.ai"
    echo "📊 GPU monitoring: nvidia-smi -l 1"
else
    echo "❌ Training failed to start. Check training.log for details."
fi
EOF

chmod +x run_training.sh

log_success "Training environment ready"
echo ""

# Step 10: Final Instructions
log_info "Step 10: Deployment completed! 🎉"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. 🚀 Start Training:"
echo "   ./run_training.sh"
echo ""
echo "2. 📊 Monitor Progress:"
echo "   - W&B Dashboard: https://wandb.ai/$(whoami)/movielens-hybrid-vae-a100"
echo "   - Training logs: tail -f training.log"
echo "   - GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "3. 🎯 Expected Results:"
echo "   - Target RMSE: 0.52-0.55 (improvement from 0.5996)"
echo "   - Training time: 2-3 hours on A100"
echo "   - Memory usage: ~15-20GB GPU"
echo "   - Cost: ~$2-4 for full training"
echo ""
echo "4. 📁 Output Files:"
echo "   - Best model: $MODELS_DIR/hybrid_vae_a100_best.pt"
echo "   - Training summary: $MODELS_DIR/training_summary.json"
echo "   - Training logs: $(pwd)/training.log"
echo ""
echo "5. 🔍 Troubleshooting:"
echo "   - OOM errors: Reduce batch_size in config"
echo "   - NaN values: Training script has stability fixes"
echo "   - Slow training: Check TF32 is enabled"
echo ""

# Create quick monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "🔍 A100 Training Monitor"
echo "======================="

while true; do
    clear
    echo "🔍 A100 Training Monitor - $(date)"
    echo "================================="
    echo ""
    
    if pgrep -f "python.*training" > /dev/null; then
        echo "✅ Training Status: RUNNING"
    else
        echo "❌ Training Status: NOT RUNNING"
    fi
    
    echo ""
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
    
    echo ""
    echo "📈 Latest Training Progress:"
    tail -5 training.log 2>/dev/null | grep -E "(Epoch|RMSE|Loss)" || echo "   No training logs yet"
    
    echo ""
    echo "🔗 W&B Dashboard: https://wandb.ai/$(whoami)/movielens-hybrid-vae-a100"
    echo ""
    echo "Press Ctrl+C to exit monitor"
    
    sleep 10
done
EOF

chmod +x monitor.sh

log_success "Deployment completed successfully!"
echo ""
echo "🎉 Your A100 training environment is ready!"
echo "   Run: ./run_training.sh to start training"
echo "   Run: ./monitor.sh for real-time monitoring"
echo ""
echo "💡 Pro tip: Keep this terminal open for monitoring"