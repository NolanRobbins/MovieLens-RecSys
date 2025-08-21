#!/bin/bash

# RunPod Neural CF Training Entrypoint Script
# Automated training setup for A6000 GPU instances

set -e  # Exit on any error

echo "🚀 Neural CF Training on RunPod"
echo "================================"
echo "🖥️  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "💾 Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "📅 Started: $(date)"
echo "================================"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to handle errors
error_exit() {
    log "❌ ERROR: $1"
    exit 1
}

# Set up environment variables
export PYTHONPATH="/workspace:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parse command line arguments
MODEL_TYPE="ncf"
CONFIG_FILE=""
WANDB_ENABLED=true
AUTO_SETUP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_ENABLED=false
            shift
            ;;
        --no-setup)
            AUTO_SETUP=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL     Model type (ncf, ss4rec) [default: ncf]"
            echo "  --config FILE     Custom config file"
            echo "  --no-wandb        Disable W&B logging"
            echo "  --no-setup        Skip automatic setup"
            echo "  --help            Show this help"
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

log "🎯 Model: $MODEL_TYPE"
log "📊 W&B Enabled: $WANDB_ENABLED"
log "⚠️  Note: Test data excluded from training (as it should be!)"

# Ensure we're in the project directory
if [[ ! -f "runpod_entrypoint.sh" ]]; then
    error_exit "Not in project directory! Please run: cd MovieLens-RecSys && ./runpod_entrypoint.sh"
fi

log "📁 Running from: $(pwd)"

# Automatic setup if enabled
if [ "$AUTO_SETUP" = true ]; then
    log "🔧 Running automatic setup..."
    
    # Check if setup has been run before
    if [ ! -f ".runpod_setup_complete" ]; then
        log "📦 First time setup - installing dependencies..."
        
        # Update system packages
        apt-get update -qq || log "⚠️  apt-get update failed (continuing...)"
        
        # Update package lists and install system dependencies
        log "📦 Updating system packages..."
        apt-get update -qq || log "⚠️  apt-get update failed (continuing...)"

        
        # Install uv if not present
        if ! command -v uv &> /dev/null; then
            log "📦 Installing uv package manager..."
            pip install uv || error_exit "Failed to install uv"
        fi
        
        # Create virtual environment with uv
        if [ ! -d ".venv" ]; then
            log "🐍 Creating virtual environment with uv..."
            uv venv || error_exit "Failed to create virtual environment"
        else
            log "🐍 Virtual environment already exists"
        fi
        
        # Mark setup as complete
        touch .runpod_setup_complete
        log "✅ Setup completed and marked"
    else
        log "✅ Setup already completed (found .runpod_setup_complete)"
    fi
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    log "🐍 Activating virtual environment..."
    source .venv/bin/activate || error_exit "Failed to activate virtual environment"
    
    # Verify activation worked
    which python
    python --version
else
    error_exit "Virtual environment not found at .venv"
fi

# Install requirements using uv pip
log "📦 Installing requirements with uv pip..."
log "🔍 Current working directory: $(pwd)"
log "🔍 Available requirements files:"
ls -la requirements*.txt 2>/dev/null || log "   No requirements*.txt files found"

if [ "$MODEL_TYPE" = "ss4rec" ] && [ -f "requirements_ss4rec.txt" ]; then
    log "🧠 Installing SS4Rec requirements with uv..."
    
    # Install core dependencies first
    log "🔧 Installing core ML dependencies..."
    uv pip install torch>=2.0.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 || error_exit "Failed to install core dependencies"
    
    # Install state space dependencies individually
    log "📦 Installing causal-conv1d..."
    uv pip install "causal-conv1d>=1.2.0" || error_exit "Failed to install causal-conv1d"
    
    log "📦 Installing mamba-ssm (with torch available)..."
    uv pip install mamba-ssm==2.2.2 || error_exit "Failed to install mamba-ssm"
    
    log "📦 Installing remaining SS4Rec dependencies..."
    uv pip install s5-pytorch==0.2.1 recbole==1.2.0 || error_exit "Failed to install s5-pytorch and recbole"
    
    # Install monitoring and dev tools
    log "📦 Installing monitoring and development tools..."
    uv pip install "wandb>=0.15.0" "tensorboard>=2.10.0" "tqdm>=4.64.0" "pyyaml>=6.0" || error_exit "Failed to install monitoring tools"
    uv pip install "matplotlib>=3.5.0" "seaborn>=0.11.0" "streamlit>=1.25.0" "plotly>=5.10.0" || error_exit "Failed to install visualization tools"
    uv pip install "fastapi>=0.95.0" "uvicorn>=0.20.0" || error_exit "Failed to install API tools"
    uv pip install "pytest>=7.0.0" "black>=22.0.0" "isort>=5.10.0" "mypy>=0.900" "ruff>=0.0.200" || error_exit "Failed to install dev tools"
elif [ -f "requirements.txt" ]; then
    log "📦 Installing base requirements for $MODEL_TYPE with uv..."
    uv pip install -r requirements.txt || error_exit "Failed to install base requirements"
else
    log "❌ Available files in current directory:"
    ls -la
    error_exit "No requirements file found (requirements.txt or requirements_ss4rec.txt)"
fi

# Verify installation worked
log "✅ Verifying Python environment..."
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

# Additional verification for SS4Rec
if [ "$MODEL_TYPE" = "ss4rec" ]; then
    log "🧠 Verifying SS4Rec dependencies..."
    python -c "import mamba_ssm; print(f'Mamba-SSM version: {mamba_ssm.__version__}')" || log "⚠️  Mamba-SSM import failed - may need additional setup"
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
fi

# W&B setup
if [ "$WANDB_ENABLED" = true ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo ""
        echo "🔑 Weights & Biases Setup Required"
        echo "=================================="
        echo "Please enter your W&B API key (get it from: https://wandb.ai/authorize)"
        echo "Or press Enter to skip W&B logging:"
        read -p "W&B API Key: " user_wandb_key
        
        if [ -z "$user_wandb_key" ]; then
            log "⚠️  No W&B API key provided, disabling W&B logging"
            WANDB_ENABLED=false
        else
            export WANDB_API_KEY="$user_wandb_key"
            log "📊 W&B API key set, enabling logging"
            wandb login || log "⚠️  W&B login failed, continuing without logging"
        fi
    else
        log "📊 W&B API key found in environment, enabling logging"
        wandb login || log "⚠️  W&B login failed, continuing without logging"
    fi
fi

# Determine config file
if [ -z "$CONFIG_FILE" ]; then
    if [ "$MODEL_TYPE" = "ncf" ]; then
        CONFIG_FILE="configs/ncf_runpod.yaml"
    else
        CONFIG_FILE="configs/${MODEL_TYPE}.yaml"
    fi
fi

log "⚙️  Using config: $CONFIG_FILE"

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log "⚠️  Config file not found: $CONFIG_FILE"
    log "📁 Available configs:"
    ls -la configs/ || log "configs/ directory not found"
    
    # Use fallback config
    FALLBACK_CONFIG="configs/ncf_baseline.yaml"
    if [ -f "$FALLBACK_CONFIG" ]; then
        log "🔄 Using fallback config: $FALLBACK_CONFIG"
        CONFIG_FILE="$FALLBACK_CONFIG"
    else
        error_exit "No valid config file found"
    fi
fi

# Google Drive download
log "📥 Downloading training data from Google Drive..."

# Ensure gdown is installed
log "📦 Installing gdown for Google Drive downloads..."
uv pip install -q gdown || error_exit "Failed to install gdown"

mkdir -p data/processed

# Download files from Google Drive using file IDs
gdown --id 1a3KsSZWcPpSF5Qu_cb2rh7KtR57ypfjS -O data/processed/train_data.csv || error_exit "Failed to download train_data.csv"
gdown --id 1GUXQGSVdm_pc_iqh05lvu3nVKRKij_jU -O data/processed/val_data.csv || error_exit "Failed to download val_data.csv"
gdown --id 1hm8PM5DdPhlmAl6r8TsSekr-n5MGXmQh -O data/processed/data_mappings.pkl || error_exit "Failed to download data_mappings.pkl"


# Verify required training files exist
log "✅ Verifying training data files..."

# Check for required files
REQUIRED_FILES=("data/processed/train_data.csv" "data/processed/val_data.csv" "data/processed/data_mappings.pkl")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error_exit "Required file missing after Google Drive download: $file"
    fi
done

# Check if data has correct temporal columns
TRAIN_HEADER=$(head -1 data/processed/train_data.csv)
EXPECTED_COLS=("user_idx" "movie_idx" "rating" "timestamp" "rating_date" "rating_year" "rating_month" "rating_weekday")

log "✅ Validating data format with temporal features..."
log "   📄 Found header: $TRAIN_HEADER"

# Verify all required columns are present
MISSING_COLS=()
for col in "${EXPECTED_COLS[@]}"; do
    if [[ "$TRAIN_HEADER" != *"$col"* ]]; then
        MISSING_COLS+=("$col")
    fi
done

if [ ${#MISSING_COLS[@]} -eq 0 ]; then
    log "✅ Training data has correct temporal column format"
    log "   📄 $(wc -l < data/processed/train_data.csv) training samples"
    log "   📄 $(wc -l < data/processed/val_data.csv) validation samples"
    log "   📄 Data mappings: $(ls -lh data/processed/data_mappings.pkl | awk '{print $5}')"
else
    log "❌ Training data missing required columns: ${MISSING_COLS[*]}"
    log "   Expected: user_idx, movie_idx, rating, timestamp, rating_date, rating_year, rating_month, rating_weekday"
    log "   Found: $TRAIN_HEADER"
    error_exit "Training data format is incompatible"
fi

# GPU memory optimization
log "🚀 Setting up GPU optimizations..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0  # For performance

# Start training
log "🎬 Starting $MODEL_TYPE training..."
log "================================"

# Build training command - use auto_train for Discord notifications
TRAIN_CMD="python auto_train_ss4rec.py --model $MODEL_TYPE --config $CONFIG_FILE"

if [ "$WANDB_ENABLED" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --no-wandb"
fi

log "🚀 Executing: $TRAIN_CMD"
log "📱 Discord notifications enabled (if DISCORD_WEBHOOK_URL is set)"
if [ -z "$DISCORD_WEBHOOK_URL" ]; then
    log "⚠️  To enable Discord notifications, set: export DISCORD_WEBHOOK_URL='your_webhook_url'"
else
    log "✅ Discord webhook configured - notifications will be sent"
fi
log "================================"

# Execute training with error handling
if eval "$TRAIN_CMD"; then
    log "================================"
    log "🎉 Training completed successfully!"
    log "📅 Completed: $(date)"
    
    # Post-training file organization and download guidance
    log "================================"
    log "📁 POST-TRAINING: File Download Guide"
    log "================================"
    
    # Create download summary
    DOWNLOAD_SUMMARY="training_download_guide.txt"
    cat > "$DOWNLOAD_SUMMARY" << EOF
🎯 ESSENTIAL FILES TO DOWNLOAD FROM RUNPOD
==========================================

📍 Access via Jupyter Lab (Port 8888) → File Browser → Right-click → Download

PRIORITY 1 - ESSENTIAL MODEL FILES:
$(find results -name "best_model.pt" 2>/dev/null | head -5)

PRIORITY 2 - TRAINING LOGS:
$(find logs -name "training.log" 2>/dev/null | head -5)
$(find logs -name "*.log" 2>/dev/null | head -5)

PRIORITY 3 - CONFIGURATION & METADATA:
$(find configs -name "*.yaml" 2>/dev/null | head -5)
$(find data/processed -name "metadata.json" 2>/dev/null)
$(find data/processed -name "data_mappings.pkl" 2>/dev/null)

PRIORITY 4 - BACKUP CHECKPOINTS (Optional - Large files):
$(find results -name "checkpoint_epoch_*.pt" 2>/dev/null | tail -3)

📊 Training Results Summary:
Model Type: $MODEL_TYPE
Completed: $(date)
Results Directory: $(ls -la results/ 2>/dev/null | wc -l) files
W&B Dashboard: Check your W&B project for detailed metrics

🔗 Next Steps:
1. Download Priority 1-3 files via Jupyter Lab
2. Terminate RunPod to save costs
3. Push training metadata to GitHub (not model files)
4. Plan next training phase (SS4Rec target: RMSE < 0.70)
EOF
    
    log "📄 Download guide created: $DOWNLOAD_SUMMARY"
    cat "$DOWNLOAD_SUMMARY"
    
    # Show detailed results summary
    if [ -d "results" ]; then
        log "📊 Detailed results summary:"
        find results -type f | while read file; do
            SIZE=$(ls -lh "$file" | awk '{print $5}')
            log "   📄 $file ($SIZE)"
        done
    fi
    
    # Show logs summary
    if [ -d "logs" ]; then
        log "📝 Training logs summary:"
        find logs -name "*.log" | while read file; do
            SIZE=$(ls -lh "$file" | awk '{print $5}')
            log "   📄 $file ($SIZE)"
        done
    fi
    
    # Git setup for easy metadata pushing
    log "================================"
    log "🔧 Git Configuration for Metadata Push"
    log "================================"
    
    # Setup git config if not already set
    if ! git config user.name &>/dev/null; then
        log "⚠️  Git user not configured. To push metadata later, run:"
        log "   git config user.name 'Your Name'"
        log "   git config user.email 'your.email@example.com'"
    fi
    
    # Add ignore for large model files
    if ! grep -q "*.pt" .gitignore 2>/dev/null; then
        echo "*.pt" >> .gitignore
        echo "*.tar.gz" >> .gitignore
        log "✅ Added model files to .gitignore"
    fi
    
    # Create easy commit script
    cat > "commit_training_metadata.sh" << 'EOF'
#!/bin/bash
# Easy script to commit training metadata (not large model files)
git add logs/ configs/ data/processed/metadata.json data/processed/data_mappings.pkl .gitignore training_download_guide.txt
git add .runpod_setup_complete
git status
echo ""
echo "🚀 Ready to commit training metadata. Run:"
echo "git commit -m 'Training session complete - $(date)'"
echo "git push origin main"
EOF
    chmod +x commit_training_metadata.sh
    log "📄 Commit script created: ./commit_training_metadata.sh"
    
    log "================================"
    log "🎯 IMMEDIATE NEXT STEPS:"
    log "1. 📥 Download files listed in $DOWNLOAD_SUMMARY via Jupyter Lab"
    log "2. 🔧 Configure git identity if needed"
    log "3. 📤 Run ./commit_training_metadata.sh to push metadata"
    log "4. 🛑 Terminate RunPod to save costs"
    log "5. 📊 Analyze results and plan next training phase"
    log "================================"
    
else
    error_exit "Training failed"
fi

log "🏁 RunPod training session complete"