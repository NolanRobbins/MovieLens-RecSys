#!/bin/bash

# RunPod MovieLens RecSys Training Entrypoint Script
# Automated training setup for A6000 GPU instances
# Supports: NCF Baseline, Official SS4Rec, Custom SS4Rec (deprecated)

set -e  # Exit on any error

echo "🚀 MovieLens RecSys Training on RunPod"
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

DEBUG_LOGGING=false

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
        --debug)
            DEBUG_LOGGING=true
            shift
            ;;
        --production)
            DEBUG_LOGGING=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL     Model type (ncf, ss4rec, ss4rec-official) [default: ncf]"
            echo "  --config FILE     Custom config file"
            echo "  --no-wandb        Disable W&B logging"
            echo "  --no-setup        Skip automatic setup"
            echo "  --debug           Enable comprehensive debug logging for NaN detection"
            echo "  --production      Disable debug logging for optimal performance"
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
log "🔍 Debug Logging: $DEBUG_LOGGING"
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
        
        # Install gdown for Google Drive downloads
        if ! command -v gdown &> /dev/null; then
            log "📦 Installing gdown for Google Drive downloads..."
            pip install gdown || log "⚠️ Failed to install gdown, will fallback to wget/curl"
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

# Download RecBole format data (train+val combined, test excluded for future ETL)
download_recbole_data() {
    local data_dir="data/recbole_format"
    local inter_file="$data_dir/movielens.inter"
    local config_file="$data_dir/movielens_recbole_config.yaml"
    local stats_file="$data_dir/movielens_stats.json"
    
    log "📊 Checking RecBole format data..."
    
    # Create directory if it doesn't exist
    mkdir -p "$data_dir"
    
    # Check if data already exists and is valid
    if [ -f "$inter_file" ] && [ -f "$config_file" ] && [ -s "$inter_file" ]; then
        local file_size=$(stat -f%z "$inter_file" 2>/dev/null || stat -c%s "$inter_file" 2>/dev/null || echo "0")
        if [ "$file_size" -gt 10000000 ]; then  # > 10MB indicates valid data
            log "✅ RecBole format data already exists (${file_size} bytes)"
            return 0
        fi
    fi
    
    log "📥 Downloading RecBole format data from Google Drive..."
    log "🎯 Data: Train + Val combined (20M+ interactions, ~400MB)"
    log "⚠️  Test data excluded for future ETL pipeline evaluation"
    
    # TODO: Replace with your actual Google Drive download links
    # You'll need to replace these URLs with the actual Google Drive download links
    local GDRIVE_INTER_URL="https://drive.google.com/uc?export=download&id=1HsS87bMnAkomOzBa3IdxbWeintPnao0C"
    local GDRIVE_CONFIG_URL="https://drive.google.com/uc?export=download&id=1Uxe42ENupS6cM5GYDxgARdx-ucPJzPof"
    local GDRIVE_STATS_URL="https://drive.google.com/uc?export=download&id=1dVKkIuDZrMFBahKLLKmhvoG1gs4-ayVG"
    
    # Download interaction file (.inter) - Handle Google Drive large file download
    log "📥 Downloading movielens.inter (~400MB)..."
    
    # Extract file ID from URL
    local file_id=$(echo "$GDRIVE_INTER_URL" | sed 's/.*id=\([^&]*\).*/\1/')
    log "📋 File ID: $file_id"
    
    # Use gdown if available (best for large Google Drive files)
    if command -v gdown >/dev/null 2>&1; then
        log "📥 Using gdown for large file download..."
        gdown --id "$file_id" -O "$inter_file" || error_exit "Failed to download .inter file with gdown"
    elif command -v wget >/dev/null 2>&1; then
        log "📥 Using wget with Google Drive workaround..."
        # First try direct download
        wget --no-check-certificate -O "$inter_file" "$GDRIVE_INTER_URL" 2>/dev/null || {
            # If that fails, try the large file workaround
            log "🔄 Trying large file download method..."
            local confirm_url="https://drive.google.com/uc?export=download&confirm=t&id=$file_id"
            wget --no-check-certificate --load-cookies /tmp/cookies.txt -O "$inter_file" "$confirm_url" || error_exit "Failed to download .inter file"
        }
    elif command -v curl >/dev/null 2>&1; then
        log "📥 Using curl with Google Drive workaround..."
        # Try direct download first
        curl -L -o "$inter_file" "$GDRIVE_INTER_URL" 2>/dev/null || {
            # If that fails, try the large file workaround
            log "🔄 Trying large file download method..."
            local confirm_url="https://drive.google.com/uc?export=download&confirm=t&id=$file_id"
            curl -L -o "$inter_file" "$confirm_url" || error_exit "Failed to download .inter file"
        }
    else
        error_exit "No download tool available (wget, curl, or gdown)"
    fi
    
    # Download config file
    log "📥 Downloading RecBole config..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$config_file" "$GDRIVE_CONFIG_URL" || log "⚠️ Config download failed, will use default"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$config_file" "$GDRIVE_CONFIG_URL" || log "⚠️ Config download failed, will use default"
    fi
    
    # Download stats file
    log "📥 Downloading dataset statistics..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$stats_file" "$GDRIVE_STATS_URL" || log "⚠️ Stats download failed, continuing"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$stats_file" "$GDRIVE_STATS_URL" || log "⚠️ Stats download failed, continuing"
    fi
    
    # Verify downloaded data
    if [ -f "$inter_file" ] && [ -s "$inter_file" ]; then
        local downloaded_size=$(stat -f%z "$inter_file" 2>/dev/null || stat -c%s "$inter_file" 2>/dev/null || echo "0")
        log "✅ RecBole data downloaded (${downloaded_size} bytes)"
        
        # Check if file is too small (indicates HTML error page instead of data)
        if [ "$downloaded_size" -lt 1000000 ]; then  # Less than 1MB indicates error
            log "⚠️ Downloaded file seems too small, checking content..."
            
            # Check if it's an HTML file (Google Drive error page)
            if head -5 "$inter_file" | grep -q -i "html\|<!DOCTYPE\|<title"; then
                log "❌ Downloaded HTML error page instead of data file"
                log "🔍 This usually means Google Drive requires virus scan confirmation"
                log "💡 Solutions:"
                log "   1. Try a different cloud provider (Dropbox, OneDrive)"
                log "   2. Split the file into smaller chunks (<100MB each)"
                log "   3. Use a different download method"
                
                # Show first few lines for debugging
                log "📄 File content (first 5 lines):"
                head -5 "$inter_file" | sed 's/^/   /'
                
                error_exit "❌ Downloaded data is corrupted (HTML page instead of data)"
            fi
        fi
        
        # Quick validation for proper format
        local line_count=$(head -10 "$inter_file" | wc -l)
        if [ "$line_count" -gt 5 ]; then
            # Check if first line contains expected headers
            local first_line=$(head -1 "$inter_file")
            if echo "$first_line" | grep -q "user_id.*item_id.*timestamp"; then
                log "✅ Data format validation passed"
            else
                log "⚠️ Header format may be incorrect: $first_line"
                log "✅ Continuing anyway (may still work)"
            fi
        else
            log "❌ Downloaded data appears corrupted (too few lines)"
            error_exit "❌ Downloaded data validation failed"
        fi
    else
        error_exit "❌ Failed to download RecBole format data"
    fi
}

# Download data for SS4Rec models
if [[ "$MODEL_TYPE" == "ss4rec"* ]]; then
    download_recbole_data
fi

# Install requirements using uv pip
log "📦 Installing requirements with uv pip..."
log "🔍 Current working directory: $(pwd)"
log "🔍 Available requirements files:"
ls -la requirements*.txt 2>/dev/null || log "   No requirements*.txt files found"

if [[ "$MODEL_TYPE" == "ss4rec"* ]] && [ -f "requirements_ss4rec.txt" ]; then
    log "🧠 Installing SS4Rec requirements with uv..."
    
    # Install core dependencies first
    log "🔧 Installing core ML dependencies..."
    uv pip install torch>=2.0.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 || error_exit "Failed to install core dependencies"
    
    # Install state space dependencies individually
    log "📦 Installing causal-conv1d..."
    uv pip install "causal-conv1d>=1.2.0" || error_exit "Failed to install causal-conv1d"
    
    log "📦 Installing mamba-ssm (with torch available)..."
    # Install additional build dependencies for mamba-ssm
    uv pip install ninja packaging wheel setuptools
    
    # Install git for cloning repository
    apt-get update -qq && apt-get install -y git
    
    # Try installing from GitHub repository with submodules (using uv pip for virtual env)
    log "🔧 Installing mamba-ssm from GitHub repository..."
    uv pip install git+https://github.com/state-spaces/mamba.git@v2.2.2 --no-build-isolation || \
    log "⚠️  GitHub installation failed, trying PyPI with fallback..." && \
    uv pip install mamba-ssm==2.2.2 --no-build-isolation || \
    uv pip install mamba-ssm==2.0.2 --no-build-isolation || \
    error_exit "Failed to install any version of mamba-ssm"
    
    log "📦 Installing remaining SS4Rec dependencies..."
    uv pip install s5-pytorch==0.2.1 recbole==1.2.0 || error_exit "Failed to install s5-pytorch and recbole"
    
    # Install monitoring and dev tools
    log "📦 Installing monitoring and development tools..."
    uv pip install "wandb>=0.15.0" "tensorboard>=2.10.0" "tqdm>=4.64.0" "pyyaml>=6.0" "diskcache>=5.4.0" || error_exit "Failed to install monitoring tools"
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
if [[ "$MODEL_TYPE" == "ss4rec"* ]]; then
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

# Discord webhook setup
if [ -z "$DISCORD_WEBHOOK_URL" ]; then
    echo ""
    echo "📱 Discord Notifications Setup"
    echo "================================"
    echo "Please enter your Discord webhook URL for training notifications"
    echo "(Create one in your Discord server: Server Settings → Integrations → Webhooks)"
    echo "Or press Enter to skip Discord notifications:"
    read -p "Discord Webhook URL: " user_discord_webhook
    
    if [ -z "$user_discord_webhook" ]; then
        log "⚠️  No Discord webhook provided, notifications disabled"
    else
        export DISCORD_WEBHOOK_URL="$user_discord_webhook"
        log "📱 Discord webhook set, notifications enabled"
    fi
else
    log "📱 Discord webhook found in environment, notifications enabled"
fi

# Determine config file
if [ -z "$CONFIG_FILE" ]; then
    if [ "$MODEL_TYPE" = "ncf" ]; then
        CONFIG_FILE="configs/ncf_runpod.yaml"
    elif [ "$MODEL_TYPE" = "ss4rec" ]; then
        CONFIG_FILE="configs/ss4rec_a6000_optimized.yaml"
    elif [ "$MODEL_TYPE" = "ss4rec-official" ]; then
        CONFIG_FILE="configs/official/ss4rec_official.yaml"
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

# Smart data download - skip if files already exist and are valid
log "📥 Checking training data availability..."

mkdir -p data/processed

# Check if files already exist and are valid
NEED_DOWNLOAD=false
REQUIRED_FILES=("data/processed/train_data.csv" "data/processed/val_data.csv" "data/processed/data_mappings.pkl")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log "❌ Missing: $file"
        NEED_DOWNLOAD=true
    else
        # Check if file is not empty
        if [ ! -s "$file" ]; then
            log "❌ Empty file: $file"
            NEED_DOWNLOAD=true
        else
            SIZE=$(ls -lh "$file" | awk '{print $5}')
            log "✅ Found: $file ($SIZE)"
        fi
    fi
done

if [ "$NEED_DOWNLOAD" = true ]; then
    log "📥 Downloading missing/invalid training data from Google Drive..."
    
    # Ensure gdown is installed
    log "📦 Installing gdown for Google Drive downloads..."
    uv pip install -q gdown || error_exit "Failed to install gdown"
    
    # Download files from Google Drive using file IDs
    gdown --id 1a3KsSZWcPpSF5Qu_cb2rh7KtR57ypfjS -O data/processed/train_data.csv || error_exit "Failed to download train_data.csv"
    gdown --id 1GUXQGSVdm_pc_iqh05lvu3nVKRKij_jU -O data/processed/val_data.csv || error_exit "Failed to download val_data.csv"
    gdown --id 1hm8PM5DdPhlmAl6r8TsSekr-n5MGXmQh -O data/processed/data_mappings.pkl || error_exit "Failed to download data_mappings.pkl"
    
    log "✅ Download completed"
else
    log "✅ All training data files already present - skipping download"
    log "💡 This saves significant time on subsequent runs!"
fi


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

# Training mode configuration
if [ "$DEBUG_LOGGING" = true ]; then
    log "🔍 ENABLING DEBUG MODE - Comprehensive NaN detection and logging"
    log "⚠️  WARNING: Debug mode significantly reduces training performance"
    log "⚠️  Use --production flag for optimal performance once debugging is complete"
    
    # Create comprehensive logging directory structure
    mkdir -p logs/debug
    
    # Configure Python logging environment variables
    export PYTHONPATH="/workspace:$PYTHONPATH"
    export SS4REC_DEBUG_LOGGING=1
    export SS4REC_LOG_LEVEL=DEBUG
    export SS4REC_LOG_FILE="logs/debug/ss4rec_training_debug.log"
    
    # Enable PyTorch anomaly detection for gradient debugging
    export TORCH_DETECT_ANOMALY=1
    
    # Create Python logging configuration
    cat > debug_logging_config.py << 'EOF'
import logging
import sys
from pathlib import Path

def setup_comprehensive_debug_logging():
    """Set up comprehensive debug logging for SS4Rec NaN detection"""
    
    # Ensure log directory exists
    log_dir = Path("logs/debug")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger with comprehensive settings
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s',
        handlers=[
            # File handler for debug logs
            logging.FileHandler(log_dir / "ss4rec_training_debug.log", mode='w'),
            # Console handler for important messages only
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set console handler to show only warnings and errors
    console_handler = logging.getLogger().handlers[1]
    console_handler.setLevel(logging.WARNING)
    
    # Configure specific loggers for maximum debug detail
    loggers_to_debug = [
        "models.sota_2025.ss4rec",
        "models.sota_2025.components.state_space_models",
        "training.train_ss4rec",
        "auto_train_ss4rec"
    ]
    
    for logger_name in loggers_to_debug:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # Prevent duplicate messages
        logger.propagate = True
    
    # Log the setup completion
    debug_logger = logging.getLogger("debug_setup")
    debug_logger.info("🔧 Comprehensive debug logging enabled")
    debug_logger.info(f"📁 Debug logs will be saved to: {log_dir / 'ss4rec_training_debug.log'}")
    debug_logger.info("🚨 NaN detection: Enabled at every tensor operation")
    debug_logger.info("🔍 Einsum logging: Enabled for all state space operations")
    debug_logger.info("⚡ PyTorch anomaly detection: Enabled for gradient debugging")
    
    return True

if __name__ == "__main__":
    setup_comprehensive_debug_logging()
EOF
    
    log "✅ Debug logging configuration created"
    log "📁 Debug logs will be saved to: logs/debug/ss4rec_training_debug.log"
    log "🚨 PyTorch anomaly detection enabled - training will be slower but catch NaN issues immediately"
    log "🔍 All tensor operations in SS4Rec will be logged with full statistics"
    log "🎯 RECOMMENDATION: Run 1-2 epochs in debug mode, then restart with --production for full training"
else
    log "🚀 PRODUCTION MODE ENABLED - Optimized for performance"
    log "🔧 Debug logging disabled for maximum training speed"
    log "⚡ PyTorch anomaly detection disabled"
    log "📈 CUDA operations optimized for throughput"
    
    # Ensure debug environment variables are unset
    unset SS4REC_DEBUG_LOGGING
    unset SS4REC_LOG_LEVEL 
    unset SS4REC_LOG_FILE
    unset TORCH_DETECT_ANOMALY
fi

# GPU memory optimization
log "🚀 Setting up GPU optimizations..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

if [ "$DEBUG_LOGGING" = true ]; then
    # Enable blocking for better error traces when debugging
    export CUDA_LAUNCH_BLOCKING=1
    log "⚠️  CUDA_LAUNCH_BLOCKING=1 enabled for precise error traces (slower performance)"
    log "🐛 GPU operations will be synchronous for debugging"
else
    export CUDA_LAUNCH_BLOCKING=0
    log "⚡ CUDA_LAUNCH_BLOCKING=0 for async GPU operations (optimal performance)"
    log "🚀 GPU memory optimizations enabled for production training"
fi

# Training mode summary
log "================================"
log "🎬 Starting $MODEL_TYPE training..."
if [ "$DEBUG_LOGGING" = true ]; then
    log "🔍 MODE: DEBUG (NaN detection enabled)"
    log "⚠️  Performance: Significantly slower"
    log "🎯 Purpose: Model validation and debugging"
    log "💡 Recommendation: Run 1-2 epochs, then restart with --production"
else
    log "🚀 MODE: PRODUCTION (Optimized performance)"
    log "⚡ Performance: Maximum speed"
    log "🎯 Purpose: Full model training"
    log "📈 Expected: Optimal throughput and memory usage"
fi
log "================================"

# Build training command
if [ "$MODEL_TYPE" = "ss4rec-official" ]; then
    # Official SS4Rec using RecBole framework
    TRAIN_CMD="python training/official/runpod_train_ss4rec_official.py --config $CONFIG_FILE --install-deps --prepare-data"
elif [ "$MODEL_TYPE" = "ss4rec" ]; then
    # Custom SS4Rec (deprecated - has gradient explosion issues)
    log "⚠️ WARNING: Using deprecated custom SS4Rec implementation"
    log "⚠️ This implementation has gradient explosion issues after epoch 3"
    log "💡 Consider using --model ss4rec-official instead"
    if [ "$DEBUG_LOGGING" = true ]; then
        TRAIN_CMD="python -c 'from debug_logging_config import setup_comprehensive_debug_logging; setup_comprehensive_debug_logging()' && python training/train_ss4rec.py --config $CONFIG_FILE"
    else
        TRAIN_CMD="python training/train_ss4rec.py --config $CONFIG_FILE"
    fi
else
    # NCF and other models
    if [ "$DEBUG_LOGGING" = true ]; then
        TRAIN_CMD="python -c 'from debug_logging_config import setup_comprehensive_debug_logging; setup_comprehensive_debug_logging()' && python auto_train_ss4rec.py --model $MODEL_TYPE --config $CONFIG_FILE"
    else
        TRAIN_CMD="python auto_train_ss4rec.py --model $MODEL_TYPE --config $CONFIG_FILE"
    fi
fi

if [ "$WANDB_ENABLED" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --no-wandb"
fi

if [ "$DEBUG_LOGGING" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --debug"
fi

log "🚀 Executing: $TRAIN_CMD"
log "📱 Discord notifications enabled (if DISCORD_WEBHOOK_URL is set)"
if [ -z "$DISCORD_WEBHOOK_URL" ]; then
    log "⚠️  To enable Discord notifications, set: export DISCORD_WEBHOOK_URL='your_webhook_url'"
else
    log "✅ Discord webhook configured - notifications will be sent"
fi
log "================================"

# Create training log file with timestamp
TRAINING_LOG="training_$(date +%Y%m%d_%H%M%S).log"
log "📝 Training output will be saved to: $TRAINING_LOG"
log "🔄 Training will run in background - safe to close terminal"
log "💡 Monitor progress via: tail -f $TRAINING_LOG"
log "📊 Check W&B dashboard for real-time metrics"
log "📱 Discord will notify when training completes"
log "================================"

# Execute training in background with comprehensive logging
log "🚀 Starting background training process..."
nohup bash -c "$TRAIN_CMD" > "$TRAINING_LOG" 2>&1 &
TRAIN_PID=$!

log "✅ Training started in background with PID: $TRAIN_PID"
log "📝 Training log: $TRAINING_LOG"
log "🔍 Monitor with: tail -f $TRAINING_LOG"
log "🛑 Stop with: kill $TRAIN_PID"
log "================================"

# Save PID for easy access
echo $TRAIN_PID > training.pid
log "💾 Process ID saved to training.pid"

# Wait a moment to check if process started successfully
sleep 5
if kill -0 $TRAIN_PID 2>/dev/null; then
    log "✅ Training process running successfully (PID: $TRAIN_PID)"
    log "🔄 Safe to close terminal - training will continue"
    log "📱 Discord will notify when complete"
    
    # Show recent log output to confirm it's working
    log "================================"
    log "🔍 RECENT TRAINING OUTPUT:"
    log "================================"
    tail -10 "$TRAINING_LOG" 2>/dev/null || log "   (Log file still initializing...)"
    log "================================"
    log "✅ SETUP COMPLETE - Training running in background"
    log "💡 Use 'tail -f $TRAINING_LOG' to monitor progress"
    log "🛑 Use 'kill $TRAIN_PID' to stop training"
else
    log "❌ Training process failed to start"
    log "📝 Check log for details: cat $TRAINING_LOG"
    exit 1
fi

# Create download guide template for when training completes
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
log "📄 Download guide created: $DOWNLOAD_SUMMARY"
log "📄 Commit script created: ./commit_training_metadata.sh"

log "================================"
log "🎯 BACKGROUND TRAINING ACTIVE"
log "================================"
log "✅ Training is now running in background (PID: $TRAIN_PID)"
log "🔄 Safe to close this terminal - training will continue"
log "📝 Monitor progress: tail -f $TRAINING_LOG"
log "📊 Check W&B dashboard for real-time metrics"
log "📱 Discord will notify when training completes"
log "🛑 Stop training: kill $TRAIN_PID"
log "📁 Download guide ready: $DOWNLOAD_SUMMARY"
log "================================"
log "🏁 RunPod setup complete - training continues in background"