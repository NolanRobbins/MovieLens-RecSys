#!/bin/bash
# Google Cloud Platform Setup Script for SS4Rec Training
# ======================================================
#
# This script sets up a Google Cloud VM for SS4Rec training with:
# - CUDA-enabled PyTorch
# - RecBole framework
# - SS4Rec dependencies
# - Proper environment configuration
#
# Usage:
#   chmod +x gcp_setup.sh
#   ./gcp_setup.sh

set -e

echo "🌟 SS4Rec Google Cloud Setup Script"
echo "===================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    software-properties-common

# Install NVIDIA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo "🔧 Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-525
else
    echo "✅ NVIDIA drivers already installed"
    nvidia-smi
fi

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv ss4rec_env
source ss4rec_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install numpy==1.24.4  # Compatible with RecBole
pip install pandas scipy scikit-learn tqdm pyyaml

# Install RecBole framework
echo "🤖 Installing RecBole framework..."
pip install recbole==1.2.0
pip install ray hyperopt kmeans-pytorch lightgbm xgboost

# Install State Space Model dependencies
echo "🌊 Installing State Space Model dependencies..."
pip install ninja packaging  # Build dependencies
pip install causal-conv1d    # Flexible version
pip install mamba-ssm       # Latest compatible version

# Install experiment tracking
echo "📊 Installing experiment tracking tools..."
pip install wandb tensorboard matplotlib seaborn

# Install data download utility
echo "📥 Installing data utilities..."
pip install gdown

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')

try:
    import recbole
    print(f'RecBole: {recbole.__version__}')
except ImportError:
    print('RecBole: Not installed')

try:
    from mamba_ssm import Mamba
    print('Mamba-SSM: Available')
except ImportError:
    print('Mamba-SSM: Not available')
"

# Create directories
echo "📁 Creating project directories..."
mkdir -p data/recbole_format/ml-25m
mkdir -p results/gcp_ss4rec
mkdir -p logs

# Set environment variables
echo "🌍 Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create activation script
cat > activate_ss4rec.sh << 'EOF'
#!/bin/bash
source ss4rec_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ SS4Rec environment activated"
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_ss4rec.sh

echo "✅ Google Cloud setup completed!"
echo ""
echo "🚀 To start training:"
echo "   source activate_ss4rec.sh"
echo "   python gcp_training.py"
echo ""
echo "📊 To monitor training:"
echo "   tail -f gcp_training.log"