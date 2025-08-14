#!/bin/bash

# Runpod A100 Environment Setup Script
# Run this script first after starting your Runpod instance

echo "🚀 Setting up Runpod A100 environment for Hybrid VAE training..."
echo "=================================================="

# Update system
apt-get update -y
apt-get install -y wget curl unzip git

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib seaborn
pip install wandb tqdm jupyter ipywidgets

# Create directory structure
mkdir -p /workspace/data
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/results

echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your data files to /workspace/data/"
echo "2. Run: python runpod_training.py"
echo ""
echo "Required data files:"
echo "- train_data.csv"
echo "- val_data.csv"  
echo "- data_mappings.pkl"
echo ""
echo "GPU Info:"
nvidia-smi

# Set environment variables for optimal A100 performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

echo "🎯 Ready for training!"