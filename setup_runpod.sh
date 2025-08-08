#!/bin/bash
# RunPod Setup Script for Hybrid VAE Training

echo "🚀 Setting up Hybrid VAE Training Environment on RunPod..."

# Update system
apt-get update && apt-get install -y wget unzip

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_cloud.txt

# Create data directory for pre-split training files
mkdir -p data

echo "📁 Upload your pre-split training files to ./data/ directory:"
echo "   - train_data.csv"
echo "   - val_data.csv"
echo "   - data_mappings.pkl"
echo ""
echo "🚨 IMPORTANT: Do NOT upload the newer (ETL test) data to prevent leakage!"

# Create directories for outputs
mkdir -p models
mkdir -p logs
mkdir -p results

# Set permissions
chmod +x cloud_training.py

echo "🎯 Setup complete! After uploading your data files, run training with:"
echo "python cloud_training.py --data_path ./data --save_path ./models/hybrid_vae_best.pt --use_wandb"

# Optional: Start jupyter lab
echo "🔬 Starting Jupyter Lab on port 8888..."
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

echo "✨ Environment ready for training!"