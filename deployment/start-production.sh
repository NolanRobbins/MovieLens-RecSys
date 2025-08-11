#!/bin/bash

# Production startup script for MovieLens Recommendation System

set -e

echo "🚀 Starting MovieLens Recommendation System in Production Mode..."

# Set production environment
export ENVIRONMENT=production

# Run system validation
echo "🔧 Running system validation..."
python validate_system.py

# Check if validation passed
if [ $? -ne 0 ]; then
    echo "❌ System validation failed. Exiting..."
    exit 1
fi

echo "✅ System validation passed!"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the Streamlit application with production settings
echo "🌟 Starting Streamlit application..."
exec streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false \
    --logger.level=info