#!/bin/bash

# Quick Demo Runner for MovieLens RecSys
# Launches the Streamlit demo interface

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🎬 MovieLens Hybrid VAE Recommender - Demo Launcher${NC}"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Please install Python 3.7+${NC}"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install required packages
echo -e "${BLUE}Installing required packages...${NC}"
pip install -q streamlit pandas numpy plotly

# Check if processed data exists
if [ ! -d "data/processed" ]; then
    echo -e "${YELLOW}Processed data not found. Demo will run in demo mode.${NC}"
    echo -e "${YELLOW}For full functionality, run the ETL pipeline first:${NC}"
    echo -e "${YELLOW}  python src/data/etl_pipeline.py --data_source ./data/raw/ml-32m${NC}"
    echo ""
fi

# Launch Streamlit demo
echo -e "${GREEN}🚀 Launching demo interface...${NC}"
echo -e "${GREEN}Demo will be available at: http://localhost:8501${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the demo${NC}"
echo ""

# Run Streamlit
streamlit run src/api/streamlit_demo.py --server.address 0.0.0.0 --server.port 8501

# Deactivate virtual environment on exit
deactivate