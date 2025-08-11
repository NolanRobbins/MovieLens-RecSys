#!/usr/bin/env python3
"""
Main entry point for MovieLens RecSys
Provides easy access to all system components
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def run_api():
    """Launch the FastAPI inference service"""
    import uvicorn
    from src.api.inference_api import app
    print("🚀 Starting MovieLens Inference API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_streamlit():
    """Launch the Streamlit demo"""
    import subprocess
    print("🎬 Starting Streamlit Demo...")
    subprocess.run(["streamlit", "run", "src/api/streamlit_demo.py"])

def run_evaluation():
    """Run advanced model evaluation"""
    from src.evaluation.advanced_evaluation import main
    print("📊 Running Advanced Evaluation...")
    main()

def run_training():
    """Run model training"""
    from src.models.ranking_optimized_training import main
    print("🧠 Starting Model Training...")
    main()

def main():
    parser = argparse.ArgumentParser(description="MovieLens RecSys - Main Entry Point")
    parser.add_argument("command", choices=["api", "demo", "eval", "train"], 
                       help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api()
    elif args.command == "demo":
        run_streamlit()
    elif args.command == "eval":
        run_evaluation()
    elif args.command == "train":
        run_training()

if __name__ == "__main__":
    main()