#!/usr/bin/env python3
"""
MovieLens RecSys Deployment Demo
Comprehensive demonstration of ETL pipeline, model deployment, and system architecture
"""

import subprocess
import time
import requests
import json
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime
import threading
from typing import Dict, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentDemo:
    """Comprehensive deployment demonstration"""
    
    def __init__(self, skip_docker: bool = False):
        self.skip_docker = skip_docker
        self.api_port = 8000
        self.demo_port = 8501
        self.api_base_url = f"http://localhost:{self.api_port}"
        self.demo_url = f"http://localhost:{self.demo_port}"
        
        # Process handles
        self.api_process = None
        self.demo_process = None
        
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"🎬 {title}")
        print("="*60)
        
    def print_step(self, step: str):
        """Print formatted step"""
        print(f"\n📋 {step}")
        print("-" * 40)
        
    def check_dependencies(self):
        """Check if all dependencies are available"""
        self.print_step("Checking Dependencies")
        
        # Check Python packages
        try:
            import torch
            import fastapi
            import streamlit
            import pandas
            import numpy
            print("✅ All Python packages available")
        except ImportError as e:
            print(f"❌ Missing Python package: {e}")
            return False
        
        # Check Docker if not skipping
        if not self.skip_docker:
            try:
                result = subprocess.run(['docker', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Docker available: {result.stdout.strip()}")
                else:
                    print("❌ Docker not available")
                    return False
            except FileNotFoundError:
                print("❌ Docker not installed")
                return False
        
        return True
    
    def demonstrate_etl_pipeline(self):
        """Demonstrate the ETL pipeline capabilities"""
        self.print_step("ETL Pipeline Demonstration")
        
        # Show data structure
        if Path("ml-32m").exists():
            print("📊 Source Data Structure:")
            for file_path in Path("ml-32m").glob("*.csv"):
                file_size = file_path.stat().st_size / 1024 / 1024
                print(f"   - {file_path.name}: {file_size:.1f} MB")
        
        # Show processed data
        if Path("processed_data").exists():
            print("\n📈 Processed Data:")
            for file_path in Path("processed_data").iterdir():
                if file_path.is_file():
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        print(f"   - {file_path.name}: {len(df):,} rows")
                    elif file_path.suffix == '.json':
                        with open(file_path) as f:
                            data = json.load(f)
                        print(f"   - {file_path.name}: {len(data)} keys")
                    else:
                        file_size = file_path.stat().st_size / 1024 / 1024
                        print(f"   - {file_path.name}: {file_size:.1f} MB")
        
        # Show ETL config
        print("\n⚙️ ETL Configuration:")
        try:
            with open("etl_config.json") as f:
                config = json.load(f)
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"   - {key}: {len(value)} settings")
                else:
                    print(f"   - {key}: {value}")
        except FileNotFoundError:
            print("   ⚠️ ETL config not found")
    
    def demonstrate_model_info(self):
        """Show information about the trained model"""
        self.print_step("Model Information")
        
        try:
            import torch
            checkpoint = torch.load("models/hybrid_vae_best.pt", map_location='cpu')
            
            print("🧠 Model Architecture:")
            config = checkpoint['config']
            print(f"   - Embeddings: {config['n_factors']}D")
            print(f"   - Hidden Layers: {config['hidden_dims']}")
            print(f"   - Latent Space: {config['latent_dim']}D")
            print(f"   - Dropout: {config['dropout_rate']}")
            
            print(f"\n📊 Training Results:")
            print(f"   - Experiment: {checkpoint['experiment_name']}")
            print(f"   - Final Epoch: {checkpoint['epoch']}")
            print(f"   - Validation Loss: {checkpoint['val_loss']:.4f}")
            print(f"   - Users: {checkpoint['n_users']:,}")
            print(f"   - Movies: {checkpoint['n_movies']:,}")
            
        except Exception as e:
            print(f"❌ Error loading model info: {e}")
    
    def start_api_service(self):
        """Start the FastAPI service"""
        self.print_step("Starting API Service")
        
        try:
            # Start API in background
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "inference_api:app", 
                "--host", "0.0.0.0", 
                "--port", str(self.api_port),
                "--log-level", "info"
            ]
            
            print(f"🚀 Starting API server on port {self.api_port}")
            self.api_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for API to be ready
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.api_base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ API service is ready!")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
                time.sleep(2)
            
            print("❌ API failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Error starting API: {e}")
            return False
    
    def start_demo_interface(self):
        """Start the Streamlit demo interface"""
        self.print_step("Starting Demo Interface")
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "streamlit_demo.py",
                "--server.port", str(self.demo_port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ]
            
            print(f"🎨 Starting Streamlit demo on port {self.demo_port}")
            self.demo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for demo to be ready
            time.sleep(5)  # Streamlit takes longer to start
            
            print("✅ Demo interface started!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting demo: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        self.print_step("Testing API Endpoints")
        
        try:
            # Health check
            response = requests.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health Check:")
                print(f"   - Status: {health_data['status']}")
                print(f"   - Model Loaded: {health_data['model_loaded']}")
                print(f"   - Uptime: {health_data['uptime_seconds']:.1f}s")
            
            # Test recommendations
            test_request = {
                "user_id": 1,
                "n_recommendations": 5,
                "exclude_seen": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/recommend",
                json=test_request
            )
            
            if response.status_code == 200:
                rec_data = response.json()
                print("\n✅ Recommendation Test:")
                print(f"   - User ID: {rec_data['user_id']}")
                print(f"   - Generated: {len(rec_data['recommendations'])} recommendations")
                print(f"   - Processing Time: {rec_data['processing_time_ms']:.1f}ms")
                
                # Show sample recommendations
                print("   - Sample Recommendations:")
                for i, rec in enumerate(rec_data['recommendations'][:3], 1):
                    print(f"     {i}. {rec['title']} (Rating: {rec['predicted_rating']:.2f})")
            else:
                print(f"❌ Recommendation test failed: {response.status_code}")
            
            # Test movie details
            response = requests.get(f"{self.api_base_url}/movies/1")
            if response.status_code == 200:
                movie_data = response.json()
                print(f"\n✅ Movie Details Test:")
                print(f"   - Title: {movie_data['title']}")
                print(f"   - Genres: {movie_data['genres']}")
            
        except Exception as e:
            print(f"❌ API testing failed: {e}")
    
    def show_system_architecture(self):
        """Display system architecture information"""
        self.print_step("System Architecture Overview")
        
        print("""
🏗️ MovieLens RecSys Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│ ETL Pipeline (etl_pipeline.py)                                 │
│ • Raw data extraction (MovieLens 32M)                          │
│ • Data quality validation                                       │
│ • Temporal splitting (train/val/test)                          │
│ • Feature engineering & mapping creation                       │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│ Hybrid VAE Model (cloud_training.py)                           │
│ • User/Movie Embeddings (150D)                                 │
│ • Encoder: [512, 256, 128] → 64D latent                       │
│ • Decoder: 64D → [128, 256, 512] → Rating                     │
│ • Advanced Loss Functions (ranking_loss_functions.py)          │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVICE LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│ FastAPI Service (inference_api.py)                             │
│ • Real-time predictions                                         │
│ • Business logic integration                                    │
│ • Caching & performance optimization                            │
│ • Health monitoring & metrics                                   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│ Streamlit Demo (streamlit_demo.py)                             │
│ • Interactive recommendation interface                          │
│ • Business impact calculator                                    │
│ • System monitoring dashboard                                   │
│ • Technical showcase for stakeholders                          │
└─────────────────────────────────────────────────────────────────┘

📊 Key Technical Features:
• Production-ready ETL with quality validation
• Advanced ML: Hybrid VAE with collaborative filtering
• Real-time inference API with caching
• Business logic integration & filtering
• Comprehensive monitoring & health checks
• Docker containerization for deployment
• A/B testing framework ready
""")
    
    def show_deployment_summary(self):
        """Show final deployment summary"""
        self.print_step("Deployment Summary")
        
        print("🎉 MovieLens RecSys Successfully Deployed!")
        print("\n📍 Access Points:")
        print(f"   • API Service:      {self.api_base_url}")
        print(f"   • API Health:       {self.api_base_url}/health")
        print(f"   • API Docs:         {self.api_base_url}/docs")
        print(f"   • Demo Interface:   {self.demo_url}")
        
        print(f"\n🔧 Management Commands:")
        print(f"   • Test API:         curl {self.api_base_url}/health")
        print(f"   • View Logs:        Check terminal output")
        print(f"   • Stop Services:    Ctrl+C or use stop_services()")
        
        print(f"\n📈 Next Steps:")
        print(f"   • Open {self.demo_url} in your browser")
        print(f"   • Test the recommendation system")
        print(f"   • Explore the business impact calculator")
        print(f"   • Review system architecture diagrams")
        
    def stop_services(self):
        """Stop all running services"""
        self.print_step("Stopping Services")
        
        if self.api_process:
            print("🛑 Stopping API service...")
            self.api_process.terminate()
            self.api_process.wait()
            print("✅ API service stopped")
        
        if self.demo_process:
            print("🛑 Stopping demo interface...")
            self.demo_process.terminate()
            self.demo_process.wait()
            print("✅ Demo interface stopped")
    
    def run_full_demo(self):
        """Run the complete deployment demonstration"""
        self.print_header("MovieLens Recommendation System - Deployment Demo")
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                print("❌ Dependency check failed. Please install missing components.")
                return False
            
            # Step 2: Show ETL pipeline
            self.demonstrate_etl_pipeline()
            
            # Step 3: Show model info
            self.demonstrate_model_info()
            
            # Step 4: Show architecture
            self.show_system_architecture()
            
            # Step 5: Start services
            if not self.start_api_service():
                print("❌ Failed to start API service")
                return False
            
            time.sleep(2)
            
            if not self.start_demo_interface():
                print("❌ Failed to start demo interface")
                return False
            
            # Step 6: Test services
            self.test_api_endpoints()
            
            # Step 7: Show summary
            self.show_deployment_summary()
            
            # Keep services running
            print(f"\n⏰ Services are now running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False
        finally:
            self.stop_services()
        
        return True

def main():
    parser = argparse.ArgumentParser(description='MovieLens RecSys Deployment Demo')
    parser.add_argument('--skip-docker', action='store_true',
                       help='Skip Docker-related checks and operations')
    parser.add_argument('--api-only', action='store_true',
                       help='Start only API service')
    parser.add_argument('--demo-only', action='store_true',
                       help='Start only demo interface')
    
    args = parser.parse_args()
    
    demo = DeploymentDemo(skip_docker=args.skip_docker)
    
    if args.api_only:
        demo.start_api_service()
        demo.test_api_endpoints()
        input("Press Enter to stop...")
        demo.stop_services()
    elif args.demo_only:
        demo.start_demo_interface()
        input("Press Enter to stop...")
        demo.stop_services()
    else:
        demo.run_full_demo()

if __name__ == "__main__":
    main()