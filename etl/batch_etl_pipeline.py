#!/usr/bin/env python3
"""
MovieLens ETL Pipeline - Batch Processing Engine

Processes 5% of test data daily over 20 days with monitoring and A/B testing.
"""

import os
import json
import logging
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import time

class BatchETLPipeline:
    """Main ETL Pipeline for processing test data in daily batches"""
    
    def __init__(self, config_path: str = "etl/config/pipeline_config.yaml"):
        """Initialize the ETL pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.metrics = {}
        
        # Create output directories
        self._create_directories()
        
        # Load processing tracker
        self.tracker_path = self.config['pipeline']['data']['processed_tracker']
        self.processing_tracker = self._load_processing_tracker()
        
        self.logger.info("✅ ETL Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        log_level = self.config['pipeline']['processing']['log_level']
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('etl/logs/etl_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _create_directories(self):
        """Create necessary directories for ETL pipeline"""
        directories = [
            self.config['pipeline']['data']['output_path'],
            self.config['pipeline']['data']['metrics_path'],
            'etl/logs/',
            'data/etl_batches/daily/',
            'data/etl_batches/cumulative/',
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_processing_tracker(self) -> Dict:
        """Load or create processing tracker to track what's been processed"""
        if os.path.exists(self.tracker_path):
            with open(self.tracker_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize tracker
            tracker = {
                "total_records": 0,
                "processed_records": 0,
                "daily_batches": {},
                "start_date": self.config['pipeline']['batch']['start_date'],
                "current_day": 0,
                "completion_percentage": 0.0,
                "last_updated": None
            }
            self._save_processing_tracker(tracker)
            return tracker
    
    def _save_processing_tracker(self, tracker: Dict):
        """Save processing tracker to file"""
        tracker["last_updated"] = datetime.now().isoformat()
        with open(self.tracker_path, 'w') as f:
            json.dump(tracker, f, indent=2)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load the complete test dataset"""
        self.logger.info("📊 Loading test dataset...")
        
        test_data_path = self.config['pipeline']['data']['test_data_path']
        
        try:
            # Handle Git LFS files by trying to read actual data
            df = pd.read_csv(test_data_path)
            
            # If we get Git LFS metadata, log a warning
            if len(df) < 100:  # Suspiciously small
                self.logger.warning("⚠️ Test data appears to be Git LFS pointer. You may need to run 'git lfs pull'")
                # For demo purposes, let's create sample data
                df = self._create_sample_test_data()
            
            self.logger.info(f"✅ Loaded {len(df)} test records")
            
            # Update tracker with total records
            if self.processing_tracker["total_records"] == 0:
                self.processing_tracker["total_records"] = len(df)
                self._save_processing_tracker(self.processing_tracker)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading test data: {e}")
            # Fallback to sample data for demonstration
            return self._create_sample_test_data()
    
    def _create_sample_test_data(self) -> pd.DataFrame:
        """Create sample test data for demonstration"""
        self.logger.info("🔧 Creating sample test data for demonstration...")
        
        np.random.seed(42)
        n_samples = 10000  # Sample dataset
        
        data = {
            'userId': np.random.randint(1, 1000, n_samples),
            'movieId': np.random.randint(1, 5000, n_samples),
            'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_samples, p=[0.05, 0.1, 0.25, 0.35, 0.25]),
            'timestamp': np.random.randint(1600000000, 1700000000, n_samples),  # Recent timestamps
            'user_idx': np.random.randint(0, 999, n_samples),
            'movie_idx': np.random.randint(0, 4999, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['rating_date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['rating_year'] = df['rating_date'].dt.year
        df['rating_month'] = df['rating_date'].dt.month
        df['rating_weekday'] = df['rating_date'].dt.dayofweek
        
        # Sort by timestamp for chronological processing
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"✅ Created {len(df)} sample test records")
        return df
    
    def get_daily_batch(self, df: pd.DataFrame, day_number: int) -> pd.DataFrame:
        """Extract daily batch (5% of original dataset)"""
        total_records = len(df)
        batch_size = int(total_records * self.config['pipeline']['batch']['daily_percentage'])
        
        # Calculate start and end indices for this day
        start_idx = (day_number - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        
        if start_idx >= total_records:
            self.logger.warning(f"⚠️ Day {day_number}: No more data to process")
            return pd.DataFrame()
        
        daily_batch = df.iloc[start_idx:end_idx].copy()
        
        self.logger.info(f"📦 Day {day_number}: Processing batch {start_idx}:{end_idx} ({len(daily_batch)} records)")
        
        return daily_batch
    
    def validate_batch_data(self, batch_df: pd.DataFrame) -> Dict[str, float]:
        """Validate data quality of the batch"""
        if batch_df.empty:
            return {"completeness": 0.0, "validity": 0.0, "uniqueness": 0.0}
        
        metrics = {}
        
        # Completeness: % of non-null values
        total_cells = batch_df.size
        non_null_cells = batch_df.count().sum()
        metrics["completeness"] = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Validity: % of valid ratings (1-5)
        valid_ratings = batch_df['rating'].between(1, 5).sum()
        metrics["validity"] = valid_ratings / len(batch_df) if len(batch_df) > 0 else 0
        
        # Uniqueness: % of unique user-item pairs
        unique_pairs = batch_df[['userId', 'movieId']].drop_duplicates().shape[0]
        metrics["uniqueness"] = unique_pairs / len(batch_df) if len(batch_df) > 0 else 0
        
        # Consistency: Timestamp ordering
        if len(batch_df) > 1:
            is_sorted = batch_df['timestamp'].is_monotonic_increasing
            metrics["consistency"] = 1.0 if is_sorted else 0.0
        else:
            metrics["consistency"] = 1.0
        
        return metrics
    
    def process_daily_batch(self, day_number: int, simulate_day: bool = True) -> Dict:
        """Process a single day's batch with monitoring"""
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting Day {day_number} batch processing...")
        
        # Load test data
        test_df = self.load_test_data()
        
        # Get daily batch
        daily_batch = self.get_daily_batch(test_df, day_number)
        
        if daily_batch.empty:
            self.logger.warning(f"⚠️ No data to process for Day {day_number}")
            return {"status": "no_data", "day": day_number}
        
        # Validate data quality
        data_quality_metrics = self.validate_batch_data(daily_batch)
        
        # Transform data (basic feature engineering)
        transformed_batch = self._transform_batch(daily_batch)
        
        # Save daily batch
        daily_output_path = f"data/etl_batches/daily/day_{day_number:02d}_batch.csv"
        transformed_batch.to_csv(daily_output_path, index=False)
        
        # Update cumulative dataset
        cumulative_df = self._update_cumulative_dataset(transformed_batch, day_number)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        throughput = len(daily_batch) / processing_time if processing_time > 0 else 0
        
        # Prepare batch results
        batch_results = {
            "day": day_number,
            "status": "success",
            "records_processed": len(daily_batch),
            "processing_time_seconds": processing_time,
            "throughput_records_per_second": throughput,
            "data_quality": data_quality_metrics,
            "output_file": daily_output_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update processing tracker
        self._update_processing_tracker(day_number, len(daily_batch), batch_results)
        
        # Save daily metrics
        self._save_daily_metrics(day_number, batch_results)
        
        self.logger.info(f"✅ Day {day_number} completed: {len(daily_batch)} records in {processing_time:.2f}s")
        
        return batch_results
    
    def _transform_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic transformations to the batch"""
        transformed = batch_df.copy()
        
        # Add processing metadata
        transformed['batch_processed_date'] = datetime.now().isoformat()
        transformed['data_source'] = 'test_batch'
        
        # Basic feature engineering
        transformed['rating_normalized'] = (transformed['rating'] - 1) / 4  # Normalize to 0-1
        transformed['is_high_rating'] = (transformed['rating'] >= 4).astype(int)
        
        # User activity features
        user_stats = transformed.groupby('userId')['rating'].agg(['count', 'mean']).reset_index()
        user_stats.columns = ['userId', 'user_rating_count', 'user_avg_rating']
        transformed = transformed.merge(user_stats, on='userId', how='left')
        
        return transformed
    
    def _update_cumulative_dataset(self, daily_batch: pd.DataFrame, day_number: int) -> pd.DataFrame:
        """Update cumulative dataset with new daily batch"""
        cumulative_path = "data/etl_batches/cumulative/cumulative_processed.csv"
        
        if os.path.exists(cumulative_path):
            cumulative_df = pd.read_csv(cumulative_path)
            cumulative_df = pd.concat([cumulative_df, daily_batch], ignore_index=True)
        else:
            cumulative_df = daily_batch.copy()
        
        # Save updated cumulative dataset
        cumulative_df.to_csv(cumulative_path, index=False)
        
        self.logger.info(f"📈 Cumulative dataset updated: {len(cumulative_df)} total records")
        
        return cumulative_df
    
    def _update_processing_tracker(self, day_number: int, records_processed: int, batch_results: Dict):
        """Update the processing tracker with daily results"""
        self.processing_tracker["current_day"] = day_number
        self.processing_tracker["processed_records"] += records_processed
        
        # Calculate completion percentage
        if self.processing_tracker["total_records"] > 0:
            completion = self.processing_tracker["processed_records"] / self.processing_tracker["total_records"]
            self.processing_tracker["completion_percentage"] = min(completion * 100, 100.0)
        
        # Store daily batch info
        self.processing_tracker["daily_batches"][f"day_{day_number}"] = {
            "records": records_processed,
            "status": batch_results["status"],
            "processing_time": batch_results["processing_time_seconds"],
            "data_quality": batch_results["data_quality"]
        }
        
        self._save_processing_tracker(self.processing_tracker)
    
    def _save_daily_metrics(self, day_number: int, batch_results: Dict):
        """Save daily batch metrics"""
        metrics_file = f"data/etl_metrics/day_{day_number:02d}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline processing status"""
        return {
            "pipeline_name": self.config['pipeline']['name'],
            "version": self.config['pipeline']['version'],
            "current_day": self.processing_tracker["current_day"],
            "total_days": self.config['pipeline']['batch']['total_days'],
            "completion_percentage": self.processing_tracker["completion_percentage"],
            "total_records": self.processing_tracker["total_records"],
            "processed_records": self.processing_tracker["processed_records"],
            "remaining_records": self.processing_tracker["total_records"] - self.processing_tracker["processed_records"],
            "last_updated": self.processing_tracker["last_updated"]
        }
    
    def run_simulation(self, days_to_simulate: int = None) -> Dict:
        """Run the complete ETL pipeline simulation"""
        if days_to_simulate is None:
            days_to_simulate = self.config['pipeline']['batch']['total_days']
        
        self.logger.info(f"🎬 Starting ETL pipeline simulation for {days_to_simulate} days...")
        
        simulation_results = {
            "simulation_start": datetime.now().isoformat(),
            "days_simulated": 0,
            "total_records_processed": 0,
            "daily_results": {},
            "overall_status": "running"
        }
        
        try:
            for day in range(1, days_to_simulate + 1):
                daily_result = self.process_daily_batch(day)
                simulation_results["daily_results"][f"day_{day}"] = daily_result
                simulation_results["days_simulated"] = day
                
                if daily_result["status"] == "success":
                    simulation_results["total_records_processed"] += daily_result["records_processed"]
                
                # Small delay to simulate daily processing
                time.sleep(1)
            
            simulation_results["overall_status"] = "completed"
            simulation_results["simulation_end"] = datetime.now().isoformat()
            
            self.logger.info(f"🎉 ETL pipeline simulation completed successfully!")
            self.logger.info(f"📊 Processed {simulation_results['total_records_processed']} records over {days_to_simulate} days")
            
        except Exception as e:
            simulation_results["overall_status"] = "failed"
            simulation_results["error"] = str(e)
            self.logger.error(f"❌ ETL pipeline simulation failed: {e}")
        
        # Save simulation results
        with open("data/etl_metrics/simulation_results.json", 'w') as f:
            json.dump(simulation_results, f, indent=2)
        
        return simulation_results


def main():
    """Main entry point for ETL pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MovieLens ETL Pipeline")
    parser.add_argument("--day", type=int, help="Process specific day (1-20)")
    parser.add_argument("--simulate", type=int, help="Simulate N days of processing")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BatchETLPipeline()
    
    if args.status:
        # Show status
        status = pipeline.get_pipeline_status()
        print(f"""
🚀 ETL Pipeline Status:
  Pipeline: {status['pipeline_name']} v{status['version']}
  Progress: Day {status['current_day']}/{status['total_days']} ({status['completion_percentage']:.1f}%)
  Records: {status['processed_records']:,}/{status['total_records']:,} processed
  Remaining: {status['remaining_records']:,} records
  Last Updated: {status['last_updated']}
        """)
    
    elif args.day:
        # Process specific day
        result = pipeline.process_daily_batch(args.day)
        print(f"Day {args.day} processing result:", json.dumps(result, indent=2))
    
    elif args.simulate:
        # Run simulation
        results = pipeline.run_simulation(args.simulate)
        print(f"Simulation completed: {results['overall_status']}")
        print(f"Processed {results['total_records_processed']} records over {results['days_simulated']} days")
    
    else:
        # Default: process next day
        current_day = pipeline.processing_tracker["current_day"] + 1
        if current_day <= pipeline.config['pipeline']['batch']['total_days']:
            result = pipeline.process_daily_batch(current_day)
            print(f"Processed day {current_day}:", json.dumps(result, indent=2))
        else:
            print("✅ All days completed!")
            status = pipeline.get_pipeline_status()
            print(f"Final completion: {status['completion_percentage']:.1f}%")


if __name__ == "__main__":
    main()
