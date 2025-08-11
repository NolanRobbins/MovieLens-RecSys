"""
Temporal Data Simulator for MovieLens RecSys
Simulates streaming data arrival using your existing temporal data split
Week 10+ data becomes "production" streaming batches for realistic testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple, Any
import logging
import json
import time
import random
from dataclasses import dataclass, asdict
from threading import Thread, Event
import queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingBatch:
    """Represents a streaming batch of interactions"""
    batch_id: str
    interactions: pd.DataFrame
    timestamp: datetime
    batch_size: int
    source_week: int
    simulation_time: float

@dataclass
class StreamingStats:
    """Statistics for streaming simulation"""
    total_batches_sent: int
    total_interactions_sent: int
    simulation_start_time: datetime
    simulation_duration_seconds: float
    avg_batch_size: float
    avg_batch_interval_seconds: float
    weeks_simulated: List[int]

class TemporalDataSimulator:
    """
    Simulates real-time data arrival using your existing temporal data
    Uses weeks 10+ as streaming data to demonstrate model drift
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        
        # Load existing temporal data
        self.train_data = pd.read_csv(self.data_dir / "train_data.csv")
        self.val_data = pd.read_csv(self.data_dir / "val_data.csv")  
        self.test_data = pd.read_csv(self.data_dir / "test_data.csv")
        
        # Load metadata to understand temporal split
        self.metadata = self._load_metadata()
        
        # Simulation configuration
        self.config = {
            'batch_size_range': (50, 200),  # Variable batch sizes
            'batch_interval_seconds': (30, 120),  # 30s to 2min between batches
            'streaming_weeks_start': 10,  # Start streaming from week 10
            'simulation_speed_multiplier': 1.0,  # 1.0 = real-time, 10.0 = 10x faster
            'add_noise': True,  # Add realistic noise to ratings
            'seasonal_patterns': True  # Simulate seasonal viewing patterns
        }
        
        self.streaming_queue = queue.Queue()
        self.simulation_active = Event()
        self.stats = StreamingStats(
            total_batches_sent=0,
            total_interactions_sent=0,
            simulation_start_time=datetime.now(),
            simulation_duration_seconds=0.0,
            avg_batch_size=0.0,
            avg_batch_interval_seconds=0.0,
            weeks_simulated=[]
        )
        
        logger.info(f"TemporalDataSimulator initialized with {len(self.test_data)} test interactions")
    
    def _load_metadata(self) -> Dict:
        """Load metadata about the temporal split"""
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        # Create default metadata if not available
        return {
            'train_weeks': '1-8',
            'val_weeks': '9',
            'test_weeks': '10+',
            'total_interactions': len(self.train_data) + len(self.val_data) + len(self.test_data)
        }
    
    def simulate_streaming_data(self, duration_minutes: int = 60) -> Generator[StreamingBatch, None, None]:
        """
        Simulate streaming data arrival for a specified duration
        Yields batches of interactions as they would arrive in production
        """
        logger.info(f"Starting streaming simulation for {duration_minutes} minutes")
        
        simulation_start = datetime.now()
        simulation_end = simulation_start + timedelta(minutes=duration_minutes)
        
        # Prepare streaming data from test set (representing weeks 10+)
        streaming_data = self.test_data.copy()
        
        # Add realistic timestamps if not present
        streaming_data = self._add_realistic_timestamps(streaming_data)
        
        # Sort by timestamp for realistic streaming
        streaming_data = streaming_data.sort_values('timestamp')
        
        batch_id = 0
        current_time = simulation_start
        data_index = 0
        
        while current_time < simulation_end and data_index < len(streaming_data):
            # Determine batch size
            batch_size = random.randint(*self.config['batch_size_range'])
            batch_size = min(batch_size, len(streaming_data) - data_index)
            
            if batch_size == 0:
                break
            
            # Create batch
            batch_data = streaming_data.iloc[data_index:data_index + batch_size].copy()
            
            # Add noise if configured
            if self.config['add_noise']:
                batch_data = self._add_realistic_noise(batch_data)
            
            # Apply seasonal patterns
            if self.config['seasonal_patterns']:
                batch_data = self._apply_seasonal_patterns(batch_data, current_time)
            
            # Create streaming batch
            batch = StreamingBatch(
                batch_id=f"batch_{batch_id:06d}",
                interactions=batch_data[['user_id', 'movie_id', 'rating']],
                timestamp=current_time,
                batch_size=batch_size,
                source_week=self._get_source_week(data_index, len(streaming_data)),
                simulation_time=current_time.timestamp()
            )
            
            # Update statistics
            self._update_streaming_stats(batch)
            
            yield batch
            
            # Move to next batch
            data_index += batch_size
            batch_id += 1
            
            # Wait for next batch (respecting simulation speed)
            wait_seconds = random.uniform(*self.config['batch_interval_seconds'])
            wait_seconds /= self.config['simulation_speed_multiplier']
            
            time.sleep(wait_seconds)
            current_time = datetime.now()
        
        self.stats.simulation_duration_seconds = (datetime.now() - simulation_start).total_seconds()
        logger.info(f"Streaming simulation completed: {self.stats.total_batches_sent} batches, "
                   f"{self.stats.total_interactions_sent} interactions")
    
    def _add_realistic_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic timestamps to data if not present"""
        data = data.copy()
        
        if 'timestamp' not in data.columns:
            # Create timestamps spanning several weeks for realism
            start_time = datetime.now() - timedelta(weeks=4)
            end_time = datetime.now()
            
            # Generate random timestamps
            time_range = (end_time - start_time).total_seconds()
            random_seconds = np.random.uniform(0, time_range, len(data))
            timestamps = [start_time + timedelta(seconds=s) for s in random_seconds]
            
            data['timestamp'] = timestamps
        
        return data
    
    def _add_realistic_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise to ratings to simulate real-world variance"""
        data = data.copy()
        
        # Add small amount of noise to ratings (±0.25 with 10% probability)
        noise_mask = np.random.random(len(data)) < 0.1
        noise_values = np.random.uniform(-0.25, 0.25, len(data))
        
        data.loc[noise_mask, 'rating'] = data.loc[noise_mask, 'rating'] + noise_values[noise_mask]
        
        # Ensure ratings stay within valid range
        data['rating'] = data['rating'].clip(0.5, 5.0)
        
        return data
    
    def _apply_seasonal_patterns(self, data: pd.DataFrame, current_time: datetime) -> pd.DataFrame:
        """Apply seasonal patterns to simulate real-world behavior"""
        data = data.copy()
        
        # Weekend boost (people watch more movies on weekends)
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            weekend_boost = np.random.uniform(0.1, 0.3, len(data))
            data['rating'] = data['rating'] + weekend_boost
        
        # Evening boost (higher ratings in the evening)
        if 18 <= current_time.hour <= 23:
            evening_boost = np.random.uniform(0.05, 0.2, len(data))
            data['rating'] = data['rating'] + evening_boost
        
        # Holiday patterns (more family-friendly content during holidays)
        # Simplified - would be more sophisticated in production
        
        # Ensure ratings stay within bounds
        data['rating'] = data['rating'].clip(0.5, 5.0)
        
        return data
    
    def _get_source_week(self, data_index: int, total_data: int) -> int:
        """Determine which week this data represents"""
        # Map data index to week (starting from week 10)
        week_progress = data_index / total_data
        return int(10 + week_progress * 10)  # Spread over weeks 10-20
    
    def _update_streaming_stats(self, batch: StreamingBatch):
        """Update streaming statistics"""
        self.stats.total_batches_sent += 1
        self.stats.total_interactions_sent += batch.batch_size
        
        if batch.source_week not in self.stats.weeks_simulated:
            self.stats.weeks_simulated.append(batch.source_week)
        
        # Calculate running averages
        self.stats.avg_batch_size = self.stats.total_interactions_sent / self.stats.total_batches_sent
    
    def simulate_model_drift_scenario(self) -> Generator[StreamingBatch, None, None]:
        """
        Simulate a specific scenario where model performance degrades over time
        Useful for testing retraining triggers
        """
        logger.info("Starting model drift simulation scenario")
        
        scenarios = [
            {
                'name': 'stable_period',
                'duration_minutes': 30,
                'rating_bias': 0.0,
                'noise_level': 0.1,
                'description': 'Normal stable performance period'
            },
            {
                'name': 'gradual_drift',
                'duration_minutes': 45,
                'rating_bias': -0.3,  # Ratings trend lower
                'noise_level': 0.2,
                'description': 'Gradual model drift - ratings trending lower'
            },
            {
                'name': 'seasonal_shift',
                'duration_minutes': 30,
                'rating_bias': 0.2,
                'noise_level': 0.15,
                'description': 'Seasonal shift - holiday movie preferences'
            },
            {
                'name': 'data_quality_issues',
                'duration_minutes': 15,
                'rating_bias': 0.0,
                'noise_level': 0.5,  # High noise
                'description': 'Data quality issues - high noise period'
            }
        ]
        
        for scenario in scenarios:
            logger.info(f"Starting scenario: {scenario['description']}")
            
            # Configure simulation for this scenario
            original_config = self.config.copy()
            self.config['add_noise'] = True
            self.config['simulation_speed_multiplier'] = 2.0  # Faster for demo
            
            # Simulate this scenario
            scenario_start = datetime.now()
            streaming_data = self.test_data.copy()
            
            # Apply scenario-specific modifications
            streaming_data = self._apply_scenario_modifications(
                streaming_data, scenario['rating_bias'], scenario['noise_level']
            )
            
            data_index = 0
            batch_id = 0
            
            while (datetime.now() - scenario_start).total_seconds() < scenario['duration_minutes'] * 60:
                if data_index >= len(streaming_data):
                    data_index = 0  # Loop if we run out of data
                
                batch_size = random.randint(25, 100)  # Smaller batches for demo
                batch_size = min(batch_size, len(streaming_data) - data_index)
                
                batch_data = streaming_data.iloc[data_index:data_index + batch_size].copy()
                
                batch = StreamingBatch(
                    batch_id=f"drift_{scenario['name']}_{batch_id:04d}",
                    interactions=batch_data[['user_id', 'movie_id', 'rating']],
                    timestamp=datetime.now(),
                    batch_size=batch_size,
                    source_week=self._get_source_week(data_index, len(streaming_data)),
                    simulation_time=datetime.now().timestamp()
                )
                
                self._update_streaming_stats(batch)
                yield batch
                
                data_index += batch_size
                batch_id += 1
                
                # Short wait between batches
                time.sleep(10 / self.config['simulation_speed_multiplier'])
            
            # Restore original configuration
            self.config = original_config
            
            logger.info(f"Completed scenario: {scenario['name']}")
    
    def _apply_scenario_modifications(self, data: pd.DataFrame, rating_bias: float, 
                                    noise_level: float) -> pd.DataFrame:
        """Apply scenario-specific modifications to data"""
        data = data.copy()
        
        # Apply rating bias
        data['rating'] = data['rating'] + rating_bias
        
        # Apply noise
        noise = np.random.normal(0, noise_level, len(data))
        data['rating'] = data['rating'] + noise
        
        # Ensure valid range
        data['rating'] = data['rating'].clip(0.5, 5.0)
        
        return data
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        stats_dict = asdict(self.stats)
        
        # Add calculated metrics
        if self.stats.total_batches_sent > 0:
            stats_dict['avg_interactions_per_batch'] = (
                self.stats.total_interactions_sent / self.stats.total_batches_sent
            )
        
        if self.stats.simulation_duration_seconds > 0:
            stats_dict['interactions_per_second'] = (
                self.stats.total_interactions_sent / self.stats.simulation_duration_seconds
            )
        
        stats_dict['config'] = self.config
        stats_dict['data_source'] = {
            'train_samples': len(self.train_data),
            'val_samples': len(self.val_data),
            'test_samples': len(self.test_data),
            'total_samples': len(self.train_data) + len(self.val_data) + len(self.test_data)
        }
        
        return stats_dict
    
    def simulate_batch_on_demand(self, batch_size: Optional[int] = None) -> StreamingBatch:
        """Generate a single batch on demand for testing"""
        batch_size = batch_size or random.randint(*self.config['batch_size_range'])
        batch_size = min(batch_size, len(self.test_data))
        
        # Random sample from test data
        sample_indices = np.random.choice(len(self.test_data), batch_size, replace=False)
        batch_data = self.test_data.iloc[sample_indices].copy()
        
        # Add realistic modifications
        if self.config['add_noise']:
            batch_data = self._add_realistic_noise(batch_data)
        
        batch = StreamingBatch(
            batch_id=f"ondemand_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            interactions=batch_data[['user_id', 'movie_id', 'rating']],
            timestamp=datetime.now(),
            batch_size=batch_size,
            source_week=random.randint(10, 15),
            simulation_time=datetime.now().timestamp()
        )
        
        return batch

def main():
    """Test the temporal data simulator"""
    simulator = TemporalDataSimulator()
    
    print("Testing on-demand batch generation...")
    test_batch = simulator.simulate_batch_on_demand(batch_size=50)
    print(f"Generated batch: {test_batch.batch_id} with {test_batch.batch_size} interactions")
    print(f"Sample interactions:\n{test_batch.interactions.head()}")
    
    print("\nTesting streaming simulation (30 seconds)...")
    batch_count = 0
    for batch in simulator.simulate_streaming_data(duration_minutes=0.5):  # 30 seconds
        batch_count += 1
        print(f"Received batch {batch.batch_id}: {batch.batch_size} interactions from week {batch.source_week}")
        
        if batch_count >= 3:  # Limit for demo
            break
    
    # Print final statistics
    stats = simulator.get_streaming_stats()
    print(f"\nStreaming simulation stats:")
    for key, value in stats.items():
        if key not in ['config', 'data_source']:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()