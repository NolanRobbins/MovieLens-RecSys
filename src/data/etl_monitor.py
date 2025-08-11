"""
ETL Pipeline Monitoring and Data Quality Dashboard
Tracks data quality metrics and pipeline performance over time
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PipelineRun:
    """Single ETL pipeline execution record"""
    timestamp: datetime
    status: str
    execution_time: float
    data_processed: Dict
    quality_metrics: Dict
    output_directory: str

class ETLMonitor:
    """
    Monitor ETL pipeline performance and data quality over time
    Useful for demonstrating production monitoring capabilities
    """
    
    def __init__(self, history_file: str = "etl_history.json"):
        self.history_file = Path(history_file)
        self.pipeline_history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load pipeline execution history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def record_pipeline_run(self, summary: Dict):
        """Record a new pipeline execution"""
        run_record = {
            'timestamp': summary.get('completed_at', datetime.now().isoformat()),
            'status': summary.get('status', 'UNKNOWN'),
            'execution_time': summary.get('execution_time_seconds', 0),
            'data_processed': summary.get('data_processed', {}),
            'quality_metrics': summary.get('quality_metrics', {}),
            'output_directory': summary.get('output_directory', ''),
        }
        
        self.pipeline_history.append(run_record)
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(self.pipeline_history, f, indent=2)
    
    def get_success_rate(self, days: int = 30) -> float:
        """Calculate pipeline success rate over specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_runs = [
            run for run in self.pipeline_history
            if datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_runs:
            return 0.0
        
        successful_runs = [run for run in recent_runs if run['status'] == 'SUCCESS']
        return len(successful_runs) / len(recent_runs) * 100
    
    def get_average_execution_time(self, days: int = 30) -> float:
        """Calculate average execution time over specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_runs = [
            run for run in self.pipeline_history
            if (datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) > cutoff_date 
                and run['status'] == 'SUCCESS')
        ]
        
        if not recent_runs:
            return 0.0
        
        execution_times = [run['execution_time'] for run in recent_runs]
        return sum(execution_times) / len(execution_times)
    
    def get_data_volume_trend(self, days: int = 30) -> pd.DataFrame:
        """Get data volume trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_runs = [
            run for run in self.pipeline_history
            if (datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) > cutoff_date 
                and run['status'] == 'SUCCESS')
        ]
        
        if not recent_runs:
            return pd.DataFrame()
        
        data = []
        for run in recent_runs:
            timestamp = run['timestamp']
            data_processed = run.get('data_processed', {})
            
            data.append({
                'timestamp': timestamp,
                'total_users': data_processed.get('total_users', 0),
                'total_movies': data_processed.get('total_movies', 0),
                'train_samples': data_processed.get('train_samples', 0),
                'val_samples': data_processed.get('val_samples', 0),
                'test_samples': data_processed.get('test_samples', 0)
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['total_samples'] = df['train_samples'] + df['val_samples'] + df['test_samples']
        
        return df
    
    def get_quality_metrics_trend(self, days: int = 30) -> pd.DataFrame:
        """Get data quality metrics trends"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_runs = [
            run for run in self.pipeline_history
            if (datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) > cutoff_date 
                and run['status'] == 'SUCCESS')
        ]
        
        if not recent_runs:
            return pd.DataFrame()
        
        data = []
        for run in recent_runs:
            timestamp = run['timestamp']
            quality = run.get('quality_metrics', {})
            
            data.append({
                'timestamp': timestamp,
                'total_ratings': quality.get('total_ratings', 0),
                'unique_users': quality.get('unique_users', 0),
                'unique_movies': quality.get('unique_movies', 0),
                'duplicate_count': quality.get('duplicate_count', 0),
                'data_freshness_days': quality.get('data_freshness_days', 0),
                'validation_passed': quality.get('validation_passed', False)
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['duplicate_rate'] = df['duplicate_count'] / df['total_ratings'] * 100
        
        return df
    
    def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report"""
        if not self.pipeline_history:
            return "No pipeline execution history available."
        
        # Recent performance metrics
        success_rate_7d = self.get_success_rate(7)
        success_rate_30d = self.get_success_rate(30)
        avg_execution_time = self.get_average_execution_time(30)
        
        # Latest run info
        latest_run = self.pipeline_history[-1]
        latest_status = latest_run['status']
        latest_timestamp = latest_run['timestamp']
        
        # Data volume trends
        volume_df = self.get_data_volume_trend(30)
        quality_df = self.get_quality_metrics_trend(30)
        
        report = f"""
📊 ETL PIPELINE MONITORING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚀 PIPELINE PERFORMANCE
├── Success Rate (7 days):  {success_rate_7d:.1f}%
├── Success Rate (30 days): {success_rate_30d:.1f}%
├── Avg Execution Time:     {avg_execution_time:.1f}s
└── Total Runs Recorded:    {len(self.pipeline_history)}

📈 LATEST RUN STATUS
├── Status: {latest_status}
├── Executed: {latest_timestamp}
└── Data Processed: {latest_run.get('data_processed', {}).get('train_samples', 'N/A')} training samples

📊 DATA VOLUME TRENDS (30 days)
"""
        
        if not volume_df.empty:
            latest_volume = volume_df.iloc[-1]
            report += f"""├── Latest Users:    {latest_volume['total_users']:,}
├── Latest Movies:   {latest_volume['total_movies']:,}
└── Latest Samples:  {latest_volume['total_samples']:,}
"""
        else:
            report += "└── No volume data available\n"
        
        report += "\n🔍 DATA QUALITY TRENDS (30 days)\n"
        
        if not quality_df.empty:
            latest_quality = quality_df.iloc[-1]
            report += f"""├── Duplicate Rate:     {latest_quality.get('duplicate_rate', 0):.2f}%
├── Data Freshness:     {latest_quality.get('data_freshness_days', 0):.1f} days
└── Validation Status:  {'✅ PASS' if latest_quality.get('validation_passed') else '❌ FAIL'}
"""
        else:
            report += "└── No quality data available\n"
        
        # Recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        
        if success_rate_7d < 90:
            report += "⚠️  Success rate below 90% - investigate recent failures\n"
        
        if avg_execution_time > 300:  # 5 minutes
            report += "⚠️  Execution time increasing - consider optimization\n"
        
        if not quality_df.empty and quality_df['duplicate_rate'].iloc[-1] > 1:
            report += "⚠️  High duplicate rate detected - check data source\n"
        
        if len([r for r in self.pipeline_history[-5:] if r['status'] != 'SUCCESS']) > 1:
            report += "⚠️  Multiple recent failures - urgent attention needed\n"
        
        if not quality_df.empty and quality_df['data_freshness_days'].iloc[-1] > 7:
            report += "⚠️  Data getting stale - check upstream data sources\n"
        
        return report
    
    def create_dashboard(self, save_path: str = "etl_dashboard.png"):
        """Create visual dashboard for ETL monitoring"""
        if not self.pipeline_history:
            print("No data available for dashboard")
            return
        
        # Get recent data
        volume_df = self.get_data_volume_trend(30)
        quality_df = self.get_quality_metrics_trend(30)
        
        if volume_df.empty and quality_df.empty:
            print("No recent data available for dashboard")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ETL Pipeline Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Execution time trend
        if len(self.pipeline_history) > 1:
            timestamps = [datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) 
                         for run in self.pipeline_history[-30:]]
            exec_times = [run['execution_time'] for run in self.pipeline_history[-30:]]
            
            axes[0, 0].plot(timestamps, exec_times, marker='o', linewidth=2, markersize=4)
            axes[0, 0].set_title('Execution Time Trend')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Data volume trend
        if not volume_df.empty:
            axes[0, 1].plot(volume_df['timestamp'], volume_df['total_samples'], 
                           marker='o', linewidth=2, markersize=4, color='green')
            axes[0, 1].set_title('Data Volume Trend')
            axes[0, 1].set_ylabel('Total Samples')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate over time
        success_history = []
        for i in range(len(self.pipeline_history)):
            recent_runs = self.pipeline_history[max(0, i-6):i+1]  # 7-day window
            success_count = sum(1 for run in recent_runs if run['status'] == 'SUCCESS')
            success_rate = success_count / len(recent_runs) * 100 if recent_runs else 0
            success_history.append(success_rate)
        
        if len(success_history) > 1:
            timestamps = [datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')) 
                         for run in self.pipeline_history]
            axes[1, 0].plot(timestamps, success_history, marker='o', linewidth=2, 
                           markersize=4, color='blue')
            axes[1, 0].set_title('Success Rate (7-day rolling)')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_ylim(0, 105)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Data quality metrics
        if not quality_df.empty:
            axes[1, 1].plot(quality_df['timestamp'], quality_df['duplicate_rate'], 
                           marker='o', linewidth=2, markersize=4, color='red', 
                           label='Duplicate Rate %')
            axes[1, 1].set_title('Data Quality Metrics')
            axes[1, 1].set_ylabel('Duplicate Rate (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to {save_path}")

def main():
    """Command-line interface for ETL monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ETL Pipeline Monitor')
    parser.add_argument('--history', type=str, default='etl_history.json', 
                       help='Pipeline history file')
    parser.add_argument('--report', action='store_true', 
                       help='Generate monitoring report')
    parser.add_argument('--dashboard', type=str, 
                       help='Generate dashboard (specify output file)')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days for analysis')
    
    args = parser.parse_args()
    
    monitor = ETLMonitor(history_file=args.history)
    
    if args.report:
        report = monitor.generate_monitoring_report()
        print(report)
    
    if args.dashboard:
        monitor.create_dashboard(save_path=args.dashboard)

if __name__ == "__main__":
    main()