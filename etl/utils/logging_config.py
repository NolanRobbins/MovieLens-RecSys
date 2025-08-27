#!/usr/bin/env python3
"""
Enhanced Logging Configuration for ETL Pipeline

Production-ready logging with structured output, metrics, and monitoring.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

class ETLFormatter(logging.Formatter):
    """Custom formatter for ETL pipeline logs with structured output"""
    
    def format(self, record):
        """Format log record with ETL-specific structure"""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add ETL-specific fields if available
        if hasattr(record, 'etl_context'):
            log_data.update(record.etl_context)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add metrics if present
        if hasattr(record, 'metrics'):
            log_data["metrics"] = record.metrics
        
        return json.dumps(log_data)


class ETLLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds ETL context to all log messages"""
    
    def __init__(self, logger, pipeline_context: Dict[str, Any]):
        super().__init__(logger, pipeline_context)
    
    def process(self, msg, kwargs):
        """Add ETL context to log record"""
        
        # Get the extra context
        extra = kwargs.get('extra', {})
        
        # Add pipeline context
        extra['etl_context'] = {
            'pipeline_name': self.extra.get('pipeline_name'),
            'pipeline_version': self.extra.get('pipeline_version'),
            'run_id': self.extra.get('run_id'),
            'day_number': self.extra.get('day_number'),
            **extra.get('etl_context', {})
        }
        
        kwargs['extra'] = extra
        return msg, kwargs
    
    def log_batch_start(self, day_number: int, batch_size: int):
        """Log batch processing start"""
        self.info(
            f"Starting batch processing for day {day_number}",
            extra={
                'etl_context': {
                    'event_type': 'batch_start',
                    'day_number': day_number,
                    'batch_size': batch_size
                }
            }
        )
    
    def log_batch_complete(self, day_number: int, processing_time: float, 
                          records_processed: int, data_quality: Dict):
        """Log batch processing completion"""
        self.info(
            f"Completed batch processing for day {day_number}",
            extra={
                'etl_context': {
                    'event_type': 'batch_complete',
                    'day_number': day_number,
                    'processing_time_seconds': processing_time,
                    'records_processed': records_processed
                },
                'metrics': {
                    'processing_time': processing_time,
                    'throughput': records_processed / processing_time if processing_time > 0 else 0,
                    'data_quality': data_quality
                }
            }
        )
    
    def log_data_quality_issue(self, issue_type: str, severity: str, details: Dict):
        """Log data quality issues"""
        log_level = getattr(logging, severity.upper(), logging.WARNING)
        
        self.log(
            log_level,
            f"Data quality issue detected: {issue_type}",
            extra={
                'etl_context': {
                    'event_type': 'data_quality_issue',
                    'issue_type': issue_type,
                    'severity': severity,
                    **details
                }
            }
        )
    
    def log_ab_test_result(self, winner: str, improvements: Dict):
        """Log A/B testing results"""
        self.info(
            f"A/B testing completed: {winner} wins",
            extra={
                'etl_context': {
                    'event_type': 'ab_test_result',
                    'winner': winner
                },
                'metrics': {
                    'improvements': improvements
                }
            }
        )


def setup_etl_logging(
    pipeline_name: str = "movieLens_etl",
    pipeline_version: str = "1.0.0",
    log_level: str = "INFO",
    log_dir: str = "etl/logs",
    run_id: str = None
) -> ETLLoggerAdapter:
    """
    Set up comprehensive logging for ETL pipeline
    
    Returns:
        ETLLoggerAdapter: Configured logger with ETL context
    """
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    logger = logging.getLogger(f"etl.{pipeline_name}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with simple format for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler with JSON format for production
    log_file = os.path.join(log_dir, f"{pipeline_name}_{run_id}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(ETLFormatter())
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Separate error log file
    error_file = os.path.join(log_dir, f"{pipeline_name}_errors_{run_id}.log")
    error_handler = logging.FileHandler(error_file)
    error_handler.setFormatter(ETLFormatter())
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # Create adapter with ETL context
    pipeline_context = {
        'pipeline_name': pipeline_name,
        'pipeline_version': pipeline_version,
        'run_id': run_id
    }
    
    etl_logger = ETLLoggerAdapter(logger, pipeline_context)
    
    # Log initialization
    etl_logger.info(
        f"ETL logging initialized for {pipeline_name} v{pipeline_version}",
        extra={
            'etl_context': {
                'event_type': 'logging_initialized',
                'log_level': log_level,
                'log_dir': log_dir,
                'log_file': log_file
            }
        }
    )
    
    return etl_logger


def setup_performance_logging(logger: ETLLoggerAdapter):
    """Set up performance monitoring hooks"""
    
    import time
    import psutil
    import threading
    
    def log_system_metrics():
        """Log system performance metrics periodically"""
        while True:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                logger.info(
                    "System performance metrics",
                    extra={
                        'etl_context': {
                            'event_type': 'system_metrics'
                        },
                        'metrics': {
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory.percent,
                            'memory_available_gb': memory.available / (1024**3),
                            'disk_free_gb': disk.free / (1024**3),
                            'disk_percent': disk.percent
                        }
                    }
                )
                
                time.sleep(30)  # Log every 30 seconds
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
                time.sleep(60)  # Wait longer on error
    
    # Start performance monitoring in background
    perf_thread = threading.Thread(target=log_system_metrics, daemon=True)
    perf_thread.start()
    
    logger.info("Performance monitoring started")


class ETLMetricsCollector:
    """Collect and aggregate ETL pipeline metrics"""
    
    def __init__(self, logger: ETLLoggerAdapter):
        self.logger = logger
        self.metrics = {
            'batches_processed': 0,
            'total_records': 0,
            'total_processing_time': 0.0,
            'data_quality_issues': 0,
            'errors': 0,
            'start_time': datetime.utcnow()
        }
    
    def increment_batch_count(self):
        """Increment batch processing count"""
        self.metrics['batches_processed'] += 1
    
    def add_records_processed(self, count: int):
        """Add to total records processed"""
        self.metrics['total_records'] += count
    
    def add_processing_time(self, time_seconds: float):
        """Add processing time"""
        self.metrics['total_processing_time'] += time_seconds
    
    def increment_data_quality_issues(self):
        """Increment data quality issue count"""
        self.metrics['data_quality_issues'] += 1
    
    def increment_errors(self):
        """Increment error count"""
        self.metrics['errors'] += 1
    
    def get_summary(self) -> Dict:
        """Get pipeline metrics summary"""
        elapsed = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        summary = {
            **self.metrics,
            'elapsed_time_seconds': elapsed,
            'avg_processing_time_per_batch': (
                self.metrics['total_processing_time'] / self.metrics['batches_processed']
                if self.metrics['batches_processed'] > 0 else 0
            ),
            'overall_throughput': (
                self.metrics['total_records'] / elapsed
                if elapsed > 0 else 0
            )
        }
        
        return summary
    
    def log_summary(self):
        """Log pipeline metrics summary"""
        summary = self.get_summary()
        
        self.logger.info(
            "ETL pipeline metrics summary",
            extra={
                'etl_context': {
                    'event_type': 'pipeline_summary'
                },
                'metrics': summary
            }
        )


def log_function_performance(func):
    """Decorator to log function performance"""
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Try to get logger from first argument (self)
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
                logger.debug(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'etl_context': {
                            'event_type': 'function_performance',
                            'function_name': func.__name__
                        },
                        'metrics': {
                            'execution_time_seconds': execution_time
                        }
                    }
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        'etl_context': {
                            'event_type': 'function_error',
                            'function_name': func.__name__,
                            'error': str(e)
                        },
                        'metrics': {
                            'execution_time_seconds': execution_time
                        }
                    },
                    exc_info=True
                )
            
            raise
    
    return wrapper


if __name__ == "__main__":
    # Test logging setup
    logger = setup_etl_logging(
        pipeline_name="test_pipeline",
        log_level="DEBUG"
    )
    
    # Test different log types
    logger.info("Testing ETL logging setup")
    logger.log_batch_start(1, 1000)
    logger.log_batch_complete(1, 45.2, 1000, {'completeness': 0.95})
    logger.log_data_quality_issue("missing_values", "WARNING", {'column': 'rating', 'missing_count': 5})
    logger.log_ab_test_result("ss4rec", {'rmse_improvement': 8.2})
    
    # Test metrics collector
    metrics = ETLMetricsCollector(logger)
    metrics.increment_batch_count()
    metrics.add_records_processed(1000)
    metrics.add_processing_time(45.2)
    metrics.log_summary()
    
    print("✅ ETL logging test completed successfully!")
