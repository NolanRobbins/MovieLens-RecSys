"""
Performance Optimization Suite
Advanced caching, batching, and optimization techniques for recommendation systems
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from collections import defaultdict, deque, OrderedDict
import hashlib
import pickle
import psutil
import torch
from functools import wraps, lru_cache
import diskcache as dc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used  
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns
    HIERARCHICAL = "hierarchical"  # Multi-level cache

class BatchStrategy(Enum):
    """Batching strategies"""
    FIXED_SIZE = "fixed_size"      # Fixed batch size
    ADAPTIVE_SIZE = "adaptive_size" # Dynamic batch size
    TIME_BASED = "time_based"      # Time window batching
    PRIORITY_BASED = "priority"    # Priority-based batching

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    
    # Latency metrics
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    batch_processing_rate: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_eviction_rate: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_rate: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Custom metrics
    model_inference_time_ms: float = 0.0
    data_loading_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    priority: int = 0

class AdaptiveCache:
    """
    Multi-level adaptive caching system with intelligent eviction
    """
    
    def __init__(self, max_memory_mb: int = 1000, 
                 disk_cache_mb: int = 10000,
                 default_ttl: int = 3600):
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        self.default_ttl = default_ttl
        
        # Memory cache (L1)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        
        # Disk cache (L2)
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache = dc.Cache(str(cache_dir), size_limit=disk_cache_mb * 1024 * 1024)
        
        # Cache statistics
        self.stats = {
            'hits': 0, 'misses': 0, 'evictions': 0,
            'memory_hits': 0, 'disk_hits': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"💾 Adaptive cache initialized: {max_memory_mb}MB memory, {disk_cache_mb}MB disk")
    
    def _generate_key(self, key_parts: List[Any]) -> str:
        """Generate consistent cache key"""
        key_str = ":".join(str(part) for part in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.now() - entry.created_at).total_seconds()
        return age_seconds > entry.ttl_seconds
    
    def _evict_from_memory(self):
        """Evict items from memory cache"""
        with self._lock:
            while (self.current_memory_usage > self.max_memory_bytes * 0.8 and 
                   self.memory_cache):
                
                # Use adaptive eviction strategy
                if len(self.memory_cache) < 100:
                    # LRU for small cache
                    key_to_evict = self.access_order.popleft()
                else:
                    # LFU for larger cache
                    key_to_evict = min(self.memory_cache.keys(), 
                                     key=lambda k: self.access_frequency[k])
                
                if key_to_evict in self.memory_cache:
                    entry = self.memory_cache[key_to_evict]
                    
                    # Move to disk cache if valuable
                    if entry.access_count > 1:
                        self.disk_cache[key_to_evict] = entry.value
                    
                    # Remove from memory
                    self.current_memory_usage -= entry.size_bytes
                    del self.memory_cache[key_to_evict]
                    self.access_frequency[key_to_evict] = 0
                    self.stats['evictions'] += 1
    
    def get(self, key_parts: List[Any]) -> Optional[Any]:
        """Get item from cache"""
        key = self._generate_key(key_parts)
        
        with self._lock:
            # Check memory cache first (L1)
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                if self._is_expired(entry):
                    del self.memory_cache[key]
                    self.current_memory_usage -= entry.size_bytes
                    self.stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.access_frequency[key] += 1
                self.access_order.append(key)
                
                self.stats['hits'] += 1
                self.stats['memory_hits'] += 1
                return entry.value
            
            # Check disk cache (L2)
            if key in self.disk_cache:
                value = self.disk_cache[key]
                
                # Promote to memory cache
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=self._estimate_size(value)
                )
                
                # Ensure memory space
                self._evict_from_memory()
                
                # Add to memory
                self.memory_cache[key] = entry
                self.current_memory_usage += entry.size_bytes
                self.access_frequency[key] += 1
                self.access_order.append(key)
                
                self.stats['hits'] += 1
                self.stats['disk_hits'] += 1
                return value
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key_parts: List[Any], value: Any, ttl_seconds: Optional[int] = None):
        """Put item in cache"""
        key = self._generate_key(key_parts)
        size_bytes = self._estimate_size(value)
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl
            )
            
            # Ensure memory space
            while (self.current_memory_usage + size_bytes > self.max_memory_bytes and 
                   self.memory_cache):
                self._evict_from_memory()
            
            # Add to memory cache
            self.memory_cache[key] = entry
            self.current_memory_usage += size_bytes
            self.access_order.append(key)
            self.access_frequency[key] += 1
    
    def invalidate(self, key_parts: List[Any]):
        """Invalidate cache entry"""
        key = self._generate_key(key_parts)
        
        with self._lock:
            # Remove from memory
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                self.current_memory_usage -= entry.size_bytes
                del self.memory_cache[key]
            
            # Remove from disk
            if key in self.disk_cache:
                del self.disk_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': 1 - hit_rate,
                'memory_hit_rate': self.stats['memory_hits'] / max(1, total_requests),
                'disk_hit_rate': self.stats['disk_hits'] / max(1, total_requests),
                'eviction_count': self.stats['evictions'],
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'memory_entries': len(self.memory_cache),
                'disk_entries': len(self.disk_cache)
            }

class BatchProcessor:
    """
    Intelligent batch processing system for recommendations
    """
    
    def __init__(self, max_batch_size: int = 64, 
                 max_wait_time_ms: int = 50,
                 strategy: BatchStrategy = BatchStrategy.ADAPTIVE_SIZE):
        
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.strategy = strategy
        
        # Batching queues
        self.request_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        
        # Processing statistics
        self.stats = {
            'batches_processed': 0,
            'requests_processed': 0,
            'avg_batch_size': 0.0,
            'batch_utilization': 0.0
        }
        
        # Worker threads
        self.batch_builder_thread = None
        self.batch_processor_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State
        self.running = False
        
        logger.info(f"⚡ Batch processor initialized: max_size={max_batch_size}, strategy={strategy.value}")
    
    def start(self):
        """Start batch processing"""
        if self.running:
            return
        
        self.running = True
        
        # Start batch builder
        self.batch_builder_thread = threading.Thread(
            target=self._batch_builder_loop,
            name="BatchBuilder",
            daemon=True
        )
        self.batch_builder_thread.start()
        
        # Start batch processor
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_loop,
            name="BatchProcessor", 
            daemon=True
        )
        self.batch_processor_thread.start()
        
        logger.info("🚀 Batch processor started")
    
    def stop(self):
        """Stop batch processing"""
        self.running = False
        if self.batch_builder_thread:
            self.batch_builder_thread.join(timeout=1.0)
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=1.0)
        
        self.executor.shutdown(wait=True)
        logger.info("🛑 Batch processor stopped")
    
    def submit_request(self, request_data: Any, 
                      processor_func: Callable,
                      priority: int = 0) -> Any:
        """Submit request for batch processing"""
        
        result_queue = queue.Queue()
        
        batch_request = {
            'data': request_data,
            'processor': processor_func,
            'priority': priority,
            'submitted_at': time.time(),
            'result_queue': result_queue
        }
        
        self.request_queue.put(batch_request)
        
        # Wait for result
        try:
            result = result_queue.get(timeout=5.0)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("Batch processing timeout")
    
    def _batch_builder_loop(self):
        """Build batches from incoming requests"""
        
        while self.running:
            try:
                current_batch = []
                batch_start_time = time.time()
                
                # Collect requests for batch
                while (len(current_batch) < self.max_batch_size and
                       (time.time() - batch_start_time) * 1000 < self.max_wait_time_ms):
                    
                    try:
                        request = self.request_queue.get(timeout=0.01)
                        current_batch.append(request)
                    except queue.Empty:
                        continue
                
                # Process batch if not empty
                if current_batch:
                    # Optimize batch based on strategy
                    if self.strategy == BatchStrategy.PRIORITY_BASED:
                        current_batch.sort(key=lambda x: x['priority'], reverse=True)
                    elif self.strategy == BatchStrategy.ADAPTIVE_SIZE:
                        # Adjust batch size based on queue length
                        queue_size = self.request_queue.qsize()
                        if queue_size > 100:
                            current_batch = current_batch[:self.max_batch_size]
                        elif queue_size < 10:
                            current_batch = current_batch[:max(1, self.max_batch_size // 2)]
                    
                    self.batch_queue.put(current_batch)
                
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"❌ Batch builder error: {e}")
    
    def _batch_processor_loop(self):
        """Process batches"""
        
        while self.running:
            try:
                batch = self.batch_queue.get(timeout=1.0)
                
                # Submit batch for processing
                future = self.executor.submit(self._process_batch, batch)
                
                self.stats['batches_processed'] += 1
                self.stats['requests_processed'] += len(batch)
                
                # Update statistics
                if self.stats['batches_processed'] > 0:
                    self.stats['avg_batch_size'] = (
                        self.stats['requests_processed'] / self.stats['batches_processed']
                    )
                    self.stats['batch_utilization'] = (
                        self.stats['avg_batch_size'] / self.max_batch_size
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Batch processor error: {e}")
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests"""
        
        try:
            # Group by processor function
            processor_groups = defaultdict(list)
            for request in batch:
                processor_groups[request['processor']].append(request)
            
            # Process each group
            for processor_func, requests in processor_groups.items():
                batch_data = [req['data'] for req in requests]
                
                # Call batch processor
                try:
                    results = processor_func(batch_data)
                    
                    # Return results to individual requests
                    for request, result in zip(requests, results):
                        request['result_queue'].put(result)
                        
                except Exception as e:
                    # Return error to all requests in this group
                    for request in requests:
                        request['result_queue'].put(e)
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            # Return error to all requests
            for request in batch:
                request['result_queue'].put(e)

class ModelOptimizer:
    """
    Model-specific optimization techniques
    """
    
    def __init__(self):
        self.compiled_models = {}
        self.quantized_models = {}
        self.onnx_models = {}
        
    def optimize_pytorch_model(self, model: torch.nn.Module, 
                             model_name: str,
                             optimization_level: str = "basic") -> torch.nn.Module:
        """Optimize PyTorch model for inference"""
        
        logger.info(f"🔧 Optimizing PyTorch model: {model_name}")
        
        optimized_model = model.eval()
        
        if optimization_level in ["basic", "advanced"]:
            # Script the model for better performance
            try:
                example_input = torch.randn(1, 100)  # Adjust based on model
                scripted_model = torch.jit.trace(optimized_model, example_input)
                optimized_model = scripted_model
                logger.info("✅ Model scripted with TorchScript")
            except Exception as e:
                logger.warning(f"⚠️ TorchScript failed: {e}")
        
        if optimization_level == "advanced":
            # Model quantization for CPU inference
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    optimized_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                self.quantized_models[model_name] = quantized_model
                logger.info("✅ Model quantized for CPU inference")
            except Exception as e:
                logger.warning(f"⚠️ Quantization failed: {e}")
        
        return optimized_model
    
    def precompute_embeddings(self, model: torch.nn.Module,
                            user_ids: List[int], 
                            item_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Precompute user and item embeddings"""
        
        logger.info("🔄 Precomputing embeddings for fast inference...")
        
        embeddings = {'users': {}, 'items': {}}
        
        model.eval()
        with torch.no_grad():
            # Precompute user embeddings
            if hasattr(model, 'user_embedding'):
                user_tensor = torch.tensor(user_ids)
                user_embeds = model.user_embedding(user_tensor)
                for i, user_id in enumerate(user_ids):
                    embeddings['users'][user_id] = user_embeds[i]
            
            # Precompute item embeddings
            if hasattr(model, 'item_embedding'):
                item_tensor = torch.tensor(item_ids)
                item_embeds = model.item_embedding(item_tensor)
                for i, item_id in enumerate(item_ids):
                    embeddings['items'][item_id] = item_embeds[i]
        
        logger.info(f"✅ Precomputed {len(embeddings['users'])} user and {len(embeddings['items'])} item embeddings")
        
        return embeddings

class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting
    """
    
    def __init__(self, monitoring_interval: int = 10):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # Thresholds
        self.thresholds = {
            'max_response_time_ms': 1000,
            'min_cache_hit_rate': 0.8,
            'max_cpu_usage': 80.0,
            'max_memory_usage': 85.0,
            'max_error_rate': 0.05
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info("📊 Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                metrics.timestamp = datetime.now()
                self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                self._check_thresholds(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) if disk_io else 0
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_io_rate=disk_io_rate
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check metrics against thresholds"""
        
        # Response time check
        if metrics.avg_response_time_ms > self.thresholds['max_response_time_ms']:
            self._generate_alert(
                'high_response_time',
                f"Average response time {metrics.avg_response_time_ms:.1f}ms exceeds threshold {self.thresholds['max_response_time_ms']}ms"
            )
        
        # CPU usage check
        if metrics.cpu_usage_percent > self.thresholds['max_cpu_usage']:
            self._generate_alert(
                'high_cpu_usage',
                f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds threshold {self.thresholds['max_cpu_usage']}%"
            )
        
        # Memory usage check
        if metrics.memory_usage_percent > self.thresholds['max_memory_usage']:
            self._generate_alert(
                'high_memory_usage',
                f"Memory usage {metrics.memory_usage_percent:.1f}% exceeds threshold {self.thresholds['max_memory_usage']}%"
            )
        
        # Cache hit rate check
        if metrics.cache_hit_rate < self.thresholds['min_cache_hit_rate']:
            self._generate_alert(
                'low_cache_hit_rate',
                f"Cache hit rate {metrics.cache_hit_rate:.2%} below threshold {self.thresholds['min_cache_hit_rate']:.2%}"
            )
    
    def _generate_alert(self, alert_type: str, message: str):
        """Generate performance alert"""
        
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        logger.warning(f"⚠️ ALERT [{alert_type}]: {message}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts 
                if alert['timestamp'] >= cutoff_time]

class PerformanceOptimizer:
    """
    Comprehensive performance optimization system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize components
        self.cache = AdaptiveCache(
            max_memory_mb=config.get('cache_memory_mb', 1000),
            disk_cache_mb=config.get('cache_disk_mb', 10000)
        )
        
        self.batch_processor = BatchProcessor(
            max_batch_size=config.get('max_batch_size', 64),
            max_wait_time_ms=config.get('max_wait_time_ms', 50)
        )
        
        self.model_optimizer = ModelOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.start_time = time.time()
        
        logger.info("🚀 Performance Optimizer initialized")
    
    def start(self):
        """Start all optimization components"""
        self.batch_processor.start()
        self.performance_monitor.start_monitoring()
        logger.info("✅ All optimization components started")
    
    def stop(self):
        """Stop all optimization components"""
        self.batch_processor.stop()
        self.performance_monitor.stop_monitoring()
        logger.info("🛑 All optimization components stopped")
    
    def cached_prediction(self, cache_key_parts: List[Any],
                         prediction_func: Callable,
                         ttl_seconds: int = 3600) -> Any:
        """Cached prediction with automatic cache management"""
        
        # Check cache first
        cached_result = self.cache.get(cache_key_parts)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache result
        result = prediction_func()
        self.cache.put(cache_key_parts, result, ttl_seconds)
        
        return result
    
    def batch_predictions(self, requests: List[Any],
                         batch_processor_func: Callable) -> List[Any]:
        """Process predictions in optimized batches"""
        
        return self.batch_processor.submit_request(
            requests, 
            batch_processor_func
        )
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance"""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                self.request_times.append(execution_time)
        
        return wrapper
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        uptime_seconds = time.time() - self.start_time
        
        # Request performance
        if self.request_times:
            avg_time = np.mean(self.request_times)
            p95_time = np.percentile(self.request_times, 95)
            p99_time = np.percentile(self.request_times, 99)
        else:
            avg_time = p95_time = p99_time = 0.0
        
        return {
            'uptime_seconds': uptime_seconds,
            'cache_stats': self.cache.get_stats(),
            'batch_stats': self.batch_processor.stats,
            'request_performance': {
                'avg_response_time_ms': avg_time,
                'p95_response_time_ms': p95_time,
                'p99_response_time_ms': p99_time,
                'total_requests': len(self.request_times)
            },
            'system_metrics': self.performance_monitor.get_current_metrics(),
            'recent_alerts': self.performance_monitor.get_alerts(hours=1)
        }

def performance_test():
    """Test the performance optimization system"""
    print("⚡ Testing Performance Optimization System...")
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer({
        'cache_memory_mb': 100,
        'cache_disk_mb': 500,
        'max_batch_size': 32
    })
    
    optimizer.start()
    
    try:
        # Test caching
        print("\n💾 Testing caching system...")
        
        def expensive_computation(x):
            time.sleep(0.01)  # Simulate computation
            return x ** 2
        
        # Cache miss
        start_time = time.time()
        result1 = optimizer.cached_prediction(['user', 123], lambda: expensive_computation(123))
        cache_miss_time = time.time() - start_time
        
        # Cache hit
        start_time = time.time()
        result2 = optimizer.cached_prediction(['user', 123], lambda: expensive_computation(123))
        cache_hit_time = time.time() - start_time
        
        print(f"Cache miss time: {cache_miss_time*1000:.2f}ms")
        print(f"Cache hit time: {cache_hit_time*1000:.2f}ms")
        print(f"Speedup: {cache_miss_time/cache_hit_time:.1f}x")
        
        # Test batch processing
        print("\n⚡ Testing batch processing...")
        
        def batch_process(requests):
            # Simulate batch computation
            time.sleep(0.005 * len(requests))
            return [req * 2 for req in requests]
        
        # Process individual requests
        individual_times = []
        for i in range(20):
            start_time = time.time()
            optimizer.batch_processor.submit_request(i, batch_process)
            individual_times.append(time.time() - start_time)
        
        print(f"Average individual request time: {np.mean(individual_times)*1000:.2f}ms")
        
        # Wait for batches to complete
        time.sleep(1.0)
        
        # Get performance summary
        summary = optimizer.get_performance_summary()
        
        print(f"\n📊 Performance Summary:")
        print(f"Cache hit rate: {summary['cache_stats']['hit_rate']:.2%}")
        print(f"Memory cache entries: {summary['cache_stats']['memory_entries']}")
        print(f"Batches processed: {summary['batch_stats']['batches_processed']}")
        print(f"Average batch size: {summary['batch_stats']['avg_batch_size']:.1f}")
        print(f"Batch utilization: {summary['batch_stats']['batch_utilization']:.2%}")
        
        if summary['recent_alerts']:
            print(f"\n⚠️ Recent alerts: {len(summary['recent_alerts'])}")
            for alert in summary['recent_alerts'][:3]:
                print(f"  - {alert['type']}: {alert['message']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        optimizer.stop()
    
    print("\n✅ Performance optimization system ready!")

def main():
    """Run performance optimization tests"""
    performance_test()

if __name__ == "__main__":
    main()