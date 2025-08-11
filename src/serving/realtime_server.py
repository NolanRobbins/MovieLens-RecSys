"""
Real-time Recommendation Serving System
High-performance, low-latency recommendation server with caching and optimization
"""

import asyncio
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServingMode(Enum):
    """Serving mode configurations"""
    LOW_LATENCY = "low_latency"    # <10ms, pre-computed recommendations
    REAL_TIME = "real_time"        # <50ms, lightweight models
    CONTEXTUAL = "contextual"      # <100ms, full contextual features
    BATCH = "batch"                # High throughput batch processing

@dataclass
class RecommendationRequest:
    """Recommendation request structure"""
    user_id: int
    request_id: str
    timestamp: datetime
    n_recommendations: int = 10
    context: Optional[Dict[str, Any]] = None
    candidate_filter: Optional[List[int]] = None
    exclude_seen: bool = True
    serving_mode: ServingMode = ServingMode.REAL_TIME

@dataclass
class RecommendationResponse:
    """Recommendation response structure"""
    request_id: str
    user_id: int
    recommendations: List[Dict[str, Any]]
    serving_mode: ServingMode
    response_time_ms: float
    model_version: str
    cache_hit: bool
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    error_count: int = 0
    throughput_per_second: float = 0.0

class RecommendationCache:
    """
    High-performance caching system for recommendations
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_order = deque()
        self._lock = threading.RLock()
    
    def _generate_key(self, user_id: int, n_recommendations: int, 
                     context_hash: Optional[str] = None) -> str:
        """Generate cache key"""
        key_parts = [str(user_id), str(n_recommendations)]
        if context_hash:
            key_parts.append(context_hash)
        return ":".join(key_parts)
    
    def _hash_context(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Hash context for caching"""
        if not context:
            return None
        
        # Sort keys for consistent hashing
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def get(self, user_id: int, n_recommendations: int, 
            context: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendations"""
        context_hash = self._hash_context(context)
        key = self._generate_key(user_id, n_recommendations, context_hash)
        
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            access_time = self.access_times.get(key, 0)
            if time.time() - access_time > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            # Update access
            self.access_times[key] = time.time()
            self.access_order.append(key)
            
            return self.cache[key]
    
    def put(self, user_id: int, n_recommendations: int, 
            recommendations: List[Dict[str, Any]], 
            context: Optional[Dict[str, Any]] = None):
        """Cache recommendations"""
        context_hash = self._hash_context(context)
        key = self._generate_key(user_id, n_recommendations, context_hash)
        
        with self._lock:
            # Evict if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                self._remove_key(oldest_key)
            
            # Store
            self.cache[key] = recommendations
            self.access_times[key] = time.time()
            self.access_order.append(key)
    
    def _remove_key(self, key: str):
        """Remove key from cache"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would track in real implementation
                'ttl_seconds': self.ttl_seconds
            }

class ModelServer:
    """
    Model serving component with multiple model support
    """
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.default_model = None
        self._lock = threading.RLock()
    
    def load_model(self, model_name: str, model_path: str, 
                   is_default: bool = False):
        """Load a model for serving"""
        try:
            logger.info(f"📁 Loading model: {model_name} from {model_path}")
            
            # Load model based on file extension
            if model_path.endswith('.pt'):
                model_data = torch.load(model_path, map_location='cpu')
                model = model_data  # Simplified for this example
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            with self._lock:
                self.models[model_name] = model
                self.model_metadata[model_name] = {
                    'path': model_path,
                    'loaded_at': datetime.now(),
                    'version': model_data.get('version', '1.0.0'),
                    'performance': model_data.get('performance_metrics', {})
                }
                
                if is_default or not self.default_model:
                    self.default_model = model_name
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def get_model(self, model_name: Optional[str] = None):
        """Get a model for inference"""
        with self._lock:
            if model_name and model_name in self.models:
                return self.models[model_name]
            elif self.default_model and self.default_model in self.models:
                return self.models[self.default_model]
            else:
                raise ValueError("No suitable model available")
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model metadata"""
        model_name = model_name or self.default_model
        with self._lock:
            if model_name in self.model_metadata:
                return self.model_metadata[model_name]
            else:
                return {}

class RecommendationEngine:
    """
    Core recommendation engine with multiple serving modes
    """
    
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server
        self.user_embeddings = {}  # Pre-computed user embeddings
        self.item_embeddings = {}  # Pre-computed item embeddings
        self.popular_items = []    # Popular items for cold start
        self._lock = threading.RLock()
    
    def precompute_embeddings(self, user_ids: List[int], item_ids: List[int]):
        """Pre-compute embeddings for fast serving"""
        logger.info("🔄 Pre-computing embeddings for fast serving...")
        
        try:
            model = self.model_server.get_model()
            
            with self._lock:
                # Pre-compute user embeddings
                for user_id in user_ids:
                    # Simplified embedding computation
                    embedding = np.random.normal(0, 1, 64)  # Placeholder
                    self.user_embeddings[user_id] = embedding
                
                # Pre-compute item embeddings
                for item_id in item_ids:
                    embedding = np.random.normal(0, 1, 64)  # Placeholder
                    self.item_embeddings[item_id] = embedding
            
            logger.info(f"✅ Pre-computed {len(user_ids)} user and {len(item_ids)} item embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to pre-compute embeddings: {e}")
    
    def set_popular_items(self, item_ids: List[int]):
        """Set popular items for cold start recommendations"""
        with self._lock:
            self.popular_items = item_ids[:50]  # Keep top 50
        logger.info(f"📊 Set {len(self.popular_items)} popular items for cold start")
    
    def generate_recommendations_fast(self, user_id: int, n_recommendations: int,
                                    candidate_items: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Generate recommendations using pre-computed embeddings (LOW_LATENCY mode)"""
        
        with self._lock:
            if user_id not in self.user_embeddings:
                # Cold start - return popular items
                recommendations = []
                for item_id in self.popular_items[:n_recommendations]:
                    recommendations.append({
                        'item_id': item_id,
                        'score': 0.5,  # Default score
                        'reason': 'popular_item'
                    })
                return recommendations
            
            user_embedding = self.user_embeddings[user_id]
            
            # Compute similarities with candidate items
            candidates = candidate_items or list(self.item_embeddings.keys())
            similarities = []
            
            for item_id in candidates:
                if item_id in self.item_embeddings:
                    item_embedding = self.item_embeddings[item_id]
                    # Cosine similarity
                    similarity = np.dot(user_embedding, item_embedding) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
                    )
                    similarities.append((item_id, similarity))
            
            # Sort and return top recommendations
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in similarities[:n_recommendations]:
                recommendations.append({
                    'item_id': item_id,
                    'score': float(score),
                    'reason': 'collaborative_filtering'
                })
            
            return recommendations
    
    def generate_recommendations_realtime(self, user_id: int, n_recommendations: int,
                                        context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate recommendations with lightweight model (REAL_TIME mode)"""
        
        try:
            model = self.model_server.get_model()
            
            # Simplified real-time inference
            recommendations = []
            
            # Mock recommendation generation
            for i in range(n_recommendations):
                item_id = np.random.randint(1, 1000)
                score = np.random.beta(2, 5)  # Realistic score distribution
                
                recommendations.append({
                    'item_id': item_id,
                    'score': float(score),
                    'reason': 'realtime_model'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Real-time recommendation failed: {e}")
            # Fallback to popular items
            return self.generate_recommendations_fast(user_id, n_recommendations)
    
    def generate_recommendations_contextual(self, user_id: int, n_recommendations: int,
                                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contextual recommendations (CONTEXTUAL mode)"""
        
        try:
            # Full contextual recommendation logic would go here
            # For now, return enhanced real-time recommendations
            recommendations = self.generate_recommendations_realtime(
                user_id, n_recommendations, context
            )
            
            # Add contextual boost
            for rec in recommendations:
                rec['score'] *= 1.1  # Context boost
                rec['reason'] = 'contextual_model'
                rec['context_applied'] = True
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Contextual recommendation failed: {e}")
            return self.generate_recommendations_realtime(user_id, n_recommendations)

class RealTimeRecommendationServer:
    """
    High-performance real-time recommendation server
    """
    
    def __init__(self, cache_size: int = 10000, cache_ttl: int = 3600):
        self.model_server = ModelServer()
        self.engine = RecommendationEngine(self.model_server)
        self.cache = RecommendationCache(cache_size, cache_ttl)
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.response_times = deque(maxlen=1000)
        self._metrics_lock = threading.Lock()
        
        # Request processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.request_queue = queue.Queue(maxsize=1000)
        
        # Start background workers
        self._start_workers()
        
        logger.info("🚀 Real-time recommendation server initialized")
    
    def _start_workers(self):
        """Start background workers for request processing"""
        for i in range(4):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"RecommendationWorker-{i}",
                daemon=True
            )
            worker_thread.start()
    
    def _worker_loop(self):
        """Background worker for processing recommendations"""
        while True:
            try:
                request, result_queue = self.request_queue.get(timeout=1.0)
                response = self._process_request(request)
                result_queue.put(response)
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
    
    def load_models(self, model_configs: List[Dict[str, Any]]):
        """Load multiple models for serving"""
        for config in model_configs:
            self.model_server.load_model(
                model_name=config['name'],
                model_path=config['path'],
                is_default=config.get('is_default', False)
            )
    
    def initialize_cache(self, user_ids: List[int], item_ids: List[int],
                        popular_items: List[int]):
        """Initialize server with pre-computed data"""
        logger.info("🔧 Initializing server cache and embeddings...")
        
        # Pre-compute embeddings
        self.engine.precompute_embeddings(user_ids, item_ids)
        
        # Set popular items
        self.engine.set_popular_items(popular_items)
        
        logger.info("✅ Server initialization complete")
    
    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get recommendations asynchronously"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_recs = self.cache.get(
                request.user_id, 
                request.n_recommendations, 
                request.context
            )
            
            if cached_recs is not None:
                response_time = (time.time() - start_time) * 1000
                self._update_metrics(response_time, cache_hit=True)
                
                return RecommendationResponse(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    recommendations=cached_recs,
                    serving_mode=request.serving_mode,
                    response_time_ms=response_time,
                    model_version=self.model_server.get_model_info().get('version', '1.0.0'),
                    cache_hit=True,
                    timestamp=datetime.now()
                )
            
            # Process request
            result_queue = queue.Queue()
            self.request_queue.put((request, result_queue))
            
            # Wait for result
            response = result_queue.get(timeout=5.0)  # 5 second timeout
            
            # Cache the result
            self.cache.put(
                request.user_id,
                request.n_recommendations,
                response.recommendations,
                request.context
            )
            
            response_time = (time.time() - start_time) * 1000
            response.response_time_ms = response_time
            self._update_metrics(response_time, cache_hit=False)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Recommendation request failed: {e}")
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(response_time, cache_hit=False, error=True)
            raise
    
    def _process_request(self, request: RecommendationRequest) -> RecommendationResponse:
        """Process a recommendation request"""
        
        # Choose serving strategy based on mode
        if request.serving_mode == ServingMode.LOW_LATENCY:
            recommendations = self.engine.generate_recommendations_fast(
                request.user_id, request.n_recommendations
            )
        elif request.serving_mode == ServingMode.CONTEXTUAL and request.context:
            recommendations = self.engine.generate_recommendations_contextual(
                request.user_id, request.n_recommendations, request.context
            )
        else:  # REAL_TIME mode (default)
            recommendations = self.engine.generate_recommendations_realtime(
                request.user_id, request.n_recommendations, request.context
            )
        
        return RecommendationResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            serving_mode=request.serving_mode,
            response_time_ms=0.0,  # Will be set by caller
            model_version=self.model_server.get_model_info().get('version', '1.0.0'),
            cache_hit=False,
            timestamp=datetime.now()
        )
    
    def _update_metrics(self, response_time_ms: float, cache_hit: bool = False, 
                       error: bool = False):
        """Update performance metrics"""
        with self._metrics_lock:
            self.metrics.total_requests += 1
            
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            if error:
                self.metrics.error_count += 1
            
            self.response_times.append(response_time_ms)
            
            # Update averages
            if len(self.response_times) > 0:
                self.metrics.avg_response_time_ms = np.mean(self.response_times)
                self.metrics.p95_response_time_ms = np.percentile(self.response_times, 95)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._metrics_lock:
            cache_rate = (self.metrics.cache_hits / 
                         max(1, self.metrics.total_requests) * 100)
            error_rate = (self.metrics.error_count / 
                         max(1, self.metrics.total_requests) * 100)
            
            return {
                'total_requests': self.metrics.total_requests,
                'cache_hit_rate_percent': cache_rate,
                'error_rate_percent': error_rate,
                'avg_response_time_ms': self.metrics.avg_response_time_ms,
                'p95_response_time_ms': self.metrics.p95_response_time_ms,
                'cache_stats': self.cache.get_stats(),
                'models_loaded': len(self.model_server.models)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Server health check"""
        try:
            # Test model availability
            model = self.model_server.get_model()
            models_healthy = model is not None
            
            # Test cache
            cache_healthy = len(self.cache.cache) < self.cache.max_size
            
            # Test queue
            queue_healthy = self.request_queue.qsize() < 500  # Not overwhelmed
            
            overall_health = models_healthy and cache_healthy and queue_healthy
            
            return {
                'status': 'healthy' if overall_health else 'degraded',
                'models_loaded': len(self.model_server.models),
                'cache_size': len(self.cache.cache),
                'queue_size': self.request_queue.qsize(),
                'uptime_seconds': time.time(),  # Simplified
                'components': {
                    'models': 'healthy' if models_healthy else 'unhealthy',
                    'cache': 'healthy' if cache_healthy else 'unhealthy', 
                    'queue': 'healthy' if queue_healthy else 'unhealthy'
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Test the real-time recommendation server"""
    print("⚡ Testing Real-time Recommendation Server...")
    
    # Initialize server
    server = RealTimeRecommendationServer()
    
    # Mock model loading
    try:
        # In a real implementation, load actual model files
        logger.info("📁 Loading mock models...")
        
        # Initialize with sample data
        user_ids = list(range(1, 101))
        item_ids = list(range(1, 1001)) 
        popular_items = list(range(1, 51))
        
        server.initialize_cache(user_ids, item_ids, popular_items)
        
        # Test recommendations
        async def test_recommendations():
            request = RecommendationRequest(
                user_id=42,
                request_id="test-001",
                timestamp=datetime.now(),
                n_recommendations=5,
                serving_mode=ServingMode.REAL_TIME
            )
            
            response = await server.get_recommendations(request)
            
            print(f"✅ Generated recommendations:")
            print(f"User: {response.user_id}")
            print(f"Response time: {response.response_time_ms:.2f}ms")
            print(f"Cache hit: {response.cache_hit}")
            print(f"Recommendations:")
            for i, rec in enumerate(response.recommendations, 1):
                print(f"  {i}. Item {rec['item_id']}: {rec['score']:.3f}")
        
        # Run async test
        asyncio.run(test_recommendations())
        
        # Check metrics
        metrics = server.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Cache hit rate: {metrics['cache_hit_rate_percent']:.1f}%")
        print(f"Avg response time: {metrics['avg_response_time_ms']:.2f}ms")
        
        # Health check
        health = server.health_check()
        print(f"\n🏥 Health Status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
        print(f"Cache size: {health['cache_size']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Real-time recommendation server ready!")

if __name__ == "__main__":
    main()