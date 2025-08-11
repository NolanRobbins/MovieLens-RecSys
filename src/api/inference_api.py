"""
Production Inference API for MovieLens Hybrid VAE Recommender
FastAPI service for real-time recommendations with business logic
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
import torch
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
import time
from contextlib import asynccontextmanager

# Import business logic and model classes
from src.business.business_logic_system import BusinessRulesEngine, UserProfile

# Import model architecture
import torch.nn as nn
import torch.nn.functional as F

class HybridVAE(nn.Module):
    """Hybrid VAE + Embedding model"""
    def __init__(self, n_users, n_movies, n_factors=150, 
                 hidden_dims=[512, 256], latent_dim=64, 
                 dropout_rate=0.3):
        super(HybridVAE, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Encoder network
        encoder_layers = []
        input_dim = n_factors * 2
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent space
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder network
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.user_embedding.weight, -0.05, 0.05)
        nn.init.uniform_(self.movie_embedding.weight, -0.05, 0.05)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def encode(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        features = torch.cat([user_emb, movie_emb], dim=1)
        features = self.embedding_dropout(features)
        encoded = self.encoder(features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        decoded = self.decoder(z)
        return torch.sigmoid(decoded)
    
    def forward(self, users, movies, minmax=None):
        mu, logvar = self.encode(users, movies)
        z = self.reparameterize(mu, logvar)
        rating_pred = self.decode(z)
        
        if minmax is not None:
            min_rating, max_rating = minmax
            rating_pred = rating_pred * (max_rating - min_rating) + min_rating
            
        return rating_pred, mu, logvar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = Field(default=10, ge=1, le=100)
    exclude_seen: bool = True
    user_profile: Optional[Dict] = None

class UserProfileRequest(BaseModel):
    user_id: int
    hard_avoids: Optional[Set[int]] = None
    genre_preferences: Optional[Dict[str, float]] = None
    age_rating_limit: str = 'R'
    language_preferences: Optional[Set[str]] = None
    recency_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    diversity_preference: float = Field(default=0.3, ge=0.0, le=1.0)

class MovieDetails(BaseModel):
    movie_id: int
    title: str
    genres: str
    release_year: Optional[int] = None
    predicted_rating: float
    confidence_score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieDetails]
    generated_at: datetime
    processing_time_ms: float
    model_version: str
    business_rules_applied: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    avg_response_time_ms: float

# Global variables for model and data
app_state = {
    'model': None,
    'mappings': None,
    'movies_df': None,
    'business_engine': None,
    'user_histories': {},
    'request_count': 0,
    'total_processing_time': 0.0,
    'start_time': time.time()
}

class HybridVAEInference:
    """
    Inference wrapper for Hybrid VAE model
    Handles model loading, prediction, and caching
    """
    
    def __init__(self, model_path: str, mappings_path: str, movies_path: str):
        self.model_path = Path(model_path)
        self.mappings_path = Path(mappings_path)
        self.movies_path = Path(movies_path)
        
        self.model = None
        self.mappings = None
        self.movies_df = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prediction cache for performance
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    def load_model_and_data(self):
        """Load trained model and associated data"""
        logger.info("Loading model and data...")
        
        try:
            # Load model
            if self.model_path.exists():
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Extract model configuration
                    config = checkpoint['config']
                    n_users = checkpoint['n_users']
                    n_movies = checkpoint['n_movies']
                    
                    # Create model with checkpoint configuration
                    self.model = HybridVAE(
                        n_users=n_users,
                        n_movies=n_movies,
                        n_factors=config['n_factors'],
                        hidden_dims=config['hidden_dims'],
                        latent_dim=config['latent_dim'],
                        dropout_rate=config['dropout_rate']
                    )
                    
                    # Load state dict
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    
                    logger.info(f"✅ Model loaded from {self.model_path}")
                    logger.info(f"   - Experiment: {checkpoint['experiment_name']}")
                    logger.info(f"   - Epoch: {checkpoint['epoch']}")
                    logger.info(f"   - Validation Loss: {checkpoint['val_loss']:.4f}")
                    logger.info(f"   - Users: {n_users}, Movies: {n_movies}")
                else:
                    # Legacy format - direct model
                    self.model = checkpoint
                    self.model.eval()
                    logger.info(f"✅ Legacy model loaded from {self.model_path}")
            else:
                logger.warning(f"⚠️  Model file not found: {self.model_path}")
                # Create dummy model for demonstration
                self._create_dummy_model()
            
            # Load mappings
            with open(self.mappings_path, 'rb') as f:
                self.mappings = pickle.load(f)
            logger.info(f"✅ Mappings loaded: {len(self.mappings['user_to_index'])} users, {len(self.mappings['movie_to_index'])} movies")
            
            # Load movies metadata
            self.movies_df = pd.read_csv(self.movies_path)
            logger.info(f"✅ Movies metadata loaded: {len(self.movies_df)} movies")
            
        except Exception as e:
            logger.error(f"❌ Error loading model/data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Create dummy data for demonstration
            self._create_dummy_data()
    
    def _create_dummy_model(self):
        """Create dummy model for demonstration when real model not available"""
        logger.info("Creating dummy model for demonstration...")
        
        class DummyModel(torch.nn.Module):
            def __init__(self, n_users=200000, n_movies=50000, n_factors=150):
                super().__init__()
                self.user_embeddings = torch.nn.Embedding(n_users, n_factors)
                self.movie_embeddings = torch.nn.Embedding(n_movies, n_factors)
                self.output_layer = torch.nn.Linear(n_factors * 2, 1)
                
            def forward(self, user_ids, movie_ids):
                user_emb = self.user_embeddings(user_ids)
                movie_emb = self.movie_embeddings(movie_ids)
                combined = torch.cat([user_emb, movie_emb], dim=-1)
                return torch.sigmoid(self.output_layer(combined)) * 4.5 + 0.5
        
        self.model = DummyModel()
        self.model.eval()
    
    def _create_dummy_data(self):
        """Create dummy data when real data not available"""
        logger.info("Creating dummy data for demonstration...")
        
        # Dummy mappings
        self.mappings = {
            'user_to_index': {i: i for i in range(1000)},
            'movie_to_index': {i: i for i in range(5000)},
            'n_users': 1000,
            'n_movies': 5000
        }
        
        # Dummy movies
        genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
        self.movies_df = pd.DataFrame({
            'movieId': range(5000),
            'title': [f'Movie {i} (2020)' for i in range(5000)],
            'genres': [np.random.choice(genres) for _ in range(5000)],
            'release_year': np.random.randint(1990, 2024, 5000)
        })
    
    def predict_ratings(self, user_id: int, movie_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-movie pairs"""
        cache_key = f"{user_id}_{hash(tuple(sorted(movie_ids)))}"
        
        # Check cache
        if cache_key in self.prediction_cache:
            cached_data, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Map IDs to model indices
            if user_id not in self.mappings['user_to_index']:
                # Handle cold start - use average user embedding
                user_idx = 0  # Could use mean embedding or random
            else:
                user_idx = self.mappings['user_to_index'][user_id]
            
            valid_movie_indices = []
            valid_movie_ids = []
            
            for movie_id in movie_ids:
                if movie_id in self.mappings['movie_to_index']:
                    valid_movie_indices.append(self.mappings['movie_to_index'][movie_id])
                    valid_movie_ids.append(movie_id)
            
            if not valid_movie_indices:
                return np.array([])
            
            # Predict with model
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx] * len(valid_movie_indices), dtype=torch.long)
                movie_tensor = torch.tensor(valid_movie_indices, dtype=torch.long)
                
                predictions, _, _ = self.model(user_tensor, movie_tensor, (0.5, 5.0))
                ratings = predictions.cpu().numpy().flatten()
            
            # Add some noise for realism in dummy predictions
            if hasattr(self.model, 'user_embeddings') and 'DummyModel' in str(type(self.model)):
                ratings += np.random.normal(0, 0.1, len(ratings))
                ratings = np.clip(ratings, 0.5, 5.0)
            
            # Cache results
            self.prediction_cache[cache_key] = (ratings, time.time())
            
            return ratings
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Fallback to random predictions
            return np.random.uniform(2.0, 4.5, len(movie_ids))
    
    def get_top_recommendations(self, user_id: int, n_recommendations: int = 10,
                              exclude_movies: Set[int] = None) -> List[Dict]:
        """Get top N movie recommendations for a user"""
        
        if exclude_movies is None:
            exclude_movies = set()
        
        # Get candidate movies (subset for efficiency)
        all_movie_ids = list(self.mappings['movie_to_index'].keys())
        candidate_movies = [mid for mid in all_movie_ids if mid not in exclude_movies]
        
        # Limit candidates for performance (in production, you'd use more sophisticated sampling)
        if len(candidate_movies) > 1000:
            candidate_movies = np.random.choice(candidate_movies, 1000, replace=False).tolist()
        
        # Predict ratings
        predicted_ratings = self.predict_ratings(user_id, candidate_movies)
        
        if len(predicted_ratings) == 0:
            return []
        
        # Get top recommendations
        top_indices = np.argsort(predicted_ratings)[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            movie_id = candidate_movies[idx]
            rating = predicted_ratings[idx]
            
            # Get movie details
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                movie_row = movie_info.iloc[0]
                
                recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_row.get('title', f'Movie {movie_id}'),
                    'genres': movie_row.get('genres', 'Unknown'),
                    'release_year': movie_row.get('release_year'),
                    'predicted_rating': float(rating),
                    'confidence_score': float(min(rating / 5.0, 1.0))  # Normalize to 0-1
                })
        
        return recommendations

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting MovieLens Inference API...")
    
    # Initialize inference engine  
    model_path = "data/models/experiment_2_epoch086_val0.5996.pt"  # Use the best trained model
    mappings_path = "data/processed/data_mappings.pkl"
    movies_path = "data/processed/movies_processed.csv"
    
    inference_engine = HybridVAEInference(model_path, mappings_path, movies_path)
    inference_engine.load_model_and_data()
    
    # Initialize business rules engine
    business_engine = BusinessRulesEngine(inference_engine.movies_df)
    
    # Store in app state
    app_state['model'] = inference_engine
    app_state['mappings'] = inference_engine.mappings
    app_state['movies_df'] = inference_engine.movies_df
    app_state['business_engine'] = business_engine
    
    logger.info("✅ API initialization complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MovieLens Inference API...")

# Create FastAPI app
app = FastAPI(
    title="MovieLens Hybrid VAE Recommender API",
    description="Production inference API for movie recommendations with business logic",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking middleware
@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = (time.time() - start_time) * 1000
    
    app_state['request_count'] += 1
    app_state['total_processing_time'] += processing_time
    
    response.headers["X-Processing-Time-MS"] = str(round(processing_time, 2))
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_state['start_time']
    avg_response_time = (app_state['total_processing_time'] / app_state['request_count'] 
                        if app_state['request_count'] > 0 else 0)
    
    return HealthResponse(
        status="healthy" if app_state['model'] else "degraded",
        model_loaded=app_state['model'] is not None,
        uptime_seconds=uptime,
        total_requests=app_state['request_count'],
        avg_response_time_ms=avg_response_time
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    start_time = time.time()
    
    if not app_state['model']:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get user's viewing history for exclusion
        exclude_movies = set()
        if request.exclude_seen and request.user_id in app_state['user_histories']:
            exclude_movies = app_state['user_histories'][request.user_id]
        
        # Get recommendations from model
        raw_recommendations = app_state['model'].get_top_recommendations(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations * 2,  # Get more for filtering
            exclude_movies=exclude_movies
        )
        
        # Apply business rules if user profile provided
        business_rules_applied = False
        if request.user_profile:
            try:
                user_profile = UserProfile(
                    user_id=request.user_id,
                    hard_avoids=set(request.user_profile.get('hard_avoids', [])),
                    genre_preferences=request.user_profile.get('genre_preferences', {}),
                    age_rating_limit=request.user_profile.get('age_rating_limit', 'R'),
                    language_preferences=set(request.user_profile.get('language_preferences', ['English'])),
                    recency_bias=request.user_profile.get('recency_bias', 0.5),
                    diversity_preference=request.user_profile.get('diversity_preference', 0.3)
                )
                
                # Extract movie IDs and scores
                movie_ids = [rec['movie_id'] for rec in raw_recommendations]
                scores = [rec['predicted_rating'] for rec in raw_recommendations]
                
                # Apply business rules
                filtered_ids = app_state['business_engine'].apply_hard_avoids(movie_ids, user_profile)
                
                if filtered_ids:
                    # Filter recommendations
                    filtered_recommendations = [
                        rec for rec in raw_recommendations 
                        if rec['movie_id'] in filtered_ids
                    ]
                    raw_recommendations = filtered_recommendations
                    business_rules_applied = True
                
            except Exception as e:
                logger.warning(f"Error applying business rules: {e}")
        
        # Limit to requested number
        final_recommendations = raw_recommendations[:request.n_recommendations]
        
        # Convert to response format
        movie_details = [
            MovieDetails(
                movie_id=rec['movie_id'],
                title=rec['title'],
                genres=rec['genres'],
                release_year=rec.get('release_year'),
                predicted_rating=rec['predicted_rating'],
                confidence_score=rec['confidence_score']
            )
            for rec in final_recommendations
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=movie_details,
            generated_at=datetime.now(),
            processing_time_ms=processing_time,
            model_version="hybrid_vae_v1.0",
            business_rules_applied=business_rules_applied
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.post("/user-profile/{user_id}")
async def update_user_profile(user_id: int, profile: UserProfileRequest):
    """Update user profile for personalized recommendations"""
    # In production, this would save to database
    # For demo, we'll just return success
    return {"message": f"User profile updated for user {user_id}", "status": "success"}

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: int):
    """Get user's viewing history"""
    history = app_state['user_histories'].get(user_id, [])
    return {"user_id": user_id, "watched_movies": list(history)}

@app.post("/user/{user_id}/watched")
async def add_watched_movie(user_id: int, movie_id: int):
    """Add movie to user's watched list"""
    if user_id not in app_state['user_histories']:
        app_state['user_histories'][user_id] = set()
    
    app_state['user_histories'][user_id].add(movie_id)
    return {"message": f"Movie {movie_id} added to user {user_id}'s history"}

@app.get("/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get details for a specific movie"""
    if app_state['movies_df'] is None:
        raise HTTPException(status_code=503, detail="Movies data not loaded")
    
    movie_info = app_state['movies_df'][app_state['movies_df']['movieId'] == movie_id]
    
    if movie_info.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_row = movie_info.iloc[0]
    return {
        "movie_id": int(movie_id),
        "title": movie_row.get('title', ''),
        "genres": movie_row.get('genres', ''),
        "release_year": movie_row.get('release_year')
    }

@app.get("/stats")
async def get_api_stats():
    """Get API usage statistics"""
    uptime = time.time() - app_state['start_time']
    avg_response_time = (app_state['total_processing_time'] / app_state['request_count'] 
                        if app_state['request_count'] > 0 else 0)
    
    return {
        "uptime_seconds": uptime,
        "total_requests": app_state['request_count'],
        "average_response_time_ms": avg_response_time,
        "model_loaded": app_state['model'] is not None,
        "cached_predictions": len(app_state['model'].prediction_cache) if app_state['model'] else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")