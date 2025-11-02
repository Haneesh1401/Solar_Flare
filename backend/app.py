from pathlib import Path
import joblib
import pickle
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any, Optional
import redis
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from model_versioning import ModelRegistry, ABTestingFramework
from auto_retrain import ModelMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Solar Flare Prediction API",
    description="Advanced solar flare prediction with multi-model support, monitoring, and real-time capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
project_root = Path(__file__).resolve().parent.parent
model_registry = ModelRegistry()
ab_testing = ABTestingFramework(model_registry)
model_monitor = ModelMonitor(
    str(project_root / "models" / "model_rf_improved.joblib"),
    str(project_root / "data" / "historical_goes_2010_2015_parsed.csv")
)

# Redis setup for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Prometheus metrics
PREDICTION_COUNTER = Counter('solar_flare_predictions_total', 'Total number of predictions made')
PREDICTION_DURATION = Histogram('solar_flare_prediction_duration_seconds', 'Time spent making predictions')
MODEL_SWITCHES = Counter('solar_flare_model_switches_total', 'Number of model switches')
CACHE_HITS = Counter('solar_flare_cache_hits_total', 'Number of cache hits')
CACHE_MISSES = Counter('solar_flare_cache_misses_total', 'Number of cache misses')
ACTIVE_MODELS = Gauge('solar_flare_active_models', 'Number of active models')

# Global variables for current models
current_models = {}
current_model_name = "solar_flare_rf"  # Default model

# Pydantic models
class FlareInput(BaseModel):
    flux: float
    month: int
    day: int
    hour: int = 0
    day_of_year: int = 0

class BatchFlareInput(BaseModel):
    inputs: List[FlareInput]

class ModelSwitchRequest(BaseModel):
    model_name: str
    version: Optional[str] = None

class PredictionResponse(BaseModel):
    model_used: str
    model_version: str
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    timestamp: str
    processing_time: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time: float
    average_confidence: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    cache_status: str
    memory_usage: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    metrics: Dict[str, float]
    created_at: str

# Utility functions
def get_cache_key(input_data: Dict) -> str:
    """Generate cache key from input data"""
    return f"pred_{hash(frozenset(input_data.items()))}"

def load_models():
    """Load all available models from registry"""
    global current_models, current_model_name

    try:
        # Get all model names from registry
        model_names = list(set([info["model_name"] for info in model_registry.list_models()]))

        loaded_models = {}
        for model_name in model_names:
            try:
                # Get best performing model for this name
                best_version = model_registry.get_best_model(model_name)
                if best_version:
                    model = model_registry.get_model(best_version)
                    loaded_models[model_name] = {
                        "model": model,
                        "version": best_version,
                        "info": model_registry.registry[best_version]
                    }
                    logger.info(f"Loaded model: {model_name} (version: {best_version})")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")

        current_models = loaded_models
        ACTIVE_MODELS.set(len(current_models))

        # Set default model if available
        if "solar_flare_rf" in current_models:
            current_model_name = "solar_flare_rf"
        elif current_models:
            current_model_name = list(current_models.keys())[0]

        logger.info(f"Loaded {len(current_models)} models. Current model: {current_model_name}")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def predict_with_model(model_info: Dict, input_data: Dict) -> Dict:
    """Make prediction with a specific model"""
    model = model_info["model"]
    model_type = model_info["info"]["model_type"]

    # Prepare input
    input_vector = [[input_data["flux"], input_data["month"], input_data["day"],
                     input_data["hour"], input_data["day_of_year"]]]

    try:
        if model_type == "sklearn":
            prediction = model.predict(input_vector)[0]
            probabilities = model.predict_proba(input_vector)[0]
        elif model_type == "tensorflow":
            # Lazy import TensorFlow only if needed
            import tensorflow as tf  # noqa: F401
            prediction = np.argmax(model.predict(np.array(input_vector), verbose=0)[0])
            probabilities = model.predict(np.array(input_vector), verbose=0)[0]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Convert numeric prediction to class name
        class_map_rev = {0: 'No Flare', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
        prediction_class = class_map_rev.get(prediction, "Unknown")

        # Calculate confidence
        confidence = float(max(probabilities))

        return {
            "prediction": prediction_class,
            "probabilities": {class_map_rev.get(i, f"Class {i}"): float(prob)
                           for i, prob in enumerate(probabilities)},
            "confidence": confidence,
            "numeric_prediction": int(prediction)
        }

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Enhanced Solar Flare Prediction API")
    load_models()
    logger.info("✅ Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("🛑 Shutting down application")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        # Check cache status
        cache_status = "connected" if redis_client.ping() else "disconnected"

        # Get memory usage (approximate)
        memory_usage = {
            "models_loaded": len(current_models),
            "cache_keys": redis_client.dbsize() if cache_status == "connected" else 0
        }

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            models_loaded=len(current_models),
            cache_status=cache_status,
            memory_usage=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Model management endpoints
@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    try:
        models = []
        for version, info in model_registry.registry.items():
            models.append(ModelInfo(
                name=info["model_name"],
                version=version,
                type=info["model_type"],
                metrics=info["metrics"],
                created_at=info["created_at"]
            ))
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        if request.model_name not in current_models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not loaded")

        global current_model_name
        old_model = current_model_name
        current_model_name = request.model_name

        MODEL_SWITCHES.inc()
        logger.info(f"Switched model from {old_model} to {current_model_name}")

        return {
            "message": f"Switched to model {current_model_name}",
            "previous_model": old_model,
            "new_model": current_model_name
        }
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/current")
async def get_current_model():
    """Get information about current active model"""
    if current_model_name not in current_models:
        raise HTTPException(status_code=404, detail="No active model")

    model_info = current_models[current_model_name]
    return {
        "name": current_model_name,
        "version": model_info["version"],
        "info": model_info["info"]
    }

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_flare(data: FlareInput):
    """Make a single prediction"""
    start_time = time.time()

    try:
        PREDICTION_COUNTER.inc()

        # Check cache first
        cache_key = get_cache_key(data.dict())
        cached_result = redis_client.get(cache_key)

        if cached_result:
            CACHE_HITS.inc()
            cached_data = json.loads(cached_result)
            processing_time = time.time() - start_time

            return PredictionResponse(
                model_used=current_model_name,
                model_version=current_models[current_model_name]["version"],
                prediction=cached_data["prediction"],
                probabilities=cached_data["probabilities"],
                confidence=cached_data["confidence"],
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
        else:
            CACHE_MISSES.inc()

        # Make prediction
        if current_model_name not in current_models:
            raise HTTPException(status_code=500, detail="No active model available")

        model_info = current_models[current_model_name]
        prediction_result = predict_with_model(model_info, data.dict())

        # Cache the result
        cache_data = {
            "prediction": prediction_result["prediction"],
            "probabilities": prediction_result["probabilities"],
            "confidence": prediction_result["confidence"]
        }
        redis_client.setex(cache_key, 300, json.dumps(cache_data))  # Cache for 5 minutes

        processing_time = time.time() - start_time
        PREDICTION_DURATION.observe(processing_time)

        return PredictionResponse(
            model_used=current_model_name,
            model_version=model_info["version"],
            prediction=prediction_result["prediction"],
            probabilities=prediction_result["probabilities"],
            confidence=prediction_result["confidence"],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: BatchFlareInput):
    """Make batch predictions"""
    start_time = time.time()

    try:
        predictions = []
        total_confidence = 0

        for input_data in data.inputs:
            # Reuse single prediction logic
            prediction = await predict_flare(input_data)
            predictions.append(prediction)
            total_confidence += prediction.confidence

        processing_time = time.time() - start_time
        average_confidence = total_confidence / len(predictions) if predictions else 0

        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time=processing_time,
            average_confidence=average_confidence
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/realtime")
async def predict_realtime():
    """Real-time prediction using latest GOES data"""
    try:
        # Fetch latest GOES data
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            raise HTTPException(status_code=503, detail="GOES API unavailable")

        data = response.json()
        df = pd.DataFrame(data)
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df.set_index("time_tag", inplace=True)
        df = df[["flux"]].resample("10min").mean().dropna()

        # Get latest data point
        latest_data = df.iloc[-1]
        latest_flux = latest_data["flux"]
        current_time = latest_data.name

        # Prepare input
        input_data = FlareInput(
            flux=float(latest_flux),
            month=current_time.month,
            day=current_time.day,
            hour=current_time.hour,
            day_of_year=current_time.dayofyear
        )

        # Make prediction
        prediction = await predict_flare(input_data)

        return {
            "time": str(current_time),
            "flux": latest_flux,
            "prediction": prediction.dict()
        }

    except Exception as e:
        logger.error(f"Real-time prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/monitoring/performance")
async def get_performance_metrics():
    """Get current model performance metrics"""
    try:
        if current_model_name not in current_models:
            raise HTTPException(status_code=404, detail="No active model")

        model_info = current_models[current_model_name]

        # Load test data for evaluation
        df = pd.read_csv(str(project_root / "data" / "historical_goes_2010_2015_parsed.csv"))
        df = df.dropna(subset=['flare_class', 'flux', 'start'])

        # Prepare features
        feature_columns = ['flux', 'month', 'day', 'hour', 'day_of_year']
        X = df[feature_columns]
        y = df['flare_class_num']

        # Get performance metrics
        current_metrics = model_monitor.check_performance(model_info["model"], X, y)

        return {
            "model": current_model_name,
            "version": model_info["version"],
            "metrics": current_metrics,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Performance monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/models/compare")
async def compare_models():
    """Compare performance of all available models"""
    try:
        comparison_results = {}

        for model_name, model_info in current_models.items():
            try:
                # Load test data
                df = pd.read_csv(str(project_root / "data" / "historical_goes_2010_2015_parsed.csv"))
                df = df.dropna(subset=['flare_class', 'flux', 'start'])

                feature_columns = ['flux', 'month', 'day', 'hour', 'day_of_year']
                X = df[feature_columns]
                y = df['flare_class_num']

                # Get performance
                metrics = model_monitor.check_performance(model_info["model"], X, y)
                comparison_results[model_name] = {
                    "version": model_info["version"],
                    "metrics": metrics
                }

            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
                comparison_results[model_name] = {"error": str(e)}

        return {
            "comparison": comparison_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for backward compatibility
@app.post("/predict/legacy")
def predict_flare_legacy(data: FlareInput):
    """Legacy prediction endpoint for backward compatibility"""
    try:
        # Use Random Forest model if available, otherwise use current model
        model_name = "solar_flare_rf" if "solar_flare_rf" in current_models else current_model_name

        if model_name not in current_models:
            raise HTTPException(status_code=500, detail="No model available")

        model_info = current_models[model_name]
        prediction_result = predict_with_model(model_info, data.dict())

        class_map_rev = {0: 'No Flare', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
        flare_class = prediction_result["prediction"]

        return {"predicted_flare_class": flare_class}

    except Exception as e:
        logger.error(f"Legacy prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
