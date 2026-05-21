"""
FastAPI MLOps Server for ML Model Serving with Prometheus Monitoring

This module provides a production-ready FastAPI server for serving machine learning
models with comprehensive monitoring, logging, health checks, and graceful error handling.

Features:
- Model serving endpoints (/predict, /evaluate)
- Prometheus metrics collection and exposure
- Health check endpoints (/health, /ready)
- Structured logging
- MLflow integration
- Docker/Kubernetes compatibility
"""

import argparse
import json
import logging
import sys
import time
from http import HTTPStatus
from typing import Any, Callable, Dict, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from madewithml import evaluate, predict
from madewithml.config import MLFLOW_TRACKING_URI, logger, mlflow

# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure structured logging for production."""
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    
    return log


log = setup_logging()

# ============================================================================
# Prometheus Metrics Definition
# ============================================================================

# Counters: monotonically increasing metrics
REQUEST_COUNT_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

PREDICTION_REQUESTS_TOTAL = Counter(
    "model_prediction_requests_total",
    "Total prediction requests processed",
    ["model_id", "prediction_label"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
)

# Gauges: point-in-time values
IN_PROGRESS_REQUESTS = Gauge(
    "http_in_progress_requests",
    "Number of HTTP requests currently in progress",
)

MODEL_CONFIDENCE_SCORE = Gauge(
    "model_confidence_score",
    "Average confidence score of model predictions",
)

# Histograms: distribution of values with quantiles
REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Summaries: quantile metrics with percentiles
RESPONSE_SIZE_BYTES = Summary(
    "http_response_size_bytes",
    "HTTP response size in bytes",
)

# ============================================================================
# Middleware: Prometheus Metrics Collection
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics and track request lifecycle."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request timing, status, and errors."""
        # Skip metrics collection for /metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Increment in-progress requests
        IN_PROGRESS_REQUESTS.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Log errors and track them
            status_code = 500
            log.error(f"Request error: {str(e)}", exc_info=True)
            raise
        finally:
            # Record timing and counts
            elapsed_time = time.time() - start_time
            REQUEST_LATENCY_SECONDS.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(elapsed_time)
            
            REQUEST_COUNT_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
            ).inc()
            
            IN_PROGRESS_REQUESTS.dec()
            
            # Log request summary
            log.info(
                f"{request.method} {request.url.path} - {status_code} - {elapsed_time:.3f}s"
            )
        
        return response


# ============================================================================
# Utility Functions
# ============================================================================

async def get_prediction_input(
    request: Request, 
    title: str = "", 
    description: str = ""
) -> Dict[str, str]:
    """Extract prediction input from JSON body or query parameters.
    
    Args:
        request: FastAPI Request object
        title: Optional title from query parameter
        description: Optional description from query parameter
    
    Returns:
        Dictionary with title and description keys
    """
    data = {}
    try:
        body = await request.body()
        if body:
            data = json.loads(body)
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse JSON body: {e}")
        pass
    
    return {
        "title": data.get("title", title) or "",
        "description": data.get("description", description) or "",
    }


def make_json_safe(results: list) -> list:
    """Convert numpy/torch values to JSON-serializable Python types.
    
    Args:
        results: List of prediction results with numpy types
    
    Returns:
        List of results with JSON-safe types
    """
    safe_results = []
    for result in results:
        try:
            safe_results.append(
                {
                    "prediction": str(result["prediction"]),
                    "probabilities": {
                        label: float(prob) 
                        for label, prob in result["probabilities"].items()
                    },
                }
            )
        except (KeyError, TypeError) as e:
            log.error(f"Error converting result to JSON-safe format: {e}")
            PREDICTION_ERRORS_TOTAL.inc()
            raise ValueError(f"Invalid prediction result format: {e}")
    
    return safe_results


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app(run_id: str, threshold: float = 0.9) -> FastAPI:
    """Create and configure FastAPI application with MLOps setup.
    
    Args:
        run_id: MLflow run ID for model serving
        threshold: Confidence threshold for "other" class classification
    
    Returns:
        Configured FastAPI application instance
    
    Raises:
        Exception: If model checkpoint cannot be loaded
    """
    local_app = FastAPI(
        title="Made With ML - MLOps Server",
        description="Production-ready ML model serving API with Prometheus monitoring",
        version="1.0.0",
    )
    
    # Add Prometheus middleware
    local_app.add_middleware(PrometheusMiddleware)
    
    # Initialize MLflow and load model
    log.info(f"Initializing MLflow with tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        log.info(f"Loading model checkpoint for run_id: {run_id}")
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)
        log.info(f"✓ Model loaded successfully for run_id: {run_id}")
    except Exception as e:
        log.error(f"Failed to load model checkpoint: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}")
    
    # ========================================================================
    # Health Check Endpoints
    # ========================================================================
    
    @local_app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint - basic liveness probe.
        
        Returns:
            Status of the service
        """
        log.debug("Health check requested")
        return {"status": "healthy", "service": "ml-server"}
    
    @local_app.get("/ready")
    async def readiness() -> Dict[str, str]:
        """Readiness check endpoint - validates model is loaded and ready.
        
        Returns:
            Readiness status and model info
        """
        log.debug("Readiness check requested")
        try:
            # Verify model is available
            assert predictor is not None, "Model predictor not initialized"
            return {
                "status": "ready",
                "model_id": run_id,
                "threshold": str(threshold),
            }
        except AssertionError as e:
            log.error(f"Readiness check failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))
    
    # ========================================================================
    # Prometheus Metrics Endpoint
    # ========================================================================
    
    @local_app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint - exposes all collected metrics.
        
        Returns:
            Prometheus-formatted metrics (text/plain)
        """
        try:
            data = generate_latest()
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            log.error(f"Error generating metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate metrics")
    
    # ========================================================================
    # API Endpoints
    # ========================================================================
    
    @local_app.get("/")
    async def index() -> Dict[str, Any]:
        """Root endpoint - API information.
        
        Returns:
            Basic service information and status
        """
        log.info("Index endpoint requested")
        return {
            "message": HTTPStatus.OK.phrase,
            "status": HTTPStatus.OK.value,
            "service": "ML Model Serving API",
            "version": "1.0.0",
        }
    
    @local_app.get("/run_id")
    async def get_run_id() -> Dict[str, str]:
        """Get the current MLflow run ID.
        
        Returns:
            Current model run ID
        """
        log.info(f"Run ID endpoint requested - returning {run_id}")
        return {"run_id": run_id}
    
    @local_app.post("/evaluate")
    async def evaluate_model(request: Request) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Args:
            request: Request containing dataset location
        
        Returns:
            Evaluation results with metrics
        """
        log.info("Evaluate endpoint called")
        try:
            data = await request.json()
            dataset_loc = data.get("dataset")
            
            if not dataset_loc:
                log.warning("Evaluate called without dataset location")
                raise HTTPException(
                    status_code=400,
                    detail="Missing required field: 'dataset'"
                )
            
            log.info(f"Evaluating model on dataset: {dataset_loc}")
            results = evaluate.evaluate(run_id=run_id, dataset_loc=dataset_loc)
            
            log.info(f"✓ Evaluation completed with results")
            return {"status": "success", "results": results}
        
        except HTTPException:
            raise
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in evaluate request: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        except Exception as e:
            log.error(f"Evaluation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
    
    @local_app.post("/predict")
    async def predict_endpoint(
        request: Request,
        title: str = "",
        description: str = ""
    ) -> Dict[str, Any]:
        """Predict model output for given project title and description.
        
        Args:
            request: HTTP request (can contain JSON body)
            title: Optional project title from query parameter
            description: Optional project description from query parameter
        
        Returns:
            Prediction results with probabilities for each label
        
        Raises:
            HTTPException: On validation or prediction errors
        """
        log.info("Predict endpoint called")
        prediction_start = time.time()
        
        try:
            # Extract input data
            data = await get_prediction_input(request, title, description)
            
            # Validate inputs
            if not data["title"] and not data["description"]:
                log.warning("Predict called with empty title and description")
                raise HTTPException(
                    status_code=400,
                    detail="At least one of 'title' or 'description' must be provided"
                )
            
            log.info(
                f"Processing prediction - title_len={len(data['title'])}, "
                f"desc_len={len(data['description'])}"
            )
            
            # Create DataFrame for model
            sample_df = pd.DataFrame([{
                "title": data["title"],
                "description": data["description"],
                "tag": ""  # Placeholder for training compatibility
            }])
            
            # Run model prediction
            try:
                results = predict.predict_proba(df=sample_df, predictor=predictor)
            except Exception as e:
                log.error(f"Model prediction failed: {e}", exc_info=True)
                PREDICTION_ERRORS_TOTAL.inc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {str(e)}"
                )
            
            # Apply threshold and collect metrics
            confidence_scores = []
            for i, result in enumerate(results):
                pred = result["prediction"]
                prob = result["probabilities"]
                
                # Store confidence for metric
                top_confidence = float(max(prob.values()))
                confidence_scores.append(top_confidence)
                
                # Apply threshold
                if top_confidence < threshold:
                    results[i]["prediction"] = "other"
                    log.debug(
                        f"Prediction below threshold ({top_confidence:.3f} < {threshold}), "
                        f"changed to 'other'"
                    )
                
                # Track prediction
                PREDICTION_REQUESTS_TOTAL.labels(
                    model_id=run_id,
                    prediction_label=results[i]["prediction"]
                ).inc()
            
            # Record metrics
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                MODEL_CONFIDENCE_SCORE.set(avg_confidence)
            
            # Record prediction latency
            pred_latency = time.time() - prediction_start
            PREDICTION_LATENCY_SECONDS.observe(pred_latency)
            
            # Convert to JSON-safe format
            safe_results = make_json_safe(results)
            
            log.info(
                f"✓ Prediction completed - latency={pred_latency:.3f}s, "
                f"predictions={[r['prediction'] for r in safe_results]}"
            )
            
            return {
                "status": "success",
                "results": safe_results,
                "latency_seconds": pred_latency,
            }
        
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Prediction endpoint error: {e}", exc_info=True)
            PREDICTION_ERRORS_TOTAL.inc()
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    # ========================================================================
    # Startup & Shutdown Events
    # ========================================================================
    
    @local_app.on_event("startup")
    async def startup_event():
        """Initialize resources on application startup."""
        log.info("=" * 70)
        log.info("🚀 MLOps Server Starting")
        log.info("=" * 70)
        log.info(f"Run ID: {run_id}")
        log.info(f"Threshold: {threshold}")
        log.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        log.info("Prometheus metrics endpoint: http://localhost:8000/metrics")
        log.info("Health check endpoint: http://localhost:8000/health")
        log.info("=" * 70)
    
    @local_app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on application shutdown."""
        log.info("=" * 70)
        log.info("🛑 MLOps Server Shutting Down")
        log.info("=" * 70)
    
    return local_app


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastAPI MLOps Server for Model Serving"
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="MLflow run ID for the model to serve"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for 'other' class classification (default: 0.9)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers (default: 1)"
    )
    
    args = parser.parse_args()
    
    log.info(f"Starting MLOps Server on {args.host}:{args.port}")
    
    # Create and run application
    app = create_app(run_id=args.run_id, threshold=args.threshold)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
