"""
Mock Prometheus metrics exporter for testing Grafana dashboards.
This service exposes synthetic ML metrics without requiring the full hugging-face-serve.
"""

import random
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server


# Define metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

IN_PROGRESS = Gauge("in_progress_requests", "Number of in-progress requests")

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model", "prediction"],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total model prediction errors",
)

PREDICTION_CONFIDENCE = Summary(
    "model_prediction_confidence",
    "Prediction confidence scores",
)


def generate_mock_data():
    """Generate realistic mock metrics for dashboard testing."""
    
    # Simulate API requests with varying patterns
    print("[Mock Metrics] Generating API request metrics...")
    
    # Success requests (200)
    for _ in range(random.randint(300, 500)):
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict/",
            http_status="200"
        ).inc()
    
    for _ in range(random.randint(100, 200)):
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/",
            http_status="200"
        ).inc()
    
    # Client errors (400)
    for _ in range(random.randint(5, 15)):
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict/",
            http_status="400"
        ).inc()
    
    # Server errors (500)
    for _ in range(random.randint(2, 8)):
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict/",
            http_status="500"
        ).inc()
    
    # Generate request latencies
    print("[Mock Metrics] Generating latency metrics...")
    for _ in range(200):
        REQUEST_LATENCY.labels(endpoint="/predict/").observe(
            random.uniform(0.1, 2.5)  # 100ms to 2.5s
        )
    
    for _ in range(100):
        REQUEST_LATENCY.labels(endpoint="/").observe(
            random.uniform(0.01, 0.1)  # 10ms to 100ms
        )
    
    # Generate prediction metrics
    print("[Mock Metrics] Generating model prediction metrics...")
    tags = ["web", "mobile", "backend", "other"]
    
    for tag in tags:
        for _ in range(random.randint(80, 250)):
            PREDICTION_COUNT.labels(
                model="default-model",
                prediction=tag
            ).inc()
    
    # Generate confidence scores
    for _ in range(600):
        PREDICTION_CONFIDENCE.observe(random.uniform(0.65, 0.98))
    
    # Simulate prediction errors
    for _ in range(random.randint(10, 25)):
        PREDICTION_ERRORS.inc()
    
    # Set in-progress requests (varies over time)
    IN_PROGRESS.set(random.randint(0, 5))
    
    print("[Mock Metrics] ✓ Mock data generation complete!")


def run_simulator():
    """Run background simulator to update metrics periodically."""
    print("[Mock Metrics] Starting metric simulator...")
    while True:
        # Update in-progress requests (simulates real traffic pattern)
        IN_PROGRESS.set(random.randint(0, 8))
        
        # Occasionally add new requests (simulating real traffic)
        if random.random() > 0.7:
            status = random.choice(["200", "200", "200", "400", "500"])
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/predict/",
                http_status=status
            ).inc()
        
        # Occasionally add latency observations
        if random.random() > 0.6:
            REQUEST_LATENCY.labels(endpoint="/predict/").observe(
                random.uniform(0.05, 2.0)
            )
        
        # Occasionally add predictions
        if random.random() > 0.5:
            tag = random.choice(["web", "mobile", "backend", "other"])
            PREDICTION_COUNT.labels(
                model="default-model",
                prediction=tag
            ).inc()
            PREDICTION_CONFIDENCE.observe(random.uniform(0.65, 0.98))
        
        time.sleep(5)  # Update every 5 seconds


if __name__ == "__main__":
    print("=" * 60)
    print("Mock Prometheus Metrics Exporter")
    print("=" * 60)
    print("[Mock Metrics] Starting HTTP server on port 9091...")
    start_http_server(9091)
    
    # Generate initial mock data
    generate_mock_data()
    
    # Run continuous simulator
    print("[Mock Metrics] Running metrics simulator...")
    print("[Mock Metrics] Metrics will be available at http://localhost:9091/metrics")
    run_simulator()
