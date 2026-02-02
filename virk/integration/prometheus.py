import time
try:
    from prometheus_client import start_http_server, Gauge, Counter
    HAS_PROM = True
except ImportError:
    HAS_PROM = False

# Singleton metrics (to avoid reregistration)
_DRIFT_GAUGE = None
_DRIFT_COUNTER = None

def init_prometheus(port: int = 8000):
    """Start Prometheus HTTP server and register metrics."""
    global _DRIFT_GAUGE, _DRIFT_COUNTER
    if not HAS_PROM:
        print("Warning: prometheus_client not installed. Exporter disabled.")
        return

    _DRIFT_GAUGE = Gauge('virk_drift_magnitude', 'Magnitude of detected drift', ['ref_id'])
    _DRIFT_COUNTER = Counter('virk_drift_detected_total', 'Total drift events detected', ['cause'])
    
    try:
        start_http_server(port)
        print(f"Prometheus metrics exposed on port {port}")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

def record_drift(magnitude: float, is_detected: bool, cause: str = "unknown", ref_id: str = "baseline"):
    """Update prometheus metrics."""
    if not HAS_PROM or _DRIFT_GAUGE is None:
        return
        
    _DRIFT_GAUGE.labels(ref_id=ref_id).set(magnitude)
    if is_detected:
         _DRIFT_COUNTER.labels(cause=cause).inc()
