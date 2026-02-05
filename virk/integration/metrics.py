from typing import Protocol, Any, Dict, Optional
import logging

logger = logging.getLogger("virk.metrics")

class MetricsBackend(Protocol):
    """Abstract interface for metric emission."""
    
    def record_drift(self, value: float, labels: Dict[str, str]):
        """Record the magnitude of drift (Histogram/Gauge)."""
        ...
        
    def increment_drift_event(self, labels: Dict[str, str]):
        """Increment count of drift events (Counter)."""
        ...

class NoOpBackend(MetricsBackend):
    def record_drift(self, value: float, labels: Dict[str, str]):
        pass
    def increment_drift_event(self, labels: Dict[str, str]):
        pass

class OTelBackend(MetricsBackend):
    def __init__(self, service_name: str):
        try:
            from opentelemetry import metrics
            self.meter = metrics.get_meter(service_name)
            self.drift_hist = self.meter.create_histogram(
                "virk_drift_magnitude", 
                description="Magnitude of detected drift (MMD)"
            )
            self.drift_counter = self.meter.create_counter(
                "virk_drift_events", 
                description="Count of drift events detected"
            )
        except ImportError:
            # Should be handled by factory, but safe guard
            self.drift_hist = None
            self.drift_counter = None

    def record_drift(self, value: float, labels: Dict[str, str]):
        if self.drift_hist:
            self.drift_hist.record(value, labels)

    def increment_drift_event(self, labels: Dict[str, str]):
        if self.drift_counter:
            self.drift_counter.add(1, labels)

class PrometheusBackend(MetricsBackend):
    def __init__(self, port: int = None):
        try:
            from prometheus_client import Histogram, Counter, start_http_server
            
            # Start server if port provided (and not already running, simplified)
            if port:
                try:
                    start_http_server(port)
                except OSError:
                    pass # Port likely usage
                    
            self.drift_hist = Histogram(
                "virk_drift_magnitude", 
                "Magnitude of detected drift (MMD)",
                ["ref"]
            )
            self.drift_counter = Counter(
                "virk_drift_events",
                "Count of drift events detected",
                ["cause"]
            )
        except ImportError:
             self.drift_hist = None
             self.drift_counter = None

    def record_drift(self, value: float, labels: Dict[str, str]):
        if self.drift_hist:
            # Prometheus python client labels handling
            lbls = {"ref": labels.get("ref", "unknown")}
            self.drift_hist.labels(**lbls).observe(value)

    def increment_drift_event(self, labels: Dict[str, str]):
        if self.drift_counter:
            lbls = {"cause": labels.get("cause", "unknown")}
            self.drift_counter.labels(**lbls).inc()

def get_backend(service_name: str, backend_type: str = "prometheus", port: int = 8000) -> MetricsBackend:
    """
    Factory to get the requested backend with robust fallback.
    Order: Requested -> Prometheus (Fallback) -> NoOp.
    """
    backend = None
    active_type = "none"

    # 1. Try Requested
    if backend_type == "otel":
        try:
            import opentelemetry
            backend = OTelBackend(service_name)
            active_type = "otel"
        except ImportError:
            logger.warning("OpenTelemetry requested but not installed. Falling back to Prometheus.")
            backend_type = "prometheus" # Fallback

    # 2. Try Prometheus (Requested or Fallback)
    if backend_type == "prometheus":
        try:
            import prometheus_client
            backend = PrometheusBackend(port=port)
            active_type = "prometheus"
        except ImportError:
            logger.warning("Prometheus requested but not installed. Falling back to NoOp.")
            backend = NoOpBackend()
            active_type = "noop"
            
    # 3. Default/None
    if backend is None:
        logger.warning("Metrics backend unavailable. Falling back to NoOp.")
        backend = NoOpBackend()
        active_type = "noop"
        
    logger.info(f"VIRK Metrics Initialized: Backend={active_type.upper()}")
    return backend
