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
            logger.warning("OpenTelemetry not installed. Metrics will be dropped.")
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
             logger.warning("Prometheus client not installed.")
             self.drift_hist = None
             self.drift_counter = None

    def record_drift(self, value: float, labels: Dict[str, str]):
        if self.drift_hist:
            # Prometheus python client labels handling
            # Filter labels to match expected keys if stricter, but here we just pass 'ref'
            lbls = {"ref": labels.get("ref", "unknown")}
            self.drift_hist.labels(**lbls).observe(value)

    def increment_drift_event(self, labels: Dict[str, str]):
        if self.drift_counter:
            lbls = {"cause": labels.get("cause", "unknown")}
            self.drift_counter.labels(**lbls).inc()
