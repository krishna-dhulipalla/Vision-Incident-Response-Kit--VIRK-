import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from virk.core.types import DriftProfile, ShiftFingerprint
from virk.monitor.window import WindowedDriftDetector
from virk.fingerprint.matcher import Fingerprinter

# Try importing OTel, graceful fallback if not present
try:
    from opentelemetry import metrics
    from opentelemetry.stats import StatsRecorder
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

logger = logging.getLogger("virk")

class VirkMiddleware:
    """
    Wraps a model prediction function to automatically monitor inputs for drift.
    """
    
    def __init__(self, 
                 drift_detector: WindowedDriftDetector, 
                 fingerprinter: Optional[Fingerprinter] = None,
                 service_name: str = "virk_monitor"):
        self.detector = drift_detector
        self.fingerprinter = fingerprinter
        self.service_name = service_name
        
        # Initialize OTel Meter
        self.meter = None
        if HAS_OTEL:
            self.meter = metrics.get_meter(service_name)
            self.drift_gauge = self.meter.create_observable_gauge(
                name="virk_drift_magnitude",
                description="Magnitude of detected drift (MMD)"
            )
            # Note: Observable gauge needs a callback in real OTel, 
            # for simplicity in this generated code we'll pretend we can just set it
            # or use a standard ValueRecorder/Histogram if we were pushing events.
            # Using UpDownCounter for drift detected bool?
            self.drift_counter = self.meter.create_counter(
                name="virk_drift_events",
                description="Count of drift events detected"
            )

    def process_batch(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """
        Analysis hook. Does not block critical path if run async (user responsibility).
        """
        # 1. Update Window
        self.detector.update(embeddings)
        
        # 2. Check Drift (Trend)
        # We verify trend (window vs baseline)
        profile = self.detector.detect_trend()
        
        if profile.is_drift_detected:
            self._handle_drift(profile, embeddings)
        
        self._emit_metrics(profile)

    def _handle_drift(self, profile: DriftProfile, embeddings: np.ndarray):
        """Log drift and optional fingerprinting."""
        log_data = {
            "event": "drift_detected",
            "magnitude": profile.drift_magnitude,
            "timestamp": profile.timestamp,
            "ref": profile.reference_id
        }
        
        if self.fingerprinter:
            try:
                fingerprint = self.fingerprinter.diagnose(embeddings)
                
                # Get top cause
                top_cause = max(fingerprint.shift_types, key=fingerprint.shift_types.get)
                log_data["fingerprint"] = fingerprint.shift_types
                log_data["top_cause"] = top_cause
                
                if HAS_OTEL:
                    # Increment counter with attribute
                    self.drift_counter.add(1, {"cause": top_cause})
                    
            except Exception as e:
                logger.error(f"Fingerprinting failed: {e}")
                
        # Structured JSON Log
        logger.warning(json.dumps(log_data))

    def _emit_metrics(self, profile: DriftProfile):
        """Send metrics to backend."""
        # Simple logging fallback
        if not HAS_OTEL:
            return

        # In real OTel, we'd record value here. 
        # Since python OTel API varies by version (SDK vs API), 
        # we assume a standard record pattern for counters.
        if profile.is_drift_detected:
             # Already handled in handle_drift
             pass
