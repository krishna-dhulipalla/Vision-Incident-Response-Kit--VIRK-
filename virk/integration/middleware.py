import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import numpy as np

from virk.core.types import DriftProfile, Incident
from virk.monitor.window import WindowedDriftDetector
from virk.fingerprint.matcher import Fingerprinter
from virk.repro.bundler import ReproBundler
from virk.repro.storage import ArtifactStorage
from virk.slicing.slicer import SliceEngine

# Try importing OTel, graceful fallback if not present
try:
    from opentelemetry import metrics
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
                 bundler: Optional[ReproBundler] = None,
                 storage: Optional[ArtifactStorage] = None,
                 slicer: Optional[SliceEngine] = None,
                 service_name: str = "virk_monitor",
                 incident_threshold: float = 0.05):
        """
        Args:
            drift_detector: Stateful detector (windowed).
            fingerprinter: Optional engine to diagnose shift type.
            bundler: Optional generator for repro zip packs.
            storage: Optional backend to persist bundles (S3/Local).
            slicer: Optional engine to find contributing slices.
            incident_threshold: MMD score above which an Incident is declared and bundled.
        """
        self.detector = drift_detector
        self.fingerprinter = fingerprinter
        self.bundler = bundler
        self.storage = storage
        self.slicer = slicer
        self.service_name = service_name
        self.incident_threshold = incident_threshold
        
        # Initialize OTel Meter
        self.meter = None
        if HAS_OTEL:
            self.meter = metrics.get_meter(service_name)
            # Use Histogram for drift magnitude (Event-based recording)
            self.drift_histogram = self.meter.create_histogram(
                name="virk_drift_magnitude",
                description="Magnitude of detected drift (MMD)"
            )
            
            self.drift_counter = self.meter.create_counter(
                name="virk_drift_events",
                description="Count of drift events detected"
            )

    def process_batch(self, embeddings: np.ndarray, image_paths: List[str] = None, metadata: List[Dict[str, Any]] = None):
        """
        Analysis hook. Does not block critical path if run async (user responsibility).
        
        Args:
            embeddings: (Batch, Dim) feature vectors.
            image_paths: (Batch,) paths to source images (required for bundling).
            metadata: (Batch,) dicts of metadata (required for slicing).
        """
        # 1. Update Window
        self.detector.update(embeddings)
        
        # 2. Check Drift (Trend)
        # We verify trend (window vs baseline)
        profile = self.detector.detect_trend()
        
        if profile.is_drift_detected:
            self._handle_drift(profile, embeddings, image_paths, metadata)
        
        self._emit_metrics(profile)

    def _handle_drift(self, profile: DriftProfile, embeddings: np.ndarray, image_paths: List[str], metadata: List[Dict[str, Any]]):
        """Log drift, fingerprint, slice, and bundle if critical."""
        log_data = {
            "event": "drift_detected",
            "magnitude": profile.drift_magnitude,
            "timestamp": profile.timestamp,
            "ref": profile.reference_id
        }
        
        incident_data = {
            "drift_profile": profile.__dict__,
            "fingerprint": {},
            "top_slices": [],
            "metadata": {}
        }
        
        # 1. Fingerprint (Diagnose Cause)
        if self.fingerprinter:
            try:
                fingerprint = self.fingerprinter.diagnose(embeddings)
                
                # Get top cause
                top_cause = max(fingerprint.shift_types, key=fingerprint.shift_types.get)
                log_data["fingerprint"] = fingerprint.shift_types
                log_data["top_cause"] = top_cause
                incident_data["fingerprint"] = fingerprint  # Keep as Object
                
                if HAS_OTEL:
                    self.drift_counter.add(1, {"cause": top_cause})
            except Exception as e:
                logger.error(f"Fingerprinting failed: {e}")

        # 2. Slicing (Attribution)
        if self.slicer and metadata:
            try:
                slices = self.slicer.analyze(embeddings, metadata)
                log_data["top_slices"] = [s.__dict__ for s in slices[:3]]
                incident_data["top_slices"] = slices # Keep as Objects
            except Exception as e:
                logger.error(f"Slicing failed: {e}")

        # 3. Incident Bundling (If Critical)
        if (profile.drift_magnitude > self.incident_threshold) and self.bundler and self.storage and image_paths:
            logger.warning(f"CRITICAL DRIFT ({profile.drift_magnitude:.4f} > {self.incident_threshold}). Initiating Incident Response.")
            
            try:
                # Create Incident Object
                import uuid
                incident_id = str(uuid.uuid4())[:8]
                
                # Handle missing diagnosis if components failed/absent
                from virk.core.types import ShiftFingerprint
                fp_obj = incident_data.get("fingerprint") or ShiftFingerprint()
                slices_obj = incident_data.get("top_slices") or []
                
                incident = Incident(
                    incident_id=incident_id,
                    timestamp=profile.timestamp,
                    drift_profile=profile,
                    fingerprint=fp_obj,
                    top_slices=slices_obj,
                    affected_sample_ids=[str(Path(p).name) for p in image_paths],
                    metadata={"trigger": "auto_threshold"}
                )
                
                # Bundle
                bundle_path = self.bundler.bundle(incident, image_paths)
                
                # Upload
                remote_key = f"incident_{incident_id}.zip"
                uri = self.storage.save(Path(bundle_path), remote_key)
                
                log_data["incident_uri"] = uri
                logger.info(f"Incident Bundled & Uploaded: {uri}")
                
            except Exception as e:
                logger.error(f"Incident bundling/upload failed: {e}")

        # Structured JSON Log
        logger.warning(json.dumps(log_data, default=str))

    def _emit_metrics(self, profile: DriftProfile):
        """Send metrics to backend."""
        # Simple logging fallback
        if not HAS_OTEL:
            return

        # Record drift magnitude
        self.drift_histogram.record(profile.drift_magnitude, {"ref": profile.reference_id})
