import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from virk.core.types import DriftProfile, Incident, ShiftFingerprint
from virk.monitor.window import WindowedDriftDetector
from virk.fingerprint.matcher import Fingerprinter
from virk.repro.bundler import ReproBundler
from virk.repro.storage import ArtifactStorage, LocalStorage, S3Storage
from virk.slicing.slicer import SliceEngine
from virk.core.concurrency import AsyncWorker
from virk.integration.metrics import MetricsBackend, NoOpBackend, OTelBackend, PrometheusBackend

logger = logging.getLogger("virk")

class VirkMiddleware:
    """
    Wraps a model prediction function to automatically monitor inputs for drift.
    Supports asynchronous processing for incident response (Fingerprinting/Bundling).
    """
    
    def __init__(self, 
                 drift_detector: WindowedDriftDetector, 
                 fingerprinter: Optional[Fingerprinter] = None,
                 bundler: Optional[ReproBundler] = None,
                 storage: Optional[ArtifactStorage] = None,
                 slicer: Optional[SliceEngine] = None,
                 service_name: str = "virk_monitor",
                 incident_threshold: float = 0.05,
                 metrics_backend: Optional[MetricsBackend] = None,
                 async_processing: bool = True):
        
        self.detector = drift_detector
        self.fingerprinter = fingerprinter
        self.bundler = bundler
        self.storage = storage
        self.slicer = slicer
        self.service_name = service_name
        self.incident_threshold = incident_threshold
        
        # Metrics Setup
        self.metrics = metrics_backend or NoOpBackend()
        
        # Async Setup
        self.async_processing = async_processing
        self.worker = AsyncWorker() if async_processing else None
        
    @classmethod
    def create(cls, 
               reference_embeddings: np.ndarray,
               extractor: Any, 
               reference_paths: List[str] = None,
               output_dir: str = "incidents",
               s3_bucket: str = None,
               metrics_type: str = "prometheus",
               service_name: str = "virk_service",
               async_processing: bool = True) -> 'VirkMiddleware':
        """
        Factory method for 'One-Line' integration with sensible defaults.
        
        Args:
            reference_embeddings: Baseline data for drift detection.
            extractor: Feature extractor (e.g. TimmExtractor) for Fingerprinter.
            reference_paths: Paths to baseline images (needed for Fingerprint calibration).
            output_dir: Local directory for bundles.
            s3_bucket: Optional S3 bucket name. If set, uses S3Storage.
            metrics_type: 'prometheus', 'otel', or 'none'.
            service_name: Name of the service for metrics.
            async_processing: Whether to run heavy tasks in background.
        """
        # 1. Components
        detector = WindowedDriftDetector(reference_embeddings)
        
        fingerprinter = None
        if reference_paths and extractor:
            fingerprinter = Fingerprinter(extractor, reference_paths)
            fingerprinter.calibrate()
            
        bundler = ReproBundler(output_dir=output_dir)
        
        storage = None
        if s3_bucket:
             storage = S3Storage(s3_bucket)
        else:
             storage = LocalStorage(output_dir)
             
        slicer = SliceEngine(detector)
        
        # 2. Metrics
        backend = NoOpBackend()
        if metrics_type == "otel":
            backend = OTelBackend(service_name)
        elif metrics_type == "prometheus":
            backend = PrometheusBackend(port=8000) # Default port
            
        return cls(
            drift_detector=detector,
            fingerprinter=fingerprinter,
            bundler=bundler,
            storage=storage,
            slicer=slicer,
            service_name=service_name,
            metrics_backend=backend,
            async_processing=async_processing
        )

    def process_batch(self, embeddings: np.ndarray, image_paths: List[str] = None, metadata: List[Dict[str, Any]] = None):
        """
        Analysis hook. Runs Drift Detection synchronously (fast O(1)), 
        then offloads heavy tasks (Fingerprint/Bundle) if drift is detected.
        """
        # 1. Fast Path: Drift Detection
        self.detector.update(embeddings)
        profile = self.detector.detect_trend()
        
        # Emit metric immediately
        self.metrics.record_drift(profile.drift_magnitude, {"ref": profile.reference_id})
        
        if profile.is_drift_detected:
            # 2. Slow Path: Incident Response
            if self.async_processing and self.worker:
                # Offload to background thread
                self.worker.submit(self._handle_drift_sync, profile, embeddings, image_paths, metadata)
            else:
                # Block
                self._handle_drift_sync(profile, embeddings, image_paths, metadata)

    def _handle_drift_sync(self, profile: DriftProfile, embeddings: np.ndarray, image_paths: List[str], metadata: List[Dict[str, Any]]):
        """
        Heavy lifting: Fingerprint -> Slice -> Bundle -> Upload.
        Executed in worker thread if async.
        """
        log_data = {
            "event": "drift_detected",
            "magnitude": profile.drift_magnitude,
            "timestamp": profile.timestamp
        }
        
        incident_data = {
            "drift_profile": profile,
            "fingerprint": ShiftFingerprint(),
            "top_slices": [],
            "metadata": {}
        }
        
        # A. Fingerprint
        if self.fingerprinter:
            try:
                fp = self.fingerprinter.diagnose(embeddings)
                incident_data["fingerprint"] = fp
                
                top_cause = max(fp.shift_types, key=fp.shift_types.get)
                log_data["cause"] = top_cause
                self.metrics.increment_drift_event({"cause": top_cause})
            except Exception as e:
                logger.error(f"Fingerprinting failed: {e}")

        # B. Slice
        if self.slicer and metadata:
            try:
                slices = self.slicer.analyze(embeddings, metadata)
                incident_data["top_slices"] = slices
                log_data["slices"] = [s.slice_name for s in slices[:3]]
            except Exception as e:
                 logger.error(f"Slicing failed: {e}")

        # C. Bundle & Upload (If Critical)
        if (profile.drift_magnitude > self.incident_threshold) and self.bundler and self.storage and image_paths:
            logger.warning(f"CRITICAL DRIFT ({profile.drift_magnitude:.4f}). Creating Incident...")
            try:
                import uuid
                incident_id = str(uuid.uuid4())[:8]
                
                incident = Incident(
                    incident_id=incident_id,
                    timestamp=profile.timestamp,
                    drift_profile=profile,
                    fingerprint=incident_data["fingerprint"],
                    top_slices=incident_data["top_slices"],
                    affected_sample_ids=[str(Path(p).name) for p in image_paths],
                    metadata={"trigger": "auto_threshold"}
                )
                
                bundle_path = self.bundler.bundle(incident, image_paths)
                remote_key = f"incident_{incident_id}.zip"
                uri = self.storage.save(Path(bundle_path), remote_key)
                
                log_data["incident_uri"] = uri
                logger.info(f"Incident Bundled: {uri}")
                
            except Exception as e:
                logger.error(f"Bundling failed: {e}")

        logger.warning(json.dumps(log_data, default=str))

    def shutdown(self):
        if self.worker:
            self.worker.shutdown()
