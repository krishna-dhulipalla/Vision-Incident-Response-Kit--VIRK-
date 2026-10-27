import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np

from virk.core.types import DriftProfile, Incident, ShiftFingerprint, DetectionEvidence
from virk.monitor.window import WindowedDriftDetector
from virk.repro.bundler import ReproBundler
from virk.repro.storage import ArtifactStorage, LocalStorage, S3Storage
from virk.slicing.slicer import SliceEngine
from virk.core.concurrency import AsyncWorker
from virk.integration.metrics import MetricsBackend, NoOpBackend

if TYPE_CHECKING:
    from virk.fingerprint.matcher import Fingerprinter

logger = logging.getLogger("virk")


class VirkMiddleware:
    """
    Wraps a model prediction function to automatically monitor inputs for drift.
    Supports asynchronous processing for incident response (Fingerprinting/Bundling).
    """

    def __init__(
        self,
        drift_detector: WindowedDriftDetector,
        fingerprinter: Optional["Fingerprinter"] = None,
        bundler: Optional[ReproBundler] = None,
        storage: Optional[ArtifactStorage] = None,
        slicer: Optional[SliceEngine] = None,
        service_name: str = "virk_monitor",
        incident_threshold: float = 0.05,
        metrics_backend: Optional[MetricsBackend] = None,
        async_processing: bool = True,
        drift_threshold: float = 0.02,
    ):

        self.detector = drift_detector
        self.fingerprinter = fingerprinter
        self.bundler = bundler
        self.storage = storage
        self.slicer = slicer
        self.service_name = service_name
        self.incident_threshold = incident_threshold
        if not np.isfinite(drift_threshold) or drift_threshold < 0:
            raise ValueError("drift_threshold must be finite and nonnegative")
        self.drift_threshold = drift_threshold

        # Metrics Setup
        self.metrics = metrics_backend or NoOpBackend()

        # Async Setup
        self.async_processing = async_processing
        self.worker = AsyncWorker() if async_processing else None

    @classmethod
    def create(
        cls,
        reference_embeddings: np.ndarray,
        extractor: Any,
        reference_paths: List[str] = None,
        output_dir: str = "incidents",
        s3_bucket: str = None,
        metrics_type: str = "prometheus",
        service_name: str = "virk_service",
        async_processing: bool = True,
        drift_threshold: float = 0.02,
        incident_threshold: float = 0.05,
        window_size: int = 1000,
        min_window_size: Optional[int] = None,
        max_samples: int = 1000,
        seed: Optional[int] = None,
        replay_config: Optional[Dict[str, Any]] = None,
        capture_model_state: bool = False,
        model_state_path: Optional[str] = None,
    ) -> "VirkMiddleware":
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
            drift_threshold: Threshold for window drift checks; validate for this monitoring setup.
            incident_threshold: Threshold for bundling detected drift.
            window_size: Number of recent samples retained.
            min_window_size: Samples required before scoring; default 25% fill.
            max_samples: Maximum reference/current samples in each MMD computation.
            seed: Optional reproducible subsampling seed.
            replay_config: Custom model identity/preprocessing for non-TIMM extractors.
            capture_model_state: Save exact weights once and include them in incidents.
            model_state_path: Existing weights file for custom extractors.
        """
        # 1. Components
        detector = WindowedDriftDetector(
            reference_embeddings,
            window_size=window_size,
            min_window_size=min_window_size,
            max_samples=max_samples,
            random_state=seed,
        )

        fingerprinter = None
        if reference_paths and extractor:
            from virk.fingerprint.matcher import Fingerprinter

            fingerprinter = Fingerprinter(extractor, reference_paths)
            fingerprinter.calibrate()

        config = dict(replay_config or {})
        config.setdefault(
            "extractor",
            (
                extractor.describe()
                if hasattr(extractor, "describe")
                else {
                    "class": type(extractor).__name__,
                    "preprocessing": "Not supplied",
                }
            ),
        )
        config["monitoring"] = {
            "window_size": window_size,
            "min_window_size": detector.min_window_size,
            "max_samples": max_samples,
            "seed": seed,
            "drift_threshold": drift_threshold,
            "incident_threshold": incident_threshold,
        }
        if capture_model_state:
            if model_state_path is not None:
                raise ValueError(
                    "Choose capture_model_state or model_state_path, not both"
                )
            if not hasattr(extractor, "save_model_state"):
                raise ValueError(
                    "Extractor must implement save_model_state or supply model_state_path"
                )
            import uuid

            model_state_path = extractor.save_model_state(
                Path(output_dir) / f"model_state_{uuid.uuid4().hex}.pt"
            )
        bundler = ReproBundler(
            output_dir=output_dir,
            replay_config=config,
            model_state_path=model_state_path,
        )

        storage = None
        if s3_bucket:
            storage = S3Storage(s3_bucket)
        else:
            storage = LocalStorage(output_dir)

        slicer = SliceEngine(detector)

        # 2. Metrics
        from virk.integration.metrics import get_backend

        backend = get_backend(service_name, metrics_type)

        return cls(
            drift_detector=detector,
            fingerprinter=fingerprinter,
            bundler=bundler,
            storage=storage,
            slicer=slicer,
            service_name=service_name,
            metrics_backend=backend,
            async_processing=async_processing,
            drift_threshold=drift_threshold,
            incident_threshold=incident_threshold,
        )

    def process_batch(
        self,
        embeddings: np.ndarray,
        image_paths: List[str] = None,
        metadata: List[Dict[str, Any]] = None,
    ):
        """
        Analysis hook. Runs bounded MMD drift detection synchronously,
        then offloads heavy tasks (Fingerprint/Bundle) if drift is detected.
        """
        # 1. Fast Path: Drift Detection
        self.detector.update(embeddings)
        profile, evidence = self.detector.detect_trend_with_evidence(
            threshold=self.drift_threshold
        )

        # Emit metric immediately
        self.metrics.record_drift(
            profile.drift_magnitude, {"ref": profile.reference_id}
        )

        if profile.is_drift_detected:
            # 2. Slow Path: Incident Response
            if self.async_processing and self.worker:
                # Offload to background thread
                from copy import deepcopy

                self.worker.submit(
                    self._handle_drift_sync,
                    profile,
                    embeddings.copy(),
                    list(image_paths) if image_paths is not None else None,
                    deepcopy(metadata),
                    evidence,
                )
            else:
                # Block
                self._handle_drift_sync(
                    profile, embeddings, image_paths, metadata, evidence
                )

    def _handle_drift_sync(
        self,
        profile: DriftProfile,
        embeddings: np.ndarray,
        image_paths: List[str],
        metadata: List[Dict[str, Any]],
        evidence: Optional[DetectionEvidence] = None,
    ):
        """
        Heavy lifting: Fingerprint -> Slice -> Bundle -> Upload.
        Executed in worker thread if async.
        """
        log_data = {
            "event": "drift_detected",
            "magnitude": profile.drift_magnitude,
            "timestamp": profile.timestamp,
        }

        incident_data = {
            "drift_profile": profile,
            "fingerprint": ShiftFingerprint(),
            "top_slices": [],
            "metadata": {},
        }

        # A. Fingerprint
        if self.fingerprinter:
            try:
                fp = self.fingerprinter.diagnose(embeddings)
                incident_data["fingerprint"] = fp

                top_cause = max(fp.shift_types, key=fp.shift_types.get)
                log_data["cause"] = top_cause
            except Exception as e:
                logger.error(f"Fingerprinting failed: {e}")

        self.metrics.increment_drift_event({"cause": log_data.get("cause", "unknown")})

        # B. Slice
        if self.slicer and metadata:
            try:
                slices = self.slicer.analyze(embeddings, metadata)
                incident_data["top_slices"] = slices
                log_data["slices"] = [s.slice_name for s in slices[:3]]
            except Exception as e:
                logger.error(f"Slicing failed: {e}")

        # C. Bundle & Upload (If Critical)
        if (
            (profile.drift_magnitude > self.incident_threshold)
            and self.bundler
            and self.storage
            and image_paths
        ):
            logger.warning(
                f"CRITICAL DRIFT ({profile.drift_magnitude:.4f}). Creating Incident..."
            )
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
                    metadata={"trigger": "auto_threshold"},
                )

                bundle_path = self.bundler.bundle(
                    incident, image_paths, evidence=evidence
                )
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
