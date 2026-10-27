from unittest.mock import Mock
import zipfile

import numpy as np
import pytest

from virk.core.types import DriftProfile
from virk.integration.middleware import VirkMiddleware


def test_middleware_uses_configured_threshold_and_counts_unattributed_drift():
    detector, metrics = Mock(), Mock()
    detector.detect_trend_with_evidence.return_value = (
        DriftProfile(True, 0.04, "baseline", "now"),
        None,
    )
    monitor = VirkMiddleware(
        detector, metrics_backend=metrics, async_processing=False, drift_threshold=0.03
    )
    embeddings = np.ones((4, 2))
    monitor.process_batch(embeddings)
    detector.detect_trend_with_evidence.assert_called_once_with(threshold=0.03)
    metrics.increment_drift_event.assert_called_once_with({"cause": "unknown"})


def test_middleware_factory_forwards_threshold(tmp_path):
    monitor = VirkMiddleware.create(
        np.ones((4, 2)),
        extractor=None,
        metrics_type="none",
        output_dir=str(tmp_path),
        async_processing=False,
        drift_threshold=0.07,
    )
    assert monitor.drift_threshold == 0.07


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_drift_threshold_rejected_before_starting_worker(threshold):
    with pytest.raises(ValueError, match="drift_threshold"):
        VirkMiddleware(Mock(), drift_threshold=threshold)


def test_async_incident_is_bundled_and_saved_locally(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: b"numpy\n")
    image = tmp_path / "frame.png"
    image.write_bytes(b"incident evidence")
    output_dir = tmp_path / "incidents"
    monitor = VirkMiddleware.create(
        np.zeros((300, 2)),
        extractor=None,
        output_dir=str(output_dir),
        metrics_type="none",
        async_processing=True,
    )
    try:
        monitor.process_batch(np.full((250, 2), 2.0), image_paths=[str(image)])
    finally:
        monitor.shutdown()
    bundles = list(output_dir.glob("incident_*.zip"))
    assert len(bundles) == 1
    assert "Bundling failed" not in caplog.text
    assert "Incident Bundled" in caplog.text or "incident_uri" in caplog.text
    with zipfile.ZipFile(bundles[0]) as archive:
        assert archive.read("images/frame.png") == b"incident evidence"
