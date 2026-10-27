import json
import zipfile

import numpy as np
import pytest

from virk.core.types import Incident, ShiftFingerprint
from virk.monitor.drift import DriftDetector
from virk.repro.bundler import ReproBundler
from virk.repro.replay import inspect_bundle


def test_exact_subsampled_score_replay_and_artifact_integrity(tmp_path):
    rng = np.random.default_rng(42)
    detector = DriftDetector(rng.normal(size=(30, 4)), max_samples=8, random_state=7)
    profile, evidence = detector.detect_with_evidence(
        rng.normal(3, 1, (40, 4)), threshold=0.1
    )
    original_score = profile.drift_magnitude
    # Later detector calls cannot alter the queued incident evidence.
    detector.detect(rng.normal(size=(20, 4)))
    incident = Incident("replay", "now", profile, ShiftFingerprint(), [], [])
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"custom checkpoint")
    bundler = ReproBundler(
        str(tmp_path / "out"),
        replay_config={"extractor": {"model_name": "custom"}},
        model_state_path=weights,
    )
    archive_path = bundler.bundle(incident, [], evidence=evidence)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    context = json.loads((extracted / "replay_context.json").read_text())
    assert context["reference_shape"] == [8, 4]
    assert context["current_shape"] == [8, 4]
    assert context["model_state_included"]
    result = inspect_bundle(extracted / "manifest.json", verify_drift=True)
    assert result["score"] == pytest.approx(original_score)
    (extracted / "current_embeddings.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        inspect_bundle(extracted / "manifest.json", verify_drift=True)


def test_old_bundle_has_explicit_missing_context_error(tmp_path):
    data = {
        "incident_id": "old",
        "drift_profile": {"drift_magnitude": 0.1, "is_drift_detected": True},
        "fingerprint": {"shift_types": {}},
        "top_slices": [],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    assert inspect_bundle(manifest)["incident_id"] == "old"
    with pytest.raises(ValueError, match="no recorded drift replay context"):
        inspect_bundle(manifest, verify_drift=True)
