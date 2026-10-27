"""Real CPU pipeline check, enabled in the pinned model-smoke CI job."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIRK_MODEL_SMOKE") != "1",
    reason="Set VIRK_MODEL_SMOKE=1 in the pinned research environment",
)


def test_timm_extraction_calibration_and_model_replay(tmp_path):
    import json
    import zipfile
    import numpy as np
    import torch
    from PIL import Image
    from virk.eval.calibration import CalibrationEngine
    from virk.integration.middleware import VirkMiddleware
    from virk.monitor.extractor import TimmExtractor
    from virk.repro.replay import inspect_bundle

    torch.set_num_threads(2)
    torch.manual_seed(42)
    extractor = TimmExtractor("resnet18", pretrained=False)
    clean, test = tmp_path / "clean", tmp_path / "test"
    clean.mkdir()
    test.mkdir()
    rng = np.random.default_rng(42)
    paths = []
    for i in range(12):
        path = (clean if i < 8 else test) / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    features = extractor.extract(paths[:8])
    assert features.shape == (8, 512)
    assert np.isfinite(features).all()
    report = CalibrationEngine(str(clean), extractor=extractor).calibrate(
        batch_size=2, num_batches=2, test_dataset_path=str(test)
    )
    assert report["independent_test"]["num_batches"] == 2
    json.dumps(report, allow_nan=False)
    monitor = VirkMiddleware.create(
        features[:4],
        extractor=extractor,
        output_dir=str(tmp_path / "incidents"),
        metrics_type="none",
        async_processing=False,
        window_size=4,
        min_window_size=4,
        drift_threshold=0,
        incident_threshold=0,
        capture_model_state=True,
        seed=42,
    )
    try:
        monitor.process_batch(features[4:], image_paths=paths[4:8])
    finally:
        monitor.shutdown()
    (bundle,) = (tmp_path / "incidents").glob("incident_*.zip")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extracted)
    context = json.loads((extracted / "replay_context.json").read_text())
    assert context["configuration"]["extractor"]["model_name"] == "resnet18"
    assert "preprocessing" in context["configuration"]["extractor"]
    assert inspect_bundle(extracted / "manifest.json", verify_drift=True)[
        "is_drift_detected"
    ]
    restored = TimmExtractor("resnet18", pretrained=False)
    restored.model.load_state_dict(
        torch.load(extracted / "model_state.pt", map_location="cpu", weights_only=True)
    )
    np.testing.assert_allclose(
        restored.extract(paths[4:8]), features[4:], rtol=1e-5, atol=1e-6
    )
