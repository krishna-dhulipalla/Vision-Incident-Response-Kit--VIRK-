import numpy as np
import pytest
from virk.monitor.drift import DriftDetector

def test_drift_detector_initialization():
    ref = np.random.rand(100, 10)
    detector = DriftDetector(ref, max_samples=50)
    
    assert len(detector.reference_embeddings) == 50, "Should subsample reference on init"
    assert detector.max_samples == 50

def test_drift_detection_logic():
    # Baseline: Gaussian centered at 0
    ref = np.random.normal(0, 1, (100, 10))
    detector = DriftDetector(ref)
    
    # Test 1: Same distribution (should be low MMD)
    same_dist = np.random.normal(0, 1, (50, 10))
    profile = detector.detect(same_dist, threshold=0.1)
    # MMD is never exactly 0 due to finite sampling, but should be small
    assert not profile.is_drift_detected, f"False positive detected: {profile.drift_magnitude}"
    
    # Test 2: Shifted distribution (should conform)
    shifted_dist = np.random.normal(2, 1, (50, 10))
    profile = detector.detect(shifted_dist, threshold=0.01)
    assert profile.is_drift_detected, "Failed to detect shift"
    assert profile.drift_magnitude > 0.1

def test_large_batch_subsampling():
    ref = np.random.rand(50, 10)
    detector = DriftDetector(ref, max_samples=20)
    
    # Large batch
    large_batch = np.random.rand(100, 10)
    # This should not raise error and run fast
    profile = detector.detect(large_batch)
    
    # We can't easily assert internal state without mocking, 
    # but execution success implies subsampling worked (or full calc worked)
    assert isinstance(profile.drift_magnitude, float)


@pytest.mark.parametrize("embeddings", [
    np.empty((0, 2)), np.empty((2, 0)), np.array([1.0, 2.0]),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([["bad"]]),
])
def test_invalid_embeddings_rejected(embeddings):
    with pytest.raises(ValueError):
        DriftDetector(embeddings)
    with pytest.raises(ValueError):
        DriftDetector(np.ones((2, 2))).detect(embeddings)


def test_feature_dimensions_must_match():
    with pytest.raises(ValueError, match="same feature dimension"):
        DriftDetector(np.ones((2, 2))).detect(np.ones((2, 3)))


def test_seed_controls_both_reference_and_current_subsampling():
    rng = np.random.default_rng(42)
    reference, current = rng.normal(size=(80, 3)), rng.normal(size=(100, 3))
    first = DriftDetector(reference, max_samples=10, random_state=3)
    second = DriftDetector(reference, max_samples=10, random_state=3)
    np.testing.assert_array_equal(first.reference_embeddings, second.reference_embeddings)
    assert first.detect(current).drift_magnitude == second.detect(current).drift_magnitude


def test_reference_is_not_modified_by_caller():
    reference = np.ones((4, 2))
    detector = DriftDetector(reference)
    reference[:] = 10
    assert detector.detect(np.ones((4, 2))).drift_magnitude == pytest.approx(0)


def test_global_seed_still_controls_default_subsampling():
    state = np.random.get_state()
    try:
        reference = np.arange(60, dtype=float).reshape(20, 3)
        np.random.seed(7)
        first = DriftDetector(reference, max_samples=5)
        np.random.seed(7)
        second = DriftDetector(reference, max_samples=5)
        np.testing.assert_array_equal(first.reference_embeddings, second.reference_embeddings)
    finally:
        np.random.set_state(state)
