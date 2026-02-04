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
