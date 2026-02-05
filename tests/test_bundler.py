import pytest
import os
import json
import zipfile
from pathlib import Path
from virk.repro.bundler import ReproBundler
from virk.core.types import Incident, DriftProfile, ShiftFingerprint

def test_bundler_integrity(tmp_path):
    # 1. Setup
    bundler = ReproBundler(output_dir=str(tmp_path))
    
    # Create mock images
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test1.jpg"
    img1.write_bytes(b"image data 1")
    img2 = img_dir / "test2.jpg"
    img2.write_bytes(b"image data 2")
    
    incident = Incident(
        incident_id="test_incident",
        timestamp="2024-01-01",
        drift_profile=DriftProfile(
            is_drift_detected=True, 
            drift_magnitude=0.5,
            reference_id="ref_01",
            timestamp="2024-01-01"
        ),
        fingerprint=ShiftFingerprint(shift_types={"blur": 0.8}),
        top_slices=[],
        affected_sample_ids=["test1.jpg", "test2.jpg"]
    )
    
    # 2. Run Bundle
    zip_path = bundler.bundle(incident, [str(img1), str(img2)])
    
    assert os.path.exists(zip_path)
    
    # 3. Verify Contents
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        
        # Check structure
        assert "manifest.json" in names
        assert "replay.py" in names
        assert "requirements.txt" in names # New!
        assert "images/test1.jpg" in names
        
        # Verify Manifest Schema
        with zf.open("manifest.json") as f:
            data = json.load(f)
            assert data["incident_id"] == "test_incident"
            assert data["metadata"]["schema_version"] == "1.0.0" # New!
            assert "data_hash" in data["metadata"]
            
        # Verify Requirements
        # Note: pip freeze might fail in some test envs, or produce empty output if no packages.
        # We check file existence (critical) and content if possible.
        if "requirements.txt" in names:
            # Just verify it opens
            with zf.open("requirements.txt") as f:
                _ = f.read()
