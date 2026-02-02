import os
import shutil
import uuid
import numpy as np
from PIL import Image
from virk.monitor.extractor import TimmExtractor
from virk.monitor.drift import DriftDetector
from virk.fingerprint.matcher import Fingerprinter
from virk.slicing.slicer import SliceEngine
from virk.repro.bundler import ReproBundler
from virk.core.types import Incident

def create_dummy_dataset(output_dir: str, num_images: int = 10, pattern: str = "noise"):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(num_images):
        img_path = os.path.join(output_dir, f"img_{i:03d}.jpg")
        if pattern == "noise":
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        elif pattern == "black":
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
        elif pattern == "white":
            arr = np.full((224, 224, 3), 255, dtype=np.uint8)
        
        img = Image.fromarray(arr)
        img.save(img_path)
        paths.append(img_path)
    return paths

def main():
    print("Initializing VIRK Demo...")
    
    # 0. Setup directories
    base_dir = "demo_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    ref_dir = os.path.join(base_dir, "ref")
    prod_dir = os.path.join(base_dir, "prod")
    
    # 1. Create Data
    print("Generating synthetic data...")
    ref_paths = create_dummy_dataset(ref_dir, num_images=5, pattern="noise")
    # Simulate drift: make prod images darker (brightness shift)
    prod_paths = create_dummy_dataset(prod_dir, num_images=10, pattern="noise")
    # Apply corruption manually to prod images to ensure drift
    from virk.fingerprint.corruptions import Corruptions
    for p in prod_paths:
        img = Image.open(p)
        # Apply severe brightness reduction
        corrupted = Corruptions.brightness(img, severity=5) 
        corrupted.save(p)
        
    # Metadata for slicing
    prod_metadata = []
    for i in range(10):
        # Biased metadata: all corrupted ones are from cam_01
        prod_metadata.append({
            "camera_id": "cam_01" if i < 8 else "cam_02",
            "time": "night"
        })
    
    # 2. Pipeline Execution
    print("Running Pipeline...")
    
    # A. Monitor
    extractor = TimmExtractor(model_name='resnet18', pretrained=True)
    ref_embeddings = extractor.extract(ref_paths)
    prod_embeddings = extractor.extract(prod_paths)
    
    detector = DriftDetector(ref_embeddings)
    drift_profile = detector.detect(prod_embeddings, threshold=0.001)
    
    print(f"Drift Detected: {drift_profile.is_drift_detected} (Magnitude: {drift_profile.drift_magnitude:.4f})")
    
    if drift_profile.is_drift_detected:
        # B. Fingerprint
        print("Diagnosing cause...")
        fingerprinter = Fingerprinter(extractor, ref_paths)
        print("Calibrating fingerprinter...")
        fingerprinter.calibrate()
        fingerprint = fingerprinter.diagnose(prod_embeddings)
        print("Fingerprint:", fingerprint.shift_types)
        
        # C. Slicing
        print("Analyzing slices...")
        slicer = SliceEngine(detector)
        slices = slicer.analyze(prod_embeddings, prod_metadata)
        print("Top Slices:", [s.slice_name for s in slices])
        
        # D. Bundle
        print("Generating Repro Pack...")
        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            timestamp=drift_profile.timestamp,
            drift_profile=drift_profile,
            fingerprint=fingerprint,
            top_slices=slices,
            affected_sample_ids=[os.path.basename(p) for p in prod_paths],
            metadata={"description": "Demo Incident"}
        )
        
        bundler = ReproBundler(output_dir="demo_output")
        zip_path = bundler.bundle(incident, prod_paths)
        print(f"Bundle created at: {zip_path}")
        
        return zip_path
    
    return None

if __name__ == "__main__":
    main()
