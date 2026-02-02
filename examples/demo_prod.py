import os
import shutil
import time
import numpy as np
from PIL import Image
from virk.monitor.extractor import TimmExtractor
from virk.monitor.window import WindowedDriftDetector
from virk.fingerprint.matcher import Fingerprinter
from virk.slicing.slicer import SliceEngine
from virk.integration.middleware import VirkMiddleware
from virk.fingerprint.corruptions import Corruptions

def create_stream_data(output_dir: str, num_samples: int = 50):
    """Simulate a stream of data where drift happens in the middle."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    metadata = []
    
    for i in range(num_samples):
        img_path = os.path.join(output_dir, f"stream_{i:03d}.jpg")
        
        # Base image: noise
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        
        # Metadata stuff
        # Camera 1 is clean, Camera 2 gets dirty later
        cam_id = "cam_01" if i % 2 == 0 else "cam_02"
        # Time of day cycles
        tod = "day" if i % 10 < 5 else "night"
        
        # Drift injection:
        # After index 30, EVERYTHING drifts to show signal clearly
        is_drifted = False
        if i > 30:
            img = Corruptions.gaussian_blur(img, severity=4)
            is_drifted = True
            
        img.save(img_path)
        paths.append(img_path)
        metadata.append({"camera_id": cam_id, "time": tod})
        
    return paths, metadata

def main():
    print("Initializing VIRK Production Demo...")
    base_dir = "demo_prod_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    # 1. Setup Components
    # Load model
    extractor = TimmExtractor(model_name='resnet18', pretrained=True)
    
    # Create reference set (first 10 synthetic images)
    ref_paths, _ = create_stream_data(os.path.join(base_dir, "ref"), num_samples=10)
    ref_embs = extractor.extract(ref_paths)
    
    # Initialize Detectors
    # Use Windowed Detector
    window_detector = WindowedDriftDetector(ref_embs, window_size=20)
    
    # Initialize Calibrated Fingerprinter
    fingerprinter = Fingerprinter(extractor, ref_paths)
    fingerprinter.calibrate()
    
    # Initialize Middleware
    middleware = VirkMiddleware(window_detector, fingerprinter, service_name="vir_demo")
    
    # Initialize Slicer for ad-hoc analysis
    slicer = SliceEngine(window_detector)
    
    # 2. Simulate Production Stream
    print("Simulating Traffic Stream...")
    stream_paths, stream_meta = create_stream_data(os.path.join(base_dir, "prod"), num_samples=50)
    
    # Process batch by batch to simulate time
    batch_size = 5
    drift_batches = []
    
    for i in range(0, len(stream_paths), batch_size):
        batch_paths = stream_paths[i:i+batch_size]
        batch_meta = stream_meta[i:i+batch_size]
        
        # Model Inference (Simulated)
        embs = extractor.extract(batch_paths)
        
        # VIRK Middleware Hook
        print(f"Processing batch {i//batch_size}...")
        middleware.process_batch(embs, batch_meta)
        
        # Check internal state for demo purpose
        # In real life, we'd check logs/metrics
        profile = window_detector.detect_trend()
        if profile.is_drift_detected:
            print(f"  [ALERT] Drift Detected! Mag: {profile.drift_magnitude:.4f}")
            drift_batches.extend(range(i, i+batch_size))
            
            # 3. Ad-hoc Diagnostics on Drifted Data
            # If we accumulated enough drifted data, run diagnosis
            # (In middleware this happens per batch, but let's do a bigger slice analysis here)
            if len(drift_batches) >= 10:
                print("  Running Deep Diagnostics on recent drift...")
                recent_indices = drift_batches[-20:] # Analyze last 20 drifted items
                
                # Get embeddings for these
                start_idx = recent_indices[0]
                end_idx = recent_indices[-1] + 1
                # Re-extract or slice... simplified for demo, assuming contiguous memory in synthetic list
                # actually we need to gather them.
                # Let's just use the current batch for slicing demo
                
                # Multi-dim Slicing
                # The issue is cam_02 AND night
                slices = slicer.analyze(embs, batch_meta, top_k=3, max_dims=2)
                for s in slices:
                     print(f"    Possible culprit slice: {s.slice_name} (Score: {s.contribution_score:.4f})")
                     
    print("Demo Complete.")

if __name__ == "__main__":
    main()
