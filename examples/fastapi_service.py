import time
import uuid
import shutil
import numpy as np
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks

# VIRK Imports
from virk.monitor.drift import DriftDetector
from virk.monitor.window import WindowedDriftDetector
from virk.fingerprint.matcher import Fingerprinter
from virk.slicing.slicer import SliceEngine
from virk.repro.bundler import ReproBundler
from virk.repro.storage import LocalStorage
from virk.integration.middleware import VirkMiddleware
from virk.integration.prometheus import init_prometheus

# Mock Model for Demo
class ResNetClassifier:
    def __init__(self):
        # Simulator: Returns 512-dim embeddings
        self.dim = 512
        
    def encode(self, image_paths: List[str]) -> np.ndarray:
        # Simulate embeddings
        # If filename contains "drift", shift the distribution
        
        batch_size = len(image_paths)
        base = np.random.normal(0, 1, (batch_size, self.dim))
        
        for i, p in enumerate(image_paths):
            if "drift" in p:
                # Add shift to simulate Blur/Noise impact
                base[i] += np.random.normal(2.0, 0.5, self.dim)
                
        return base.astype(np.float32)

    def predict(self, embs: np.ndarray):
        return np.argmax(embs[:, :10], axis=1).tolist()

# --- Setup ---
app = FastAPI(title="VIRK Demo Service")
model = ResNetClassifier()

# 1. Initialize Baseline (Reference)
print("Initializing VIRK Monitor...")
# Real app: Load from disk
ref_data = np.random.normal(0, 1, (200, 512)).astype(np.float32) 

# 2. Monitor Setup (The New Way: Factory Pattern)
# Note: For demo simplicity, we use the same ResNetClassifier as "extractor".
# In reality, you'd pass the actual feature extractor (timm model).
monitor = VirkMiddleware.create(
    reference_embeddings=ref_data,
    extractor=model, # Mock extractor
    reference_paths=None, # Skip fingerprinter for this mocked demo
    output_dir="./incidents",
    metrics_type="prometheus", # Built-in!
    service_name="demo-vision-service",
    async_processing=True # Non-blocking!
)

@app.post("/predict")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Inference endpoint. 
    Accepts images -> Saves temp -> Predicts -> cleanup.
    """
    # 1. Save uploaded files temporarily (Production usually has them on disk/S3)
    batch_id = str(uuid.uuid4())
    temp_dir = Path(f"temp_inference/{batch_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for file in files:
        p = temp_dir / file.filename
        with open(p, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_paths.append(str(p.absolute()))
    
    # 2. Mock Metadata (e.g. from headers)
    # We pretend half come from "cam_01" and half "cam_02"
    metadata = []
    for i, _ in enumerate(image_paths):
        metadata.append({
            "camera_id": "cam_01" if i % 2 == 0 else "cam_02",
            "time": "night"
        })

    # 3. Inference
    embeddings = model.encode(image_paths)
    preds = model.predict(embeddings)
    
    # 4. VIRK Hook
    # We pass paths so bundler can zip them if drift detected
    monitor.process_batch(embeddings, image_paths=image_paths, metadata=metadata)
    
    # Cleanup (In real app, maybe keep for bundling? 
    # Current Middleware runs synchronously for demo, preventing race condition on delete.
    # Ideally middleware is async/background task)
    # shutil.rmtree(temp_dir) 
    
    return {"batch_id": batch_id, "predictions": preds, "drift_status": "monitored"}

if __name__ == "__main__":
    import uvicorn
    print("Running Demo Service on port 8080...")
    print("Send requests with 'drift' in filename to trigger incidents.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
