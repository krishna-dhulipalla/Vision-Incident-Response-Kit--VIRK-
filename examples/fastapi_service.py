import time
import uuid
import shutil
import numpy as np
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# 1. Real ML Imports
try:
    import torch
    import timm
    from torchvision import transforms
except ImportError:
    print("Error: This demo requires 'timm', 'torch', and 'torchvision'.")
    print("pip install torch torchvision timm")
    exit(1)

# VIRK Imports
from virk.integration.middleware import VirkMiddleware

# --- Model Wrapper (Real) ---
class VisionService:
    def __init__(self):
        # Load a real, lightweight ResNet18
        print("Loading ResNet18 (pretrained)...")
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=0) # num_classes=0 gives embeddings
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_paths: List[str]) -> torch.Tensor:
        batch = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            batch.append(self.transform(img))
        return torch.stack(batch)

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        # VIRK specific: Takes paths, returns numpy embeddings
        # This is the "extractor" signature VIRK expects (or similar)
        # Note: VIRK doesn't enforce signature on extractor, but you need to call it yourself 
        # inside your inference loop, OR let VIRK call it if you pass it for fingerprinting.
        
        # For fingerprinting, VIRK expects: func(image_path) -> embedding
        # But here we use it for batch inference too.
        
        with torch.no_grad():
             tensors = self.preprocess(image_paths)
             features = self.model(tensors) # [B, 512]
        return features.numpy()

# --- Setup ---
app = FastAPI(title="VIRK Real-World Demo")
service = VisionService()

# 2. VIRK Initialization
print("Initializing VIRK Monitor...")

# Generate a synthetic baseline that MATCHES ResNet18 distribution (conceptually)
# In prod, load this from 'baseline_embeddings.npy'
# ResNet18 output is 512 dim.
ref_data = np.random.normal(0, 1, (500, 512)).astype(np.float32)

# Create Middleware
monitor = VirkMiddleware.create(
    reference_embeddings=ref_data,
    # Pass the extraction function. 
    # NOTE: VirkMiddleware only uses 'extractor' to generate reference embeddings 
    # if reference_paths provided OR for Fingerprinting analysis on drift.
    # It must handle a list of image paths/objects.
    extractor=service.extract_features, 
    output_dir="./incidents",
    metrics_type="prometheus",
    service_name="resnet-production-v1",
    async_processing=True 
)

@app.post("/predict")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Real inference pipeline: Use Torch + TIMM.
    """
    batch_id = str(uuid.uuid4())
    temp_dir = Path(f"temp_inference/{batch_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    try:
        # 1. Save Batch
        for file in files:
            p = temp_dir / file.filename
            with open(p, "wb") as f:
                shutil.copyfileobj(file.file, f)
            image_paths.append(str(p.absolute()))
        
        # 2. Inference (Get Embeddings)
        embeddings = service.extract_features(image_paths)
        
        # 3. Simulate Logic (Classification)
        # Just creating dummy predictions from embeddings for demo
        preds = [int(np.argmax(e[:10])) for e in embeddings]
        
        # 4. VIRK Hook (Non-blocking)
        # We assume metadata is available locally
        metadata = [{"source": "api", "filesize": Path(p).stat().st_size} for p in image_paths]
        
        monitor.process_batch(
            embeddings=embeddings, 
            image_paths=image_paths, 
            metadata=metadata
        )
        
        return {
            "batch_id": batch_id, 
            "predictions": preds, 
            "model": "resnet18",
            "monitoring": "active"
        }
    
    except Exception as e:
        return {"error": str(e)}
    # Note: We rely on VIRK async worker to read files, so we can't delete immediately here 
    # if we want 100% safety. In production, use S3 paths or a delayed cleanup job.
    # For this demo, VIRK's fingerprinting (if triggered) reads file. 
    # Middleware.process_batch is usually fast, but if async, the worker needs the file to exist.
    # Solution: Only delete after some time, or use a "Keep" policy.

if __name__ == "__main__":
    import uvicorn
    print("\n✅ Service Ready! Send images to http://localhost:8080/predict")
    print("Example: curl -X POST -F files=@dog.jpg http://localhost:8080/predict")
    uvicorn.run(app, host="0.0.0.0", port=8080)
